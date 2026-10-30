"""Tracing with the real Langfuse SDK, exporting spans to memory instead of the cloud."""

import json
from functools import partial
from itertools import count

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from chat.llm import ChatService, Completed, Retrieval, Source
from chat.models import Message
from chat.tracing import ReplyTrace, get_langfuse
from tests.fakes import (
    THINKING,
    FakeClient,
    Script,
    api_message,
    midstream_overloaded_error,
    overloaded_error,
)

pytestmark = [
    pytest.mark.django_db(transaction=True),
    pytest.mark.usefixtures("close_async_db_connections"),
]

_keys = count()


class Recorded:
    def __init__(self, client, exporter):
        self.client, self.exporter = client, exporter

    def spans(self):
        self.client.flush()
        return {s.name: s for s in self.exporter.get_finished_spans()}

    def all_spans(self):
        self.client.flush()
        return list(self.exporter.get_finished_spans())


@pytest.fixture
def langfuse():
    from langfuse import Langfuse

    exporter = InMemorySpanExporter()
    client = Langfuse(
        public_key=f"pk-test-{next(_keys)}",  # the SDK keeps one client per key
        secret_key="sk-test",
        base_url="http://localhost:9",  # never contacted: spans go to the exporter
        span_exporter=exporter,
        tracer_provider=TracerProvider(),
    )
    return Recorded(client, exporter)


SOURCE = Source(
    position=0,
    chunk_id=1,
    document_id=1,
    document_name="Policy.pdf",
    chunk_index=0,
    location="p. 2",
    text="Notice is two months.",
    score=0.5,
    vector_rank=1,
)


async def one_source(user_id, query):
    return Retrieval(sources=[SOURCE], embedding_model="voyage-4", embedding_tokens=7)


def service(langfuse, *scripts, **kwargs):
    return ChatService(
        FakeClient(*scripts),
        model="claude-opus-5-5",
        effort="low",
        retry_base_delay=0,
        retriever=one_source,
        tracer=partial(ReplyTrace, langfuse.client),
        **kwargs,
    )


async def reply(service, conversation):
    await Message.objects.acreate(conversation=conversation, role="user", content="Notice?")
    return [e async for e in service.stream_reply(conversation)]


def attr(span, name):
    value = span.attributes.get(f"langfuse.observation.{name}")
    return json.loads(value) if isinstance(value, str) and value[:1] in "[{" else value


async def test_reply_is_one_trace_with_retrieval_and_generation(langfuse, conversation):
    final = api_message("Two months.", input_tokens=1500, output_tokens=300, cache_read_tokens=1000)
    events = await reply(
        service(langfuse, Script([THINKING, "Two months."], final=final)), conversation
    )
    assert isinstance(events[-1], Completed)

    spans = langfuse.spans()
    root, retrieval, generation = spans["chat-reply"], spans["retrieval"], spans["claude"]
    assert {retrieval.context.trace_id, generation.context.trace_id} == {root.context.trace_id}
    assert retrieval.parent.span_id == generation.parent.span_id == root.context.span_id
    for span in spans.values():
        assert span.attributes["user.id"] == str(conversation.user_id)
        assert span.attributes["session.id"] == f"conversation-{conversation.pk}"

    assert attr(root, "input") == "Notice?"
    assert attr(root, "output") == "Two months."
    assert attr(retrieval, "type") == "retriever"
    assert attr(retrieval, "output")[0]["title"] == "Policy.pdf (p. 2)"
    assert attr(generation, "type") == "generation"
    assert attr(generation, "model.name") == "claude-opus-5-5"
    assert attr(generation, "usage_details") == {
        "input": 1500,
        "cache_read_input_tokens": 1000,
        "output": 300,
    }
    assert attr(generation, "cost_details") == {"total": 0.0122}  # + 1000 * $0.20 / 1M
    assert attr(generation, "completion_start_time")  # time to first token
    assert attr(generation, "input")["messages"][-1]["content"][0]["type"] == "search_result"


async def test_retried_attempt_is_a_failed_generation(langfuse, conversation):
    scripts = [Script([midstream_overloaded_error()]), Script(["Recovered"])]
    await reply(service(langfuse, *scripts), conversation)

    generations = [s for s in langfuse.all_spans() if s.name == "claude"]
    assert len(generations) == 2
    failed, ok = sorted(generations, key=lambda s: s.start_time)
    assert attr(failed, "level") == "ERROR"
    assert attr(ok, "level") is None
    assert attr(ok, "output") == "Recovered"


async def test_fallback_is_a_second_generation_with_the_other_model(langfuse, conversation):
    final = api_message("Two months.", model="claude-sonnet-5-5")
    scripts = [Script(open_error=overloaded_error()), Script(["Two months."], final=final)]
    await reply(service(langfuse, *scripts, fallback_model="claude-sonnet-5-5"), conversation)

    generations = sorted(
        (s for s in langfuse.all_spans() if s.name == "claude"), key=lambda s: s.start_time
    )
    assert [attr(g, "model.name") for g in generations] == ["claude-opus-5-5", "claude-sonnet-5-5"]
    assert [attr(g, "level") for g in generations] == ["ERROR", None]


async def test_rewrite_is_a_generation_before_retrieval(langfuse, conversation):
    from decimal import Decimal

    from chat.rewrite import Rewrite

    class Rewriter:
        model = "claude-haiku-5-5"

        async def rewrite(self, history, question):
            return Rewrite(
                query="Notice period?",
                rewritten=True,
                model=self.model,
                input_tokens=1000,
                output_tokens=40,
                cost_usd=Decimal("0.00012"),
            )

    await Message.objects.acreate(conversation=conversation, role="user", content="Policy?")
    await Message.objects.acreate(conversation=conversation, role="assistant", content="It...")
    await reply(service(langfuse, Script(["Two months."]), rewriter=Rewriter()), conversation)

    spans = langfuse.spans()
    rewrite, retrieval = spans["rewrite"], spans["retrieval"]
    assert rewrite.parent.span_id == spans["chat-reply"].context.span_id
    assert rewrite.end_time <= retrieval.start_time
    assert attr(rewrite, "type") == "generation"
    assert attr(rewrite, "model.name") == "claude-haiku-5-5"
    assert attr(rewrite, "output") == "Notice period?"
    assert attr(rewrite, "usage_details") == {"input": 1000, "output": 40}
    assert attr(rewrite, "cost_details") == {"total": 0.00012}
    assert attr(retrieval, "input") == "Notice period?"


async def test_content_capture_can_be_turned_off(langfuse, conversation, settings):
    settings.LANGFUSE_CAPTURE_CONTENT = False
    await reply(service(langfuse, Script(["Two months."])), conversation)

    for span in langfuse.all_spans():
        assert attr(span, "input") is None
        assert attr(span, "output") is None
    assert attr(langfuse.spans()["claude"], "usage_details")  # numbers are still sent


async def test_tracing_errors_never_break_the_reply(conversation):
    class BrokenClient:
        def start_observation(self, **kwargs):
            raise RuntimeError("langfuse is down")

    chat = ChatService(
        FakeClient(Script(["Two months."])),
        retriever=one_source,
        tracer=partial(ReplyTrace, BrokenClient()),
    )
    events = await reply(chat, conversation)
    assert events[-1].message.content == "Two months."


def test_tracing_is_off_without_keys():
    assert get_langfuse() is None  # the autouse fixture cleared the keys
