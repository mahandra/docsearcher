from decimal import Decimal

import pytest

from chat.llm import (
    NO_SOURCES_NOTE,
    ChatService,
    Completed,
    Failed,
    SourcesFound,
    TextDelta,
    ThinkingStarted,
    build_messages,
    load_prompt,
)
from chat.models import Message
from tests.fakes import (
    THINKING,
    FakeClient,
    Script,
    api_message,
    auth_error,
    midstream_overloaded_error,
    rate_limit_error,
)

# Async ORM calls run in a worker thread, outside the test's transaction,
# so these tests need real commits (transaction=True).
pytestmark = [
    pytest.mark.django_db(transaction=True),
    pytest.mark.usefixtures("close_async_db_connections"),
]


def add(conversation, role, content, **kwargs):
    return Message.objects.create(conversation=conversation, role=role, content=content, **kwargs)


def make_service(*scripts, **kwargs):
    client = FakeClient(*scripts)
    kwargs.setdefault("model", "claude-opus-5-5")
    kwargs.setdefault("effort", "low")
    kwargs.setdefault("retry_base_delay", 0)
    return ChatService(client, **kwargs), client


async def collect(service, conversation):
    return [event async for event in service.stream_reply(conversation)]


# --- build_messages -----------------------------------------------------------------


async def test_build_messages_maps_history_in_order(conversation):
    await Message.objects.acreate(conversation=conversation, role="user", content="Hi")
    await Message.objects.acreate(conversation=conversation, role="assistant", content="Hello!")
    await Message.objects.acreate(conversation=conversation, role="user", content="How are you?")

    assert await build_messages(conversation) == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "How are you?"},
    ]


async def test_build_messages_window_never_starts_with_assistant(conversation):
    for i in range(3):
        await Message.objects.acreate(conversation=conversation, role="user", content=f"q{i}")
        await Message.objects.acreate(conversation=conversation, role="assistant", content=f"a{i}")
    await Message.objects.acreate(conversation=conversation, role="user", content="q3")

    # Last 4 messages are a1, q2, a2, q3 — the leading assistant turn is dropped.
    assert await build_messages(conversation, max_messages=4) == [
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "q3"},
    ]


async def test_build_messages_skips_failed_replies_and_merges_user_turns(conversation):
    await Message.objects.acreate(conversation=conversation, role="user", content="first")
    await Message.objects.acreate(
        conversation=conversation, role="assistant", content="", status="error"
    )
    await Message.objects.acreate(conversation=conversation, role="user", content="retry")

    assert await build_messages(conversation) == [
        {"role": "user", "content": "first\n\nretry"},
    ]


# --- stream_reply: happy path ------------------------------------------------------


async def test_stream_reply_streams_text_and_saves_usage_and_cost(conversation):
    await Message.objects.acreate(conversation=conversation, role="user", content="Hi")
    final = api_message("Hello there", input_tokens=1500, output_tokens=300)
    service, client = make_service(Script([THINKING, "Hello", " there"], final=final))

    events = await collect(service, conversation)

    assert isinstance(events[0], SourcesFound)  # retrieval runs first
    assert isinstance(events[1], ThinkingStarted)
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["Hello", " there"]
    assert isinstance(events[-1], Completed)
    saved = events[-1].message
    await saved.arefresh_from_db()
    assert saved.role == "assistant"
    assert saved.content == "Hello there"
    assert saved.status == Message.Status.COMPLETE
    assert saved.stop_reason == "end_turn"
    assert (saved.input_tokens, saved.output_tokens) == (1500, 300)
    assert saved.cost_usd == Decimal("0.012000")  # 1500*$4/1M + 300*$20/1M
    assert client.messages.streams[0].closed


async def test_stream_reply_sends_system_prompt_history_and_config(conversation):
    await Message.objects.acreate(conversation=conversation, role="user", content="Hi")
    service, client = make_service(Script(["ok"]), max_tokens=1234)

    await collect(service, conversation)

    params = client.messages.calls[0]
    assert params["model"] == "claude-opus-5-5"
    assert params["max_tokens"] == 1234
    assert params["system"] == load_prompt("system")
    # No documents: the model is told that the search found nothing.
    assert params["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": NO_SOURCES_NOTE},
                {"type": "text", "text": "Hi"},
            ],
        }
    ]
    assert params["output_config"] == {"effort": "low"}
    assert "temperature" not in params


async def test_effort_is_omitted_when_empty(conversation):
    await Message.objects.acreate(conversation=conversation, role="user", content="Hi")
    service, client = make_service(Script(["ok"]), effort="")

    await collect(service, conversation)

    assert "output_config" not in client.messages.calls[0]


async def test_truncated_answer_records_max_tokens_stop_reason(conversation):
    await Message.objects.acreate(conversation=conversation, role="user", content="Hi")
    final = api_message("cut", stop_reason="max_tokens")
    service, _ = make_service(Script(["cut"], final=final))

    events = await collect(service, conversation)

    assert events[-1].message.stop_reason == "max_tokens"


async def test_requires_trailing_user_message(conversation):
    service, _ = make_service()
    with pytest.raises(ValueError):
        await collect(service, conversation)


# --- stream_reply: errors and retries ----------------------------------------------


async def test_error_when_opening_stream_saves_error_message(conversation):
    await Message.objects.acreate(conversation=conversation, role="user", content="Hi")
    # The SDK has already retried this rate limit max_retries times before raising.
    service, client = make_service(Script(open_error=rate_limit_error()))

    events = await collect(service, conversation)

    assert len(events) == 2 and isinstance(events[1], Failed)
    assert "Too many requests" in events[1].error
    assert events[1].message.status == Message.Status.ERROR
    assert events[1].message.content == ""
    # Not retried again by us: the SDK owns retries of the initial request.
    assert len(client.messages.calls) == 1


async def test_non_retryable_error_is_not_retried(conversation):
    await Message.objects.acreate(conversation=conversation, role="user", content="Hi")
    service, client = make_service(Script([auth_error()]))

    events = await collect(service, conversation)

    assert isinstance(events[-1], Failed)
    assert "API key" in events[-1].error
    assert len(client.messages.calls) == 1


async def test_midstream_overload_before_any_text_is_retried(conversation):
    await Message.objects.acreate(conversation=conversation, role="user", content="Hi")
    service, client = make_service(
        Script([THINKING, midstream_overloaded_error()]),
        Script(["Recovered"]),
    )

    events = await collect(service, conversation)

    assert len(client.messages.calls) == 2
    assert isinstance(events[-1], Completed)
    assert events[-1].message.content == "Recovered"
    assert await Message.objects.filter(role="assistant").acount() == 1


async def test_retries_are_bounded(conversation):
    await Message.objects.acreate(conversation=conversation, role="user", content="Hi")
    scripts = [Script([midstream_overloaded_error()]) for _ in range(3)]
    service, client = make_service(*scripts, max_stream_retries=2)

    events = await collect(service, conversation)

    assert len(client.messages.calls) == 3  # first try + 2 retries
    assert isinstance(events[-1], Failed)
    assert events[-1].message.status == Message.Status.ERROR


async def test_midstream_error_after_text_is_not_retried_and_keeps_partial(conversation):
    await Message.objects.acreate(conversation=conversation, role="user", content="Hi")
    service, client = make_service(
        Script(["Partial", " answer", midstream_overloaded_error()], input_tokens=200),
    )

    events = await collect(service, conversation)

    assert len(client.messages.calls) == 1
    failed = events[-1]
    assert isinstance(failed, Failed)
    assert failed.message.status == Message.Status.INTERRUPTED
    assert failed.message.content == "Partial answer"
    # Usage is taken from the stream snapshot: we still pay for what was generated.
    assert (failed.message.input_tokens, failed.message.output_tokens) == (200, 2)


async def test_client_disconnect_saves_partial_answer_and_closes_stream(conversation):
    await Message.objects.acreate(conversation=conversation, role="user", content="Hi")
    service, client = make_service(Script(["One", " two", " three"]))

    gen = service.stream_reply(conversation)
    assert isinstance(await anext(gen), SourcesFound)
    assert (await anext(gen)).text == "One"
    await gen.aclose()  # what Django does when the browser goes away

    saved = await Message.objects.aget(role="assistant")
    assert saved.status == Message.Status.INTERRUPTED
    assert saved.content == "One"
    assert client.messages.streams[0].closed
