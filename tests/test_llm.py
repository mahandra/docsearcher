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
    cached,
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
    overloaded_error,
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


async def history(conversation, *contents):
    """Alternating user/assistant messages, starting with the user."""
    for i, content in enumerate(contents):
        role = "user" if i % 2 == 0 else "assistant"
        await Message.objects.acreate(conversation=conversation, role=role, content=content)


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
    assert params["system"] == cached(load_prompt("system"))
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


async def test_cache_breakpoints_on_system_prompt_and_last_history_message(conversation):
    await history(conversation, "First?", "First answer.", "Second?", "Second answer.", "Third?")
    service, client = make_service(Script(["ok"]))

    await collect(service, conversation)

    params = client.messages.calls[0]
    assert params["system"][0]["cache_control"] == {"type": "ephemeral"}
    messages = params["messages"]
    # Older history stays plain; the next request repeats it all as a prefix.
    assert messages[:3] == [
        {"role": "user", "content": "First?"},
        {"role": "assistant", "content": "First answer."},
        {"role": "user", "content": "Second?"},
    ]
    assert messages[3] == {"role": "assistant", "content": cached("Second answer.")}
    # The latest turn (search results + question) never recurs: no breakpoint.
    assert "cache_control" not in str(messages[4])


async def test_prompt_caching_can_be_turned_off(conversation):
    await history(conversation, "First?", "First answer.", "Second?")
    service, client = make_service(Script(["ok"]), prompt_caching=False)

    await collect(service, conversation)

    params = client.messages.calls[0]
    assert params["system"] == load_prompt("system")
    assert "cache_control" not in str(params["messages"])


async def test_cache_usage_is_saved_and_priced(conversation):
    await Message.objects.acreate(conversation=conversation, role="user", content="Hi")
    final = api_message(
        "Hi", input_tokens=100, output_tokens=300, cache_write_tokens=1000, cache_read_tokens=4000
    )
    service, _ = make_service(Script(["Hi"], final=final))

    events = await collect(service, conversation)

    saved = events[-1].message
    assert (saved.cache_write_tokens, saved.cache_read_tokens) == (1000, 4000)
    assert saved.prompt_tokens == 5100
    # 100*$4 + 1000*$5 (write) + 4000*$0.20 (read) + 300*$20, per 1M
    assert saved.cost_usd == Decimal("0.012200")


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


# --- Model fallback -------------------------------------------------------------------


async def test_falls_back_to_another_model_when_the_main_one_stays_overloaded(conversation):
    await Message.objects.acreate(conversation=conversation, role="user", content="Hi")
    fallback = api_message("Hello", input_tokens=1500, output_tokens=300, model="claude-sonnet-5-5")
    service, client = make_service(
        Script(open_error=overloaded_error()),
        Script(["Hello"], final=fallback),
        fallback_model="claude-sonnet-5-5",
    )

    events = await collect(service, conversation)

    assert [c["model"] for c in client.messages.calls] == ["claude-opus-5-5", "claude-sonnet-5-5"]
    # Everything but the model is the same request.
    assert {**client.messages.calls[1], "model": "claude-opus-5-5"} == client.messages.calls[0]
    saved = events[-1].message
    assert isinstance(events[-1], Completed)
    assert saved.model == "claude-sonnet-5-5"
    assert saved.cost_usd == Decimal("0.006000")  # priced as Sonnet: 1500*$2 + 300*$10, per 1M


async def test_falls_back_after_mid_stream_retries_are_used_up(conversation):
    await Message.objects.acreate(conversation=conversation, role="user", content="Hi")
    service, client = make_service(
        Script([midstream_overloaded_error()]),
        Script([midstream_overloaded_error()]),
        Script(["Hello"]),
        max_stream_retries=1,
        fallback_model="claude-sonnet-5-5",
    )

    events = await collect(service, conversation)

    models = [c["model"] for c in client.messages.calls]
    assert models == ["claude-opus-5-5", "claude-opus-5-5", "claude-sonnet-5-5"]
    assert isinstance(events[-1], Completed)


async def test_no_fallback_for_errors_another_model_would_repeat(conversation):
    await Message.objects.acreate(conversation=conversation, role="user", content="Hi")
    service, client = make_service(
        Script(open_error=auth_error()), fallback_model="claude-sonnet-5-5"
    )

    events = await collect(service, conversation)

    assert len(client.messages.calls) == 1
    assert isinstance(events[-1], Failed)


async def test_no_fallback_once_text_was_streamed(conversation):
    await Message.objects.acreate(conversation=conversation, role="user", content="Hi")
    service, client = make_service(
        Script(["Partial", midstream_overloaded_error()]), fallback_model="claude-sonnet-5-5"
    )

    events = await collect(service, conversation)

    assert len(client.messages.calls) == 1
    assert events[-1].message.status == Message.Status.INTERRUPTED


async def test_falls_back_only_once(conversation):
    await Message.objects.acreate(conversation=conversation, role="user", content="Hi")
    service, client = make_service(
        Script(open_error=overloaded_error()),
        Script(open_error=overloaded_error()),
        fallback_model="claude-sonnet-5-5",
    )

    events = await collect(service, conversation)

    assert len(client.messages.calls) == 2
    assert isinstance(events[-1], Failed)
    assert events[-1].message.model == "claude-sonnet-5-5"  # the model that failed last


async def test_fallback_is_off_without_a_model_or_when_it_is_the_main_one(conversation):
    service, _ = make_service(fallback_model="")
    assert service.fallback_model == ""
    service, _ = make_service(fallback_model="claude-opus-5-5")
    assert service.fallback_model == ""
