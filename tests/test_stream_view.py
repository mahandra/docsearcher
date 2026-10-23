import json

import pytest
from django.test import AsyncClient
from django.urls import reverse
from django.utils import timezone

from chat import views
from chat.llm import ChatService
from chat.models import Conversation, Message
from tests.fakes import THINKING, FakeClient, Script, rate_limit_error

pytestmark = [
    pytest.mark.django_db(transaction=True),
    pytest.mark.usefixtures("close_async_db_connections"),
]


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


@pytest.fixture
def fake_llm(monkeypatch):
    """Point the view at a ChatService backed by scripted fake streams."""

    def install(*scripts):
        client = FakeClient(*scripts)
        service = ChatService(client, model="claude-opus-5-5", effort="low", retry_base_delay=0)
        monkeypatch.setattr(views, "get_chat_service", lambda: service)
        return client

    return install


@pytest.fixture
async def client(user):
    c = AsyncClient()
    await c.aforce_login(user)
    return c


async def read_stream(response) -> str:
    return "".join([chunk.decode() async for chunk in response.streaming_content])


def stream_url(conversation):
    return reverse("chat:message_stream", args=[conversation.pk])


async def test_streams_tokens_then_done(client, conversation, fake_llm):
    await Message.objects.acreate(conversation=conversation, role="user", content="Hi")
    fake_llm(Script([THINKING, "Hel", "lo"]))

    response = await client.get(stream_url(conversation))

    assert response.status_code == 200
    assert response["Content-Type"] == "text/event-stream"
    events = parse_sse(await read_stream(response))
    assert [name for name, _ in events] == ["thinking", "token", "token", "done"]
    assert "".join(d["text"] for name, d in events if name == "token") == "Hello"
    done = events[-1][1]
    assert done["status"] == "complete"
    assert done["input_tokens"] == 100 and done["output_tokens"] == 20
    assert done["cost_usd"] == "0.000800"

    await conversation.arefresh_from_db()
    assert conversation.generation_started_at is None  # lock released


async def test_failure_is_sent_as_failed_event(client, conversation, fake_llm):
    await Message.objects.acreate(conversation=conversation, role="user", content="Hi")
    fake_llm(Script(open_error=rate_limit_error()))

    response = await client.get(stream_url(conversation))

    events = parse_sse(await read_stream(response))
    assert events[-1][0] == "failed"
    assert "Too many requests" in events[-1][1]["error"]


async def test_no_generation_when_already_answered(client, conversation, fake_llm):
    await Message.objects.acreate(conversation=conversation, role="user", content="Hi")
    await Message.objects.acreate(conversation=conversation, role="assistant", content="Hello")
    fake = fake_llm()

    response = await client.get(stream_url(conversation))

    # 204 makes EventSource stop reconnecting; nothing is generated (and nothing is billed).
    assert response.status_code == 204
    assert fake.messages.calls == []


async def test_no_duplicate_generation_while_one_is_running(client, conversation, fake_llm):
    await Message.objects.acreate(conversation=conversation, role="user", content="Hi")
    await Conversation.objects.filter(pk=conversation.pk).aupdate(
        generation_started_at=timezone.now()
    )
    fake = fake_llm()

    response = await client.get(stream_url(conversation))

    assert response.status_code == 204
    assert fake.messages.calls == []


async def test_stale_generation_lock_is_taken_over(client, conversation, fake_llm):
    await Message.objects.acreate(conversation=conversation, role="user", content="Hi")
    stale = timezone.now() - views.GENERATION_LOCK_TTL * 2
    await Conversation.objects.filter(pk=conversation.pk).aupdate(generation_started_at=stale)
    fake_llm(Script(["ok"]))

    response = await client.get(stream_url(conversation))

    assert response.status_code == 200
    await read_stream(response)


async def test_other_users_conversation_is_not_found(conversation, django_user_model, fake_llm):
    await Message.objects.acreate(conversation=conversation, role="user", content="Hi")
    bob = await django_user_model.objects.acreate_user(username="bob", password="pw-bob-123")
    c = AsyncClient()
    await c.aforce_login(bob)

    response = await c.get(stream_url(conversation))

    assert response.status_code == 404


async def test_anonymous_user_is_rejected(conversation):
    response = await AsyncClient().get(stream_url(conversation))
    assert response.status_code == 401
