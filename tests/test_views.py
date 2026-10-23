import pytest
from django.urls import reverse

from chat.models import Conversation, Message

pytestmark = pytest.mark.django_db


@pytest.fixture
def auth_client(client, user):
    client.force_login(user)
    return client


def test_pages_require_login(client, conversation):
    for url in (
        reverse("chat:conversation_list"),
        reverse("chat:conversation_detail", args=[conversation.pk]),
    ):
        response = client.get(url)
        assert response.status_code == 302
        assert response.url.startswith(reverse("login"))


def test_list_shows_only_own_conversations(auth_client, conversation, django_user_model):
    bob = django_user_model.objects.create_user(username="bob", password="pw-bob-123")
    Conversation.objects.create(user=bob, title="Bob's secret")

    response = auth_client.get(reverse("chat:conversation_list"))

    assert "Test" in response.text
    assert "Bob&#x27;s secret" not in response.text


def test_create_conversation_redirects_to_it(auth_client, user):
    response = auth_client.post(reverse("chat:conversation_create"))

    conversation = Conversation.objects.get(user=user)
    assert response.url == reverse("chat:conversation_detail", args=[conversation.pk])


def test_message_create_returns_message_and_stream_placeholder(auth_client, user):
    conversation = Conversation.objects.create(user=user)
    url = reverse("chat:message_create", args=[conversation.pk])

    response = auth_client.post(url, {"content": "  What is RAG?  "})

    assert response.status_code == 200
    assert "What is RAG?" in response.text
    stream_url = reverse("chat:message_stream", args=[conversation.pk])
    assert f'data-stream-url="{stream_url}"' in response.text
    message = Message.objects.get(conversation=conversation)
    assert (message.role, message.content) == ("user", "What is RAG?")
    conversation.refresh_from_db()
    assert conversation.title == "What is RAG?"


def test_empty_message_is_ignored(auth_client, conversation):
    url = reverse("chat:message_create", args=[conversation.pk])

    response = auth_client.post(url, {"content": "   "})

    assert response.status_code == 204
    assert not Message.objects.exists()


def test_detail_renders_placeholder_only_when_reply_is_missing(auth_client, conversation):
    url = reverse("chat:conversation_detail", args=[conversation.pk])
    Message.objects.create(conversation=conversation, role="user", content="Hi")
    assert "data-stream-url" in auth_client.get(url).text

    Message.objects.create(
        conversation=conversation,
        role="assistant",
        content="Hello",
        model="claude-opus-5-5",
        input_tokens=10,
        output_tokens=5,
    )
    response = auth_client.get(url)
    assert "data-stream-url" not in response.text
    assert "10 in / 5 out" in response.text


def test_llm_output_is_html_escaped(auth_client, conversation):
    Message.objects.create(conversation=conversation, role="user", content="Hi")
    Message.objects.create(
        conversation=conversation, role="assistant", content="<script>alert(1)</script>"
    )

    response = auth_client.get(reverse("chat:conversation_detail", args=[conversation.pk]))

    assert "<script>alert(1)</script>" not in response.text
    assert "&lt;script&gt;" in response.text


def test_other_users_conversation_is_404(client, conversation, django_user_model):
    bob = django_user_model.objects.create_user(username="bob", password="pw-bob-123")
    client.force_login(bob)

    for name in ("conversation_detail", "message_create", "conversation_delete"):
        url = reverse(f"chat:{name}", args=[conversation.pk])
        method = client.get if name == "conversation_detail" else client.post
        assert method(url, {"content": "x"}).status_code == 404


def test_delete_conversation(auth_client, conversation):
    auth_client.post(reverse("chat:conversation_delete", args=[conversation.pk]))
    assert not Conversation.objects.exists()


def test_signup_creates_user_and_logs_in(client, django_user_model):
    response = client.post(
        reverse("chat:signup"),
        {"username": "carol", "password1": "a-long-pass-123", "password2": "a-long-pass-123"},
    )

    assert response.url == reverse("chat:conversation_list")
    assert django_user_model.objects.filter(username="carol").exists()
    assert client.get(reverse("chat:conversation_list")).status_code == 200
