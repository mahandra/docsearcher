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


def test_uncited_answer_carries_a_warning(auth_client, conversation):
    Message.objects.create(conversation=conversation, role="user", content="Hi")
    Message.objects.create(conversation=conversation, role="assistant", content="Two months.")
    url = reverse("chat:conversation_detail", args=[conversation.pk])
    assert "No citations" not in auth_client.get(url).text

    Message.objects.filter(role="assistant").update(uncited=True)
    assert "No citations: check this answer" in auth_client.get(url).text


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


def test_detail_renders_citation_markers_and_footnotes(auth_client, conversation, user):
    from chat.models import MessageSource
    from documents.models import Document

    document = Document.objects.create(
        user=user, original_name="contract.pdf", sha256="x", size_bytes=1, status="ready"
    )
    Message.objects.create(conversation=conversation, role="user", content="Notice?")
    answer = Message.objects.create(
        conversation=conversation,
        role="assistant",
        content="Two months, in writing.",
        # Cites position 3 first, then 0: footnotes are numbered by first appearance.
        citations=[
            {"offset": 10, "source": 3, "cited_text": "Notice is two months."},
            {"offset": 23, "source": 0, "cited_text": "<b>In writing</b>"},
        ],
    )
    for position, doc in [(0, None), (3, document)]:
        MessageSource.objects.create(
            message=answer,
            position=position,
            document=doc,
            document_name="old.docx" if doc is None else "contract.pdf",
            chunk_index=5,
            location="p. 3",
            score=0.1,
            cited=True,
        )

    html = auth_client.get(reverse("chat:conversation_detail", args=[conversation.pk])).text

    content = (
        f'Two months<a class="cite" href="/documents/{document.pk}/#chunk-5" '
        'title="Notice is two months.">[1]</a>, in writing.'
        '<span class="cite" title="&lt;b&gt;In writing&lt;/b&gt;">[2]</span>'
    )
    assert content in html
    assert html.index("contract.pdf</a> · p. 3") < html.index("old.docx")
    assert "(deleted)" in html
