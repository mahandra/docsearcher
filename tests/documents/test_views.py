import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

from documents import tasks
from documents.models import Chunk, Document
from tests.documents.factories import make_docx, make_pdf

pytestmark = pytest.mark.django_db


@pytest.fixture
def auth_client(client, user):
    client.force_login(user)
    return client


@pytest.fixture
def queued(monkeypatch):
    calls = []
    monkeypatch.setattr(tasks.process_document, "delay", calls.append)
    return calls


def upload(client, data: bytes, name: str, captured):
    with captured(execute=True):
        return client.post(
            reverse("documents:document_upload"),
            {"file": SimpleUploadedFile(name, data)},
            HTTP_HX_REQUEST="true",
        )


def test_upload_pdf_creates_document_and_queues_processing(
    auth_client, user, queued, django_capture_on_commit_callbacks, media_root
):
    response = upload(
        auth_client, make_pdf([["hi"]]), "My Report.pdf", django_capture_on_commit_callbacks
    )

    assert response.status_code == 200
    document = Document.objects.get(user=user)
    assert document.original_name == "My Report.pdf"
    assert document.status == Document.Status.UPLOADED
    assert queued == [document.pk]
    assert (media_root / document.file.name).exists()
    # The new row is swapped in out-of-band and polls for status updates.
    assert 'hx-swap-oob="afterbegin:#document-rows"' in response.text
    assert 'hx-trigger="every 2s"' in response.text


def test_upload_docx(auth_client, user, queued, django_capture_on_commit_callbacks):
    upload(auth_client, make_docx([("p", "x")]), "a.docx", django_capture_on_commit_callbacks)
    assert Document.objects.filter(user=user, original_name="a.docx").exists()


@pytest.mark.parametrize(
    ("data", "name", "error"),
    [
        (b"plain text", "notes.txt", "Only PDF and DOCX"),
        (b"MZ\x90\x00 executable", "invoice.pdf", "does not match"),
        (make_docx([("p", "x")]), "renamed.pdf", "does not match"),
    ],
)
def test_upload_rejects_bad_files(
    auth_client, queued, django_capture_on_commit_callbacks, data, name, error
):
    response = upload(auth_client, data, name, django_capture_on_commit_callbacks)

    assert error in response.text
    assert not Document.objects.exists()
    assert queued == []


def test_upload_rejects_too_large_file(
    auth_client, settings, queued, django_capture_on_commit_callbacks
):
    settings.DOCUMENT_MAX_UPLOAD_MB = 0
    response = upload(auth_client, make_pdf([["hi"]]), "a.pdf", django_capture_on_commit_callbacks)

    assert "larger than 0 MB" in response.text
    assert not Document.objects.exists()


def test_upload_rejects_duplicate(auth_client, queued, django_capture_on_commit_callbacks):
    data = make_pdf([["same"]])
    upload(auth_client, data, "a.pdf", django_capture_on_commit_callbacks)

    response = upload(auth_client, data, "copy.pdf", django_capture_on_commit_callbacks)

    assert "already uploaded" in response.text
    assert Document.objects.count() == 1
    assert len(queued) == 1


def test_list_shows_only_own_documents(auth_client, make_document, django_user_model):
    bob = django_user_model.objects.create_user(username="bob", password="pw-bob-123")
    make_document(b"%PDF mine", name="mine.pdf")
    make_document(b"%PDF bobs", name="bobs-secret.pdf", owner=bob)

    response = auth_client.get(reverse("documents:document_list"))

    assert "mine.pdf" in response.text
    assert "bobs-secret.pdf" not in response.text


def test_row_polls_until_final_status(auth_client, make_document):
    document = make_document(status=Document.Status.PROCESSING)
    url = reverse("documents:document_row", args=[document.pk])
    assert 'hx-trigger="every 2s"' in auth_client.get(url).text

    Document.objects.filter(pk=document.pk).update(status=Document.Status.READY)
    assert "hx-trigger" not in auth_client.get(url).text  # polling stops


def test_detail_shows_chunks_with_location(auth_client, make_document):
    document = make_document(status=Document.Status.READY, chunk_count=1)
    Chunk.objects.create(
        document=document,
        index=0,
        text="Chunk <b>text</b>",
        page_start=2,
        page_end=3,
        token_count=4,
        embedding=[0.5] * 1024,
    )

    response = auth_client.get(reverse("documents:document_detail", args=[document.pk]))

    assert "pp. 2–3" in response.text
    assert "Chunk &lt;b&gt;text&lt;/b&gt;" in response.text
    assert "1024 values" in response.text


def test_delete_removes_document_and_file(
    auth_client, make_document, media_root, django_capture_on_commit_callbacks
):
    document = make_document()
    path = media_root / document.file.name
    assert path.exists()

    with django_capture_on_commit_callbacks(execute=True):
        auth_client.post(reverse("documents:document_delete", args=[document.pk]))

    assert not Document.objects.exists()
    assert not path.exists()


def test_reprocess_failed_document(
    auth_client, make_document, queued, django_capture_on_commit_callbacks
):
    document = make_document(status=Document.Status.FAILED, error="boom")

    with django_capture_on_commit_callbacks(execute=True):
        auth_client.post(reverse("documents:document_reprocess", args=[document.pk]))

    document.refresh_from_db()
    assert document.status == Document.Status.UPLOADED
    assert document.error == ""
    assert queued == [document.pk]


def test_ready_document_cannot_be_reprocessed(auth_client, make_document, queued):
    document = make_document(status=Document.Status.READY)
    url = reverse("documents:document_reprocess", args=[document.pk])
    assert auth_client.post(url).status_code == 404


def test_other_users_document_is_404(client, make_document, django_user_model):
    document = make_document()
    bob = django_user_model.objects.create_user(username="bob", password="pw-bob-123")
    client.force_login(bob)

    for name in ("document_detail", "document_row"):
        assert client.get(reverse(f"documents:{name}", args=[document.pk])).status_code == 404
    for name in ("document_delete", "document_reprocess"):
        assert client.post(reverse(f"documents:{name}", args=[document.pk])).status_code == 404
    assert Document.objects.exists()


def test_documents_require_login(client):
    response = client.get(reverse("documents:document_list"))
    assert response.status_code == 302


def test_delete_survives_file_removal_error(
    auth_client, make_document, monkeypatch, django_capture_on_commit_callbacks
):
    document = make_document()

    def broken_delete(name):
        raise PermissionError("read-only volume")

    monkeypatch.setattr(document.file.storage, "delete", broken_delete)

    with django_capture_on_commit_callbacks(execute=True):
        response = auth_client.post(reverse("documents:document_delete", args=[document.pk]))

    assert response.status_code == 302
    assert not Document.objects.exists()


# --- Search playground ----------------------------------------------------------------


@pytest.fixture
def fake_embedder(fake_query_embedder):
    return fake_query_embedder


def test_search_page_requires_login(client):
    response = client.get(reverse("documents:search_debug"))
    assert response.status_code == 302


def test_search_page_without_query_does_not_embed(auth_client, fake_embedder):
    response = auth_client.get(reverse("documents:search_debug"))

    assert response.status_code == 200
    assert fake_embedder.queries == []


def test_search_page_shows_ranked_candidates_and_top_k_cutoff(
    auth_client, user, make_document, fake_embedder, settings
):
    settings.RETRIEVAL_TOP_K = 1
    document = make_document(status=Document.Status.READY)
    for i, text in enumerate(["The notice period is one month.", "Notice must be written."]):
        Chunk.objects.create(
            document=document, index=i, text=text, token_count=5, embedding=[0.1] * 1024
        )

    response = auth_client.get(reverse("documents:search_debug"), {"q": "notice period"})

    assert fake_embedder.queries == ["notice period"]
    content = response.content.decode()
    assert content.index("The notice period is one month.") < content.index("top 1 go into")
    assert content.index("top 1 go into") < content.index("Notice must be written.")
    assert "#chunk-0" in content
