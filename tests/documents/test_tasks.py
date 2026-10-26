import pytest
from celery.exceptions import Retry

from documents import tasks
from documents.embeddings import EmbeddingAuthError, EmbeddingError, TransientEmbeddingError
from documents.models import Chunk, Document
from tests.documents.factories import make_docx, make_pdf
from tests.documents.fakes import FakeEmbedder

pytestmark = pytest.mark.django_db


@pytest.fixture
def embedder(monkeypatch):
    def install(*errors):
        fake = FakeEmbedder(errors=list(errors))
        monkeypatch.setattr(tasks, "get_embedder", lambda: fake)
        return fake

    return install


@pytest.fixture(autouse=True)
def small_chunks(settings):
    settings.CHUNK_SIZE_TOKENS = 60
    settings.CHUNK_OVERLAP_TOKENS = 10


def long_pdf(pages: int = 3) -> bytes:
    return make_pdf(
        [
            [f"Page {p} line {i} with a few more words to fill it." for i in range(20)]
            for p in range(1, pages + 1)
        ]
    )


def run(document):
    tasks.process_document.apply(args=[document.pk], throw=True)
    document.refresh_from_db()
    return document


def test_pdf_is_parsed_chunked_embedded_and_stored(make_document, embedder):
    fake = embedder()
    document = make_document(long_pdf(), name="report.pdf")

    document = run(document)

    assert document.status == Document.Status.READY
    assert document.page_count == 3
    chunks = list(document.chunks.all())
    assert len(chunks) == document.chunk_count > 3
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert chunks[0].page_start == 1 and chunks[-1].page_end == 3
    assert len(chunks[0].embedding) == 1024
    assert fake.calls == [[c.text for c in chunks]]
    assert document.embedding_model == "voyage-4"
    assert document.embedding_tokens > 0
    assert document.embedding_cost_usd is not None
    assert document.processed_at is not None


def test_docx_chunks_carry_sections(make_document, embedder):
    embedder()
    data = make_docx([("h1", "Введение"), ("p", "Текст."), ("h1", "Results"), ("p", "Data.")])
    document = run(make_document(data, name="paper.docx"))

    assert [c.section for c in document.chunks.all()] == ["Введение", "Results"]


def test_processing_twice_does_not_duplicate_or_rebill(make_document, embedder):
    fake = embedder()
    document = run(make_document(long_pdf()))
    count = document.chunk_count

    run(document)  # e.g. redelivered after a worker crash

    assert Chunk.objects.filter(document=document).count() == count
    assert len(fake.calls) == 1


def test_reprocessing_failed_document_replaces_chunks(make_document, embedder):
    embedder()
    document = make_document(long_pdf())
    document.status = Document.Status.FAILED
    document.save()
    Chunk.objects.create(
        document=document, index=0, text="stale", token_count=1, embedding=[0.0] * 1024
    )

    document = run(document)

    assert not Chunk.objects.filter(text="stale").exists()
    assert document.status == Document.Status.READY


def test_scanned_pdf_fails_with_message(make_document, embedder):
    fake = embedder()
    document = run(make_document(make_pdf([[]]), name="scan.pdf"))

    assert document.status == Document.Status.FAILED
    assert "scanned" in document.error
    assert fake.calls == []  # nothing was sent (or billed)


def test_unsupported_file(make_document, embedder):
    embedder()
    document = run(make_document(b"just text", name="notes.pdf"))

    assert document.status == Document.Status.FAILED
    assert document.error == "Unsupported file type."


def test_transient_embedding_error_schedules_retry(make_document, embedder):
    embedder(TransientEmbeddingError("rate limited"))
    document = make_document(long_pdf())

    with pytest.raises(Retry):
        tasks.process_document.apply(args=[document.pk], throw=True)

    document.refresh_from_db()
    assert document.status == Document.Status.PROCESSING  # not failed yet


def test_transient_error_recovers_on_retry(make_document, embedder, settings):
    # In eager mode Celery executes retries immediately and in-process.
    fake = embedder(TransientEmbeddingError("blip"))
    document = make_document(long_pdf())
    settings.CELERY_TASK_EAGER_PROPAGATES = False

    tasks.process_document.apply(args=[document.pk])

    document.refresh_from_db()
    assert document.status == Document.Status.READY
    assert len(fake.calls) == 2


def test_gives_up_after_max_retries(make_document, embedder):
    errors = [TransientEmbeddingError("down")] * (tasks.MAX_TASK_RETRIES + 1)
    embedder(*errors)
    document = make_document(long_pdf())

    tasks.process_document.apply(args=[document.pk])

    document.refresh_from_db()
    assert document.status == Document.Status.FAILED
    assert "unavailable" in document.error


def test_permanent_embedding_error_fails_without_retry(make_document, embedder):
    fake = embedder(EmbeddingError("invalid api key"))
    document = run(make_document(long_pdf()))

    assert document.status == Document.Status.FAILED
    assert "rejected" in document.error
    assert len(fake.calls) == 1


def test_deleted_document_is_ignored(embedder):
    embedder()
    tasks.process_document.apply(args=[999_999], throw=True)


def test_enqueue_runs_only_after_commit(
    make_document, monkeypatch, django_capture_on_commit_callbacks
):
    queued = []
    monkeypatch.setattr(tasks.process_document, "delay", queued.append)
    document = make_document(long_pdf())

    with django_capture_on_commit_callbacks(execute=False) as callbacks:
        tasks.enqueue_processing(document)
        assert queued == []  # nothing is sent while the transaction is open

    for callback in callbacks:
        callback()
    assert queued == [document.pk]


def test_invalid_api_key_is_reported_as_configuration_problem(make_document, embedder):
    embedder(EmbeddingAuthError("Provided API key is invalid."))
    document = run(make_document(long_pdf()))

    assert document.status == Document.Status.FAILED
    assert "API key" in document.error
