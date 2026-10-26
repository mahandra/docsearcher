"""Background ingestion: file -> text -> chunks -> embeddings -> pgvector."""

import logging
import random

from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded
from django.conf import settings
from django.db import transaction
from django.utils import timezone

from chat.pricing import estimate_embedding_cost

from .chunking import chunk_sections
from .embeddings import (
    EmbeddingAuthError,
    EmbeddingError,
    TransientEmbeddingError,
    get_embedder,
)
from .models import Chunk, Document
from .parsing import ParsingError, detect_kind, parse_document

logger = logging.getLogger(__name__)

MAX_TASK_RETRIES = 5


def enqueue_processing(document: Document) -> None:
    """Schedule processing once the current transaction commits.

    Calling .delay() directly inside a transaction is a classic race: the worker may
    pick up the task before the commit and not find the document.
    """
    transaction.on_commit(lambda: process_document.delay(document.pk))


def retry_delay(retries: int) -> float:
    """Exponential backoff with jitter: ~30s, 60s, 120s, ... capped at 10 minutes."""
    return min(30 * 2**retries, 600) * random.uniform(1, 1.25)


@shared_task(bind=True, max_retries=MAX_TASK_RETRIES)
def process_document(self, document_id: int) -> None:
    document = Document.objects.filter(pk=document_id).first()
    if document is None:
        logger.info("Document %s was deleted before processing", document_id)
        return
    if document.status == Document.Status.READY:
        # Idempotency: a redelivered task (acks_late) must not re-embed and re-bill.
        logger.info("Document %s is already processed", document_id)
        return

    Document.objects.filter(pk=document.pk).update(status=Document.Status.PROCESSING, error="")
    try:
        ingest(document)
    except ParsingError as exc:
        _mark_failed(document, str(exc))
    except TransientEmbeddingError as exc:
        if self.request.retries < self.max_retries:
            delay = retry_delay(self.request.retries)
            logger.warning(
                "Embedding temporarily failed for document %s (retry %s in %.0fs): %s",
                document.pk,
                self.request.retries + 1,
                delay,
                exc,
            )
            # Raises celery.exceptions.Retry: the task is rescheduled, not failed.
            raise self.retry(exc=exc, countdown=delay) from exc
        _mark_failed(document, "The embedding service is unavailable. Please try again later.")
    except EmbeddingAuthError as exc:
        logger.error("Embedding API key rejected for document %s: %s", document.pk, exc)
        _mark_failed(document, "The embedding service is not configured correctly (API key).")
    except EmbeddingError as exc:
        logger.error("Embedding failed for document %s: %s", document.pk, exc)
        _mark_failed(document, "The embedding service rejected the document.")
    except SoftTimeLimitExceeded:
        _mark_failed(document, "Processing took too long.")
    except Exception:
        _mark_failed(document, "Unexpected error while processing the document.")
        raise  # keep the traceback in the worker logs


def ingest(document: Document) -> None:
    path = document.file.path
    with open(path, "rb") as f:
        kind = detect_kind(f.read(8))
    if kind is None:
        raise ParsingError("Unsupported file type.")

    parsed = parse_document(path, kind)
    chunks = chunk_sections(
        parsed.sections,
        chunk_size=settings.CHUNK_SIZE_TOKENS,
        overlap=settings.CHUNK_OVERLAP_TOKENS,
    )
    logger.info(
        "Document %s: %s characters -> %s chunks", document.pk, parsed.text_length, len(chunks)
    )

    # Slow network calls happen *outside* the transaction, so no locks are held meanwhile.
    result = get_embedder().embed_documents([c.text for c in chunks])

    with transaction.atomic():
        # Replace, don't append: re-running the task yields the same result.
        Chunk.objects.filter(document=document).delete()
        Chunk.objects.bulk_create(
            Chunk(
                document=document,
                index=i,
                text=c.text,
                page_start=c.page_start,
                page_end=c.page_end,
                section=c.section[:500],
                token_count=c.token_count,
                embedding=vector,
            )
            for i, (c, vector) in enumerate(zip(chunks, result.vectors, strict=True))
        )
        Document.objects.filter(pk=document.pk).update(
            status=Document.Status.READY,
            error="",
            page_count=parsed.page_count,
            chunk_count=len(chunks),
            embedding_model=result.model,
            embedding_tokens=result.total_tokens,
            embedding_cost_usd=estimate_embedding_cost(result.model, result.total_tokens),
            processed_at=timezone.now(),
        )
    logger.info(
        "Document %s ready: %s tokens embedded with %s",
        document.pk,
        result.total_tokens,
        result.model,
    )


def _mark_failed(document: Document, error: str) -> None:
    logger.warning("Document %s failed: %s", document.pk, error)
    Document.objects.filter(pk=document.pk).update(
        status=Document.Status.FAILED, error=error, processed_at=timezone.now()
    )
