"""Embedding service layer — the only module that talks to the Voyage AI SDK.

Anthropic has no embeddings API, so a second provider is needed. Callers depend on the
small `Embedder` protocol; tests use a deterministic fake.
"""

import logging
from dataclasses import dataclass
from functools import cache
from typing import Any, Protocol

import voyageai
import voyageai.error as voyage_errors
from django.conf import settings

from .chunking import estimate_tokens
from .models import EMBEDDING_DIMENSIONS

logger = logging.getLogger(__name__)

# Voyage limits a request to 1000 texts and 320K tokens (voyage-4). We stay far below
# the token limit because our token counts are only estimates.
MAX_BATCH_TOKENS = 100_000


class EmbeddingError(Exception):
    """Embedding failed permanently (bad key, invalid input): retrying will not help."""


class EmbeddingAuthError(EmbeddingError):
    """Invalid or missing API key — a configuration problem, not the document's fault."""


class TransientEmbeddingError(EmbeddingError):
    """Embedding failed temporarily (rate limit, outage, network): retry later."""


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: list[list[float]]
    total_tokens: int  # exact, as billed by the provider
    model: str


class Embedder(Protocol):
    model: str

    def embed_documents(self, texts: list[str]) -> EmbeddingResult: ...

    def embed_query(self, text: str) -> EmbeddingResult: ...


TRANSIENT_ERRORS = (
    voyage_errors.RateLimitError,
    voyage_errors.ServiceUnavailableError,
    voyage_errors.Timeout,
    voyage_errors.APIConnectionError,
    voyage_errors.ServerError,
    voyage_errors.TryAgain,
)


class VoyageEmbedder:
    def __init__(
        self,
        client: Any,
        *,
        model: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.client = client
        self.model = model or settings.EMBEDDING_MODEL
        self.batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE

    def embed_documents(self, texts: list[str]) -> EmbeddingResult:
        vectors: list[list[float]] = []
        total_tokens = 0
        for batch in self._batches(texts):
            result = self._embed_batch(batch, input_type="document")
            vectors.extend(result.embeddings)
            total_tokens += result.total_tokens
        return EmbeddingResult(vectors=vectors, total_tokens=total_tokens, model=self.model)

    def embed_query(self, text: str) -> EmbeddingResult:
        result = self._embed_batch([text], input_type="query")
        return EmbeddingResult(
            vectors=result.embeddings, total_tokens=result.total_tokens, model=self.model
        )

    def _batches(self, texts: list[str]):
        """Group texts so each request stays under both the count and token limits."""
        batch: list[str] = []
        batch_tokens = 0
        for text in texts:
            tokens = estimate_tokens(text)
            if batch and (
                len(batch) >= self.batch_size or batch_tokens + tokens > MAX_BATCH_TOKENS
            ):
                yield batch
                batch, batch_tokens = [], 0
            batch.append(text)
            batch_tokens += tokens
        if batch:
            yield batch

    def _embed_batch(self, batch: list[str], *, input_type: str) -> Any:
        try:
            result = self.client.embed(
                batch,
                model=self.model,
                # Asymmetric retrieval: a short question and a long passage that answers
                # it are phrased very differently. Voyage prepends a different instruction
                # for "query" and "document", so both land close in the vector space.
                input_type=input_type,
                output_dimension=EMBEDDING_DIMENSIONS,
                # Fail loudly instead of silently embedding a truncated chunk.
                truncation=False,
            )
        except TRANSIENT_ERRORS as exc:
            raise TransientEmbeddingError(str(exc)) from exc
        except voyage_errors.AuthenticationError as exc:
            raise EmbeddingAuthError(str(exc)) from exc
        except voyage_errors.VoyageError as exc:
            raise EmbeddingError(str(exc)) from exc

        if len(result.embeddings) != len(batch):
            raise EmbeddingError(f"Expected {len(batch)} embeddings, got {len(result.embeddings)}")
        if any(len(v) != EMBEDDING_DIMENSIONS for v in result.embeddings):
            raise EmbeddingError(f"Embeddings must have {EMBEDDING_DIMENSIONS} dimensions")
        return result


@cache
def get_embedder() -> Embedder:
    client = voyageai.Client(
        api_key=settings.VOYAGE_API_KEY or None,
        # Fast retries (seconds) for rate limits and blips; longer outages are retried
        # by the Celery task (minutes).
        max_retries=settings.EMBEDDING_MAX_RETRIES,
        timeout=settings.EMBEDDING_TIMEOUT_SECONDS,
    )
    return VoyageEmbedder(client)
