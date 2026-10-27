"""Fakes for the embedding provider — no network calls in tests."""

import hashlib
from types import SimpleNamespace

import numpy as np

from documents.chunking import estimate_tokens
from documents.embeddings import EmbeddingResult
from documents.models import EMBEDDING_DIMENSIONS


def fake_vector(text: str, dimensions: int = EMBEDDING_DIMENSIONS) -> list[float]:
    """Deterministic unit vector derived from the text."""
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
    vector = np.random.default_rng(seed).normal(size=dimensions)
    return (vector / np.linalg.norm(vector)).tolist()


class FakeVoyageClient:
    """Mimics voyageai.Client.embed(); `errors` are raised by successive calls."""

    def __init__(self, errors: list[Exception] | None = None, dimensions=EMBEDDING_DIMENSIONS):
        self.errors = list(errors or [])
        self.dimensions = dimensions
        self.calls: list[dict] = []

    def embed(self, texts, **kwargs):
        self.calls.append({"texts": list(texts), **kwargs})
        if self.errors:
            raise self.errors.pop(0)
        return SimpleNamespace(
            embeddings=[fake_vector(t, self.dimensions) for t in texts],
            total_tokens=sum(estimate_tokens(t) for t in texts),
        )


class FakeEmbedder:
    """Implements the Embedder protocol directly; used by task and view tests."""

    model = "voyage-4"

    def __init__(self, errors: list[Exception] | None = None):
        self.errors = list(errors or [])
        self.calls: list[list[str]] = []
        self.queries: list[str] = []

    def embed_documents(self, texts: list[str]) -> EmbeddingResult:
        self.calls.append(list(texts))
        if self.errors:
            raise self.errors.pop(0)
        return EmbeddingResult(
            vectors=[fake_vector(t) for t in texts],
            total_tokens=sum(estimate_tokens(t) for t in texts),
            model=self.model,
        )

    def embed_query(self, text: str) -> EmbeddingResult:
        self.queries.append(text)
        if self.errors:
            raise self.errors.pop(0)
        return EmbeddingResult(
            vectors=[fake_vector(text)], total_tokens=estimate_tokens(text), model=self.model
        )
