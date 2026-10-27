from decimal import Decimal

import pytest
import voyageai.error as voyage_errors

from chat.pricing import estimate_embedding_cost
from documents.embeddings import (
    EmbeddingAuthError,
    EmbeddingError,
    TransientEmbeddingError,
    VoyageEmbedder,
)
from documents.models import EMBEDDING_DIMENSIONS
from tests.documents.fakes import FakeVoyageClient


def make_embedder(client, **kwargs):
    kwargs.setdefault("model", "voyage-4")
    kwargs.setdefault("batch_size", 128)
    return VoyageEmbedder(client, **kwargs)


def test_embeds_documents_with_document_input_type():
    client = FakeVoyageClient()

    result = make_embedder(client).embed_documents(["alpha", "beta"])

    assert len(result.vectors) == 2
    assert len(result.vectors[0]) == EMBEDDING_DIMENSIONS
    assert result.model == "voyage-4"
    assert result.total_tokens > 0
    call = client.calls[0]
    assert call["model"] == "voyage-4"
    assert call["input_type"] == "document"
    assert call["output_dimension"] == EMBEDDING_DIMENSIONS
    assert call["truncation"] is False


def test_embeds_query_with_query_input_type():
    client = FakeVoyageClient()

    result = make_embedder(client).embed_query("what is the notice period?")

    assert len(result.vectors) == 1
    assert len(result.vectors[0]) == EMBEDDING_DIMENSIONS
    assert result.total_tokens > 0
    call = client.calls[0]
    assert call["texts"] == ["what is the notice period?"]
    assert call["input_type"] == "query"
    assert call["output_dimension"] == EMBEDDING_DIMENSIONS


def test_query_errors_are_mapped_like_document_errors():
    client = FakeVoyageClient(errors=[voyage_errors.RateLimitError("slow down")])
    with pytest.raises(TransientEmbeddingError):
        make_embedder(client).embed_query("x")


def test_batches_by_count_and_keeps_order():
    client = FakeVoyageClient()
    texts = [f"text {i}" for i in range(7)]

    result = make_embedder(client, batch_size=3).embed_documents(texts)

    assert [len(c["texts"]) for c in client.calls] == [3, 3, 1]
    assert [t for c in client.calls for t in c["texts"]] == texts
    assert len(result.vectors) == 7


def test_batches_by_token_budget(monkeypatch):
    monkeypatch.setattr("documents.embeddings.MAX_BATCH_TOKENS", 100)
    client = FakeVoyageClient()
    texts = ["x" * 210] * 4  # 60 estimated tokens each

    make_embedder(client).embed_documents(texts)

    assert [len(c["texts"]) for c in client.calls] == [1, 1, 1, 1]


@pytest.mark.parametrize(
    "error",
    [
        voyage_errors.RateLimitError("slow down"),
        voyage_errors.ServiceUnavailableError("down"),
        voyage_errors.APIConnectionError("no route"),
        voyage_errors.ServerError("boom"),
        voyage_errors.Timeout("timeout"),
    ],
)
def test_transient_errors(error):
    with pytest.raises(TransientEmbeddingError):
        make_embedder(FakeVoyageClient(errors=[error])).embed_documents(["x"])


@pytest.mark.parametrize(
    "error",
    [voyage_errors.AuthenticationError("bad key"), voyage_errors.InvalidRequestError("bad")],
)
def test_permanent_errors(error):
    with pytest.raises(EmbeddingError) as info:
        make_embedder(FakeVoyageClient(errors=[error])).embed_documents(["x"])
    assert not isinstance(info.value, TransientEmbeddingError)


def test_wrong_dimensions_are_rejected():
    with pytest.raises(EmbeddingError, match="dimensions"):
        make_embedder(FakeVoyageClient(dimensions=256)).embed_documents(["x"])


def test_embedding_cost():
    assert estimate_embedding_cost("voyage-4", 1_000_000) == Decimal("0.060000")
    assert estimate_embedding_cost("voyage-4", 50_000) == Decimal("0.003000")
    assert estimate_embedding_cost("unknown-model", 10) is None


def test_auth_error_is_reported_separately():
    with pytest.raises(EmbeddingAuthError):
        make_embedder(
            FakeVoyageClient(errors=[voyage_errors.AuthenticationError("bad key")])
        ).embed_documents(["x"])
