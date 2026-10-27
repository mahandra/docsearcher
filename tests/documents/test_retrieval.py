import numpy as np
import pytest
from django.contrib.auth import get_user_model

from documents.embeddings import EmbeddingAuthError
from documents.models import Chunk, Document
from documents.retrieval import (
    SearchHit,
    build_search_query,
    keyword_search,
    reciprocal_rank_fusion,
    search,
)
from tests.documents.fakes import FakeEmbedder, fake_vector

pytestmark = pytest.mark.django_db

QUESTION = "What is the notice period?"


@pytest.fixture
def ready_document(make_document):
    def make(**kwargs):
        kwargs.setdefault("status", Document.Status.READY)
        kwargs.setdefault("content", f"file {Document.objects.count()}".encode())
        return make_document(**kwargs)

    return make


def add_chunk(document, text, embedding=None):
    """`embedding` defaults to an unrelated random vector (far from any query)."""
    return Chunk.objects.create(
        document=document,
        index=document.chunks.count(),
        text=text,
        token_count=10,
        embedding=embedding if embedding is not None else fake_vector(f"unrelated: {text}"),
    )


def ids(hits):
    return [h.chunk.pk for h in hits]


# --- Reciprocal Rank Fusion -----------------------------------------------------------


def test_rrf_prefers_chunks_found_by_both_searches():
    a, b, c = (Chunk(pk=i) for i in (1, 2, 3))
    vector = [SearchHit(a, 0, vector_rank=1), SearchHit(b, 0, vector_rank=2)]
    keyword = [SearchHit(c, 0, keyword_rank=1), SearchHit(b, 0, keyword_rank=2)]

    fused = reciprocal_rank_fusion(vector, keyword)

    assert ids(fused) == [2, 1, 3]  # b: 1/62 + 1/62 beats 1/61
    assert fused[0].vector_rank == 2 and fused[0].keyword_rank == 2
    assert fused[1].keyword_rank is None


def test_rrf_with_one_empty_list_keeps_the_other_order():
    hits = [SearchHit(Chunk(pk=i), 0, vector_rank=i) for i in (5, 3, 9)]
    assert ids(reciprocal_rank_fusion(hits, [])) == [5, 3, 9]


# --- Keyword search -------------------------------------------------------------------


@pytest.mark.parametrize("query", ["", "   ", "?!", "—"])
def test_query_without_words_builds_no_search(query):
    assert build_search_query(query) is None


def test_keyword_search_matches_stemmed_english_and_ignores_stop_words(ready_document):
    document = ready_document()
    notice = add_chunk(document, "Notice periods are two months.")
    add_chunk(document, "What is the meaning of the cat?")  # only stop words + "cat"

    hits = keyword_search(Chunk.objects.all(), QUESTION, limit=10)

    # "what", "is", "the" are English stop words: only "notice"/"period" can match.
    assert ids(hits) == [notice.pk]


def test_keyword_search_matches_russian_word_forms(ready_document):
    document = ready_document()
    chunk = add_chunk(document, "Срок уведомления о расторжении договора — два месяца.")
    add_chunk(document, "Что это и как это работает?")

    hits = keyword_search(Chunk.objects.all(), "Какой срок для расторжения договоров?", 10)

    assert ids(hits) == [chunk.pk]


def test_keyword_search_ranks_chunks_with_more_query_words_higher(ready_document):
    document = ready_document()
    one = add_chunk(document, "The notice must be written.")
    both = add_chunk(document, "The notice period is one month.")

    hits = keyword_search(Chunk.objects.all(), QUESTION, limit=10)

    assert ids(hits) == [both.pk, one.pk]


# --- Hybrid search --------------------------------------------------------------------


def test_vector_search_finds_semantically_close_chunk_without_shared_words(ready_document):
    document = ready_document()
    # The fake embedder maps the question to fake_vector(QUESTION): this chunk is "close".
    close = add_chunk(
        document, "Either party may leave with 60 days warning.", fake_vector(QUESTION)
    )
    add_chunk(document, "Payment is due monthly.")

    result = search(document.user, QUESTION, limit=1, embedder=FakeEmbedder())

    assert ids(result.hits) == [close.pk]
    assert result.hits[0].vector_rank == 1
    assert result.hits[0].keyword_rank is None
    assert result.hits[0].vector_distance == pytest.approx(0, abs=1e-6)


def test_hybrid_search_ranks_chunk_found_by_both_searches_first(ready_document):
    document = ready_document()
    near = np.array(fake_vector(QUESTION))
    vector_only = add_chunk(document, "Leave with 60 days warning.", near)
    keyword_only = add_chunk(document, "The notice period is one month.")
    both = add_chunk(document, "Notice period: see clause 7.", near + 0.01)

    # Vector search always returns `candidates` chunks, relevant or not; with 2 the
    # unrelated keyword-only chunk stays out of the vector list, as in a real corpus.
    result = search(document.user, QUESTION, candidates=2, embedder=FakeEmbedder())

    assert ids(result.vector_hits) == [vector_only.pk, both.pk]
    assert set(ids(result.keyword_hits)) == {keyword_only.pk, both.pk}
    assert ids(result.hits)[0] == both.pk
    assert set(ids(result.hits)) == {both.pk, vector_only.pk, keyword_only.pk}


def test_search_only_sees_the_users_ready_documents(ready_document, user):
    other_user = get_user_model().objects.create_user("other", password="x")
    near = fake_vector(QUESTION)
    mine = add_chunk(ready_document(), "notice period", near)
    add_chunk(ready_document(owner=other_user), "notice period", near)
    add_chunk(ready_document(status=Document.Status.PROCESSING), "notice period", near)

    result = search(user, QUESTION, embedder=FakeEmbedder())

    assert ids(result.hits) == [mine.pk]


def test_search_returns_at_most_limit_hits(ready_document):
    document = ready_document()
    for i in range(12):
        add_chunk(document, f"notice period variant {i}")

    result = search(document.user, QUESTION, limit=5, embedder=FakeEmbedder())

    assert len(result.hits) == 5


def test_search_records_query_embedding_usage(ready_document):
    document = ready_document()
    embedder = FakeEmbedder()

    result = search(document.user, QUESTION, embedder=embedder)

    assert embedder.queries == [QUESTION]
    assert result.embedding_model == "voyage-4"
    assert result.embedding_tokens > 0
    assert result.vector_error == ""


def test_search_falls_back_to_keywords_when_embedding_fails(ready_document):
    document = ready_document()
    chunk = add_chunk(document, "The notice period is one month.")
    embedder = FakeEmbedder(errors=[EmbeddingAuthError("bad key")])

    result = search(document.user, QUESTION, embedder=embedder)

    assert ids(result.hits) == [chunk.pk]
    assert result.vector_error == "bad key"
    assert result.vector_hits == []
