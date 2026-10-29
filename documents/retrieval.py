"""Hybrid retrieval: vector search + full-text search, merged with Reciprocal Rank Fusion.

Why two searches:
- Vectors find passages with the same *meaning* in other words ("dismissal" ~ "termination"),
  across languages too, but are weak at exact tokens: names, codes, numbers, rare terms.
- Full-text search is the opposite: exact (stemmed) words, no notion of meaning.

Their scores are on unrelated scales (cosine distance vs ts_rank), so they are merged by
*rank*, not by score: RRF(chunk) = sum over lists of 1 / (k + rank). A chunk found by both
searches outranks one found by a single search; k damps the dominance of the top ranks.

k = 60 (the RRF paper's value) suits fused lists of ~1000 results. With our 20 candidates
it flattens the ranks so much that any keyword match beats the vector top hit: Russian
questions about an English contract lost their best chunk to Russian chunks of unrelated
documents. A small k keeps the top of each list decisive (see RETRIEVAL_RRF_K, chosen
with `manage.py eval_retrieval`).
"""

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from django.conf import settings
from django.contrib.postgres.search import SearchQuery, SearchRank
from django.db import connection, transaction
from django.db.models import F
from pgvector.django import CosineDistance

from .embeddings import Embedder, EmbeddingError, get_embedder
from .models import Chunk, Document

logger = logging.getLogger(__name__)

MAX_QUERY_TERMS = 32

_WORD = re.compile(r"\w+")
_CYRILLIC = re.compile(r"[а-яё]", re.IGNORECASE)


@dataclass
class SearchHit:
    chunk: Chunk
    score: float  # RRF score — only meaningful for ordering
    vector_rank: int | None = None  # 1-based; None = not found by this search
    vector_distance: float | None = None
    keyword_rank: int | None = None
    keyword_score: float | None = None


@dataclass
class RetrievalResult:
    query: str
    hits: list[SearchHit]
    vector_hits: list[SearchHit] = field(default_factory=list)  # for the debug page
    keyword_hits: list[SearchHit] = field(default_factory=list)
    embedding_model: str = ""
    embedding_tokens: int = 0
    # Set when the vector half failed and the result is keyword-only (degraded).
    vector_error: str = ""


def search(
    user,
    query: str,
    *,
    limit: int | None = None,
    candidates: int | None = None,
    embedder: Embedder | None = None,
) -> RetrievalResult:
    """Find the chunks of the user's ready documents most relevant to the query."""
    limit = limit or settings.RETRIEVAL_TOP_K
    candidates = candidates or settings.RETRIEVAL_CANDIDATES
    result = RetrievalResult(query=query, hits=[])
    chunks = Chunk.objects.filter(
        document__user=user, document__status=Document.Status.READY
    ).select_related("document")

    try:
        embedding = (embedder or get_embedder()).embed_query(query)
    except EmbeddingError as exc:
        # Degrade gracefully: keyword search alone is still better than no answer.
        logger.warning("Query embedding failed, falling back to keyword search: %s", exc)
        result.vector_error = str(exc)
    else:
        result.embedding_model = embedding.model
        result.embedding_tokens = embedding.total_tokens
        result.vector_hits = vector_search(chunks, embedding.vectors[0], candidates)

    result.keyword_hits = keyword_search(chunks, query, candidates)
    result.hits = reciprocal_rank_fusion(result.vector_hits, result.keyword_hits)[:limit]
    return result


def vector_search(chunks, vector: Sequence[float], limit: int) -> list[SearchHit]:
    """Nearest chunks by cosine distance (HNSW index)."""
    with transaction.atomic(), connection.cursor() as cursor:
        # HNSW + WHERE is a trap: the index returns the ef_search nearest chunks of *all*
        # users, then the filter drops other users' chunks — possibly leaving almost
        # nothing. Iterative scan (pgvector >= 0.8) keeps scanning until enough rows pass
        # the filter. SET LOCAL lasts until the end of this transaction only.
        cursor.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
        cursor.execute("SET LOCAL hnsw.ef_search = %s", [max(40, limit * 2)])
        rows = list(
            chunks.annotate(distance=CosineDistance("embedding", vector)).order_by("distance")[
                :limit
            ]
        )
    # relaxed_order may return rows slightly out of order; restore exact ordering.
    rows.sort(key=lambda c: c.distance)
    return [
        SearchHit(chunk=c, score=0.0, vector_rank=i, vector_distance=c.distance)
        for i, c in enumerate(rows, start=1)
    ]


def keyword_search(chunks, query: str, limit: int) -> list[SearchHit]:
    """Chunks containing the query's words (stemmed), ranked by ts_rank."""
    search_query = build_search_query(query)
    if search_query is None:
        return []
    rows = (
        chunks.filter(search_vector=search_query)
        .annotate(rank=SearchRank(F("search_vector"), search_query))
        .order_by("-rank", "pk")[:limit]
    )
    return [
        SearchHit(chunk=c, score=0.0, keyword_rank=i, keyword_score=c.rank)
        for i, c in enumerate(rows, start=1)
    ]


def build_search_query(query: str) -> SearchQuery | None:
    """OR of the query's words, each stemmed with the config of its own script.

    - OR, not AND: a natural-language question rarely has *all* its words in one chunk;
      chunks matching more words simply rank higher.
    - Per-word config: Cyrillic words go through the Russian stemmer and stop-word list,
      the rest through the English one. Applying the Russian config to "the" would keep it
      as a term (it is not a Russian stop word) and match nearly every English chunk.
    """
    words = _WORD.findall(query.lower())[:MAX_QUERY_TERMS]
    search_query = None
    for word in dict.fromkeys(words):  # dedupe, keep order
        config = "russian" if _CYRILLIC.search(word) else "english"
        term = SearchQuery(word, config=config, search_type="plain")
        search_query = term if search_query is None else search_query | term
    return search_query


def reciprocal_rank_fusion(*rankings: list[SearchHit], k: int | None = None) -> list[SearchHit]:
    """Merge ranked lists by RRF; per-search ranks and scores are kept for inspection."""
    k = k or settings.RETRIEVAL_RRF_K
    merged: dict[int, SearchHit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            current = merged.setdefault(hit.chunk.pk, SearchHit(chunk=hit.chunk, score=0.0))
            current.score += 1 / (k + rank)
            if hit.vector_rank is not None:
                current.vector_rank, current.vector_distance = hit.vector_rank, hit.vector_distance
            if hit.keyword_rank is not None:
                current.keyword_rank, current.keyword_score = hit.keyword_rank, hit.keyword_score
    return sorted(merged.values(), key=lambda h: h.score, reverse=True)
