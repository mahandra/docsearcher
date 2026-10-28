"""Retrieval metrics. Inputs are ranked chunk texts; relevance comes from evidence quotes.

- hit@k: is at least one relevant chunk in the top k? ("can the model answer at all")
- recall@k: share of the evidence quotes found in the top k. Matters for questions whose
  answer is spread over several places (multi-hop).
- reciprocal rank: 1 / rank of the first relevant chunk, 0 if none. Averaged over the
  cases it is MRR — it rewards putting the right chunk *first*, not just somewhere.
"""

from collections.abc import Sequence
from statistics import mean

from .dataset import Evidence, contains


def relevant_ranks(chunks: Sequence[str], evidence: Sequence[Evidence]) -> list[int]:
    """1-based ranks of the chunks that contain at least one evidence quote."""
    return [
        rank
        for rank, text in enumerate(chunks, start=1)
        if any(contains(text, quote) for quote in evidence)
    ]


def hit_at(chunks: Sequence[str], evidence: Sequence[Evidence], k: int) -> bool:
    return any(rank <= k for rank in relevant_ranks(chunks, evidence))


def recall_at(chunks: Sequence[str], evidence: Sequence[Evidence], k: int) -> float:
    if not evidence:
        return 0.0
    top = chunks[:k]
    found = sum(any(contains(text, quote) for text in top) for quote in evidence)
    return found / len(evidence)


def reciprocal_rank(chunks: Sequence[str], evidence: Sequence[Evidence]) -> float:
    ranks = relevant_ranks(chunks, evidence)
    return 1 / ranks[0] if ranks else 0.0


def average(values: Sequence[float]) -> float | None:
    """Mean that tolerates an empty group (e.g. a category with no cases)."""
    return mean(values) if values else None
