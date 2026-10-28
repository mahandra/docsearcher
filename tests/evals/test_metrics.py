import pytest

from evals.dataset import Evidence
from evals.metrics import average, hit_at, recall_at, reciprocal_rank, relevant_ranks

NOTICE = Evidence("doc", "notice is two months")
INTEREST = Evidence("doc", "interest of 1.5%")
CHUNKS = [
    "Payment is due in 30 days.",
    "The notice is two\nmonths.",  # relevant, rank 2
    "Unrelated.",
    "Late payment: interest of 1.5% per month.",  # relevant, rank 4
]


def test_relevant_ranks_are_one_based():
    assert relevant_ranks(CHUNKS, [NOTICE, INTEREST]) == [2, 4]


@pytest.mark.parametrize("k, expected", [(1, False), (2, True), (10, True)])
def test_hit_at_k(k, expected):
    assert hit_at(CHUNKS, [NOTICE], k) is expected


def test_recall_counts_quotes_not_chunks():
    assert recall_at(CHUNKS, [NOTICE, INTEREST], 2) == 0.5
    assert recall_at(CHUNKS, [NOTICE, INTEREST], 4) == 1.0
    # Two quotes in one chunk: both found.
    assert recall_at(["notice is two months; interest of 1.5%"], [NOTICE, INTEREST], 1) == 1.0


def test_reciprocal_rank_uses_the_first_relevant_chunk():
    assert reciprocal_rank(CHUNKS, [NOTICE, INTEREST]) == 0.5
    assert reciprocal_rank(CHUNKS, [Evidence("doc", "absent")]) == 0.0


def test_metrics_without_evidence_or_chunks():
    assert recall_at(CHUNKS, [], 5) == 0.0
    assert hit_at([], [NOTICE], 5) is False
    assert average([]) is None
    assert average([1.0, 0.0]) == 0.5
