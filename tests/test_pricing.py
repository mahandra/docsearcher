from decimal import Decimal

from chat.pricing import estimate_cost


def test_estimate_cost_uses_separate_input_and_output_prices():
    # Opus 5.5: $4 / 1M input, $20 / 1M output.
    assert estimate_cost("claude-opus-5-5", 1_000_000, 0) == Decimal("4.000000")
    assert estimate_cost("claude-opus-5-5", 0, 1_000_000) == Decimal("20.000000")
    assert estimate_cost("claude-opus-5-5", 1500, 300) == Decimal("0.012000")


def test_estimate_cost_returns_none_for_unknown_model():
    assert estimate_cost("some-future-model", 100, 100) is None


def test_cache_writes_cost_more_and_cache_reads_cost_less_than_input():
    # Opus 5.5: writes 1.25 x $4 = $5 / 1M, reads $0.20 / 1M.
    assert estimate_cost("claude-opus-5-5", 0, 0, cache_write_tokens=1_000_000) == Decimal(
        "5.000000"
    )
    assert estimate_cost("claude-opus-5-5", 0, 0, cache_read_tokens=1_000_000) == Decimal(
        "0.200000"
    )
    # All four parts add up: 100 in, 1000 written, 4000 read, 300 out.
    assert estimate_cost(
        "claude-opus-5-5", 100, 300, cache_write_tokens=1000, cache_read_tokens=4000
    ) == Decimal("0.012200")
