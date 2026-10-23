from decimal import Decimal

from chat.pricing import estimate_cost


def test_estimate_cost_uses_separate_input_and_output_prices():
    # Opus 5.5: $4 / 1M input, $20 / 1M output.
    assert estimate_cost("claude-opus-5-5", 1_000_000, 0) == Decimal("4.000000")
    assert estimate_cost("claude-opus-5-5", 0, 1_000_000) == Decimal("20.000000")
    assert estimate_cost("claude-opus-5-5", 1500, 300) == Decimal("0.012000")


def test_estimate_cost_returns_none_for_unknown_model():
    assert estimate_cost("some-future-model", 100, 100) is None
