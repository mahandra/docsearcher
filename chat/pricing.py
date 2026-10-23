"""Model prices and cost estimation.

Prices are USD per 1M tokens (Anthropic first-party API rates, checked 2026-10).
They change over time — treat this table as configuration and keep it up to date.

Phase 1 sends no prompt-cache markers, so only plain input/output tokens are priced.
Cache read/write pricing is added together with prompt caching (phase 5).
"""

import logging
from dataclasses import dataclass
from decimal import Decimal

logger = logging.getLogger(__name__)

PER_MILLION = Decimal(1_000_000)


@dataclass(frozen=True)
class ModelPrice:
    input_per_mtok: Decimal
    output_per_mtok: Decimal


PRICES: dict[str, ModelPrice] = {
    "claude-fable-5-1": ModelPrice(Decimal("10.00"), Decimal("50.00")),
    "claude-opus-5-5": ModelPrice(Decimal("4.00"), Decimal("20.00")),
    "claude-sonnet-5-5": ModelPrice(Decimal("2.00"), Decimal("10.00")),
    # Haiku 5.5 is priced higher above 100K input tokens; chat prompts stay well below.
    "claude-haiku-5-5": ModelPrice(Decimal("0.10"), Decimal("0.50")),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> Decimal | None:
    """Return the estimated cost in USD, or None if the model's price is unknown."""
    price = PRICES.get(model)
    if price is None:
        logger.warning("No price configured for model %r; cost not recorded", model)
        return None
    cost = (
        Decimal(input_tokens) * price.input_per_mtok
        + Decimal(output_tokens) * price.output_per_mtok
    ) / PER_MILLION
    return cost.quantize(Decimal("0.000001"))
