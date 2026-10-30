"""Model prices and cost estimation (LLM and embeddings).

Prices are USD per 1M tokens (Anthropic and Voyage AI list prices, checked 2026-10).
They change over time — treat this table as configuration and keep it up to date.

With prompt caching, the input is billed in three parts: `input_tokens` (not cached,
full price), `cache_creation_input_tokens` (written to the cache, a premium over the
input price) and `cache_read_input_tokens` (read from it, a fraction of the input price).
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
    # The cache-read discount differs by model (not a fixed share of the input price).
    cache_read_per_mtok: Decimal


# Writing to the 5-minute cache costs 1.25x the input price (we use no 1-hour entries).
CACHE_WRITE_MULTIPLIER = Decimal("1.25")

PRICES: dict[str, ModelPrice] = {
    "claude-fable-5-1": ModelPrice(Decimal("10.00"), Decimal("50.00"), Decimal("0.25")),
    "claude-opus-5-5": ModelPrice(Decimal("4.00"), Decimal("20.00"), Decimal("0.20")),
    "claude-sonnet-5-5": ModelPrice(Decimal("2.00"), Decimal("10.00"), Decimal("0.20")),
    # Haiku 5.5 is priced higher above 100K input tokens; chat prompts stay well below.
    "claude-haiku-5-5": ModelPrice(Decimal("0.10"), Decimal("0.50"), Decimal("0.01")),
}


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> Decimal | None:
    """Return the estimated cost in USD, or None if the model's price is unknown.

    `input_tokens` is the uncached remainder only, as the API reports it.
    """
    price = PRICES.get(model)
    if price is None:
        logger.warning("No price configured for model %r; cost not recorded", model)
        return None
    cost = (
        Decimal(input_tokens) * price.input_per_mtok
        + Decimal(cache_write_tokens) * price.input_per_mtok * CACHE_WRITE_MULTIPLIER
        + Decimal(cache_read_tokens) * price.cache_read_per_mtok
        + Decimal(output_tokens) * price.output_per_mtok
    ) / PER_MILLION
    return cost.quantize(Decimal("0.000001"))


# --- Embeddings ----------------------------------------------------------------------
# Voyage AI, USD per 1M input tokens (checked 2026-10; first 200M tokens of the 4-series
# are free per account, which is not reflected here).
EMBEDDING_PRICES: dict[str, Decimal] = {
    "voyage-4-large": Decimal("0.12"),
    "voyage-4": Decimal("0.06"),
    "voyage-4-lite": Decimal("0.02"),
}


def estimate_embedding_cost(model: str, tokens: int) -> Decimal | None:
    price = EMBEDDING_PRICES.get(model)
    if price is None:
        logger.warning("No price configured for embedding model %r; cost not recorded", model)
        return None
    return (Decimal(tokens) * price / PER_MILLION).quantize(Decimal("0.000001"))
