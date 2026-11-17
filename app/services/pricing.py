"""Model pricing: the single source of truth for USD cost estimates.

Oraculum runs open models on its own hardware, so no model has a per-token
price and every estimate is None (never a fabricated zero). The table stays
so the usage ledger, budgets and eval reports keep one shared source if a
priced model server is ever configured: add (input, output) USD per 1M
tokens here.
"""

# (input_usd_per_1m, output_usd_per_1m)
PRICES_PER_1M: dict[str, tuple[float, float]] = {}


def cost_usd(model: str, tokens_in: int, tokens_out: int = 0) -> float | None:
    """Estimated USD for a call; None when the model has no known price."""
    prices = PRICES_PER_1M.get(model)
    if prices is None:
        return None
    in_price, out_price = prices
    return round(tokens_in / 1e6 * in_price + tokens_out / 1e6 * out_price, 6)
