"""
opportunity_detector.py
Given a list of RouteQuotes for the same pair (from different poll cycles
or from direct vs aggregated routes), detects spread opportunities and
computes realistic P&L including all fee layers.
"""

from dataclasses import dataclass
from typing import Optional
import time

from price_fetcher import RouteQuote

# Fee model (all in USD or fraction of trade)
PRIORITY_FEE_SOL   = 0.0001      # medium priority fee
SOL_PRICE_FALLBACK = 160.0       # used only if no quote available
JITO_TIP_SOL       = 0.0005      # Jito bundle tip
DEX_FEE_RATE       = 0.00022     # 2.2 bps per leg (Jupiter measured)
FAILED_TX_FACTOR   = 0.20        # 20% of fixed cost buffer for failed txs
RENT_ALREADY_PAID  = True        # assume token accounts exist (sim mode)

# Minimum spread to even log as opportunity (noise filter)
MIN_LOG_SPREAD_BPS = 5


@dataclass
class Opportunity:
    ts: float
    pair: str
    price_a: float
    price_b: float
    spread_bps: float
    trade_size_usd: float
    gross_profit_usd: float
    fee_cost_usd: float
    net_profit_usd: float
    viable: bool              # net > 0 after ALL fees
    note: str = ""


def compute_fee_cost(trade_size_usd: float, sol_price: float) -> float:
    """Total fixed + variable fees for a 2-leg arbitrage (buy + sell)."""
    # Fixed network costs
    fixed_sol = (PRIORITY_FEE_SOL * 2) + JITO_TIP_SOL   # both legs + tip
    fixed_usd = fixed_sol * sol_price

    # Add failure buffer
    fixed_usd *= (1 + FAILED_TX_FACTOR)

    # Variable DEX fees (both legs)
    variable_usd = trade_size_usd * DEX_FEE_RATE * 2

    return fixed_usd + variable_usd


def detect_opportunity(
    pair: str,
    quote_now: RouteQuote,
    quote_prev: RouteQuote,
    trade_size_usd: float = 25.0,
    sol_price: Optional[float] = None,
) -> Optional[Opportunity]:
    """
    Compare two quotes for the same pair taken at different times
    (or from different routes). If spread > fee threshold, flag it.

    In simulation, quote_now vs quote_prev acts as a proxy for
    "price on DEX A vs price on DEX B" since Jupiter surfaces the
    best aggregated route -- discrepancy between poll cycles captures
    the opportunity window.
    """
    if quote_now is None or quote_prev is None:
        return None

    price_a = quote_prev.price
    price_b = quote_now.price

    if price_a <= 0 or price_b <= 0:
        return None

    # Always compute spread as (higher - lower) / lower
    high, low = max(price_a, price_b), min(price_a, price_b)
    spread_bps = ((high - low) / low) * 10_000

    if spread_bps < MIN_LOG_SPREAD_BPS:
        return None

    sp = sol_price or SOL_PRICE_FALLBACK
    gross = (spread_bps / 10_000) * trade_size_usd
    fees  = compute_fee_cost(trade_size_usd, sp)
    net   = gross - fees
    viable = net > 0

    return Opportunity(
        ts=quote_now.ts,
        pair=pair,
        price_a=price_a,
        price_b=price_b,
        spread_bps=spread_bps,
        trade_size_usd=trade_size_usd,
        gross_profit_usd=gross,
        fee_cost_usd=fees,
        net_profit_usd=net,
        viable=viable,
        note=f"dex={quote_now.dex_label}",
    )


def min_viable_spread_bps(trade_size_usd: float, sol_price: float) -> float:
    """What spread (bps) do we need to break even at this trade size?"""
    fees = compute_fee_cost(trade_size_usd, sol_price)
    return (fees / trade_size_usd) * 10_000
