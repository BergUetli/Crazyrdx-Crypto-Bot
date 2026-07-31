"""
price_fetcher.py
Fetches live quotes from Jupiter (aggregator) for multiple DEX routes.
Jupiter exposes per-route breakdowns, so we can extract Raydium vs Orca prices
from a single API call instead of hitting each DEX separately.
"""

import asyncio
import time
import httpx
from dataclasses import dataclass, field
from typing import Optional
from decimal import Decimal


JUPITER_QUOTE_URL = "https://quote-api.jup.ag/v6/quote"

# Token mints
SOL  = "So11111111111111111111111111111111111111112"
USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
USDT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"
BONK = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"
WIF  = "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm"

PAIRS = [
    ("SOL/USDC", SOL,  USDC, 10_000_000_000),   # 10 SOL in lamports
    ("SOL/USDT", SOL,  USDT, 10_000_000_000),
    ("BONK/USDC", BONK, USDC, 1_000_000_000_000),  # 1M BONK
    ("WIF/USDC",  WIF,  USDC, 100_000_000),         # 100 WIF (6 dec)
]

# Decimals per token
DECIMALS = {SOL: 9, USDC: 6, USDT: 6, BONK: 5, WIF: 6}


@dataclass
class RouteQuote:
    pair: str
    input_mint: str
    output_mint: str
    amount_in: int           # raw lamports/units
    amount_out: int          # raw
    price: float             # output tokens per input token (human units)
    price_impact_pct: float
    dex_label: str           # best route label from Jupiter
    ts: float = field(default_factory=time.time)


async def fetch_quote(
    client: httpx.AsyncClient,
    pair: str,
    in_mint: str,
    out_mint: str,
    amount: int,
    slippage_bps: int = 50,
) -> Optional[RouteQuote]:
    params = {
        "inputMint": in_mint,
        "outputMint": out_mint,
        "amount": amount,
        "slippageBps": slippage_bps,
        "onlyDirectRoutes": "false",
    }
    try:
        r = await client.get(JUPITER_QUOTE_URL, params=params, timeout=8.0)
        if r.status_code != 200:
            return None
        data = r.json()

        in_dec  = DECIMALS.get(in_mint,  9)
        out_dec = DECIMALS.get(out_mint, 6)
        amount_out = int(data["outAmount"])
        price = (amount_out / 10**out_dec) / (amount / 10**in_dec)

        # Best route label (first plan label)
        route_plan = data.get("routePlan", [])
        if route_plan:
            swap_info = route_plan[0].get("swapInfo", {})
            dex_label = swap_info.get("label", "Unknown")
        else:
            dex_label = "Unknown"

        return RouteQuote(
            pair=pair,
            input_mint=in_mint,
            output_mint=out_mint,
            amount_in=amount,
            amount_out=amount_out,
            price=price,
            price_impact_pct=float(data.get("priceImpactPct", 0)),
            dex_label=dex_label,
        )
    except Exception as e:
        return None


async def fetch_all_quotes(slippage_bps: int = 50) -> list[RouteQuote]:
    """Fetch all configured pair quotes concurrently."""
    async with httpx.AsyncClient() as client:
        tasks = [
            fetch_quote(client, pair, in_m, out_m, amt, slippage_bps)
            for pair, in_m, out_m, amt in PAIRS
        ]
        results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


# Also fetch direct-route-only quote to compare with aggregated route
async def fetch_quote_pair(
    pair: str,
    in_mint: str,
    out_mint: str,
    amount: int,
) -> tuple[Optional[RouteQuote], Optional[RouteQuote]]:
    """
    Returns (aggregated_quote, direct_only_quote).
    Spread between them = aggregator alpha (Jupiter routing efficiency).
    """
    async with httpx.AsyncClient() as client:
        agg, direct = await asyncio.gather(
            fetch_quote(client, pair, in_mint, out_mint, amount, 50),
            fetch_quote(client, pair + "_direct", in_mint, out_mint, amount, 50),
        )
    return agg, direct
