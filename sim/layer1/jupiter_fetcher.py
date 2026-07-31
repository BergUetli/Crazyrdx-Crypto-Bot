"""
jupiter_fetcher.py
Polls Jupiter API for quotes across all configured pairs.
"""

import asyncio
import time
import httpx
from typing import Optional, Dict, Any
from dataclasses import dataclass

from config import PAIRS, JUPITER_QUOTE_URL
from layer1.db import insert_quote


@dataclass
class JupiterQuote:
    ts: float
    pair: str
    input_mint: str
    output_mint: str
    amount_in: int
    amount_out: int
    price: float
    price_impact_pct: Optional[float]
    dex_label: Optional[str]
    slippage_bps: Optional[int]


async def fetch_single_quote(
    client: httpx.AsyncClient,
    pair_name: str,
    pair_config: Dict[str, Any],
    slippage_bps: int = 50,
) -> Optional[JupiterQuote]:
    """Fetch one quote from Jupiter."""
    params = {
        "inputMint": pair_config["input_mint"],
        "outputMint": pair_config["output_mint"],
        "amount": pair_config["trade_amount_raw"],
        "slippageBps": slippage_bps,
        "onlyDirectRoutes": "false",
    }

    try:
        r = await client.get(JUPITER_QUOTE_URL, params=params, timeout=8.0)
        if r.status_code != 200:
            return None

        data = r.json()

        amount_out = int(data["outAmount"])
        in_dec = pair_config["input_decimals"]
        out_dec = pair_config["output_decimals"]
        price = (amount_out / 10**out_dec) / (pair_config["trade_amount_raw"] / 10**in_dec)

        route_plan = data.get("routePlan", [])
        if route_plan:
            swap_info = route_plan[0].get("swapInfo", {})
            dex_label = swap_info.get("label", "Unknown")
        else:
            dex_label = "Unknown"

        return JupiterQuote(
            ts=time.time(),
            pair=pair_name,
            input_mint=pair_config["input_mint"],
            output_mint=pair_config["output_mint"],
            amount_in=pair_config["trade_amount_raw"],
            amount_out=amount_out,
            price=price,
            price_impact_pct=float(data.get("priceImpactPct", 0)),
            dex_label=dex_label,
            slippage_bps=slippage_bps,
        )

    except Exception:
        return None


async def fetch_all_pairs() -> list[JupiterQuote]:
    """Fetch quotes for all configured pairs concurrently."""
    async with httpx.AsyncClient() as client:
        tasks = [
            fetch_single_quote(client, name, cfg)
            for name, cfg in PAIRS.items()
        ]
        results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


async def store_quotes(quotes: list[JupiterQuote]):
    """Store fetched quotes in SQLite."""
    for q in quotes:
        insert_quote(
            ts=q.ts,
            pair=q.pair,
            input_mint=q.input_mint,
            output_mint=q.output_mint,
            amount_in=q.amount_in,
            amount_out=q.amount_out,
            price=q.price,
            price_impact_pct=q.price_impact_pct,
            dex_label=q.dex_label,
            slippage_bps=q.slippage_bps,
        )


async def run_cycle():
    """One complete fetch-and-store cycle."""
    quotes = await fetch_all_pairs()
    if quotes:
        await store_quotes(quotes)
    return len(quotes)
