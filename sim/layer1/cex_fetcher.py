"""
cex_fetcher.py
Polls Binance and Coinbase for CEX prices to detect CEX-DEX spreads.
"""

import asyncio
import time
import httpx
from typing import Optional, Dict, Any, List

from config import BINANCE_TICKER_URL, COINBASE_TICKER_URL
from layer1.db import insert_cex_feed


# Map our pairs to exchange symbols
BINANCE_SYMBOLS = {
    "SOL/USDC": "SOLUSDC",
    "SOL/USDT": "SOLUSDT",
    "BONK/USDC": "BONKUSDC",
    "WIF/USDC": "WIFUSDC",
}

COINBASE_SYMBOLS = {
    "SOL/USDC": "SOL-USD",
    "SOL/USDT": "SOL-USD",  # Coinbase doesn't have USDT pair for SOL
    "BONK/USDC": "BONK-USD",
    "WIF/USDC": "WIF-USD",
}


async def fetch_binance_prices() -> List[Dict[str, Any]]:
    """Fetch prices from Binance."""
    results = []
    try:
        async with httpx.AsyncClient() as client:
            for pair_name, symbol in BINANCE_SYMBOLS.items():
                try:
                    r = await client.get(
                        BINANCE_TICKER_URL,
                        params={"symbol": symbol},
                        timeout=8.0
                    )
                    if r.status_code == 200:
                        data = r.json()
                        results.append({
                            "exchange": "binance",
                            "pair": pair_name,
                            "price": float(data["price"]),
                            "volume_24h": None,
                        })
                except Exception:
                    continue
    except Exception:
        pass
    return results


async def fetch_coinbase_prices() -> List[Dict[str, Any]]:
    """Fetch prices from Coinbase."""
    results = []
    try:
        async with httpx.AsyncClient() as client:
            for pair_name, symbol in COINBASE_SYMBOLS.items():
                try:
                    r = await client.get(
                        f"{COINBASE_TICKER_URL}/{symbol}/spot",
                        timeout=8.0
                    )
                    if r.status_code == 200:
                        data = r.json()
                        results.append({
                            "exchange": "coinbase",
                            "pair": pair_name,
                            "price": float(data["data"]["amount"]),
                            "volume_24h": None,
                        })
                except Exception:
                    continue
    except Exception:
        pass
    return results


async def store_cex_feeds():
    """Fetch and store CEX prices from all exchanges."""
    ts = time.time()
    stored = 0

    binance_prices = await fetch_binance_prices()
    for p in binance_prices:
        insert_cex_feed(ts, p["exchange"], p["pair"], p["price"], p["volume_24h"])
        stored += 1

    coinbase_prices = await fetch_coinbase_prices()
    for p in coinbase_prices:
        insert_cex_feed(ts, p["exchange"], p["pair"], p["price"], p["volume_24h"])
        stored += 1

    return stored


async def run_cycle():
    """One complete CEX fetch cycle."""
    return await store_cex_feeds()
