"""
pool_monitor.py
Polls Raydium and Orca for pool state (liquidity, volume, fees).
"""

import asyncio
import time
import httpx
from typing import Optional, Dict, Any, List

from config import RAYDIUM_POOLS_URL, ORCA_WHIRLPOOLS_URL, PAIRS
from layer1.db import insert_pool_snapshot


async def fetch_raydium_pools() -> List[Dict[str, Any]]:
    """Fetch Raydium pool data."""
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(RAYDIUM_POOLS_URL, timeout=10.0)
            if r.status_code != 200:
                return []
            data = r.json()
            return data if isinstance(data, list) else []
    except Exception:
        return []


async def fetch_orca_whirlpools() -> List[Dict[str, Any]]:
    """Fetch Orca whirlpool data."""
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(ORCA_WHIRLPOOLS_URL, timeout=10.0)
            if r.status_code != 200:
                return []
            data = r.json()
            return data.get("whirlpools", []) if isinstance(data, dict) else []
    except Exception:
        return []


async def store_pool_snapshots():
    """Fetch and store pool snapshots from all DEXs."""
    ts = time.time()
    stored = 0

    # Raydium
    raydium_pools = await fetch_raydium_pools()
    for pool in raydium_pools:
        pair = pool.get("name", "").replace("/", "-")
        if not pair:
            continue
        insert_pool_snapshot(
            ts=ts,
            dex="raydium",
            pool_id=pool.get("ammId", ""),
            pair=pair,
            liquidity_usd=pool.get("liquidity"),
            volume_24h=pool.get("volume24h"),
            fee_tier=pool.get("fee"),
            token_a_reserve=None,
            token_b_reserve=None,
        )
        stored += 1

    # Orca
    orca_pools = await fetch_orca_whirlpools()
    for pool in orca_pools:
        token_a = pool.get("tokenA", {}).get("symbol", "")
        token_b = pool.get("tokenB", {}).get("symbol", "")
        pair = f"{token_a}-{token_b}" if token_a and token_b else ""
        if not pair:
            continue
        insert_pool_snapshot(
            ts=ts,
            dex="orca",
            pool_id=pool.get("address", ""),
            pair=pair,
            liquidity_usd=pool.get("tvl"),
            volume_24h=pool.get("volume", {}).get("day"),
            fee_tier=pool.get("feeRate"),
            token_a_reserve=pool.get("tokenVaultA", {}).get("amount"),
            token_b_reserve=pool.get("tokenVaultB", {}).get("amount"),
        )
        stored += 1

    return stored


async def run_cycle():
    """One complete pool monitoring cycle."""
    return await store_pool_snapshots()
