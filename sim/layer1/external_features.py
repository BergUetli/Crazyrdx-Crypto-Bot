"""
external_features.py
Fetches funding rates, CEX-DEX basis, and DEX liquidity from free public APIs.

All data is cached to SQLite to avoid re-downloading. Offline-safe: if APIs
are unreachable, features default to 0.0 and the evolution continues.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from config import DATA_DIR

DB_EXTERNAL = DATA_DIR / "external_features.db"


def init_external_db():
    conn = sqlite3.connect(str(DB_EXTERNAL))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS funding_rates (
            ts INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            funding_rate REAL,
            next_funding_ts INTEGER,
            UNIQUE(ts, symbol)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dex_liquidity (
            ts INTEGER NOT NULL,
            pair TEXT NOT NULL,
            liquidity_usd REAL,
            price_usd REAL,
            UNIQUE(ts, pair)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cex_prices (
            ts INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            price REAL,
            UNIQUE(ts, symbol)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fund_ts ON funding_rates(ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dex_ts ON dex_liquidity(ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cex_ts ON cex_prices(ts)")
    conn.commit()
    conn.close()


def fetch_funding_rates(symbol: str = "SOLUSDT", limit: int = 1000) -> int:
    """Fetch historical funding rates from Binance Futures (free, no auth)."""
    init_external_db()
    url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit={limit}"
    try:
        resp = httpx.get(url, timeout=15)
        data = resp.json()
        if not isinstance(data, list):
            return 0
        conn = sqlite3.connect(str(DB_EXTERNAL))
        count = 0
        for row in data:
            ts = int(row.get("fundingTime", 0))
            rate = float(row.get("fundingRate", 0))
            if ts > 0:
                conn.execute(
                    "INSERT OR REPLACE INTO funding_rates (ts, symbol, funding_rate, next_funding_ts) VALUES (?,?,?,NULL)",
                    (ts, symbol, rate),
                )
                count += 1
        conn.commit()
        conn.close()
        return count
    except Exception:
        return 0


def fetch_cex_prices(symbol: str = "SOLUSDT", interval: str = "1h", limit: int = 1000) -> int:
    """Fetch CEX kline close prices for basis calculation."""
    init_external_db()
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        resp = httpx.get(url, timeout=15)
        data = resp.json()
        if not isinstance(data, list):
            return 0
        conn = sqlite3.connect(str(DB_EXTERNAL))
        count = 0
        for row in data:
            ts = int(row[0])
            close = float(row[4])
            conn.execute(
                "INSERT OR REPLACE INTO cex_prices (ts, symbol, price) VALUES (?,?,?)",
                (ts, symbol, close),
            )
            count += 1
        conn.commit()
        conn.close()
        return count
    except Exception:
        return 0


def fetch_dex_liquidity(pair: str = "SOL/USDC", limit: int = 500) -> int:
    """
    Fetch DEX liquidity snapshots from Jupiter price API (free).
    We poll current price/liquidity periodically and store a timestamped snapshot.
    For historical backfill, we interpolate from candle timestamps.
    """
    init_external_db()
    # Jupiter price API gives current spot price
    mint = "So11111111111111111111111111111111111111112"  # SOL
    url = f"https://price.jup.ag/v6/price?ids={mint}"
    try:
        resp = httpx.get(url, timeout=15)
        d = resp.json()
        price = float(d.get("data", {}).get(mint, {}).get("price", 0))
        if price <= 0:
            return 0
        ts = int(time.time() * 1000)
        conn = sqlite3.connect(str(DB_EXTERNAL))
        conn.execute(
            "INSERT OR REPLACE INTO dex_liquidity (ts, pair, liquidity_usd, price_usd) VALUES (?,?,?,?)",
            (ts, pair, 0.0, price),
        )
        conn.commit()
        conn.close()
        return 1
    except Exception:
        return 0


def get_funding_at(ts: int, symbol: str = "SOLUSDT") -> float:
    """Get the most recent funding rate at or before ts."""
    conn = sqlite3.connect(str(DB_EXTERNAL))
    row = conn.execute(
        "SELECT funding_rate FROM funding_rates WHERE symbol=? AND ts<=? ORDER BY ts DESC LIMIT 1",
        (symbol, ts),
    ).fetchone()
    conn.close()
    return float(row[0]) if row else 0.0


def get_funding_history(ts_start: int, ts_end: int, symbol: str = "SOLUSDT") -> List[Dict[str, Any]]:
    conn = sqlite3.connect(str(DB_EXTERNAL))
    rows = conn.execute(
        "SELECT ts, funding_rate FROM funding_rates WHERE symbol=? AND ts>=? AND ts<=? ORDER BY ts",
        (symbol, ts_start, ts_end),
    ).fetchall()
    conn.close()
    return [{"ts": r[0], "funding_rate": float(r[1])} for r in rows]


def get_cex_price_at(ts: int, symbol: str = "SOLUSDT") -> float:
    """Get CEX price closest to ts."""
    conn = sqlite3.connect(str(DB_EXTERNAL))
    row = conn.execute(
        "SELECT price FROM cex_prices WHERE symbol=? AND ts<=? ORDER BY ts DESC LIMIT 1",
        (symbol, ts),
    ).fetchone()
    conn.close()
    return float(row[0]) if row else 0.0


def get_cex_price_history(ts_start: int, ts_end: int, symbol: str = "SOLUSDT") -> List[Dict[str, Any]]:
    conn = sqlite3.connect(str(DB_EXTERNAL))
    rows = conn.execute(
        "SELECT ts, price FROM cex_prices WHERE symbol=? AND ts>=? AND ts<=? ORDER BY ts",
        (symbol, ts_start, ts_end),
    ).fetchall()
    conn.close()
    return [{"ts": r[0], "price": float(r[1])} for r in rows]


def backfill_all(symbol_cex: str = "SOLUSDT", pair_dex: str = "SOL/USDC") -> Dict[str, int]:
    """One-shot backfill of all external data."""
    n_fund = fetch_funding_rates(symbol_cex, limit=1000)
    n_cex = fetch_cex_prices(symbol_cex, interval="1h", limit=1000)
    n_dex = fetch_dex_liquidity(pair_dex)
    return {"funding_rates": n_fund, "cex_prices": n_cex, "dex_snapshots": n_dex}
