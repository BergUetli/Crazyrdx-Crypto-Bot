#!/usr/bin/env python3
"""
derivatives_collector.py
Collect free Binance futures derivatives data: funding rates, open interest,
top-trader and global long/short ratios, taker buy/sell ratio.

Why: these are the evidence-backed predictor families (funding extremes ->
mean reversion; OI build-ups -> cascade risk; positioning ratios -> crowding)
that the search cannot use until history exists. Binance's /futures/data/*
endpoints only retain ~30 days, so collection must run regularly (cron/hourly
alongside the candle downloader) — every missed month is unrecoverable.

Usage (single shot, safe to re-run — rows are upserted):
    python3 sim/layer1/derivatives_collector.py

Data lands in sim/data/derivatives.db, table derivs(symbol, metric, ts, value).
Feature-engine integration comes later; collection starts now so history
accumulates in the meantime.
"""

from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from config import DATA_DIR

DB_DERIVS = DATA_DIR / "derivatives.db"
FAPI = "https://fapi.binance.com"
SYMBOLS = ["SOLUSDT", "BTCUSDT", "ETHUSDT"]

# metric -> (path, params, timestamp field, value field)
ENDPOINTS: Dict[str, Tuple[str, Dict[str, Any], str, str]] = {
    "funding_rate": (
        "/fapi/v1/fundingRate", {"limit": 1000}, "fundingTime", "fundingRate"),
    "open_interest": (
        "/futures/data/openInterestHist", {"period": "1h", "limit": 500},
        "timestamp", "sumOpenInterest"),
    "open_interest_usd": (
        "/futures/data/openInterestHist", {"period": "1h", "limit": 500},
        "timestamp", "sumOpenInterestValue"),
    "top_ls_position_ratio": (
        "/futures/data/topLongShortPositionRatio", {"period": "1h", "limit": 500},
        "timestamp", "longShortRatio"),
    "global_ls_account_ratio": (
        "/futures/data/globalLongShortAccountRatio", {"period": "1h", "limit": 500},
        "timestamp", "longShortRatio"),
    "taker_buy_sell_ratio": (
        "/futures/data/takerlongshortRatio", {"period": "1h", "limit": 500},
        "timestamp", "buySellRatio"),
}


def init_db() -> sqlite3.Connection:
    DB_DERIVS.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_DERIVS))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS derivs (
            symbol  TEXT NOT NULL,
            metric  TEXT NOT NULL,
            ts      INTEGER NOT NULL,
            value   REAL NOT NULL,
            PRIMARY KEY (symbol, metric, ts)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_derivs_ts ON derivs(metric, ts)")
    return conn


def store_rows(
    conn: sqlite3.Connection,
    symbol: str,
    metric: str,
    rows: List[Dict[str, Any]],
    ts_field: str,
    val_field: str,
) -> int:
    """Upsert parsed rows; returns how many were stored. Pure — testable."""
    n = 0
    for row in rows:
        try:
            ts = int(row[ts_field])
            val = float(row[val_field])
        except (KeyError, TypeError, ValueError):
            continue
        conn.execute(
            "INSERT OR REPLACE INTO derivs (symbol, metric, ts, value) "
            "VALUES (?,?,?,?)",
            (symbol, metric, ts, val),
        )
        n += 1
    return n


def collect_once(client: httpx.Client, conn: sqlite3.Connection) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for symbol in SYMBOLS:
        for metric, (path, params, ts_field, val_field) in ENDPOINTS.items():
            try:
                resp = client.get(
                    FAPI + path, params={"symbol": symbol, **params}, timeout=20
                )
                resp.raise_for_status()
                rows = resp.json()
                if not isinstance(rows, list):
                    raise ValueError(f"unexpected payload: {str(rows)[:120]}")
                counts[f"{symbol}:{metric}"] = store_rows(
                    conn, symbol, metric, rows, ts_field, val_field
                )
            except Exception as e:
                counts[f"{symbol}:{metric}"] = -1
                print(f"  {symbol} {metric}: FAILED ({e})")
            time.sleep(0.3)  # gentle on rate limits
    conn.commit()
    return counts


def main() -> int:
    conn = init_db()
    try:
        with httpx.Client() as client:
            counts = collect_once(client, conn)
        total_rows = conn.execute("SELECT COUNT(*) FROM derivs").fetchone()[0]
        newest = conn.execute("SELECT MAX(ts) FROM derivs").fetchone()[0]
        ok = sum(1 for v in counts.values() if v >= 0)
        fail = sum(1 for v in counts.values() if v < 0)
        print(
            f"derivatives collect: {ok} feeds ok, {fail} failed, "
            f"db_rows={total_rows}, newest_ts={newest}"
        )
        return 0 if fail == 0 else 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
