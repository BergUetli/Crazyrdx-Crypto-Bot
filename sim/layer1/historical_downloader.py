"""
historical_downloader.py
Downloads historical 5-minute candles from Binance for backtesting.
"""

import json
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

import httpx

from config import DATA_DIR

DB_HISTORICAL = DATA_DIR / "historical_candles.db"


def init_historical_db():
    conn = sqlite3.connect(str(DB_HISTORICAL))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS candles (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ts              INTEGER NOT NULL,      -- open time (ms)
            pair            TEXT NOT NULL,
            open            REAL NOT NULL,
            high            REAL NOT NULL,
            low             REAL NOT NULL,
            close           REAL NOT NULL,
            volume          REAL NOT NULL,
            close_time      INTEGER NOT NULL,
            quote_volume    REAL,
            trades          INTEGER,
            taker_buy_vol   REAL,
            taker_buy_quote REAL,
            created_at      REAL DEFAULT (strftime('%s','now')),
            UNIQUE(ts, pair)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS candles_1h (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ts              INTEGER NOT NULL,
            pair            TEXT NOT NULL,
            open            REAL NOT NULL,
            high            REAL NOT NULL,
            low             REAL NOT NULL,
            close           REAL NOT NULL,
            volume          REAL NOT NULL,
            close_time      INTEGER NOT NULL,
            quote_volume    REAL,
            trades          INTEGER,
            taker_buy_vol   REAL,
            taker_buy_quote REAL,
            created_at      REAL DEFAULT (strftime('%s','now')),
            UNIQUE(ts, pair)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_candles_ts ON candles(ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_candles_pair_ts ON candles(pair, ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_candles_1h_ts ON candles_1h(ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_candles_1h_pair_ts ON candles_1h(pair, ts)")
    conn.commit()
    conn.close()


def get_binance_symbol(pair: str) -> str:
    """Convert our pair names to Binance symbols."""
    mapping = {
        "SOL/USDC": "SOLUSDC",
        "SOL/USDT": "SOLUSDT",
        "BONK/USDC": "BONKUSDC",
        "WIF/USDC": "WIFUSDC",
    }
    return mapping.get(pair, pair.replace("/", ""))


async def download_candles(
    pair: str,
    start_ts: int,
    end_ts: int,
    interval: str = "5m",
) -> int:
    """
    Download candles from Binance for a time range.

    Args:
        pair: e.g., "SOL/USDC"
        start_ts: start time in milliseconds
        end_ts: end time in milliseconds
        interval: candle interval (default 5m)

    Returns:
        Number of candles downloaded
    """
    symbol = get_binance_symbol(pair)
    url = "https://api.binance.com/api/v3/klines"

    # Choose table based on interval
    table = "candles_1h" if interval == "1h" else "candles"

    conn = sqlite3.connect(str(DB_HISTORICAL))
    total_downloaded = 0

    current_start = start_ts
    limit = 1000  # Binance max per request

    async with httpx.AsyncClient() as client:
        while current_start < end_ts:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_start,
                "endTime": end_ts,
                "limit": limit,
            }

            try:
                r = await client.get(url, params=params, timeout=15.0)
                if r.status_code != 200:
                    print(f"  Error {r.status_code} for {pair}")
                    break

                data = r.json()
                if not data:
                    break

                # Insert into DB
                inserted = 0
                for candle in data:
                    try:
                        conn.execute(f"""
                            INSERT OR IGNORE INTO {table}
                            (ts, pair, open, high, low, close, volume, close_time,
                             quote_volume, trades, taker_buy_vol, taker_buy_quote)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            candle[0], pair, candle[1], candle[2], candle[3],
                            candle[4], candle[5], candle[6], candle[7],
                            candle[8], candle[9], candle[10]
                        ))
                        inserted += 1
                    except Exception:
                        pass

                conn.commit()
                total_downloaded += inserted

                # Move to next batch
                last_ts = data[-1][0]
                current_start = last_ts + 1

                # Rate limiting
                await asyncio.sleep(0.1)

            except Exception as e:
                print(f"  Exception downloading {pair}: {e}")
                break

    conn.close()
    return total_downloaded


async def download_all_pairs(
    days: int = 30,
    interval: str = "5m",
) -> Dict[str, int]:
    """
    Download historical data for all configured pairs.

    Args:
        days: number of days of history to download
        interval: candle interval

    Returns:
        Dict of pair -> candles downloaded
    """
    init_historical_db()

    end_ts = int(time.time() * 1000)
    start_ts = end_ts - (days * 24 * 60 * 60 * 1000)

    results = {}

    for pair in ["SOL/USDC", "SOL/USDT", "BONK/USDC", "WIF/USDC"]:
        print(f"Downloading {pair}...")
        count = await download_candles(pair, start_ts, end_ts, interval)
        results[pair] = count
        print(f"  -> {count} candles")

    return results


def get_candle_count(pair: Optional[str] = None) -> int:
    """Get number of candles in DB."""
    conn = sqlite3.connect(str(DB_HISTORICAL))
    if pair:
        cursor = conn.execute("SELECT COUNT(*) FROM candles WHERE pair = ?", (pair,))
    else:
        cursor = conn.execute("SELECT COUNT(*) FROM candles")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_candles(
    pair: str,
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
    limit: Optional[int] = None,
    interval: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve candles from DB."""
    conn = sqlite3.connect(str(DB_HISTORICAL))
    conn.row_factory = sqlite3.Row

    # Choose table based on interval
    if interval == "1h":
        table = "candles_1h"
    else:
        table = "candles"

    query = f"SELECT * FROM {table} WHERE pair = ?"
    params = [pair]

    if start_ts:
        query += " AND ts >= ?"
        params.append(start_ts)
    if end_ts:
        query += " AND ts <= ?"
        params.append(end_ts)

    query += " ORDER BY ts"

    if limit:
        query += f" LIMIT {limit}"

    cursor = conn.execute(query, params)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


import asyncio  # noqa: E402


if __name__ == "__main__":
    # Quick test
    async def test():
        results = await download_all_pairs(days=1)
        print(f"Downloaded: {results}")
        print(f"Total candles: {get_candle_count()}")

    asyncio.run(test())
