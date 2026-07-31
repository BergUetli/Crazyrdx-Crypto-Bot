"""
db.py
SQLite storage layer for all simulation data.
"""

import sqlite3
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from config import DB_RAW_QUOTES, DB_POOLS, DB_CEX, DB_NETWORK, DB_LABELS, DB_FEATURES


def get_conn(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_all_dbs():
    """Initialize all databases with schemas."""
    _init_raw_quotes()
    _init_pools()
    _init_cex()
    _init_network()
    _init_labels()
    _init_features()


def _init_raw_quotes():
    conn = get_conn(DB_RAW_QUOTES)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS quotes (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ts              REAL NOT NULL,
            pair            TEXT NOT NULL,
            input_mint      TEXT NOT NULL,
            output_mint     TEXT NOT NULL,
            amount_in       INTEGER NOT NULL,
            amount_out      INTEGER NOT NULL,
            price           REAL NOT NULL,
            price_impact_pct REAL,
            dex_label       TEXT,
            slippage_bps    INTEGER,
            created_at      REAL DEFAULT (strftime('%s','now'))
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_quotes_ts ON quotes(ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_quotes_pair_ts ON quotes(pair, ts)")
    conn.commit()
    conn.close()


def _init_pools():
    conn = get_conn(DB_POOLS)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pool_snapshots (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ts              REAL NOT NULL,
            dex             TEXT NOT NULL,
            pool_id         TEXT NOT NULL,
            pair            TEXT NOT NULL,
            liquidity_usd   REAL,
            volume_24h      REAL,
            fee_tier        REAL,
            token_a_reserve REAL,
            token_b_reserve REAL,
            created_at      REAL DEFAULT (strftime('%s','now'))
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pools_ts ON pool_snapshots(ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pools_pair_ts ON pool_snapshots(pair, ts)")
    conn.commit()
    conn.close()


def _init_cex():
    conn = get_conn(DB_CEX)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cex_feeds (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ts              REAL NOT NULL,
            exchange        TEXT NOT NULL,
            pair            TEXT NOT NULL,
            price           REAL NOT NULL,
            volume_24h      REAL,
            created_at      REAL DEFAULT (strftime('%s','now'))
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cex_ts ON cex_feeds(ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cex_pair_ts ON cex_feeds(pair, ts)")
    conn.commit()
    conn.close()


def _init_network():
    conn = get_conn(DB_NETWORK)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS network_stats (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ts              REAL NOT NULL,
            slot_time_ms    INTEGER,
            tps             REAL,
            congestion_score REAL,
            compute_unit_price REAL,
            created_at      REAL DEFAULT (strftime('%s','now'))
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_network_ts ON network_stats(ts)")
    conn.commit()
    conn.close()


def _init_labels():
    conn = get_conn(DB_LABELS)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS spread_events (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ts              REAL NOT NULL,
            pair            TEXT NOT NULL,
            initial_spread_bps REAL NOT NULL,
            initial_price   REAL NOT NULL,
            label           TEXT,
            label_ts        REAL,
            spread_at_5s    REAL,
            spread_at_10s   REAL,
            spread_at_30s   REAL,
            spread_at_60s   REAL,
            max_spread      REAL,
            duration_s      REAL,
            labeled         INTEGER DEFAULT 0,
            created_at      REAL DEFAULT (strftime('%s','now'))
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_labels_ts ON spread_events(ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_labels_labeled ON spread_events(labeled)")
    conn.commit()
    conn.close()


def _init_features():
    conn = get_conn(DB_FEATURES)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feature_vectors (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ts              REAL NOT NULL,
            pair            TEXT NOT NULL,
            features_json   TEXT NOT NULL,
            created_at      REAL DEFAULT (strftime('%s','now'))
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_features_ts ON feature_vectors(ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_features_pair_ts ON feature_vectors(pair, ts)")
    conn.commit()
    conn.close()


# --- Insert helpers ---

def insert_quote(ts: float, pair: str, input_mint: str, output_mint: str,
                 amount_in: int, amount_out: int, price: float,
                 price_impact_pct: Optional[float], dex_label: Optional[str],
                 slippage_bps: Optional[int]):
    conn = get_conn(DB_RAW_QUOTES)
    conn.execute(
        "INSERT INTO quotes (ts,pair,input_mint,output_mint,amount_in,amount_out,"
        "price,price_impact_pct,dex_label,slippage_bps) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (ts, pair, input_mint, output_mint, amount_in, amount_out,
         price, price_impact_pct, dex_label, slippage_bps)
    )
    conn.commit()
    conn.close()


def insert_pool_snapshot(ts: float, dex: str, pool_id: str, pair: str,
                          liquidity_usd: Optional[float], volume_24h: Optional[float],
                          fee_tier: Optional[float], token_a_reserve: Optional[float],
                          token_b_reserve: Optional[float]):
    conn = get_conn(DB_POOLS)
    conn.execute(
        "INSERT INTO pool_snapshots (ts,dex,pool_id,pair,liquidity_usd,volume_24h,"
        "fee_tier,token_a_reserve,token_b_reserve) VALUES (?,?,?,?,?,?,?,?,?)",
        (ts, dex, pool_id, pair, liquidity_usd, volume_24h, fee_tier,
         token_a_reserve, token_b_reserve)
    )
    conn.commit()
    conn.close()


def insert_cex_feed(ts: float, exchange: str, pair: str, price: float,
                     volume_24h: Optional[float]):
    conn = get_conn(DB_CEX)
    conn.execute(
        "INSERT INTO cex_feeds (ts,exchange,pair,price,volume_24h) VALUES (?,?,?,?,?)",
        (ts, exchange, pair, price, volume_24h)
    )
    conn.commit()
    conn.close()


def insert_network_stats(ts: float, slot_time_ms: Optional[int],
                          tps: Optional[float], congestion_score: Optional[float],
                          compute_unit_price: Optional[float]):
    conn = get_conn(DB_NETWORK)
    conn.execute(
        "INSERT INTO network_stats (ts,slot_time_ms,tps,congestion_score,compute_unit_price) "
        "VALUES (?,?,?,?,?)",
        (ts, slot_time_ms, tps, congestion_score, compute_unit_price)
    )
    conn.commit()
    conn.close()


def insert_spread_event(ts: float, pair: str, initial_spread_bps: float,
                         initial_price: float):
    conn = get_conn(DB_LABELS)
    cursor = conn.execute(
        "INSERT INTO spread_events (ts,pair,initial_spread_bps,initial_price) "
        "VALUES (?,?,?,?)",
        (ts, pair, initial_spread_bps, initial_price)
    )
    event_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return event_id


def update_spread_event_label(event_id: int, label: str, label_ts: float,
                                spread_at_5s: Optional[float], spread_at_10s: Optional[float],
                                spread_at_30s: Optional[float], spread_at_60s: Optional[float],
                                max_spread: Optional[float], duration_s: Optional[float]):
    conn = get_conn(DB_LABELS)
    conn.execute(
        "UPDATE spread_events SET label=?, label_ts=?, spread_at_5s=?, spread_at_10s=?, "
        "spread_at_30s=?, spread_at_60s=?, max_spread=?, duration_s=?, labeled=1 "
        "WHERE id=?",
        (label, label_ts, spread_at_5s, spread_at_10s, spread_at_30s, spread_at_60s,
         max_spread, duration_s, event_id)
    )
    conn.commit()
    conn.close()


def insert_feature_vector(ts: float, pair: str, features_json: str):
    conn = get_conn(DB_FEATURES)
    conn.execute(
        "INSERT INTO feature_vectors (ts,pair,features_json) VALUES (?,?,?)",
        (ts, pair, features_json)
    )
    conn.commit()
    conn.close()


# --- Query helpers ---

def get_quotes_since(ts: float, pair: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_conn(DB_RAW_QUOTES)
    if pair:
        cursor = conn.execute(
            "SELECT * FROM quotes WHERE ts > ? AND pair = ? ORDER BY ts",
            (ts, pair)
        )
    else:
        cursor = conn.execute(
            "SELECT * FROM quotes WHERE ts > ? ORDER BY ts",
            (ts,)
        )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_unlabeled_events(older_than_ts: float) -> List[Dict[str, Any]]:
    conn = get_conn(DB_LABELS)
    cursor = conn.execute(
        "SELECT * FROM spread_events WHERE labeled = 0 AND ts < ? ORDER BY ts",
        (older_than_ts,)
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_latest_price(pair: str) -> Optional[float]:
    conn = get_conn(DB_RAW_QUOTES)
    cursor = conn.execute(
        "SELECT price FROM quotes WHERE pair = ? ORDER BY ts DESC LIMIT 1",
        (pair,)
    )
    row = cursor.fetchone()
    conn.close()
    return row["price"] if row else None


def get_spread_history(pair: str, window_s: int = 300) -> List[float]:
    """Get spread history for a pair over the last N seconds."""
    cutoff = time.time() - window_s
    conn = get_conn(DB_RAW_QUOTES)
    cursor = conn.execute(
        "SELECT price FROM quotes WHERE pair = ? AND ts > ? ORDER BY ts",
        (pair, cutoff)
    )
    prices = [r["price"] for r in cursor.fetchall()]
    conn.close()
    if len(prices) < 2:
        return []
    spreads = []
    for i in range(1, len(prices)):
        spread = abs(prices[i] - prices[i-1]) / prices[i-1] * 10000
        spreads.append(spread)
    return spreads
