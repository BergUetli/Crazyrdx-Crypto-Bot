"""
db.py
SQLite persistence layer for the simulation.
Creates tables on first run, appends on subsequent runs.
"""

import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).parent / "sim_data.db"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS quotes (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          REAL NOT NULL,
            pair        TEXT NOT NULL,
            price       REAL NOT NULL,
            price_impact REAL,
            amount_in   INTEGER,
            amount_out  INTEGER,
            dex_label   TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS opportunities (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ts              REAL NOT NULL,
            pair            TEXT NOT NULL,
            price_a         REAL NOT NULL,
            price_b         REAL NOT NULL,
            spread_bps      REAL NOT NULL,
            trade_size_usd  REAL NOT NULL,
            gross_profit_usd REAL NOT NULL,
            fee_cost_usd    REAL NOT NULL,
            net_profit_usd  REAL NOT NULL,
            viable          INTEGER NOT NULL,   -- 1 if net > 0
            note            TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS sim_trades (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ts              REAL NOT NULL,
            pair            TEXT NOT NULL,
            direction       TEXT NOT NULL,      -- 'long' or 'short'
            entry_price     REAL NOT NULL,
            exit_price      REAL NOT NULL,
            trade_size_usd  REAL NOT NULL,
            gross_pnl_usd   REAL NOT NULL,
            fee_cost_usd    REAL NOT NULL,
            net_pnl_usd     REAL NOT NULL,
            cumulative_pnl  REAL NOT NULL,
            spread_bps      REAL NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def insert_quote(ts, pair, price, price_impact, amount_in, amount_out, dex_label):
    conn = get_conn()
    conn.execute(
        "INSERT INTO quotes (ts,pair,price,price_impact,amount_in,amount_out,dex_label) "
        "VALUES (?,?,?,?,?,?,?)",
        (ts, pair, price, price_impact, amount_in, amount_out, dex_label)
    )
    conn.commit()
    conn.close()


def insert_opportunity(ts, pair, price_a, price_b, spread_bps,
                        trade_size_usd, gross_profit_usd, fee_cost_usd,
                        net_profit_usd, viable, note=""):
    conn = get_conn()
    conn.execute(
        "INSERT INTO opportunities "
        "(ts,pair,price_a,price_b,spread_bps,trade_size_usd,gross_profit_usd,"
        "fee_cost_usd,net_profit_usd,viable,note) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (ts, pair, price_a, price_b, spread_bps, trade_size_usd,
         gross_profit_usd, fee_cost_usd, net_profit_usd, viable, note)
    )
    conn.commit()
    conn.close()


def insert_sim_trade(ts, pair, direction, entry_price, exit_price,
                     trade_size_usd, gross_pnl, fee_cost, net_pnl,
                     cumulative_pnl, spread_bps):
    conn = get_conn()
    conn.execute(
        "INSERT INTO sim_trades "
        "(ts,pair,direction,entry_price,exit_price,trade_size_usd,"
        "gross_pnl_usd,fee_cost_usd,net_pnl_usd,cumulative_pnl,spread_bps) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (ts, pair, direction, entry_price, exit_price, trade_size_usd,
         gross_pnl, fee_cost, net_pnl, cumulative_pnl, spread_bps)
    )
    conn.commit()
    conn.close()


def get_summary(since_ts: float = 0) -> dict:
    conn = get_conn()
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM quotes WHERE ts > ?", (since_ts,))
    quote_count = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM opportunities WHERE ts > ? AND viable=1", (since_ts,))
    viable_count = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM opportunities WHERE ts > ?", (since_ts,))
    opp_count = c.fetchone()[0]

    c.execute("SELECT SUM(net_pnl_usd), COUNT(*), MAX(cumulative_pnl) FROM sim_trades WHERE ts > ?", (since_ts,))
    row = c.fetchone()
    total_pnl = row[0] or 0.0
    trade_count = row[1] or 0
    peak_pnl = row[2] or 0.0

    c.execute("SELECT MAX(spread_bps) FROM opportunities WHERE ts > ?", (since_ts,))
    max_spread = c.fetchone()[0] or 0.0

    conn.close()
    return {
        "quotes": quote_count,
        "opportunities": opp_count,
        "viable": viable_count,
        "trades": trade_count,
        "total_pnl_usd": total_pnl,
        "peak_pnl_usd": peak_pnl,
        "max_spread_bps": max_spread,
    }
