#!/usr/bin/env python3
"""
execution_probe.py
Hourly reality-check of the simulator's cost model against REAL Jupiter quotes.

The sim assumes 2.2 bps taker + $0.03 fixed + modeled MEV per side. This probe
fetches actual executable quotes for SOL/USDC at the book's trade sizes
($125 / $250), compares against the Binance mid, and records the true
round-trip cost in bps. No orders are ever placed — quotes only.

Usage (single shot; scheduled hourly via LaunchAgent):
    python3 sim/layer1/execution_probe.py

Data: sim/data/execution_probe.db, table quotes(...). After a week this gives
an empirical cost curve; if it diverges from the sim's assumptions, the sim
gets recalibrated BEFORE any live decision.
"""

from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from config import DATA_DIR, PAIRS

DB_PROBE = DATA_DIR / "execution_probe.db"
QUOTE_URLS = [
    "https://lite-api.jup.ag/swap/v1/quote",  # keyless tier
    "https://api.jup.ag/swap/v1/quote",
]
BINANCE_TICKER = "https://api.binance.com/api/v3/ticker/price?symbol=SOLUSDT"
SIZES_USD = [125.0, 250.0]
SOL_MINT = PAIRS["SOL/USDC"]["input_mint"]
USDC_MINT = PAIRS["SOL/USDC"]["output_mint"]


def init_db() -> sqlite3.Connection:
    DB_PROBE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PROBE))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS quotes (
            ts            REAL NOT NULL,
            side          TEXT NOT NULL,   -- buy_sol | sell_sol
            size_usd      REAL NOT NULL,
            binance_mid   REAL,
            eff_price     REAL,            -- USDC per SOL actually quoted
            cost_bps      REAL,            -- vs binance mid, positive = worse
            impact_pct    REAL,            -- Jupiter priceImpactPct
            route_hops    INTEGER,
            source        TEXT,            -- which quote endpoint answered
            error         TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_probe_ts ON quotes(ts)")
    cols = {r[1] for r in conn.execute("PRAGMA table_info(quotes)")}
    if "instrument" not in cols:
        conn.execute("ALTER TABLE quotes ADD COLUMN instrument TEXT "
                     "NOT NULL DEFAULT 'SOL/USDC'")
        conn.commit()
    return conn


def binance_mid(client: httpx.Client) -> Optional[float]:
    try:
        r = client.get(BINANCE_TICKER, timeout=10)
        r.raise_for_status()
        return float(r.json()["price"])
    except Exception:
        return None


def jupiter_quote(
    client: httpx.Client, in_mint: str, out_mint: str, amount_raw: int
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Try quote endpoints in order; return (payload, source)."""
    params = {
        "inputMint": in_mint, "outputMint": out_mint,
        "amount": str(amount_raw), "slippageBps": "50",
    }
    last_err = "no endpoint"
    for url in QUOTE_URLS:
        try:
            r = client.get(url, params=params, timeout=15)
            r.raise_for_status()
            d = r.json()
            if d.get("outAmount"):
                return d, url.split("//")[1].split("/")[0]
            last_err = f"empty quote from {url}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
    return None, last_err


def _binance_mid_sym(client: httpx.Client, symbol: str) -> Optional[float]:
    try:
        r = client.get("https://api.binance.com/api/v3/ticker/price",
                       params={"symbol": symbol}, timeout=10)
        r.raise_for_status()
        return float(r.json()["price"])
    except Exception:
        return None


def probe_once(client: httpx.Client, conn: sqlite3.Connection) -> Dict[str, Any]:
    """One measurement pass over every tradeable instrument: real Jupiter
    buy and sell quotes at each size vs the Binance mid. Keeps each
    instrument's fee assumption (instruments.py) honest over time."""
    from instruments import INSTRUMENTS
    now = time.time()
    results = []
    per_inst: Dict[str, list] = {}
    for inst, meta in INSTRUMENTS.items():
        mint, dec = meta["mint"], 10 ** int(meta["decimals"])
        legacy = inst == "SOL/USDC"  # SOL rows keep their original side names
        buy_side, sell_side = ("buy_sol", "sell_sol") if legacy else ("buy", "sell")
        mid = _binance_mid_sym(client, meta["binance"])
        for size in SIZES_USD:
            q, src = jupiter_quote(client, USDC_MINT, mint, int(size * 1e6))
            if q and mid:
                qty = int(q["outAmount"]) / dec
                eff = size / qty if qty > 0 else None
                cost = (eff / mid - 1.0) * 1e4 if eff else None
                row = (now, buy_side, size, mid, eff, cost,
                       float(q.get("priceImpactPct") or 0),
                       len(q.get("routePlan") or []), src, None, inst)
            else:
                row = (now, buy_side, size, mid, None, None, None, None,
                       None, src, inst)
            conn.execute("INSERT INTO quotes VALUES (?,?,?,?,?,?,?,?,?,?,?)", row)
            results.append(row)
            if mid:
                qty_in = size / mid
                q, src = jupiter_quote(client, mint, USDC_MINT, int(qty_in * dec))
                if q:
                    usdc_out = int(q["outAmount"]) / 1e6
                    eff = usdc_out / qty_in if qty_in > 0 else None
                    cost = (1.0 - eff / mid) * 1e4 if eff else None
                    row = (now, sell_side, size, mid, eff, cost,
                           float(q.get("priceImpactPct") or 0),
                           len(q.get("routePlan") or []), src, None, inst)
                else:
                    row = (now, sell_side, size, mid, None, None, None, None,
                           None, src, inst)
                conn.execute("INSERT INTO quotes VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                             row)
                results.append(row)
        per_inst[inst] = [r[5] for r in results if r[10] == inst and r[5] is not None]
        time.sleep(0.3)  # stay polite to the keyless quote tier
    conn.commit()
    ok = [r for r in results if r[5] is not None]
    return {
        "n": len(results), "ok": len(ok),
        "per_instrument_bps": {k: round(sum(v) / len(v), 2)
                               for k, v in per_inst.items() if v},
        "avg_cost_bps": (sum(r[5] for r in ok) / len(ok)) if ok else None,
    }


def summary(days: float = 7.0) -> Dict[str, Any]:
    """Empirical cost curve vs the sim's assumptions (for reports/dashboard)."""
    try:
        conn = sqlite3.connect(str(DB_PROBE))
        cut = time.time() - days * 86400
        rows = conn.execute(
            "SELECT size_usd, AVG(cost_bps), COUNT(*) FROM quotes "
            "WHERE ts > ? AND cost_bps IS NOT NULL AND instrument='SOL/USDC' "
            "GROUP BY size_usd", (cut,)
        ).fetchall()
        inst_rows = conn.execute(
            "SELECT instrument, AVG(cost_bps), COUNT(*) FROM quotes "
            "WHERE ts > ? AND cost_bps IS NOT NULL GROUP BY instrument", (cut,)
        ).fetchall()
        conn.close()
        return {
            "per_size_bps": {r[0]: round(r[1], 2) for r in rows},
            "per_instrument_bps": {r[0]: round(r[1], 2) for r in inst_rows},
            "n": sum(r[2] for r in rows),
            "sim_assumption_bps_per_side": "2.2 + fixed $0.03 + MEV model",
        }
    except Exception as e:
        return {"error": str(e)}


def main() -> int:
    conn = init_db()
    try:
        with httpx.Client() as client:
            s = probe_once(client, conn)
        print(f"execution probe: {s['ok']}/{s['n']} quotes ok, one-side "
              f"cost bps by instrument: {s['per_instrument_bps']}")
        return 0 if s["ok"] else 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
