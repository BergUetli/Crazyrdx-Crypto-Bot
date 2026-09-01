#!/usr/bin/env python3
"""
paper_trader.py — the PAPER stage: live-quote shadow trading of frozen champions.

Every run (scheduled ~15min via LaunchAgent):
  1. Enroll new champion families from champions.json (frozen copy, 30-day
     term, max 8 concurrent, each with its own virtual BOOK_USD book).
  2. On each NEW closed 1h bar since the last run: evaluate each enrolled
     genome's entry signal; open positions filled at the REAL Jupiter quote
     (fallback: Binance mid, marked); check exit rules (SL/TP/trail/time,
     gap-aware, engine-consistent) against the bar's OHLC.
  3. Record trades and equity; grade each enrollment against the PAPER bars
     from success_criteria (days, trades, net, drawdown).
  4. Write sim/data/paper_status.json for the dashboard.

NO real orders are ever placed. This is measurement software.
v1 fidelity notes: entries at next available real quote after a signal bar
closes; sizing = genome fraction capped at 50% of book (no vol overlay);
network fixed cost applied per side; Jupiter quote embeds route fees/impact.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SIM = Path(__file__).resolve().parent
sys.path.insert(0, str(SIM))

from config import DATA_DIR
from success_criteria import (
    BOOK_USD, FIXED_COST_PER_SIDE_USD,
    PAPER_MIN_DAYS, PAPER_MIN_TRADES, PAPER_MIN_NET_PNL_USD,
    PAPER_MAX_DRAWDOWN_HARD,
)

DB_PAPER = DATA_DIR / "paper_trader.db"
STATUS_JSON = DATA_DIR / "paper_status.json"
MAX_CONCURRENT = 8
TERM_DAYS = 35  # trade a few days past 30 so the 30d window is fully covered
MIN_HOLD_BARS = 2
COOLDOWN_BARS = 4


def _conn() -> sqlite3.Connection:
    DB_PAPER.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PAPER), timeout=15)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=15000")
    conn.execute("""CREATE TABLE IF NOT EXISTS enrollments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        genome_id TEXT, family TEXT UNIQUE, genome_json TEXT,
        enrolled_ts REAL, status TEXT DEFAULT 'active',
        cash REAL, last_bar_ts INTEGER DEFAULT 0,
        cooldown_until_ts INTEGER DEFAULT 0)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS positions (
        enrollment_id INTEGER PRIMARY KEY,
        entry_ts INTEGER, entry_price REAL, size_usd REAL, sol_qty REAL,
        best_price REAL, entry_fill_source TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        enrollment_id INTEGER, entry_ts INTEGER, exit_ts INTEGER,
        entry_price REAL, exit_price REAL, size_usd REAL,
        net_pnl REAL, exit_reason TEXT, fill_source TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS equity (
        enrollment_id INTEGER, ts INTEGER, equity REAL,
        PRIMARY KEY (enrollment_id, ts))""")
    return conn


# ---------------------------------------------------------------------------
# Market access (quotes only, never orders)
# ---------------------------------------------------------------------------

def live_price(size_usd: float, side: str) -> Dict[str, Any]:
    """Real executable price via the Jupiter probe helpers; Binance fallback."""
    try:
        import httpx
        from layer1.execution_probe import (
            jupiter_quote, binance_mid, SOL_MINT, USDC_MINT,
        )
        with httpx.Client() as client:
            mid = binance_mid(client)
            if side == "buy":
                q, src = jupiter_quote(client, USDC_MINT, SOL_MINT,
                                       int(size_usd * 1e6))
                if q:
                    sol = int(q["outAmount"]) / 1e9
                    if sol > 0:
                        return {"price": size_usd / sol, "source": f"jupiter:{src}",
                                "mid": mid}
            else:
                if mid:
                    q, src = jupiter_quote(client, SOL_MINT, USDC_MINT,
                                           int(size_usd / mid * 1e9))
                    if q:
                        usdc = int(q["outAmount"]) / 1e6
                        sol_in = size_usd / mid
                        if sol_in > 0:
                            return {"price": usdc / sol_in,
                                    "source": f"jupiter:{src}", "mid": mid}
            if mid:
                return {"price": mid, "source": "binance_mid", "mid": mid}
    except Exception:
        pass
    return {"price": None, "source": "unavailable", "mid": None}


# ---------------------------------------------------------------------------
# Core tick
# ---------------------------------------------------------------------------

def enroll_new(conn: sqlite3.Connection, verbose: bool = True) -> int:
    from evolution.strategy_log import family_key
    n_active = conn.execute(
        "SELECT COUNT(*) FROM enrollments WHERE status='active'").fetchone()[0]
    if n_active >= MAX_CONCURRENT:
        return 0
    try:
        champs = json.loads(
            (SIM / "evolution" / "champions.json").read_text()
        ).get("champions", [])
    except Exception:
        return 0
    added = 0
    for c in champs:
        if n_active + added >= MAX_CONCURRENT:
            break
        g = c.get("genome") or {}
        fam = family_key(g.get("entry_logic") or "?", [
            x.get("indicator") for x in g.get("entry_conditions", [])])
        try:
            conn.execute(
                "INSERT INTO enrollments (genome_id, family, genome_json, "
                "enrolled_ts, cash) VALUES (?,?,?,?,?)",
                (c.get("genome_id"), fam, json.dumps(g), time.time(), BOOK_USD))
            added += 1
            if verbose:
                print(f"  [paper] enrolled {g.get('entry_logic')} "
                      f"{str(c.get('genome_id'))[:28]} (30d clock started)")
        except sqlite3.IntegrityError:
            continue  # family already enrolled (ever) — one shot per family
    conn.commit()
    return added


def _exit_rules(genome) -> List[Dict[str, Any]]:
    from layer1.backtest_engine import BacktestEngine
    return BacktestEngine()._normalize_exit_rules(genome.exit_rules, bar_hours=1.0)


def process_bars(conn: sqlite3.Connection, verbose: bool = True) -> Dict[str, int]:
    """Advance every active enrollment over any new closed bars."""
    from evolution.genome import StrategyGenome
    from evolution.evaluator import GenomeEvaluator
    from layer1.historical_feature_engine_1h import get_historical_features_1h

    features = get_historical_features_1h("SOL/USDC", limit=250)
    if len(features) < 60:
        return {"bars": 0, "entries": 0, "exits": 0}
    stats = {"bars": 0, "entries": 0, "exits": 0}
    ev = GenomeEvaluator(features, augment=True)
    feats = ev.features

    for (eid, gjson, cash, last_bar, cooldown) in conn.execute(
        "SELECT id, genome_json, cash, last_bar_ts, cooldown_until_ts "
        "FROM enrollments WHERE status='active'").fetchall():
        genome = StrategyGenome.from_dict(json.loads(gjson))
        signal_fn = None
        try:
            from layer1.fast_signals import build_array_signal_fn
            signal_fn = build_array_signal_fn(genome, feats)
        except Exception:
            pass
        if signal_fn is None:
            signal_fn = ev._build_signal_fn(genome)
        rules = _exit_rules(genome)
        tp = next((r["value"] for r in rules if r["type"] == "profit_target"), None)
        sl = next((r["value"] for r in rules if r["type"] == "stop_loss"), None)
        trail = next((r["value"] for r in rules if r["type"] == "trailing_stop"), None)
        max_hold = max([int(r["value"]) for r in rules
                        if r["type"] == "time_stop"] or [24])

        pos = conn.execute(
            "SELECT entry_ts, entry_price, size_usd, sol_qty, best_price "
            "FROM positions WHERE enrollment_id=?", (eid,)).fetchone()

        new_bars = [(i, f) for i, f in enumerate(feats)
                    if f["ts"] > last_bar and i >= 60]
        for i, bar in new_bars:
            stats["bars"] += 1
            fv = bar["features"]
            px, hi = fv["close"], fv.get("high", fv["close"])
            lo, op = fv.get("low", fv["close"]), fv.get("open", fv["close"])

            if pos:  # manage open position
                entry_ts, entry_px, size_usd, sol_qty, best = pos
                bars_held = (bar["ts"] - entry_ts) // 3_600_000
                exit_px = exit_reason = None
                if bars_held >= MIN_HOLD_BARS:
                    if sl and lo <= entry_px * (1 - sl):
                        exit_px, exit_reason = min(op, entry_px * (1 - sl)), "stop_loss"
                    elif tp and hi >= entry_px * (1 + tp):
                        exit_px, exit_reason = max(op, entry_px * (1 + tp)), "profit_target"
                    elif trail and lo <= best * (1 - trail):
                        exit_px, exit_reason = min(op, best * (1 - trail)), "trailing"
                    elif bars_held >= max_hold:
                        # market exit at REAL quote
                        lq = live_price(size_usd, "sell")
                        exit_px = lq["price"] or px
                        exit_reason = f"time_stop({lq['source']})"
                best = max(best, hi)
                if exit_px:
                    gross = (exit_px - entry_px) / entry_px * size_usd
                    net = gross - FIXED_COST_PER_SIDE_USD  # exit-side network fee
                    cash += size_usd + net
                    conn.execute(
                        "INSERT INTO trades (enrollment_id, entry_ts, exit_ts, "
                        "entry_price, exit_price, size_usd, net_pnl, exit_reason, "
                        "fill_source) VALUES (?,?,?,?,?,?,?,?,?)",
                        (eid, entry_ts, bar["ts"], entry_px, exit_px, size_usd,
                         net, exit_reason, "paper"))
                    conn.execute("DELETE FROM positions WHERE enrollment_id=?", (eid,))
                    cooldown = bar["ts"] + COOLDOWN_BARS * 3_600_000
                    pos = None
                    stats["exits"] += 1
                    if verbose:
                        print(f"  [paper] #{eid} EXIT {exit_reason} net=${net:+.2f}")
                else:
                    conn.execute(
                        "UPDATE positions SET best_price=? WHERE enrollment_id=?",
                        (best, eid))
                    pos = (entry_ts, entry_px, size_usd, sol_qty, best)
            elif bar["ts"] > cooldown:  # look for entry
                sig = signal_fn(feats, i)
                if sig and sig[0] == "long" and sig[1] >= 0.5:
                    size_usd = min(max(sig[2], 0.0), 0.5) * cash
                    if size_usd >= 5.0:
                        lq = live_price(size_usd, "buy")
                        if lq["price"]:
                            entry_px = lq["price"]
                            size_after_fee = size_usd - FIXED_COST_PER_SIDE_USD
                            cash -= size_usd
                            conn.execute(
                                "INSERT OR REPLACE INTO positions VALUES "
                                "(?,?,?,?,?,?,?)",
                                (eid, bar["ts"], entry_px, size_after_fee,
                                 size_after_fee / entry_px, entry_px,
                                 lq["source"]))
                            pos = (bar["ts"], entry_px, size_after_fee,
                                   size_after_fee / entry_px, entry_px)
                            stats["entries"] += 1
                            if verbose:
                                print(f"  [paper] #{eid} ENTER ${size_usd:.0f} "
                                      f"@ {entry_px:.2f} ({lq['source']})")

            # equity snapshot at each bar
            eq = cash + (pos[2] * (px / pos[1]) if pos else 0.0)
            conn.execute("INSERT OR REPLACE INTO equity VALUES (?,?,?)",
                         (eid, bar["ts"], eq))
            conn.execute(
                "UPDATE enrollments SET cash=?, last_bar_ts=?, "
                "cooldown_until_ts=? WHERE id=?",
                (cash, bar["ts"], cooldown, eid))
        conn.commit()
    return stats


def grade_and_publish(conn: sqlite3.Connection) -> Dict[str, Any]:
    """PAPER verdicts per enrollment + status JSON for the dashboard."""
    out = []
    now = time.time()
    for (eid, gid, gjson, ets, status) in conn.execute(
        "SELECT id, genome_id, genome_json, enrolled_ts, status "
        "FROM enrollments").fetchall():
        days = (now - ets) / 86400
        trades = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(net_pnl),0) FROM trades "
            "WHERE enrollment_id=?", (eid,)).fetchone()
        eq = [r[0] for r in conn.execute(
            "SELECT equity FROM equity WHERE enrollment_id=? ORDER BY ts",
            (eid,)).fetchall()]
        peak, dd = BOOK_USD, 0.0
        for v in eq:
            peak = max(peak, v)
            dd = max(dd, (peak - v) / peak)
        n_tr, net = int(trades[0]), float(trades[1])
        verdict = "RUNNING"
        if days >= PAPER_MIN_DAYS:
            passed = (n_tr >= PAPER_MIN_TRADES
                      and net >= PAPER_MIN_NET_PNL_USD
                      and dd <= PAPER_MAX_DRAWDOWN_HARD)
            verdict = "PASS" if passed else "FAIL"
            if status == "active" and days >= TERM_DAYS:
                conn.execute("UPDATE enrollments SET status=? WHERE id=?",
                             (f"completed_{verdict.lower()}", eid))
        logic = (json.loads(gjson) or {}).get("entry_logic", "?")
        out.append({"id": eid, "genome_id": (gid or "")[:30], "logic": logic,
                    "days": round(days, 1), "trades": n_tr,
                    "net_pnl": round(net, 2), "max_dd_pct": round(dd * 100, 1),
                    "verdict": verdict, "status": status})
    conn.commit()
    payload = {
        "updated_ts": now, "book_usd": BOOK_USD,
        "bars": {"min_days": PAPER_MIN_DAYS, "min_trades": PAPER_MIN_TRADES,
                 "min_net_usd": PAPER_MIN_NET_PNL_USD,
                 "max_dd": PAPER_MAX_DRAWDOWN_HARD},
        "enrollments": out,
    }
    try:
        STATUS_JSON.write_text(json.dumps(payload, indent=2))
    except Exception:
        pass
    return payload


def main() -> int:
    conn = _conn()
    try:
        n_new = enroll_new(conn)
        stats = process_bars(conn)
        payload = grade_and_publish(conn)
        active = sum(1 for e in payload["enrollments"] if e["status"] == "active")
        print(f"paper trader: enrolled+{n_new} active={active} "
              f"new_bars={stats['bars']} entries={stats['entries']} "
              f"exits={stats['exits']}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
