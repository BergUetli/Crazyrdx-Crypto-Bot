"""
strategy_log.py
Persistent log of EVERY strategy the engine evaluates — the search's lab
notebook. Powers: (1) the exploration tax (diminishing returns per family),
(2) indicator-coverage-driven frontier immigrants, (3) the /explore dashboard
("MEANREV: 200 sub-strategies tried, 62% viable — click for the log"), and
(4) the audit trail so no idea is ever tried and forgotten.

Volume: ~50-80k rows/day. SQLite handles this fine; prune() keeps ~45 days.
DB: sim/data/strategy_log.db (gitignored, stays local).
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import DATA_DIR

LOG_DB = DATA_DIR / "strategy_log.db"
_WRITE_WARNED = False


def family_key(logic: str, indicators) -> str:
    """Coarse family identity: logic + sorted indicator set."""
    return f"{logic}|{','.join(sorted(set(indicators)))}"


def genome_family(genome) -> str:
    return family_key(
        genome.entry_logic,
        [c.indicator for c in (genome.entry_conditions or [])],
    )


def _conn() -> sqlite3.Connection:
    LOG_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(LOG_DB), timeout=15)
    # WAL + NORMAL sync: resilient to unclean shutdown (reboot mid-write
    # corrupted the rollback-journal DB once) and to concurrent readers.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=15000")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS strategies (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          REAL NOT NULL,
            cycle       INTEGER,
            generation  INTEGER,
            genome_id   TEXT,
            logic       TEXT NOT NULL,
            family      TEXT NOT NULL,
            indicators  TEXT NOT NULL,   -- json list
            n_conds     INTEGER,
            sizing      TEXT,
            fitness     REAL,
            trades      INTEGER,
            pnl         REAL,
            source      TEXT NOT NULL,   -- 'search' | 'funnel'
            verdict     TEXT             -- funnel: failed_at or 'PASS'; search: NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_strat_ts ON strategies(ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_strat_family ON strategies(family)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_strat_logic_ts ON strategies(logic, ts)")
    return conn


def log_rows(rows: List[Dict[str, Any]]) -> int:
    """Batch-insert evaluated strategies. Never raises into the caller."""
    if not rows:
        return 0
    try:
        conn = _conn()
        conn.executemany(
            "INSERT INTO strategies (ts, cycle, generation, genome_id, logic, "
            "family, indicators, n_conds, sizing, fitness, trades, pnl, "
            "source, verdict) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    r.get("ts", time.time()), r.get("cycle"), r.get("generation"),
                    r.get("genome_id"), r.get("logic"), r.get("family"),
                    json.dumps(r.get("indicators") or []), r.get("n_conds"),
                    r.get("sizing"), r.get("fitness"), r.get("trades"),
                    r.get("pnl"), r.get("source", "search"), r.get("verdict"),
                )
                for r in rows
            ],
        )
        conn.commit()
        conn.close()
        return len(rows)
    except Exception as e:
        global _WRITE_WARNED
        if not _WRITE_WARNED:
            _WRITE_WARNED = True
            print(f"  WARNING: strategy log writes failing ({e}) — "
                  f"exploration stats will be blind until fixed")
        return 0


def log_genome(genome, result: Dict[str, Any], cycle: int, generation: int,
               source: str = "search", verdict: Optional[str] = None) -> Dict[str, Any]:
    """Build one row dict from a genome + its evaluation result."""
    inds = [c.indicator for c in (genome.entry_conditions or [])]
    return {
        "ts": time.time(), "cycle": cycle, "generation": generation,
        "genome_id": genome.genome_id, "logic": genome.entry_logic,
        "family": genome_family(genome), "indicators": inds,
        "n_conds": len(inds), "sizing": genome.sizing_method,
        "fitness": float(result.get("fitness") or 0.0),
        "trades": int(result.get("total_trades") or 0),
        "pnl": float(result.get("total_pnl") or 0.0),
        "source": source, "verdict": verdict,
    }


def family_counts(days: float = 3.0) -> Dict[str, int]:
    """How often each family was evaluated recently (exploration-tax input)."""
    try:
        conn = _conn()
        cut = time.time() - days * 86400
        out = dict(conn.execute(
            "SELECT family, COUNT(*) FROM strategies WHERE ts > ? GROUP BY family",
            (cut,),
        ).fetchall())
        conn.close()
        return out
    except Exception:
        return {}


def indicator_usage(days: float = 7.0) -> Counter:
    """Per-indicator evaluation counts (frontier-immigrant input)."""
    usage: Counter = Counter()
    try:
        conn = _conn()
        cut = time.time() - days * 86400
        for (inds,) in conn.execute(
            "SELECT indicators FROM strategies WHERE ts > ?", (cut,)
        ):
            try:
                for i in json.loads(inds):
                    usage[i] += 1
            except Exception:
                continue
        conn.close()
    except Exception:
        pass
    return usage


def summary_by_logic(days: float = 7.0) -> List[Dict[str, Any]]:
    """Per-logic aggregates for the /explore cards."""
    try:
        conn = _conn()
        cut = time.time() - days * 86400
        rows = conn.execute("""
            SELECT logic, COUNT(*), COUNT(DISTINCT family),
                   SUM(CASE WHEN trades >= 30 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN fitness > 0 THEN 1 ELSE 0 END),
                   MAX(fitness),
                   SUM(CASE WHEN source='funnel' AND verdict='PASS' THEN 1 ELSE 0 END),
                   SUM(CASE WHEN source='funnel' AND verdict IS NOT NULL
                            AND verdict != 'PASS' THEN 1 ELSE 0 END)
            FROM strategies WHERE ts > ? GROUP BY logic ORDER BY COUNT(*) DESC
        """, (cut,)).fetchall()
        conn.close()
        return [
            {"logic": r[0], "tried": r[1], "families": r[2], "viable": r[3],
             "positive": r[4], "best_fitness": r[5] or 0.0,
             "funnel_pass": r[6] or 0, "funnel_fail": r[7] or 0}
            for r in rows
        ]
    except Exception:
        return []


def family_table(logic: str, days: float = 7.0, limit: int = 60) -> List[Dict[str, Any]]:
    """Per-family aggregates within one logic (drill-down level 1)."""
    try:
        conn = _conn()
        cut = time.time() - days * 86400
        rows = conn.execute("""
            SELECT family, COUNT(*), MAX(fitness), AVG(trades), MAX(ts),
                   SUM(CASE WHEN fitness > 0 THEN 1 ELSE 0 END)
            FROM strategies WHERE ts > ? AND logic = ?
            GROUP BY family ORDER BY COUNT(*) DESC LIMIT ?
        """, (cut, logic, limit)).fetchall()
        conn.close()
        return [
            {"family": r[0], "tried": r[1], "best_fitness": r[2] or 0.0,
             "avg_trades": r[3] or 0.0, "last_ts": r[4],
             "positive": r[5] or 0}
            for r in rows
        ]
    except Exception:
        return []


def recent_log(logic: Optional[str] = None, family: Optional[str] = None,
               limit: int = 120) -> List[Dict[str, Any]]:
    """Latest individual strategies (drill-down level 2 — the raw lab log)."""
    try:
        conn = _conn()
        q = ("SELECT ts, cycle, generation, genome_id, logic, indicators, "
             "n_conds, fitness, trades, pnl, source, verdict FROM strategies")
        cond, params = [], []
        if logic:
            cond.append("logic = ?"); params.append(logic)
        if family:
            cond.append("family = ?"); params.append(family)
        if cond:
            q += " WHERE " + " AND ".join(cond)
        q += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(q, params).fetchall()
        conn.close()
        out = []
        for r in rows:
            try:
                inds = json.loads(r[5])
            except Exception:
                inds = []
            out.append({"ts": r[0], "cycle": r[1], "generation": r[2],
                        "genome_id": r[3], "logic": r[4], "indicators": inds,
                        "n_conds": r[6], "fitness": r[7], "trades": r[8],
                        "pnl": r[9], "source": r[10], "verdict": r[11]})
        return out
    except Exception:
        return []


def prune(keep_days: float = 45.0) -> int:
    try:
        conn = _conn()
        cur = conn.execute(
            "DELETE FROM strategies WHERE ts < ?",
            (time.time() - keep_days * 86400,),
        )
        conn.commit()
        n = cur.rowcount
        conn.close()
        return n
    except Exception:
        return 0
