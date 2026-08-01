"""
vintage_ledger.py
Frozen-champion forward ledger — the honest "is the engine getting smarter?" measure.

Idea:
  Every cycle's best genome is FROZEN with the timestamp of the last candle it
  ever saw ("vintage"). As new candles arrive later, each vintage is backtested
  ONLY on data newer than its freeze point. No amount of overfitting can help a
  frozen strategy on candles that did not exist when it was frozen, so the
  trend of forward performance across vintages is a true learning curve.

Control cohorts:
  Forward results also move with market regime, so once per day we freeze a
  cohort of random genomes plus two dumb baselines (buy & hold, SMA 5/20
  cross). A champion is reported as a SKILL PERCENTILE against the random
  cohort frozen on the same day and scored on the same future data. Rising
  percentile = genuine learning, regime-proof.

State lives in sim/data/vintage_ledger.db (gitignored, stays on the mini).
"""

from __future__ import annotations

import json
import sqlite3
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import DATA_DIR

LEDGER_DB = DATA_DIR / "vintage_ledger.db"

RANDOM_COHORT_SIZE = 20
RANDOM_COHORT_MIN_GAP_S = 20 * 3600  # at most one control cohort per ~day
MIN_FORWARD_BARS = 72                # need ≥3 days of unseen 1h candles to score
BASELINE_BOOK_FRACTION = 0.5         # baselines invest 50% of the $100 book
FEE_RATE = 0.00022                   # same 2.2 bps/side as the search


def _conn() -> sqlite3.Connection:
    LEDGER_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(LEDGER_DB))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS vintages (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            kind           TEXT NOT NULL,     -- champion | random | baseline_bh | baseline_sma
            genome_id      TEXT,
            genome_json    TEXT,
            signature      TEXT,
            frozen_wall_ts REAL NOT NULL,     -- wall clock at freeze
            frozen_data_ts INTEGER NOT NULL,  -- ts of last candle the genome ever saw
            cohort_day     TEXT NOT NULL      -- YYYY-MM-DD of frozen_data_ts (UTC)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS forward_scores (
            vintage_id  INTEGER PRIMARY KEY,
            scored_ts   REAL NOT NULL,
            first_ts    INTEGER, last_ts INTEGER,
            n_bars      INTEGER NOT NULL,
            trades      INTEGER NOT NULL,
            net_pnl     REAL NOT NULL,
            pnl_per_30d REAL NOT NULL,
            max_dd      REAL NOT NULL,
            FOREIGN KEY (vintage_id) REFERENCES vintages(id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_vint_kind_day ON vintages(kind, cohort_day)")
    return conn


def _cohort_day(data_ts_ms: int) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(data_ts_ms / 1000))


def _signature_str(genome) -> str:
    from evolution.genome import dna_signature
    return json.dumps(dna_signature(genome), default=str)


# ---------------------------------------------------------------------------
# Freezing
# ---------------------------------------------------------------------------

def freeze_cycle(best_genome, features: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Freeze this cycle's champion (if structurally new) + daily control cohort.

    Returns a small status dict for logging.
    """
    if not features:
        return {"frozen_champion": False, "frozen_controls": 0}
    frozen_data_ts = int(features[-1]["ts"])
    day = _cohort_day(frozen_data_ts)
    now = time.time()
    out = {"frozen_champion": False, "frozen_controls": 0}

    conn = _conn()
    try:
        # Champion: skip if identical DNA to the most recently frozen champion
        sig = _signature_str(best_genome)
        row = conn.execute(
            "SELECT signature FROM vintages WHERE kind='champion' "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row is None or row[0] != sig:
            conn.execute(
                "INSERT INTO vintages (kind, genome_id, genome_json, signature, "
                "frozen_wall_ts, frozen_data_ts, cohort_day) VALUES (?,?,?,?,?,?,?)",
                ("champion", best_genome.genome_id, best_genome.to_json(), sig,
                 now, frozen_data_ts, day),
            )
            out["frozen_champion"] = True

        # Daily control cohort: randoms + baselines, at most once per ~day
        row = conn.execute(
            "SELECT MAX(frozen_wall_ts) FROM vintages WHERE kind='random'"
        ).fetchone()
        last_random = float(row[0] or 0)
        if now - last_random >= RANDOM_COHORT_MIN_GAP_S:
            from evolution.genome import random_genome
            for _ in range(RANDOM_COHORT_SIZE):
                g = random_genome()
                conn.execute(
                    "INSERT INTO vintages (kind, genome_id, genome_json, signature, "
                    "frozen_wall_ts, frozen_data_ts, cohort_day) VALUES (?,?,?,?,?,?,?)",
                    ("random", g.genome_id, g.to_json(), _signature_str(g),
                     now, frozen_data_ts, day),
                )
            for kind in ("baseline_bh", "baseline_sma"):
                conn.execute(
                    "INSERT INTO vintages (kind, genome_id, genome_json, signature, "
                    "frozen_wall_ts, frozen_data_ts, cohort_day) VALUES (?,?,?,?,?,?,?)",
                    (kind, kind, None, kind, now, frozen_data_ts, day),
                )
            out["frozen_controls"] = RANDOM_COHORT_SIZE + 2
        conn.commit()
    finally:
        conn.close()
    return out


# ---------------------------------------------------------------------------
# Baselines (computed directly; no genome machinery)
# ---------------------------------------------------------------------------

def _baseline_scores(kind: str, fwd: List[Dict[str, Any]]) -> Dict[str, float]:
    closes = [f["features"]["close"] for f in fwd]
    book = 100.0 * BASELINE_BOOK_FRACTION
    if kind == "baseline_bh":
        gross = (closes[-1] / closes[0] - 1.0) * book
        net = gross - book * FEE_RATE * 2
        # equity path for drawdown
        equity = [book * (c / closes[0]) for c in closes]
    else:  # baseline_sma: long when sma_5 > sma_20, flat otherwise
        net = 0.0
        pos_entry = None
        equity, eq = [], 0.0
        n_switches = 0
        for f in fwd:
            fv = f["features"]
            c = fv["close"]
            long_now = fv.get("sma_5", c) > fv.get("sma_20", c)
            if long_now and pos_entry is None:
                pos_entry = c
                n_switches += 1
            elif not long_now and pos_entry is not None:
                net += (c / pos_entry - 1.0) * book
                pos_entry = None
                n_switches += 1
            eq = net + ((c / pos_entry - 1.0) * book if pos_entry else 0.0)
            equity.append(book + eq)
        if pos_entry is not None:
            net += (closes[-1] / pos_entry - 1.0) * book
        net -= n_switches * book * FEE_RATE
    peak, max_dd = max(equity[0], 1e-9), 0.0
    for v in equity:
        peak = max(peak, v)
        max_dd = max(max_dd, (peak - v) / peak)
    return {"trades": 1, "net_pnl": float(net), "max_dd": float(max_dd)}


# ---------------------------------------------------------------------------
# Forward scoring
# ---------------------------------------------------------------------------

def score_vintages(features: List[Dict[str, Any]], verbose: bool = False) -> int:
    """Re-score every vintage on candles NEWER than its freeze point.

    Idempotent: replaces each vintage's forward score with the latest cumulative
    result. Returns the number of vintages scored this call.
    """
    if not features:
        return 0
    from evolution.evaluator import GenomeEvaluator
    from evolution.genome import StrategyGenome

    ev = GenomeEvaluator(features, augment=False)
    conn = _conn()
    scored = 0
    try:
        rows = conn.execute(
            "SELECT id, kind, genome_json, frozen_data_ts FROM vintages"
        ).fetchall()
        for vid, kind, gjson, frozen_ts in rows:
            fwd = [f for f in features if f["ts"] > frozen_ts]
            if len(fwd) < MIN_FORWARD_BARS:
                continue
            try:
                if kind in ("baseline_bh", "baseline_sma"):
                    b = _baseline_scores(kind, fwd)
                    trades, net, dd = b["trades"], b["net_pnl"], b["max_dd"]
                else:
                    g = StrategyGenome.from_dict(json.loads(gjson))
                    r = ev._run_raw(g, fwd)
                    trades, net, dd = r.total_trades, r.total_pnl, r.max_drawdown
            except Exception:
                continue
            pnl30 = float(net) * (720.0 / len(fwd))  # normalize to $/30d (720 1h bars)
            conn.execute(
                "INSERT OR REPLACE INTO forward_scores "
                "(vintage_id, scored_ts, first_ts, last_ts, n_bars, trades, "
                "net_pnl, pnl_per_30d, max_dd) VALUES (?,?,?,?,?,?,?,?,?)",
                (vid, time.time(), fwd[0]["ts"], fwd[-1]["ts"], len(fwd),
                 int(trades), float(net), pnl30, float(dd or 0.0)),
            )
            scored += 1
        conn.commit()
    finally:
        conn.close()
    if verbose and scored:
        print(f"  [vintage] forward-scored {scored} frozen strategies")
    return scored


# ---------------------------------------------------------------------------
# Summary for the dashboard
# ---------------------------------------------------------------------------

def _percentile_of(value: float, cohort: List[float]) -> float:
    if not cohort:
        return 50.0
    below = sum(1 for c in cohort if c < value)
    equal = sum(1 for c in cohort if c == value)
    return 100.0 * (below + 0.5 * equal) / len(cohort)


def ledger_summary(max_weeks: int = 26) -> Dict[str, Any]:
    """Weekly cohorts of champion skill percentile vs same-day randoms."""
    if not LEDGER_DB.exists():
        return {"ok": False, "reason": "no ledger yet"}
    conn = _conn()
    try:
        rows = conn.execute("""
            SELECT v.kind, v.cohort_day, s.pnl_per_30d
            FROM vintages v JOIN forward_scores s ON s.vintage_id = v.id
        """).fetchall()
    finally:
        conn.close()
    if not rows:
        return {"ok": False, "reason": "frozen but no forward data yet"}

    randoms_by_day: Dict[str, List[float]] = {}
    champs_by_day: Dict[str, List[float]] = {}
    baselines_by_day: Dict[str, Dict[str, float]] = {}
    for kind, day, pnl30 in rows:
        if kind == "random":
            randoms_by_day.setdefault(day, []).append(pnl30)
        elif kind == "champion":
            champs_by_day.setdefault(day, []).append(pnl30)
        else:
            baselines_by_day.setdefault(day, {})[kind] = pnl30

    def week_of(day: str) -> str:
        t = time.strptime(day, "%Y-%m-%d")
        return time.strftime("%G-W%V", t)

    random_days = sorted(randoms_by_day)

    def nearest_random_day(day: str) -> Optional[str]:
        if not random_days:
            return None
        return min(random_days, key=lambda d: abs(
            time.mktime(time.strptime(d, "%Y-%m-%d"))
            - time.mktime(time.strptime(day, "%Y-%m-%d"))))

    weeks: Dict[str, Dict[str, List[float]]] = {}
    for day, champ_pnls in champs_by_day.items():
        rd = nearest_random_day(day)
        cohort = randoms_by_day.get(rd, []) if rd else []
        wk = weeks.setdefault(week_of(day), {"pct": [], "champ": [], "rand": [], "bh": []})
        for p in champ_pnls:
            wk["pct"].append(_percentile_of(p, cohort))
            wk["champ"].append(p)
        wk["rand"].extend(cohort)
        bh = baselines_by_day.get(rd or day, {}).get("baseline_bh")
        if bh is not None:
            wk["bh"].append(bh)

    cohorts = []
    for wk in sorted(weeks)[-max_weeks:]:
        d = weeks[wk]
        cohorts.append({
            "week": wk,
            "n": len(d["champ"]),
            "med_pct": statistics.median(d["pct"]) if d["pct"] else 50.0,
            "med_champ_pnl30": statistics.median(d["champ"]) if d["champ"] else 0.0,
            "med_rand_pnl30": statistics.median(d["rand"]) if d["rand"] else 0.0,
            "bh_pnl30": statistics.median(d["bh"]) if d["bh"] else None,
        })
    if not cohorts:
        return {"ok": False, "reason": "no scored champions yet"}

    pcts = [c["med_pct"] for c in cohorts]
    verdict = None
    if len(cohorts) >= 4:
        half = len(pcts) // 2
        prev_m, rec_m = statistics.mean(pcts[:half]), statistics.mean(pcts[half:])
        if rec_m >= prev_m + 8:
            verdict = ("SMARTER",
                       f"Champions now beat {rec_m:.0f}% of same-day random strategies "
                       f"on unseen future data, up from {prev_m:.0f}%.")
        elif rec_m <= prev_m - 8:
            verdict = ("WEAKER",
                       f"Champion skill percentile on unseen data fell "
                       f"{prev_m:.0f}% → {rec_m:.0f}%.")
        elif statistics.mean(pcts) <= 58:
            verdict = ("NO EDGE YET",
                       f"Champions perform like random strategies on future data "
                       f"(~{statistics.mean(pcts):.0f}th percentile). The search is not "
                       f"finding a persistent edge with the current features.")
        else:
            verdict = ("FLAT",
                       f"Champion skill percentile is steady around "
                       f"{statistics.mean(pcts):.0f}% — better than random, "
                       f"but not improving.")
    return {
        "ok": True,
        "cohorts": cohorts,
        "n_weeks": len(cohorts),
        "verdict": verdict[0] if verdict else "TOO EARLY",
        "verdict_text": verdict[1] if verdict else
        f"Only {len(cohorts)} weekly cohort(s) have forward data. "
        f"Need ~4 weeks for a trustworthy trend.",
        "latest_pct": pcts[-1],
    }
