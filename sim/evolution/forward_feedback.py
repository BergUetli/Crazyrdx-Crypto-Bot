"""
forward_feedback.py — the autopilot goal-seeking loop, done safely.

Closes the outer learning loop: search effort automatically reallocates
toward strategy structures that have made REAL forward money, using reward
signals the optimizer cannot corrupt:
  - the vintage ledger (frozen strategies scored only on candles that
    arrived after they were created), and
  - the paper trader (fills at live Jupiter quotes).

What it adapts (bounded): logic-family sampling weights, indicator sampling
weights for frontier immigrants, and a capped per-family selection bonus.

What it NEVER touches (by design, asserted by tests): the exam gates, the
cost model, the verdict logic. Softening the measuring stick does not create
profit — it creates lies; the 2026-08 beta-mirage incident is the standing
proof. The reward channel stays external; only search ALLOCATION adapts.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

from config import DATA_DIR

LEDGER_DB = DATA_DIR / "vintage_ledger.db"
PAPER_DB = DATA_DIR / "paper_trader.db"
FEEDBACK_JSON = DATA_DIR / "forward_feedback.json"

SHRINK_K = 8            # Bayesian shrinkage: excess * n/(n+K)
WEIGHT_MIN, WEIGHT_MAX = 0.5, 2.0
FAMILY_BONUS_CAP = 15.0  # fitness points; small vs the ±250 fitness scale
SCALE_PNL = 20.0         # $/30d of forward excess that maps to full tilt


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute() -> Dict[str, Any]:
    """Aggregate forward outcomes by logic / indicator / family."""
    logic_pnls: Dict[str, list] = defaultdict(list)
    ind_pnls: Dict[str, list] = defaultdict(list)
    fam_pnls: Dict[str, list] = defaultdict(list)
    random_pnls: list = []
    try:
        conn = sqlite3.connect(str(LEDGER_DB))
        for kind, gjson, pnl in conn.execute("""
            SELECT v.kind, v.genome_json, s.pnl_per_30d
            FROM forward_scores s JOIN vintages v ON v.id = s.vintage_id
            WHERE v.kind IN ('champion', 'random')"""):
            if kind == "random":
                random_pnls.append(pnl)
                continue
            try:
                g = json.loads(gjson)
            except Exception:
                continue
            logic = g.get("entry_logic") or "?"
            inds = sorted({c.get("indicator") for c in
                           g.get("entry_conditions", []) if c.get("indicator")})
            logic_pnls[logic].append(pnl)
            for i in inds:
                ind_pnls[i].append(pnl)
            fam_pnls[f"{logic}|{','.join(inds)}"].append(pnl)
        conn.close()
    except Exception:
        pass
    # Paper results carry double weight once trades exist (real quotes)
    try:
        conn = sqlite3.connect(str(PAPER_DB))
        for gjson, net in conn.execute("""
            SELECT e.genome_json, SUM(t.net_pnl) FROM enrollments e
            JOIN trades t ON t.enrollment_id = e.id GROUP BY e.id"""):
            try:
                g = json.loads(gjson)
            except Exception:
                continue
            logic = g.get("entry_logic") or "?"
            inds = sorted({c.get("indicator") for c in
                           g.get("entry_conditions", []) if c.get("indicator")})
            for _ in range(2):
                logic_pnls[logic].append(float(net or 0))
                for i in inds:
                    ind_pnls[i].append(float(net or 0))
        conn.close()
    except Exception:
        pass

    base = (sum(random_pnls) / len(random_pnls)) if random_pnls else 0.0

    def tilt(pnls: list) -> float:
        n = len(pnls)
        if n == 0:
            return 1.0
        excess = (sum(pnls) / n - base) * (n / (n + SHRINK_K))
        return _clip(1.0 + excess / SCALE_PNL, WEIGHT_MIN, WEIGHT_MAX)

    def bonus(pnls: list) -> float:
        n = len(pnls)
        if n == 0:
            return 0.0
        excess = (sum(pnls) / n - base) * (n / (n + SHRINK_K))
        return _clip(excess, -FAMILY_BONUS_CAP, FAMILY_BONUS_CAP)

    return {
        "updated_ts": time.time(),
        "random_baseline_pnl30": round(base, 3),
        "n_random": len(random_pnls),
        "logic_weights": {k: round(tilt(v), 3) for k, v in logic_pnls.items()},
        "indicator_weights": {k: round(tilt(v), 3) for k, v in ind_pnls.items()},
        "family_bonus": {k: round(bonus(v), 2) for k, v in fam_pnls.items()
                         if abs(bonus(v)) > 0.5},
    }


def compute_and_write() -> Dict[str, Any]:
    fb = compute()
    try:
        tmp = FEEDBACK_JSON.with_suffix(".tmp")
        tmp.write_text(json.dumps(fb, indent=2))
        tmp.replace(FEEDBACK_JSON)
    except Exception:
        pass
    return fb


def load() -> Dict[str, Any]:
    try:
        return json.loads(FEEDBACK_JSON.read_text())
    except Exception:
        return {}
