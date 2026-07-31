"""
kill_archive.py
Tabu / killed-DNA memory so evolution does not keep retesting near-duplicate failures.

Darwin-style idea:
- On funnel REJECT, store a coarse structural signature + fail reason.
- On propose (init / mutate / immigrant / funnel intake), reject or reshuffle
  if the neighborhood is already known-dead.

Signatures are HOOD-level, not float-exact:
  logic + sorted (indicator, operator, threshold_bin)

threshold_bin shrinks the continuous space so ±10% retunes count as the same fail.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from evolution.genome import (
    INDICATORS,
    LOGIC_OPS,
    StrategyGenome,
    get_threshold_range,
    mutate,
    random_genome,
)

SIM_DIR = Path(__file__).resolve().parent.parent
EVO_DIR = SIM_DIR / "evolution"
KILL_PATH = EVO_DIR / "killed_dna.json"
FUNNEL_DIR = EVO_DIR / "funnel_results"

# How many bins along each indicator's range (coarser = stronger ban)
N_BINS = 8

# Fail reasons that enter the kill archive
KILL_REASONS = {
    "walk_forward",
    "oos",
    "fee_stress",
    "perturbation",
    "mev",
    "dsr",
    "feasibility",
}

# Soft vs hard: after this many kills on same neighborhood, force reshuffle
HARD_STRIKES = 1

# Max retries when generating a non-killed genome
MAX_RESAMPLE = 40


def _bin_threshold(indicator: str, threshold: float) -> int:
    lo, hi = get_threshold_range(indicator)
    if hi <= lo:
        return 0
    t = float(threshold)
    # clamp
    if t <= lo:
        return 0
    if t >= hi:
        return N_BINS - 1
    frac = (t - lo) / (hi - lo)
    return min(N_BINS - 1, max(0, int(frac * N_BINS)))


def structure_key(genome: StrategyGenome) -> str:
    """Coarse neighborhood key used for tabu matching."""
    logic = genome.entry_logic or "AND"
    # Sort conditions so order doesn't create fake novelty
    parts: List[str] = []
    for c in sorted(
        genome.entry_conditions or [],
        key=lambda x: (x.indicator or "", x.operator or "", float(x.threshold or 0)),
    ):
        ind = c.indicator or "?"
        op = c.operator or "?"
        b = _bin_threshold(ind, float(c.threshold or 0))
        parts.append(f"{ind}|{op}|b{b}")
    # Cap condition count noise: drop pure sizing/exits from tabu key
    # Keep logic + primary conditions only
    body = "+".join(parts[:4]) if parts else "nocon"
    return f"{logic}::{body}"


def exact_signature(genome: StrategyGenome) -> Tuple:
    """Finer signature for debugging only."""
    conds = tuple(
        (c.indicator, c.operator, round(float(c.threshold), 4))
        for c in (genome.entry_conditions or [])
    )
    return (genome.entry_logic, conds)


class KillArchive:
    """Persistent tabu list of failed strategy neighborhoods."""

    def __init__(self, path: Path = KILL_PATH):
        self.path = path
        self._data: Dict[str, Any] = {
            "updated_ts": 0.0,
            "n": 0,
            "entries": {},  # key -> record
        }
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text())
            if isinstance(raw, dict) and isinstance(raw.get("entries"), dict):
                self._data = raw
        except Exception:
            pass

    def save(self) -> None:
        EVO_DIR.mkdir(parents=True, exist_ok=True)
        self._data["n"] = len(self._data.get("entries") or {})
        self._data["updated_ts"] = time.time()
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, default=str))
        tmp.replace(self.path)

    @property
    def size(self) -> int:
        return len(self._data.get("entries") or {})

    def is_killed(self, genome: StrategyGenome) -> bool:
        key = structure_key(genome)
        rec = (self._data.get("entries") or {}).get(key)
        if not rec:
            return False
        return int(rec.get("strikes", 0)) >= HARD_STRIKES

    def kill_info(self, genome: StrategyGenome) -> Optional[Dict[str, Any]]:
        key = structure_key(genome)
        return (self._data.get("entries") or {}).get(key)

    def record_kill(
        self,
        genome: StrategyGenome,
        reason: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Register a funnel/selection failure into the archive."""
        reason = reason or "unknown"
        if reason not in KILL_REASONS and reason not in ("RANDOM banned", "lottery_ban"):
            # still record, but unknown reasons get soft weight
            pass

        key = structure_key(genome)
        entries = self._data.setdefault("entries", {})
        rec = entries.get(key) or {
            "key": key,
            "strikes": 0,
            "reasons": {},
            "first_ts": time.time(),
            "last_ts": time.time(),
            "example_id": genome.genome_id,
            "logic": genome.entry_logic,
            "conditions": [
                {
                    "indicator": c.indicator,
                    "operator": c.operator,
                    "threshold": c.threshold,
                    "bin": _bin_threshold(c.indicator, float(c.threshold or 0)),
                }
                for c in (genome.entry_conditions or [])[:4]
            ],
        }
        rec["strikes"] = int(rec.get("strikes", 0)) + 1
        reasons = rec.setdefault("reasons", {})
        reasons[reason] = int(reasons.get(reason, 0)) + 1
        rec["last_ts"] = time.time()
        rec["last_reason"] = reason
        rec["example_id"] = genome.genome_id or rec.get("example_id")
        if meta:
            rec["last_meta"] = meta
        entries[key] = rec
        # Bound archive growth: keep most-struck + recent
        if len(entries) > 5000:
            self._prune(limit=4000)
        self.save()
        return rec

    def _prune(self, limit: int = 4000) -> None:
        entries = self._data.get("entries") or {}
        items = list(entries.items())
        items.sort(
            key=lambda kv: (int(kv[1].get("strikes", 0)), float(kv[1].get("last_ts", 0))),
            reverse=True,
        )
        self._data["entries"] = dict(items[:limit])

    def random_genome_clean(self, generation: int = 0) -> StrategyGenome:
        """Sample a genome not in the kill archive (best-effort)."""
        g = random_genome(generation=generation)
        for _ in range(MAX_RESAMPLE):
            if not self.is_killed(g):
                return g
            g = random_genome(generation=generation)
        # Last resort: mutate hard out of neighborhood
        for _ in range(MAX_RESAMPLE):
            g = mutate(g, mutation_rate=0.9)
            if not self.is_killed(g):
                return g
        return g

    def mutate_clean(
        self, genome: StrategyGenome, mutation_rate: float = 0.25
    ) -> StrategyGenome:
        """Mutate until result is not tabu (best-effort)."""
        child = mutate(genome, mutation_rate)
        for _ in range(MAX_RESAMPLE):
            if not self.is_killed(child):
                return child
            child = mutate(genome, min(1.0, mutation_rate * 1.5))
        return child

    def summary(self) -> Dict[str, Any]:
        entries = self._data.get("entries") or {}
        by_reason: Dict[str, int] = {}
        by_logic: Dict[str, int] = {}
        for rec in entries.values():
            for r, n in (rec.get("reasons") or {}).items():
                by_reason[r] = by_reason.get(r, 0) + int(n)
            logic = rec.get("logic") or "?"
            by_logic[logic] = by_logic.get(logic, 0) + 1
        top = sorted(
            entries.values(),
            key=lambda r: int(r.get("strikes", 0)),
            reverse=True,
        )[:8]
        return {
            "n": len(entries),
            "by_reason": by_reason,
            "by_logic": by_logic,
            "top": [
                {
                    "key": t.get("key"),
                    "strikes": t.get("strikes"),
                    "last_reason": t.get("last_reason"),
                    "logic": t.get("logic"),
                }
                for t in top
            ],
        }

    def bootstrap_from_funnel(self) -> int:
        """Seed archive from existing funnel_results/*.json rejects."""
        if not FUNNEL_DIR.exists():
            return 0
        added = 0
        for path in FUNNEL_DIR.glob("funnel_*.json"):
            try:
                d = json.loads(path.read_text())
            except Exception:
                continue
            if d.get("all_passed"):
                continue
            reason = d.get("failed_at") or "unknown"
            gdata = d.get("genome")
            if not gdata:
                continue
            try:
                genome = StrategyGenome.from_dict(gdata)
            except Exception:
                continue
            before = self.size
            self.record_kill(genome, reason=str(reason), meta={"source": path.name})
            if self.size >= before:
                added += 1
        return added


# Module-level singleton for engine use
_ARCHIVE: Optional[KillArchive] = None


def get_archive() -> KillArchive:
    global _ARCHIVE
    if _ARCHIVE is None:
        _ARCHIVE = KillArchive()
    return _ARCHIVE


def reload_archive() -> KillArchive:
    global _ARCHIVE
    _ARCHIVE = KillArchive()
    return _ARCHIVE
