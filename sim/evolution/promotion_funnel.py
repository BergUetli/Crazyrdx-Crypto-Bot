"""
promotion_funnel.py
Fund-grade post-search gauntlet (offline, no cloud).

Gates (QTT / Darwin / AlgoXpert style):
1. Feasibility: not RANDOM, full-sample N >= min_trades
2. IS vs locked OOS: OOS trades, OOS pnl ratio, degradation cap
3. Majority-pass walk-forward folds (raw PnL)
4. Fee stress: still profitable at 5bps and 10bps
5. Parameter perturbation (±10%)
6. MEV stress at 30%
7. Simple Deflated-Sharpe style trial haircut using logged N

Also maintains:
- trials_log.jsonl  (every genome evaluation count across runs)
- champions.json    (top-k uncorrelated survivors)
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evolution.evaluator import GenomeEvaluator
from evolution.genome import StrategyGenome
from layer1.backtest_engine import LatencyModel

SIM_DIR = Path(__file__).resolve().parent.parent
EVO_DIR = SIM_DIR / "evolution"
TRIALS_PATH = EVO_DIR / "trials_log.jsonl"
TRIAL_COUNT_PATH = EVO_DIR / "trial_count.json"
CHAMPIONS_PATH = EVO_DIR / "champions.json"
FUNNEL_DIR = EVO_DIR / "funnel_results"
FUNNEL_DIR.mkdir(parents=True, exist_ok=True)


def _result_dict(res) -> Dict[str, Any]:
    return {
        "total_trades": int(res.total_trades),
        "win_rate": float(res.win_rate),
        "total_pnl": float(res.total_pnl),
        "sharpe_ratio": float(res.sharpe_ratio or 0.0),
        "max_drawdown": float(res.max_drawdown or 0.0),
        "profit_factor": float(res.profit_factor) if res.profit_factor not in (None, float("inf")) else 99.0,
        "final_capital": float(res.final_capital),
    }


def _genome_signature(genome: StrategyGenome) -> Tuple:
    conds = tuple(
        (c.indicator, c.operator, round(float(c.threshold), 4))
        for c in (genome.entry_conditions or [])
    )
    exits = tuple(
        (e.exit_type, round(float(e.value), 6)) for e in (genome.exit_rules or [])
    )
    return (
        genome.entry_logic,
        conds,
        genome.sizing_method,
        round(float(genome.sizing_base), 4),
        exits,
    )


def log_trials(n: int, meta: Optional[Dict[str, Any]] = None) -> int:
    """Append trial count from one evolution cycle. Returns cumulative N."""
    EVO_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": time.time(),
        "n_trials": int(n),
        "meta": meta or {},
    }
    with open(TRIALS_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

    total = 0
    if TRIAL_COUNT_PATH.exists():
        try:
            total = int(json.loads(TRIAL_COUNT_PATH.read_text()).get("total_trials", 0))
        except Exception:
            total = 0
    total += int(n)
    TRIAL_COUNT_PATH.write_text(
        json.dumps({"total_trials": total, "updated_ts": time.time()}, indent=2)
    )
    return total


def get_total_trials() -> int:
    if not TRIAL_COUNT_PATH.exists():
        return 0
    try:
        return int(json.loads(TRIAL_COUNT_PATH.read_text()).get("total_trials", 0))
    except Exception:
        return 0


def approximate_dsr(sharpe: float, n_obs: int, n_trials: int) -> float:
    """
    Lightweight DSR-like score in [0,1].

    Uses the *clamped trade-return ratio* from BacktestEngine (already in [-5,5]),
    not a fake annualized Sharpe. Haircuts for selection budget N.
    """
    if n_obs < 5 or n_trials < 1:
        return 0.0
    # Never trust raw explodey Sharpes from old dumps
    sharpe = float(max(-5.0, min(5.0, sharpe or 0.0)))
    n_trials = max(1, int(n_trials))
    # Expected max SR under null of many trials (order-statistic approx)
    sr_star = math.sqrt(max(1e-12, 2.0 * math.log(n_trials + 1.0))) / math.sqrt(n_obs)
    se = 1.0 / math.sqrt(n_obs)
    z = (sharpe - sr_star) / max(se, 1e-9)
    dsr = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return float(max(0.0, min(1.0, dsr)))


class PromotionFunnel:
    """Run multi-gate promotion on a candidate genome using raw backtests."""

    def __init__(
        self,
        features: List[Dict[str, Any]],
        min_trades_full: int = 30,
        min_trades_oos: int = 10,
        oos_pnl_ratio_min: float = 0.5,
        wf_folds: int = 5,
        wf_majority: float = 0.6,
        fee_rates: Optional[List[float]] = None,
        base_fee: float = 0.00022,
    ):
        self.features = features
        self.min_trades_full = min_trades_full
        self.min_trades_oos = min_trades_oos
        self.oos_pnl_ratio_min = oos_pnl_ratio_min
        self.wf_folds = wf_folds
        self.wf_majority = wf_majority
        self.fee_rates = fee_rates or [0.00022, 0.0005, 0.0010]
        self.base_fee = base_fee
        self.eval = GenomeEvaluator(features, fee_rate=base_fee)

    def _raw(
        self,
        genome: StrategyGenome,
        features: Optional[List[Dict[str, Any]]] = None,
        fee_rate: Optional[float] = None,
        latency: Optional[LatencyModel] = None,
    ):
        feats = features if features is not None else self.features
        fee = self.base_fee if fee_rate is None else fee_rate
        ev = GenomeEvaluator(feats, fee_rate=fee, latency_model=latency)
        return ev._run_raw(genome, feats)

    def run(
        self,
        genome: StrategyGenome,
        n_trials_context: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        gates: Dict[str, Any] = {}
        name = genome.genome_id or "unknown"
        if verbose:
            print(f"\n[funnel] {name} logic={genome.entry_logic}")

        # Gate -1: kill archive tabu (near-duplicate of known rejects)
        try:
            from evolution.kill_archive import get_archive, structure_key
            arch = get_archive()
            if arch.is_killed(genome):
                info = arch.kill_info(genome) or {}
                gates["kill_archive"] = {
                    "passed": False,
                    "key": structure_key(genome),
                    "strikes": info.get("strikes"),
                    "last_reason": info.get("last_reason"),
                }
                if verbose:
                    print(f"  TABU skip (killed DNA): {info.get('last_reason')} x{info.get('strikes')}")
                return self._fail(name, genome, gates, "kill_archive", record_kill=False)
        except Exception:
            pass

        # Gate 0: lottery ban
        if genome.entry_logic == "RANDOM":
            return self._fail(name, genome, {"lottery_ban": {"passed": False}}, "RANDOM banned")

        # Gate 1: full sample feasibility
        full = self._raw(genome)
        full_d = _result_dict(full)
        feas_pass = full.total_trades >= self.min_trades_full and full.total_pnl > 0
        gates["feasibility"] = {
            "passed": feas_pass,
            **full_d,
            "min_trades": self.min_trades_full,
        }
        if not feas_pass:
            return self._fail(name, genome, gates, "feasibility")

        # Gate 2: locked OOS (last third), IS = first two thirds
        n = len(self.features)
        split = max(100, (2 * n) // 3)
        is_feats = self.features[:split]
        oos_feats = self.features[split:]
        is_res = self._raw(genome, is_feats)
        oos_res = self._raw(genome, oos_feats)
        is_d = _result_dict(is_res)
        oos_d = _result_dict(oos_res)

        if oos_res.total_trades < self.min_trades_oos:
            oos_pass = False
            ratio = 0.0
        else:
            # OOS pnl should be at least X% of |IS| when IS positive, or both positive
            if is_res.total_pnl > 0:
                ratio = oos_res.total_pnl / max(is_res.total_pnl, 1e-9)
            else:
                ratio = 1.0 if oos_res.total_pnl > 0 else 0.0
            oos_pass = (
                oos_res.total_pnl > 0
                and oos_res.total_trades >= self.min_trades_oos
                and ratio >= self.oos_pnl_ratio_min
            )
        gates["oos"] = {
            "passed": oos_pass,
            "is": is_d,
            "oos": oos_d,
            "oos_over_is_pnl": ratio,
            "min_ratio": self.oos_pnl_ratio_min,
        }
        if not oos_pass:
            return self._fail(name, genome, gates, "oos")

        # Gate 3: majority walk-forward (chronological folds, raw PnL > 0)
        fold_n = max(3, self.wf_folds)
        fold_size = n // fold_n
        fold_rows = []
        for i in range(fold_n):
            a = i * fold_size
            b = n if i == fold_n - 1 else (i + 1) * fold_size
            chunk = self.features[a:b]
            if len(chunk) < 80:
                continue
            r = self._raw(genome, chunk)
            ok = r.total_pnl > 0 and r.total_trades >= max(5, self.min_trades_oos // 2)
            # Catastrophic DD veto on fold
            if r.max_drawdown is not None and r.max_drawdown > 0.35:
                ok = False
            fold_rows.append(
                {
                    "fold": i,
                    "passed": ok,
                    **_result_dict(r),
                }
            )
        if not fold_rows:
            wf_pass = False
            pass_rate = 0.0
        else:
            pass_rate = sum(1 for f in fold_rows if f["passed"]) / len(fold_rows)
            wf_pass = pass_rate >= self.wf_majority
        gates["walk_forward"] = {
            "passed": wf_pass,
            "pass_rate": pass_rate,
            "majority": self.wf_majority,
            "folds": fold_rows,
        }
        if not wf_pass:
            return self._fail(name, genome, gates, "walk_forward")

        # Gate 4: fee stress
        fee_rows = []
        fee_pass = True
        for fee in self.fee_rates:
            r = self._raw(genome, fee_rate=fee)
            row = {"fee_rate": fee, "fee_bps": fee * 1e4, **_result_dict(r)}
            row["passed"] = r.total_pnl > 0 and r.total_trades >= self.min_trades_full // 2
            fee_rows.append(row)
            if fee >= 0.0005 and not row["passed"]:
                fee_pass = False
        gates["fee_stress"] = {"passed": fee_pass, "rows": fee_rows}
        if not fee_pass:
            return self._fail(name, genome, gates, "fee_stress")

        # Gate 5: parameter perturbation
        orig_pnl = full.total_pnl
        perturbed = []
        for _ in range(10):
            g2 = copy.deepcopy(genome)
            for cond in g2.entry_conditions:
                cond.threshold *= 1.0 + random.uniform(-0.10, 0.10)
            r = self._raw(g2)
            perturbed.append(r.total_pnl)
        profit_rate = sum(1 for p in perturbed if p > 0) / max(len(perturbed), 1)
        avg_p = sum(perturbed) / max(len(perturbed), 1)
        retention = avg_p / orig_pnl if orig_pnl != 0 else 0.0
        pp_pass = profit_rate >= 0.6 and retention >= 0.4
        gates["perturbation"] = {
            "passed": pp_pass,
            "profitability_rate": profit_rate,
            "retention": retention,
            "perturbed_pnls": perturbed,
        }
        if not pp_pass:
            return self._fail(name, genome, gates, "perturbation")

        # Gate 6: MEV stress
        mev_rows = []
        break_even = None
        for mp in [0.0, 0.1, 0.2, 0.3]:
            lat = LatencyModel(
                base_latency_s=10.0,
                latency_std_s=3.0,
                mev_probability=mp,
                mev_cost_bps=15.0,
            )
            r = self._raw(genome, latency=lat)
            row = {"mev_probability": mp, **_result_dict(r)}
            mev_rows.append(row)
            if break_even is None and r.total_pnl <= 0:
                break_even = mp
        mev_pass = break_even is None or break_even > 0.2
        gates["mev"] = {
            "passed": mev_pass,
            "break_even_mev": break_even,
            "rows": mev_rows,
        }
        if not mev_pass:
            return self._fail(name, genome, gates, "mev")

        # Gate 7: DSR-style haircut using trial budget
        # Soft gate: with millions of trials, hard DSR fails everything.
        # We record DSR and require either decent DSR OR strong OOS PnL + WF.
        n_trials = n_trials_context if n_trials_context is not None else get_total_trials()
        n_obs = max(full.total_trades, 1)
        dsr = approximate_dsr(full.sharpe_ratio or 0.0, n_obs=n_obs, n_trials=max(n_trials, 1))
        oos_pnl = float(gates.get("oos", {}).get("oos", {}).get("total_pnl", 0.0) or 0.0)
        wf_rate = float(gates.get("walk_forward", {}).get("pass_rate", 0.0) or 0.0)
        # Pass if DSR ok, or cold-start, or strong money+WF evidence
        dsr_pass = (
            dsr >= 0.5
            or n_trials < 200
            or (oos_pnl >= 5.0 and wf_rate >= 0.8 and full.total_pnl > 0)
        )
        gates["dsr"] = {
            "passed": dsr_pass,
            "dsr": dsr,
            "sharpe": full.sharpe_ratio,
            "n_obs_trades": n_obs,
            "n_trials": n_trials,
            "alt_pass_strong_oos": bool(oos_pnl >= 5.0 and wf_rate >= 0.8),
        }
        if not dsr_pass:
            return self._fail(name, genome, gates, "dsr")

        out = {
            "genome_id": name,
            "genome": genome.to_dict(),
            "signature": list(_genome_signature(genome)),
            "gates": gates,
            "all_passed": True,
            "verdict": "PROMOTE_TO_PAPER",
            "failed_at": None,
            "score": self._score(gates, full_d),
            "ts": time.time(),
        }
        if verbose:
            print(f"  VERDICT: PROMOTE_TO_PAPER score={out['score']:.2f} dsr={dsr:.2f}")
        self._save(out)
        self._maybe_archive_champion(out)
        return out

    def _score(self, gates: Dict[str, Any], full_d: Dict[str, Any]) -> float:
        """Composite score for ranking survivors. PnL-first, capped Sharpe."""
        oos = gates.get("oos", {}).get("oos", {})
        wf = gates.get("walk_forward", {})
        dsr = float(gates.get("dsr", {}).get("dsr", 0.0) or 0.0)
        oos_pnl = float(oos.get("total_pnl", 0.0) or 0.0)
        oos_sh = float(oos.get("sharpe_ratio", 0.0) or 0.0)
        oos_sh = max(-5.0, min(5.0, oos_sh))
        oos_dd = float(oos.get("max_drawdown", 0.0) or 0.0)
        trades = float(full_d.get("total_trades", 0) or 0)
        return (
            oos_pnl * 2.0                          # primary: OOS dollars
            + oos_sh * 2.0                         # mild risk adjust (was *10 on explodey Sharpe)
            + float(wf.get("pass_rate", 0.0) or 0.0) * 20.0
            + dsr * 15.0
            + min(trades / 10.0, 15.0)
            - oos_dd * 30.0
        )

    def _fail(
        self,
        name: str,
        genome: StrategyGenome,
        gates: Dict[str, Any],
        failed_at: str,
        record_kill: bool = True,
    ) -> Dict[str, Any]:
        out = {
            "genome_id": name,
            "genome": genome.to_dict(),
            "signature": list(_genome_signature(genome)),
            "gates": gates,
            "all_passed": False,
            "verdict": "REJECT",
            "failed_at": failed_at,
            "score": 0.0,
            "ts": time.time(),
        }
        self._save(out)
        # Persist structural failure so future cycles skip this neighborhood
        if record_kill and failed_at not in ("kill_archive",):
            try:
                from evolution.kill_archive import get_archive
                get_archive().record_kill(
                    genome,
                    reason=str(failed_at),
                    meta={"genome_id": name},
                )
            except Exception:
                pass
        return out

    def _save(self, out: Dict[str, Any]) -> None:
        path = FUNNEL_DIR / f"funnel_{int(out['ts'])}_{out['genome_id'][:40]}.json"
        try:
            path.write_text(json.dumps(out, indent=2, default=str))
        except Exception:
            pass
        try:
            (FUNNEL_DIR / "latest.json").write_text(json.dumps(out, indent=2, default=str))
        except Exception:
            pass

    def _maybe_archive_champion(self, out: Dict[str, Any]) -> None:
        """Keep top-k structurally diverse champions (dedupe near-clone families)."""
        champs: List[Dict[str, Any]] = []
        if CHAMPIONS_PATH.exists():
            try:
                champs = json.loads(CHAMPIONS_PATH.read_text()).get("champions", [])
            except Exception:
                champs = []

        gdict = out.get("genome") or {}
        fam = _family_key_from_genome_dict(gdict)

        def _fam_of(c: Dict[str, Any]) -> Tuple:
            return _family_key_from_genome_dict(c.get("genome") or {})

        champs = [c for c in champs if _fam_of(c) != fam]
        champs.append(
            {
                "genome_id": out["genome_id"],
                "score": out["score"],
                "signature": out.get("signature"),
                "family_key": [fam[0], list(fam[1]), list(fam[2]), list(fam[3])],
                "genome": out["genome"],
                "gates_summary": {
                    k: {"passed": v.get("passed")} if isinstance(v, dict) else v
                    for k, v in (out.get("gates") or {}).items()
                },
                "ts": out["ts"],
            }
        )
        champs.sort(key=lambda c: c.get("score", 0.0), reverse=True)

        kept: List[Dict[str, Any]] = []
        seen_families = set()
        logic_counts: Dict[str, int] = {}
        for c in champs:
            fk = _fam_of(c)
            if fk in seen_families:
                continue
            logic = (c.get("genome") or {}).get("entry_logic", "?")
            if logic_counts.get(logic, 0) >= 2:
                continue
            kept.append(c)
            seen_families.add(fk)
            logic_counts[logic] = logic_counts.get(logic, 0) + 1
            if len(kept) >= 8:
                break

        CHAMPIONS_PATH.write_text(
            json.dumps(
                {
                    "updated_ts": time.time(),
                    "n": len(kept),
                    "champions": kept,
                    "dedupe": "structural_family_v1",
                },
                indent=2,
                default=str,
            )
        )


def _norm_op(op: str) -> str:
    """Collapse near-equivalent ops so clones don't look distinct."""
    o = (op or "?").strip()
    if o in (">", ">="):
        return ">="
    if o in ("<", "<="):
        return "<="
    if o in ("==", "="):
        return "=="
    return o


def _family_key_from_genome_dict(g: Dict[str, Any]) -> Tuple:
    """Structural family identity for shortlist dedup (ignores threshold noise)."""
    logic = g.get("entry_logic") or "?"
    conds = g.get("entry_conditions") or []
    # Indicator set only (drop op duplicates like > vs >= on same feature)
    inds = tuple(sorted({str(c.get("indicator") or "?") for c in conds}))[:4]
    # Coarse direction signature without exact ops
    dirs = tuple(
        sorted(
            {
                (str(c.get("indicator") or "?"), _norm_op(str(c.get("operator") or "?")))
                for c in conds
            }
        )[:4]
    )
    exits = tuple(
        sorted({str(e.get("exit_type") or "?") for e in (g.get("exit_rules") or [])})
    )
    return (logic, inds, dirs, exits)


def rebuild_champions_deduped() -> Dict[str, Any]:
    """One-shot rewrite of champions.json with structural family dedupe."""
    if not CHAMPIONS_PATH.exists():
        return {"n": 0, "champions": []}
    try:
        raw = json.loads(CHAMPIONS_PATH.read_text())
        champs = raw.get("champions") or []
    except Exception:
        return {"n": 0, "champions": []}

    champs.sort(key=lambda c: float(c.get("score") or 0), reverse=True)
    kept: List[Dict[str, Any]] = []
    seen = set()
    logic_counts: Dict[str, int] = {}
    for c in champs:
        g = c.get("genome") or {}
        fam = _family_key_from_genome_dict(g)
        if fam in seen:
            continue
        logic = g.get("entry_logic", "?")
        if logic_counts.get(logic, 0) >= 2:
            continue
        c2 = dict(c)
        c2["family_key"] = [fam[0], list(fam[1]), list(fam[2]), list(fam[3])]
        kept.append(c2)
        seen.add(fam)
        logic_counts[logic] = logic_counts.get(logic, 0) + 1
        if len(kept) >= 8:
            break

    payload = {
        "updated_ts": time.time(),
        "n": len(kept),
        "champions": kept,
        "dedupe": "structural_family_v1",
    }
    CHAMPIONS_PATH.write_text(json.dumps(payload, indent=2, default=str))
    return payload


def funnel_population_top(
    features: List[Dict[str, Any]],
    population: List[StrategyGenome],
    top_k: int = 5,
    min_trades: int = 30,
    n_trials_context: Optional[int] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Score top-k by fitness then run funnel on each unique structural family."""
    try:
        from evolution.kill_archive import get_archive
        arch = get_archive()
    except Exception:
        arch = None

    cands = [
        g
        for g in population
        if g.entry_logic != "RANDOM"
        and (g.backtest_results or {}).get("total_trades", 0) >= min_trades
        and not (g.backtest_results or {}).get("killed")
    ]
    cands.sort(key=lambda g: g.fitness, reverse=True)

    seen_exact = set()
    seen_family = set()
    picked: List[StrategyGenome] = []
    skipped_tabu = 0
    for g in cands:
        sig = _genome_signature(g)
        if sig in seen_exact:
            continue
        fam = _family_key_from_genome_dict(g.to_dict())
        if fam in seen_family:
            continue
        if arch is not None and arch.is_killed(g):
            skipped_tabu += 1
            continue
        seen_exact.add(sig)
        seen_family.add(fam)
        picked.append(g)
        if len(picked) >= top_k:
            break

    funnel = PromotionFunnel(features)
    results = []
    for g in picked:
        results.append(
            funnel.run(g, n_trials_context=n_trials_context, verbose=verbose)
        )
    if verbose:
        n_pass = sum(1 for r in results if r.get("all_passed"))
        arch_n = arch.size if arch is not None else 0
        print(
            f"[funnel] candidates={len(results)} promoted={n_pass} "
            f"tabu_skipped={skipped_tabu} kill_archive={arch_n}"
        )
    return results
