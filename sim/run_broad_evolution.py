#!/usr/bin/env python3
"""Broad exploration + local exploit evolution runner + promotion funnel.

Search modes:
  EXPLORE — wide hunt (default). ~20% warm-start from champions, 20% immigrants.
  EXPLOIT — dig around a strong winner for a few cycles before going broad again.
            ~75% of pop is the focus idea + neighborhood mutants; fewer immigrants.

Trigger into EXPLOIT (after a finished cycle):
  - best fitness >= EXPLOIT_MIN_FITNESS
  - best trades >= LAB_MIN_TRADES_FULL
  - best history P&L > 0
  - optional: at least one funnel promote (preferred, not hard if score is strong)

Leave EXPLOIT early if the neighborhood stops improving for EXPLOIT_STALE_CYCLES.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from evolution.evaluator import EvolutionEngine, write_activity
from evolution.promotion_funnel import (
    funnel_population_top,
    get_total_trials,
    log_trials,
)
from evolution.kill_archive import get_archive, reload_archive
from layer1.historical_feature_engine_1h import get_historical_features_1h
from success_criteria import (
    BOOK_USD,
    LAB_MIN_TRADES_FULL,
    plain_english_summary,
)

SIM = Path(__file__).resolve().parent
RESULTS = SIM / "evolution" / "population"
RESULTS.mkdir(parents=True, exist_ok=True)
STOP = SIM / "STOP_EVOLUTION"
EXPLOIT_STATE = SIM / "evolution" / "exploit_state.json"

# When to switch from broad hunt → dig around a winner
EXPLOIT_MIN_FITNESS = 55.0
EXPLOIT_MIN_PNL = 5.0  # history $ on $100 book; modest bar, still > lab $0
EXPLOIT_MAX_CYCLES = 4  # consecutive focused searches
EXPLOIT_STALE_CYCLES = 2  # leave early if no improvement this many cycles
EXPLOIT_SEED_FRACTION = 0.75
EXPLOIT_IMMIGRANT_RATE = 0.08
EXPLOIT_MUTATION_RATE = 0.40  # poke the neighborhood a bit harder
EXPLORE_SEED_FRACTION = 0.20
EXPLORE_IMMIGRANT_RATE = 0.20
EXPLORE_MUTATION_RATE = 0.30


def load_seed_genomes(max_seeds: int = 5):
    """Warm-start seeds: past champions + last cycle's best genome."""
    from evolution.genome import StrategyGenome

    seeds = []
    champs_path = SIM / "evolution" / "champions.json"
    if champs_path.exists():
        try:
            champs = json.loads(champs_path.read_text()).get("champions", [])
            for c in champs[:max_seeds]:
                g = c.get("genome")
                if g:
                    seeds.append(StrategyGenome.from_dict(g))
        except Exception:
            pass
    best_path = SIM / "evolution" / "best_genome_latest.json"
    if best_path.exists():
        try:
            seeds.append(StrategyGenome.from_dict(json.loads(best_path.read_text())))
        except Exception:
            pass
    return seeds


def load_exploit_state() -> Dict[str, Any]:
    if not EXPLOIT_STATE.exists():
        return {"mode": "explore", "remaining": 0, "stale": 0, "best_fitness": None, "focus": []}
    try:
        return json.loads(EXPLOIT_STATE.read_text())
    except Exception:
        return {"mode": "explore", "remaining": 0, "stale": 0, "best_fitness": None, "focus": []}


def save_exploit_state(state: Dict[str, Any]) -> None:
    EXPLOIT_STATE.parent.mkdir(parents=True, exist_ok=True)
    EXPLOIT_STATE.write_text(json.dumps(state, indent=2, default=str))


def genomes_from_focus(focus_list: List[Dict[str, Any]]):
    from evolution.genome import StrategyGenome

    out = []
    for item in focus_list or []:
        g = item.get("genome") if isinstance(item, dict) else None
        if not g:
            continue
        try:
            out.append(StrategyGenome.from_dict(g))
        except Exception:
            continue
    return out


def should_enter_exploit(best, n_promoted: int, bt: Dict[str, Any]) -> bool:
    fit = float(getattr(best, "fitness", 0) or 0)
    if abs(fit) > 1000:
        return False
    trades = int(bt.get("total_trades") or 0)
    pnl = float(bt.get("total_pnl") or 0)
    if trades < LAB_MIN_TRADES_FULL:
        return False
    if pnl < EXPLOIT_MIN_PNL:
        return False
    if fit < EXPLOIT_MIN_FITNESS:
        return False
    # Prefer exam passes, but allow strong history winners too
    if n_promoted > 0:
        return True
    return fit >= EXPLOIT_MIN_FITNESS + 10 and pnl >= EXPLOIT_MIN_PNL * 2


def build_focus_list(best, funnel_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Focus parents: promoted exam-passers first, else the cycle best."""
    focus: List[Dict[str, Any]] = []
    seen = set()
    for r in funnel_results or []:
        if not r.get("all_passed"):
            continue
        g = r.get("genome")
        gid = r.get("genome_id") or ""
        if not g or gid in seen:
            continue
        seen.add(gid)
        focus.append(
            {
                "genome_id": gid,
                "genome": g,
                "score": r.get("score"),
                "source": "promoted",
            }
        )
        if len(focus) >= 3:
            break
    if best is not None:
        gid = best.genome_id
        if gid not in seen:
            focus.insert(
                0,
                {
                    "genome_id": gid,
                    "genome": best.to_dict(),
                    "score": best.fitness,
                    "source": "cycle_best",
                },
            )
    return focus[:4]


def main():
    features = get_historical_features_1h("SOL/USDC", limit=4000)
    print(f"Loaded {len(features)} 1h features, fields={len(features[0]['features'])}")
    print(
        f"Mode: explore/exploit | ban RANDOM | min-trades {LAB_MIN_TRADES_FULL} | "
        f"OOS-rank | book ${BOOK_USD:.0f} | funnel | kill archive"
    )
    print(
        f"Exploit: enter if fit>={EXPLOIT_MIN_FITNESS}, pnl>=${EXPLOIT_MIN_PNL}, "
        f"trades>={LAB_MIN_TRADES_FULL}; run up to {EXPLOIT_MAX_CYCLES} focused cycles "
        f"(leave early after {EXPLOIT_STALE_CYCLES} stale)."
    )
    print("Success criteria:")
    for line in plain_english_summary().splitlines():
        print(f"  {line}")
    print(f"Prior trials logged: {get_total_trials()}")

    arch = reload_archive()
    seeded = arch.bootstrap_from_funnel()
    print(f"Kill archive: {arch.size} neighborhoods (seeded +{seeded} from past funnel fails)")

    state = load_exploit_state()
    cycle = 0
    while True:
        if STOP.exists():
            print("STOP_EVOLUTION present — exiting.")
            break

        cycle += 1
        # Reload features each cycle so a long-running process picks up new
        # candles (also feeds the vintage ledger true forward data).
        if cycle > 1:
            try:
                fresh = get_historical_features_1h("SOL/USDC", limit=4000)
                if len(fresh) >= len(features) and fresh[-1]["ts"] >= features[-1]["ts"]:
                    features = fresh
            except Exception as e:
                print(f"  Feature reload failed (keeping previous): {e}")
        arch = get_archive()
        mode = state.get("mode") or "explore"
        if mode == "exploit" and int(state.get("remaining") or 0) <= 0:
            mode = "explore"
            state = {"mode": "explore", "remaining": 0, "stale": 0, "best_fitness": None, "focus": []}

        write_activity(
            {
                "phase": "cycle_start",
                "cycle": cycle,
                "generation": 0,
                "kill_archive_n": arch.size,
                "search_mode": mode,
                "exploit_remaining": int(state.get("remaining") or 0),
            }
        )
        print(f"\n=== Cycle {cycle} [{mode.upper()}] ===")
        print(f"  Kill archive size: {arch.size}")

        if mode == "exploit":
            seeds = genomes_from_focus(state.get("focus") or [])
            if not seeds:
                seeds = load_seed_genomes(max_seeds=5)
            # Also keep last best as backup parent
            extra = load_seed_genomes(max_seeds=2)
            for g in extra:
                if all(g.genome_id != s.genome_id for s in seeds):
                    seeds.append(g)
            seed_fraction = EXPLOIT_SEED_FRACTION
            immigrant_rate = EXPLOIT_IMMIGRANT_RATE
            mutation_rate = EXPLOIT_MUTATION_RATE
            print(
                f"  EXPLOIT focus parents: {len(seeds)} · "
                f"remaining focused cycles: {int(state.get('remaining') or 0)} · "
                f"stale: {int(state.get('stale') or 0)}"
            )
        else:
            seeds = load_seed_genomes(max_seeds=5)
            seed_fraction = EXPLORE_SEED_FRACTION
            immigrant_rate = EXPLORE_IMMIGRANT_RATE
            mutation_rate = EXPLORE_MUTATION_RATE
            if seeds:
                print(f"  Warm-start seeds available: {len(seeds)}")

        engine = EvolutionEngine(
            features,
            population_size=150,
            elite_size=15,
            mutation_rate=mutation_rate,
            crossover_rate=0.65,
            immigrant_rate=immigrant_rate,
            use_kill_archive=True,
            seed_genomes=seeds,
            seed_fraction=seed_fraction,
            mode=mode,
        )
        best = engine.evolve_continuous(
            max_duration_s=2400,
            no_improvement_limit=12 if mode == "exploit" else 15,
            verbose=True,
        )

        trials_this = max(
            engine.population_size * max(engine.generation + 1, 1),
            len(engine.population),
        )
        total_trials = log_trials(
            trials_this,
            meta={
                "cycle": cycle,
                "generations": engine.generation,
                "pop": engine.population_size,
                "best_fitness": best.fitness,
                "best_id": best.genome_id,
                "search_mode": mode,
            },
        )

        write_activity({"phase": "funnel", "cycle": cycle, "search_mode": mode})
        funnel_results = funnel_population_top(
            features,
            engine.population,
            top_k=5,
            min_trades=LAB_MIN_TRADES_FULL,
            n_trials_context=total_trials,
            verbose=True,
        )
        n_promoted = sum(1 for r in funnel_results if r.get("all_passed"))

        ts = int(time.time())
        bt = best.backtest_results or {}
        payload = {
            "timestamp": ts,
            "cycle": cycle,
            "mode": f"{mode}_oos_plus_funnel_killarch",
            "search_mode": mode,
            "generations_run": engine.generation,
            "best_fitness": best.fitness,
            "best_genome": best.to_dict(),
            "backtest": bt,
            "history": engine.history,
            "trials_this_cycle": trials_this,
            "total_trials": total_trials,
            "kill_archive_n": get_archive().size,
            "kill_hits": getattr(engine, "kill_hits", 0),
            "funnel": [
                {
                    "genome_id": r.get("genome_id"),
                    "verdict": r.get("verdict"),
                    "failed_at": r.get("failed_at"),
                    "score": r.get("score"),
                    "all_passed": r.get("all_passed"),
                }
                for r in funnel_results
            ],
            "n_promoted": n_promoted,
            "params": {
                "population_size": 150,
                "elite_size": 15,
                "mutation_rate": mutation_rate,
                "crossover_rate": 0.65,
                "immigrant_rate": immigrant_rate,
                "seed_fraction": seed_fraction,
                "search_mode": mode,
                "min_trades_full": LAB_MIN_TRADES_FULL,
                "rank": "oos_last_third",
                "random_banned": True,
                "funnel": True,
                "kill_archive": True,
            },
            "exploit_state": {
                "remaining": int(state.get("remaining") or 0),
                "stale": int(state.get("stale") or 0),
            },
        }
        out = RESULTS / f"evolution_{ts}.json"
        out.write_text(json.dumps(payload, indent=2, default=str))
        (SIM / "evolution" / "best_genome_latest.json").write_text(
            json.dumps(best.to_dict(), indent=2)
        )
        if n_promoted:
            prom = [r for r in funnel_results if r.get("all_passed")]
            prom.sort(key=lambda r: r.get("score", 0), reverse=True)
            (SIM / "evolution" / "last_promoted.json").write_text(
                json.dumps(prom[0], indent=2, default=str)
            )

# Vintage ledger: freeze this cycle's champion + daily control cohort,
        # then score only candles that arrived after each freeze point.
        # Failures here must never kill the search loop.
        try:
            from evolution.vintage_ledger import freeze_cycle, score_vintages
            vstat = freeze_cycle(best, features)
            n_scored = score_vintages(features, verbose=True)
            print(
                f"  [vintage] champion_frozen={vstat['frozen_champion']} "
                f"controls_frozen={vstat['frozen_controls']} scored={n_scored}"
            )
        except Exception as e:
            print(f"  [vintage] ledger step failed: {e}")

        # --- Explore / exploit transition ---
        fit = float(best.fitness or 0)
        prev_best = state.get("best_fitness")
        if mode == "exploit":
            remaining = max(0, int(state.get("remaining") or 0) - 1)
            stale = int(state.get("stale") or 0)
            improved = prev_best is None or fit > float(prev_best) + 1.0
            if improved:
                stale = 0
                state["focus"] = build_focus_list(best, funnel_results)
                state["best_fitness"] = fit
                print(f"  EXPLOIT improved ({prev_best} → {fit:.1f}); refresh focus parents")
            else:
                stale += 1
                print(f"  EXPLOIT no material improvement (stale {stale}/{EXPLOIT_STALE_CYCLES})")
            if remaining <= 0 or stale >= EXPLOIT_STALE_CYCLES:
                print("  EXPLOIT done → back to EXPLORE")
                state = {
                    "mode": "explore", "remaining": 0, "stale": 0,
                    "best_fitness": fit, "focus": [],
                }
            else:
                state.update({
                    "mode": "exploit", "remaining": remaining, "stale": stale,
                    "best_fitness": max(float(prev_best or fit), fit),
                })
        elif should_enter_exploit(best, n_promoted, bt):
            focus = build_focus_list(best, funnel_results)
            state = {
                "mode": "exploit", "remaining": EXPLOIT_MAX_CYCLES, "stale": 0,
                "best_fitness": fit, "focus": focus, "entered_from_cycle": cycle,
                "entered_fit": fit, "entered_pnl": float(bt.get("total_pnl") or 0),
            }
            print(
                f"  ★ Enter EXPLOIT next: fit={fit:.1f} pnl={bt.get('total_pnl')} "
                f"trades={bt.get('total_trades')} promoted={n_promoted} "
                f"focus_parents={len(focus)} for up to {EXPLOIT_MAX_CYCLES} cycles"
            )
        else:
            state = {
                "mode": "explore", "remaining": 0, "stale": 0,
                "best_fitness": fit, "focus": [],
            }
        save_exploit_state(state)

        write_activity(
            {
                "phase": "done",
                "cycle": cycle,
                "generation": engine.generation,
                "best_fitness": best.fitness,
                "best_genome": best.genome_id,
                "n_promoted": n_promoted,
                "total_trials": total_trials,
                "kill_archive_n": get_archive().size,
                "kill_hits": getattr(engine, "kill_hits", 0),
                "search_mode": mode,
                "next_mode": state.get("mode"),
                "exploit_remaining": int(state.get("remaining") or 0),
            }
        )
        print(
            f"  fitness={best.fitness:.1f} gens={engine.generation} "
            f"trades={bt.get('total_trades')} logic={best.entry_logic} "
            f"promoted={n_promoted}/{len(funnel_results)} trials={total_trials} "
            f"kill_arch={get_archive().size} mode={mode} next={state.get('mode')}"
        )
        try:
            engine.shutdown_pool()
        except Exception:
            pass
        time.sleep(5)


if __name__ == "__main__":
    main()
