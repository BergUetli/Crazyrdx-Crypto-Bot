#!/usr/bin/env python3
"""Broad exploration evolution runner + fund-grade post-cycle funnel.

Search:
- No RANDOM selection
- Hard min 30 trades + OOS ranking
- Stratified population across strategy families
- Immigrants every generation

Promotion:
- Log trial count every cycle
- Run gauntlet on top unique candidates
- Archive survivors to champions.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from evolution.evaluator import EvolutionEngine, write_activity
from evolution.promotion_funnel import (
    funnel_population_top,
    get_total_trials,
    log_trials,
)
from evolution.kill_archive import get_archive, reload_archive
from layer1.historical_feature_engine_1h import get_historical_features_1h

SIM = Path(__file__).resolve().parent
RESULTS = SIM / "evolution" / "population"
RESULTS.mkdir(parents=True, exist_ok=True)
STOP = SIM / "STOP_EVOLUTION"


def main():
    features = get_historical_features_1h("SOL/USDC", limit=4000)
    print(f"Loaded {len(features)} 1h features, fields={len(features[0]['features'])}")
    print(
        "Mode: broad explore | ban RANDOM | min-trades 30 | OOS-rank | "
        "immigrants 20% | post-cycle funnel | kill archive"
    )
    print(f"Prior trials logged: {get_total_trials()}")

    # Bootstrap kill archive from historical funnel rejects (once per process)
    arch = reload_archive()
    seeded = arch.bootstrap_from_funnel()
    print(f"Kill archive: {arch.size} neighborhoods (seeded +{seeded} from past funnel fails)")

    cycle = 0
    while True:
        if STOP.exists():
            print("STOP_EVOLUTION present — exiting.")
            break

        cycle += 1
        arch = get_archive()
        write_activity({
            "phase": "cycle_start",
            "cycle": cycle,
            "generation": 0,
            "kill_archive_n": arch.size,
        })
        print(f"\n=== Cycle {cycle} ===")
        print(f"  Kill archive size: {arch.size}")

        engine = EvolutionEngine(
            features,
            population_size=150,
            elite_size=15,
            mutation_rate=0.30,
            crossover_rate=0.65,
            immigrant_rate=0.20,
            use_kill_archive=True,
        )
        best = engine.evolve_continuous(
            max_duration_s=2400,
            no_improvement_limit=15,
            verbose=True,
        )

        # Estimate trials this cycle: roughly pop * (gens+1)
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
            },
        )

        # Gauntlet on top unique candidates from final population
        write_activity({"phase": "funnel", "cycle": cycle})
        funnel_results = funnel_population_top(
            features,
            engine.population,
            top_k=5,
            min_trades=30,
            n_trials_context=total_trials,
            verbose=True,
        )
        n_promoted = sum(1 for r in funnel_results if r.get("all_passed"))

        ts = int(time.time())
        bt = best.backtest_results or {}
        payload = {
            "timestamp": ts,
            "cycle": cycle,
            "mode": "broad_explore_oos_plus_funnel_killarch",
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
                "mutation_rate": 0.30,
                "crossover_rate": 0.65,
                "immigrant_rate": 0.20,
                "min_trades_full": 30,
                "rank": "oos_last_third",
                "random_banned": True,
                "funnel": True,
                "kill_archive": True,
            },
        }
        out = RESULTS / f"evolution_{ts}.json"
        out.write_text(json.dumps(payload, indent=2, default=str))
        (SIM / "evolution" / "best_genome_latest.json").write_text(
            json.dumps(best.to_dict(), indent=2)
        )
        # If anything promoted this cycle, mirror champion board path
        if n_promoted:
            prom = [r for r in funnel_results if r.get("all_passed")]
            prom.sort(key=lambda r: r.get("score", 0), reverse=True)
            (SIM / "evolution" / "last_promoted.json").write_text(
                json.dumps(prom[0], indent=2, default=str)
            )

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
            }
        )
        print(
            f"  fitness={best.fitness:.1f} gens={engine.generation} "
            f"trades={bt.get('total_trades')} logic={best.entry_logic} "
            f"promoted={n_promoted}/{len(funnel_results)} trials={total_trials} "
            f"kill_arch={get_archive().size} kill_hits={getattr(engine, 'kill_hits', 0)}"
        )
        time.sleep(5)


if __name__ == "__main__":
    main()
