#!/usr/bin/env python3
"""
evolution_runner.py — runs evolution continuously until killed.
Each cycle: evolve, save best with timestamp, brief pause, repeat.
Stop with: touch ~/.hermes/trading-bot/sim/STOP_EVOLUTION
"""

import json
import time
from datetime import datetime
from pathlib import Path

SIM = Path(__file__).resolve().parent
STOP_FILE = SIM / "STOP_EVOLUTION"
RESULTS_DIR = SIM / "evolution" / "population"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(SIM))

from evolution.evaluator import EvolutionEngine, write_activity
from layer1.historical_feature_engine import get_historical_features


def one_cycle(cycle: int, features) -> dict:
    """Run one evolution cycle, save results."""
    write_activity({"phase": "init", "cycle": cycle, "generation": 0,
                    "note": f"cycle {cycle} starting"})
    engine = EvolutionEngine(
        features,
        population_size=25,
        elite_size=4,
        mutation_rate=0.2,
        crossover_rate=0.7,
    )
    best = engine.evolve_continuous(
        max_duration_s=1800,       # 30 min per cycle max
        no_improvement_limit=8,
        verbose=True,
    )

    ts = int(time.time())
    result = {
        "timestamp": ts,
        "cycle": cycle,
        "generations_run": engine.generation,
        "best_fitness": best.fitness,
        "best_genome": best.to_dict(),
        "backtest": best.backtest_results,
        "history": engine.history,
    }
    out = RESULTS_DIR / f"evolution_{ts}.json"
    out.write_text(json.dumps(result, indent=2))

    # Always update latest
    (SIM / "evolution" / "best_genome_latest.json").write_text(
        json.dumps(best.to_dict(), indent=2)
    )
    write_activity({"phase": "done", "cycle": cycle,
                    "generation": engine.generation,
                    "best_fitness": best.fitness,
                    "best_genome": best.genome_id})
    return result


def main():
    STOP_FILE.unlink(missing_ok=True)  # clear stale stop flag
    print(f"Evolution runner started at {datetime.now()}")
    print(f"Stop file: touch {STOP_FILE}")

    features = get_historical_features("SOL/USDC", limit=2000)
    print(f"Loaded {len(features)} features")

    cycle = 0
    while not STOP_FILE.exists():
        cycle += 1
        print(f"\n=== Cycle {cycle} ===")
        try:
            r = one_cycle(cycle, features)
            print(f"  fitness={r['best_fitness']:.1f} gens={r['generations_run']}")
        except Exception as e:
            print(f"  cycle {cycle} failed: {e}")
            write_activity({"phase": "error", "cycle": cycle, "note": str(e)[:200]})
        time.sleep(10)  # brief pause between cycles

    write_activity({"phase": "stopped", "cycle": cycle})
    print("Stop file found, exiting")


if __name__ == "__main__":
    main()
