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
    flush_legacy_champions,
    funnel_population_top,
    get_total_trials,
    log_trials,
    revalidate_champions,
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


def prune_dir(path: Path, pattern: str, keep: int) -> int:
    """Delete oldest files matching pattern, keeping the newest `keep`.

    Unattended months must not fill the disk: population and funnel-result
    files are superseded records; champions/ledger hold what matters.
    """
    try:
        files = sorted(path.glob(pattern))
    except Exception:
        return 0
    removed = 0
    for f in files[: max(0, len(files) - keep)]:
        try:
            f.unlink()
            removed += 1
        except Exception:
            pass
    return removed


DIVERSITY_STATE = SIM / "evolution" / "diversity_state.json"


def load_streak_tax() -> dict:
    """Escalating tax for a family that keeps winning cycles.

    streak>=2 -> extra selection tax 15*streak on that family, growing each
    repeated win. This is the hard guarantee against monoculture: repetition
    gets mathematically more expensive every cycle until something else wins.
    """
    try:
        st = json.loads(DIVERSITY_STATE.read_text())
        fam, streak = st.get("streak_family"), int(st.get("streak", 0))
        if fam and streak >= 2:
            return {fam: 15.0 * streak}
    except Exception:
        pass
    return {}


def update_diversity_state(best) -> dict:
    """Record the cycle winner's family; report streaks and recent variety."""
    from evolution.strategy_log import genome_family
    fam = genome_family(best)
    winners = []
    try:
        winners = json.loads(DIVERSITY_STATE.read_text()).get("winners") or []
    except Exception:
        pass
    winners = winners[-49:] + [fam]
    streak = 0
    for f in reversed(winners):
        if f == fam:
            streak += 1
        else:
            break
    out = {
        "winners": winners,
        "streak": streak,
        "streak_family": fam,
        "distinct_last20": len(set(winners[-20:])),
        "updated_ts": time.time(),
    }
    try:
        DIVERSITY_STATE.write_text(json.dumps(out, indent=2))
    except Exception:
        pass
    return out


def load_seed_genomes(max_seeds: int = 5):
    """Warm-start seeds: past champions + last cycle's best genome."""
    from evolution.genome import StrategyGenome

    seeds = []
    champs_path = SIM / "evolution" / "champions.json"
    if champs_path.exists():
        try:
            champs = json.loads(champs_path.read_text()).get("champions", [])
            for c in champs:
                # Never seed from bug-era entries (belt-and-braces with flush)
                if abs(float(c.get("score") or 0)) > 1e6:
                    continue
                g = c.get("genome")
                if g:
                    seeds.append(StrategyGenome.from_dict(g))
                if len(seeds) >= max_seeds:
                    break
        except Exception:
            pass
    best_path = SIM / "evolution" / "best_genome_latest.json"
    if best_path.exists():
        try:
            seeds.append(StrategyGenome.from_dict(json.loads(best_path.read_text())))
        except Exception:
            pass
    # Research-agent hypotheses enter as ordinary seeds (max 2) — they face
    # the same exam/ledger/paper gauntlet as everything else, no shortcuts
    try:
        rs = json.loads((SIM / "evolution" / "research_seeds.json").read_text())
        for s in (rs.get("seeds") or [])[:2]:
            seeds.append(StrategyGenome.from_dict(s))
    except Exception:
        pass
    return seeds


def main():
    # Single-instance lock: two concurrent runners double-write shared
    # sqlite state (the suspected cause of one corruption). If another
    # runner is alive, exit quietly — launchd/KeepAlive retries later.
    import os as _os
    import subprocess as _sp
    _pids = _sp.run(["pgrep", "-f", "run_broad_evolution.py"],
                    capture_output=True, text=True).stdout.split()
    if any(p != str(_os.getpid()) for p in _pids):
        print(f"Another runner is already alive ({_pids}) — exiting.")
        return

    features = get_historical_features_1h("SOL/USDC", limit=4000)
    print(f"Loaded {len(features)} 1h features, fields={len(features[0]['features'])}")
    print(
        f"Mode: broad explore | ban RANDOM | min-trades {LAB_MIN_TRADES_FULL} | "
        f"OOS-rank | book ${BOOK_USD:.0f} | immigrants 20% | funnel | kill archive"
    )
    print("Success criteria:")
    for line in plain_english_summary().splitlines():
        print(f"  {line}")
    print(f"Prior trials logged: {get_total_trials()}")

    # One-time migration: bug-era champions (1e17 scores) must not seed
    # warm-starts; park them in champions_legacy.json
    n_flushed = flush_legacy_champions()
    if n_flushed:
        print(f"Archived {n_flushed} legacy (bug-era) champions to champions_legacy.json")

    # Bootstrap kill archive from historical funnel rejects (once per process)
    arch = reload_archive()
    seeded = arch.bootstrap_from_funnel()
    print(f"Kill archive: {arch.size} neighborhoods (seeded +{seeded} from past funnel fails)")

    # Champions promoted under older gate sets must survive the CURRENT
    # gauntlet (incl. the benchmark/beta gate) or leave the board.
    reval = revalidate_champions(features, n_trials_context=get_total_trials())
    print(f"Champion revalidation: kept={reval['kept']} demoted={reval['demoted']}")

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

        # Unattended safety: warn loudly when market data stops arriving.
        # The search still runs, but it learns nothing new and the forward
        # ledger starves — this must be visible, not silent.
        data_age_h = (time.time() - features[-1]["ts"] / 1000.0) / 3600.0
        if data_age_h > 6:
            print(
                f"  WARNING: newest candle is {data_age_h:.1f}h old — "
                f"the data pipeline may be dead. Check the downloader job."
            )

        # A transient error (sqlite lock, network blip, one bad genome) must
        # cost one cycle, never the whole unattended month.
        try:
            run_one_cycle(cycle, features)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            import traceback
            traceback.print_exc()
            write_activity({"phase": "cycle_error", "cycle": cycle, "error": str(e)})
            print(f"  CYCLE {cycle} FAILED ({e}) — sleeping 60s, then continuing")
            time.sleep(60)
        time.sleep(5)


def run_one_cycle(cycle: int, features: list) -> None:
    """One full search cycle: evolve, funnel, persist, ledger, prune."""
    engine = None
    try:
        arch = get_archive()
        write_activity({
            "phase": "cycle_start",
            "cycle": cycle,
            "generation": 0,
            "kill_archive_n": arch.size,
        })
        print(f"\n=== Cycle {cycle} ===")
        print(f"  Kill archive size: {arch.size}")

        seeds = load_seed_genomes()
        if seeds:
            print(f"  Warm-start seeds available: {len(seeds)}")
        engine = EvolutionEngine(
            features,
            population_size=150,
            elite_size=15,
            mutation_rate=0.30,
            crossover_rate=0.65,
            immigrant_rate=0.20,
            use_kill_archive=True,
            seed_genomes=seeds,
            extra_family_tax=load_streak_tax(),
        )
        engine.cycle_tag = cycle
        if engine._extra_family_tax:
            fam, tx = next(iter(engine._extra_family_tax.items()))
            print(f"  DIVERSITY GUARD: streak tax {tx:.0f} on repeat winner {fam[:60]}")
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

        # Vintage ledger: freeze this cycle's champion + daily control cohort,
        # then forward-score every frozen strategy on candles newer than its
        # freeze point. Failures here must never kill the search loop.
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

        # Autopilot: recompute forward-feedback allocation from the ledger +
        # paper results (bounded tilts; next cycle's engine loads them)
        try:
            from evolution.forward_feedback import compute_and_write
            fb = compute_and_write()
            lw = fb.get("logic_weights") or {}
            if lw:
                top = sorted(lw.items(), key=lambda x: -x[1])[:3]
                print("  FEEDBACK: logic tilts " +
                      " ".join(f"{k}={v}" for k, v in top) +
                      f" | families biased: {len(fb.get('family_bonus') or {})}")
        except Exception as e:
            print(f"  [feedback] failed: {e}")

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
        try:
            div = update_diversity_state(best)
            msg = (f"  DIVERSITY: winner_streak={div['streak']} "
                   f"distinct_winners_last20={div['distinct_last20']}")
            if div["streak"] >= 3:
                msg += "  << WARNING: same family keeps winning; streak tax escalating"
            print(msg)
        except Exception as e:
            print(f"  diversity state failed: {e}")

        # Disk retention: superseded records must not fill the disk over an
        # unattended month (~36 population + ~180 funnel files per day).
        n_pop = prune_dir(RESULTS, "evolution_*.json", keep=400)
        n_fun = prune_dir(SIM / "evolution" / "funnel_results", "funnel_*.json", keep=1000)
        if n_pop or n_fun:
            print(f"  Pruned {n_pop} population + {n_fun} funnel files")
        if cycle % 24 == 0:  # strategy-log retention, ~once a day
            try:
                from evolution.strategy_log import prune
                prune(keep_days=45)
            except Exception:
                pass
    finally:
        # evolve_continuous shuts its pool down on clean exit; this covers
        # exception paths so worker processes never leak across cycles.
        if engine is not None:
            try:
                engine.shutdown_pool()
            except Exception:
                pass


if __name__ == "__main__":
    main()
