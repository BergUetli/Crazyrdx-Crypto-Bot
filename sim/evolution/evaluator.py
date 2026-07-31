"""
evaluator.py
Backtest evaluator for strategy genomes.

Selection rules:
1. RANDOM banned from selection.
2. Hard discard genomes with < 30 trades on full sample.
3. Rank by out-of-sample fitness only (last third of data).
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from evolution.genome import EntryCondition, Filter, ExitRule, StrategyGenome
from layer1.backtest_engine import BacktestEngine, LatencyModel

ACTIVITY_FILE = Path(__file__).resolve().parent.parent / "logs" / "live_activity.json"


def write_activity(state: Dict[str, Any]) -> None:
    try:
        ACTIVITY_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = ACTIVITY_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps({**state, "ts": time.time()}, default=str))
        tmp.replace(ACTIVITY_FILE)
    except Exception:
        pass


class GenomeEvaluator:
    """Evaluates strategy genomes via backtesting."""

    MIN_TRADES_FULL = 30
    MIN_TRADES_OOS = 10

    def __init__(
        self,
        features: List[Dict[str, Any]],
        initial_capital: float = 100.0,
        fee_rate: float = 0.00022,  # 2.2 bps taker fee (Jupiter measured)
        latency_model: Optional[LatencyModel] = None,
    ):
        self.features = features
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.latency_model = latency_model or LatencyModel(
            base_latency_s=10.0, mev_probability=0.3
        )
        self.engine = BacktestEngine(
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            latency_model=self.latency_model,
        )
        self._current_genome: Optional[StrategyGenome] = None

    def evaluate(self, genome: StrategyGenome) -> Dict[str, Any]:
        """Score one genome for selection."""
        self._current_genome = genome

        # 1. Ban lottery baseline from selection
        if genome.entry_logic == "RANDOM":
            return self._empty_result(genome, fitness=-300.0)

        # Full-sample metrics (reporting + min-trade gate)
        full = self._run_raw(genome, self.features)

        # 2. Hard discard thin samples
        if full.total_trades < self.MIN_TRADES_FULL:
            return {
                "genome_id": genome.genome_id,
                "fitness": -100.0 + full.total_trades,  # -100 .. -71
                "total_trades": full.total_trades,
                "win_rate": full.win_rate,
                "total_pnl": full.total_pnl,
                "sharpe_ratio": full.sharpe_ratio,
                "max_drawdown": full.max_drawdown,
                "profit_factor": full.profit_factor,
                "final_capital": full.final_capital,
            }

        # 3. OOS-only ranking on last third
        n = len(self.features)
        split = max(100, (2 * n) // 3)
        oos_features = self.features[split:]
        if len(oos_features) < 80:
            oos_features = self.features[n // 2 :]

        oos = self._run_raw(genome, oos_features)
        if oos.total_trades < self.MIN_TRADES_OOS:
            fitness = -80.0 + oos.total_trades
        else:
            fitness = self._score_result(oos)
            # Mild consistency check vs first segment
            is_features = self.features[:split]
            if len(is_features) >= 80:
                is_res = self._run_raw(genome, is_features)
                if is_res.total_trades >= 10 and is_res.total_pnl != 0:
                    ratio = oos.total_pnl / abs(is_res.total_pnl)
                    if ratio < 0.3 and oos.total_pnl < is_res.total_pnl:
                        fitness *= 0.5

        return {
            "genome_id": genome.genome_id,
            "fitness": float(fitness),
            "total_trades": full.total_trades,
            "win_rate": full.win_rate,
            "total_pnl": full.total_pnl,
            "sharpe_ratio": full.sharpe_ratio,
            "max_drawdown": full.max_drawdown,
            "profit_factor": full.profit_factor,
            "final_capital": full.final_capital,
        }

    def _empty_result(self, genome: StrategyGenome, fitness: float) -> Dict[str, Any]:
        return {
            "genome_id": genome.genome_id,
            "fitness": fitness,
            "total_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
            "final_capital": self.initial_capital,
        }

    def _run_raw(self, genome: StrategyGenome, features: List[Dict[str, Any]]):
        """Plain backtest. Never recurses into evaluate(). Reuses engine instance."""
        if features is self.features:
            engine = self.engine
        else:
            engine = BacktestEngine(
                initial_capital=self.initial_capital,
                fee_rate=self.fee_rate,
                latency_model=self.latency_model,
            )
        signal_fn = self._build_signal_fn(genome)
        return engine.run_backtest(
            strategy_name=genome.genome_id,
            pair="SOL/USDC",
            features=features,
            signal_generator=signal_fn,
            exit_rules=genome.exit_rules,
        )

    def _score_result(self, result) -> float:
        """Score one window for OOS ranking."""
        if result.total_trades <= 0:
            return -200.0

        sharpe_score = result.sharpe_ratio * 10
        win_rate_score = result.win_rate * 20
        pnl_score = result.total_pnl * 2
        dd_penalty = result.max_drawdown * 50
        trade_score = min(result.total_trades / 5.0, 20.0)
        if result.profit_factor == float("inf"):
            pf_score = 15.0
        else:
            pf_score = min(float(result.profit_factor or 0.0), 3.0) * 5.0
        return sharpe_score + win_rate_score + pnl_score + trade_score + pf_score - dd_penalty

    def _compute_fitness(self, result) -> float:
        """Legacy helper for older callers."""
        return self._score_result(result)

    def _build_signal_fn(self, genome: StrategyGenome):
        """Map genome DNA to a candle signal function."""
        strategy_type = genome.entry_logic

        def signal_fn(
            features: List[Dict], idx: int
        ) -> Optional[Tuple[str, float, float]]:
            if idx < 50:
                return None
            f = features[idx]["features"]

            # MEAN REVERSION
            if strategy_type == "MEANREV":
                close = f.get("close", 0)
                sma20 = f.get("sma_20", close)
                vol = f.get(
                    "volatility_4h",
                    f.get("volatility_1h", f.get("volatility_1d", 50)),
                )
                if vol <= 0 or sma20 <= 0:
                    return None
                dev = (close - sma20) / (vol / 10000.0)
                threshold = (
                    genome.entry_conditions[0].threshold
                    if genome.entry_conditions
                    else 2.0
                )
                if dev < -threshold:
                    strength = min(abs(dev) / max(threshold, 1e-6), 1.0)
                    return ("long", strength, genome.sizing_base * strength)
                if dev > threshold:
                    strength = min(dev / max(threshold, 1e-6), 1.0)
                    return ("short", strength, genome.sizing_base * strength)
                return None

            # BREAKOUT
            if strategy_type == "BREAKOUT":
                hl_range = f.get("hl_range_pct", 0)
                hl_avg = f.get(
                    "hl_range_avg_4h",
                    f.get("hl_range_avg_1h", f.get("hl_range_avg_1d", 0)),
                )
                if hl_avg <= 0:
                    return None
                threshold = (
                    genome.entry_conditions[0].threshold
                    if genome.entry_conditions
                    else 1.5
                )
                range_ratio = hl_range / hl_avg
                if range_ratio > threshold:
                    roc = f.get("price_roc_1h", f.get("price_roc_5m", 0))
                    strength = min(range_ratio / max(threshold, 1e-6), 1.0)
                    direction = "long" if roc > 0 else "short"
                    return (direction, strength, genome.sizing_base * strength)
                return None

            # TREND
            if strategy_type == "TREND":
                t_a = f.get(
                    "trend_alignment_1h_4h", f.get("trend_alignment_5m_15m", 0)
                )
                t_b = f.get(
                    "trend_alignment_4h_1d", f.get("trend_alignment_15m_1h", 0)
                )
                t_c = f.get(
                    "trend_alignment_1h_1d", f.get("trend_alignment_1h_4h", 0)
                )
                all_align = f.get("trend_alignment_all", 0)
                threshold = (
                    genome.entry_conditions[0].threshold
                    if genome.entry_conditions
                    else 1.0
                )
                if all_align == 1.0 or (t_a > 0 and t_b > 0 and t_c > 0):
                    roc = f.get("price_roc_4h", f.get("price_roc_15m", 0))
                    if abs(roc) < threshold:
                        return None
                    strength = min(abs(roc) / max(threshold * 2, 1e-6), 1.0)
                    direction = "long" if roc > 0 else "short"
                    return (direction, strength, genome.sizing_base * strength)
                return None

            # TFT-assisted
            if strategy_type == "TFT":
                pred = float(f.get("tft_prediction", 0.0) or 0.0)
                conf = float(f.get("tft_confidence", 0.0) or 0.0)
                if abs(pred) > 0 or conf > 0:
                    threshold = (
                        genome.entry_conditions[0].threshold
                        if genome.entry_conditions
                        else 0.1
                    )
                    if conf < max(0.05, abs(threshold) * 0.01) and abs(pred) < abs(
                        threshold
                    ):
                        return None
                    strength = min(max(abs(pred), conf), 1.0)
                    direction = "long" if pred >= 0 else "short"
                    return (
                        direction,
                        strength,
                        genome.sizing_base * max(strength, 0.25),
                    )
                # Fall through to threshold AND if TFT blank
                logic = "AND"
            else:
                logic = strategy_type

            # RANDOM baseline (selection-banned, retained for offline control)
            if logic == "RANDOM" or strategy_type == "RANDOM":
                if random.random() < 0.05:
                    direction = "long" if random.random() > 0.5 else "short"
                    return (direction, 0.5, genome.sizing_base)
                return None

            # Threshold AND/OR
            conditions_met: List[bool] = []
            for cond in genome.entry_conditions:
                value = f.get(cond.indicator, 0.0)
                if cond.operator == ">":
                    met = value > cond.threshold
                elif cond.operator == "<":
                    met = value < cond.threshold
                elif cond.operator == ">=":
                    met = value >= cond.threshold
                elif cond.operator == "<=":
                    met = value <= cond.threshold
                else:
                    met = False
                conditions_met.append(met)

            use_logic = logic if strategy_type == "TFT" else genome.entry_logic
            if use_logic == "AND":
                entry_signal = all(conditions_met) if conditions_met else False
            else:
                entry_signal = any(conditions_met) if conditions_met else False
            if not entry_signal:
                return None

            for filt in genome.filters:
                if filt.filter_type == "time_of_day":
                    hour_sin = f.get("hour_of_day_sin", 0)
                    hour_cos = f.get("hour_of_day_cos", 0)
                    hour = (np.arctan2(hour_sin, hour_cos) / (2 * np.pi) * 24) % 24
                    if int(hour) not in filt.params.get("hours", []):
                        return None
                elif filt.filter_type == "day_of_week":
                    if f.get("day_of_week", 0) not in filt.params.get("days", []):
                        return None
                elif filt.filter_type == "volatility_regime":
                    vol = f.get("volatility_4h", f.get("volatility_1h", 0))
                    if vol < filt.params.get("min_vol", 0) or vol > filt.params.get(
                        "max_vol", 1000
                    ):
                        return None
                elif filt.filter_type == "trend":
                    sma_period = filt.params.get("sma_period", 20)
                    direction_f = filt.params.get("direction", "up")
                    sma_val = f.get(f"sma_{sma_period}", 0)
                    close = f.get("close", 0)
                    if direction_f == "up" and close < sma_val:
                        return None
                    if direction_f == "down" and close > sma_val:
                        return None

            direction = (
                "long"
                if f.get("price_roc_1h", f.get("price_roc_15m", 0)) > 0
                else "short"
            )

            if genome.sizing_method == "fixed":
                size = genome.sizing_base
            elif genome.sizing_method == "volatility_scaled":
                vol = f.get("volatility_4h", f.get("volatility_1h", 50))
                size = min(genome.sizing_base * (50 / max(vol, 1)), genome.sizing_max)
            elif genome.sizing_method == "kelly":
                win_rate, avg_win, avg_loss = 0.55, 0.01, 0.005
                kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / max(
                    avg_win * avg_loss, 1e-9
                )
                size = min(max(kelly * 0.1, 0.05), genome.sizing_max)
            else:
                size = genome.sizing_base

            strength = (
                sum(conditions_met) / len(conditions_met) if conditions_met else 0.5
            )
            return (direction, strength, size)

        return signal_fn


class EvolutionEngine:
    """Manages the evolutionary process with diversity-preserving selection."""

    def __init__(
        self,
        features: List[Dict[str, Any]],
        population_size: int = 100,
        elite_size: int = 10,
        mutation_rate: float = 0.25,
        crossover_rate: float = 0.7,
        immigrant_rate: float = 0.15,
        use_kill_archive: bool = True,
    ):
        self.features = features
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        # Fraction of each generation that is fresh random immigrants
        self.immigrant_rate = immigrant_rate
        self.use_kill_archive = use_kill_archive

        self.evaluator = GenomeEvaluator(features)
        self.population: List[StrategyGenome] = []
        self.generation = 0
        self.history: List[Dict[str, Any]] = []
        self._kill = None
        self.kill_hits = 0  # how many times we avoided a killed neighborhood
        if use_kill_archive:
            try:
                from evolution.kill_archive import get_archive
                self._kill = get_archive()
            except Exception:
                self._kill = None

    def _fresh_genome(self, generation: int = 0) -> StrategyGenome:
        """Random genome, tabu-aware when kill archive is on."""
        from evolution.genome import random_genome
        if self._kill is not None:
            g = self._kill.random_genome_clean(generation=generation)
            if self._kill.is_killed(g):
                self.kill_hits += 1
            return g
        return random_genome(generation=generation)

    def _mutate_genome(self, genome: StrategyGenome, rate: float) -> StrategyGenome:
        from evolution.genome import mutate
        if self._kill is not None:
            child = self._kill.mutate_clean(genome, mutation_rate=rate)
            if self._kill.is_killed(child):
                self.kill_hits += 1
                # last-ditch: brand new clean genome
                return self._fresh_genome(generation=genome.generation + 1)
            return child
        return mutate(genome, rate)

    def initialize_population(self):
        """Create initial random population, stratified by strategy type."""
        from evolution.genome import LOGIC_OPS

        self.population = []
        # Stratified seed so each strategy family gets fair exploration
        per_type = max(1, self.population_size // max(len(LOGIC_OPS), 1))
        for logic in LOGIC_OPS:
            for _ in range(per_type):
                g = self._fresh_genome(generation=0)
                g.entry_logic = logic
                # If assigned logic is killed, reshuffle whole DNA under that logic
                if self.md_kill_logic(g):
                    for _try in range(20):
                        g = self._fresh_genome(generation=0)
                        g.entry_logic = logic
                        if not self.md_kill_logic(g):
                            break
                self.population.append(g)
        while len(self.population) < self.population_size:
            self.population.append(self._fresh_genome(generation=0))
        self.population = self.population[: self.population_size]

    def md_kill_logic(self, genome: StrategyGenome) -> bool:
        return bool(self._kill is not None and self._kill.is_killed(genome))

    def evaluate_population(self):
        """Evaluate all genomes and assign fitness."""
        print(f"  Evaluating {len(self.population)} genomes...")
        total = len(self.population)
        kill_n = self._kill.size if self._kill else 0
        for i, genome in enumerate(self.population):
            # Heartbeat every 10 genomes (not every one) — big I/O win
            if i == 0 or (i + 1) % 10 == 0 or (i + 1) == total:
                write_activity(
                    {
                        "phase": "evaluate",
                        "generation": self.generation,
                        "genome_index": i + 1,
                        "population": total,
                        "current_genome": genome.genome_id,
                        "kill_archive_n": kill_n,
                    }
                )
            if genome.backtest_results is None:
                # Hard short-circuit: known-killed neighborhoods get tabu fitness
                if self.md_kill_logic(genome):
                    self.kill_hits += 1
                    genome.fitness = -400.0
                    genome.backtest_results = {
                        "genome_id": genome.genome_id,
                        "fitness": -400.0,
                        "total_trades": 0,
                        "win_rate": 0.0,
                        "total_pnl": 0.0,
                        "sharpe_ratio": 0.0,
                        "max_drawdown": 0.0,
                        "profit_factor": 0.0,
                        "final_capital": 100.0,
                        "killed": True,
                    }
                else:
                    result = self.evaluator.evaluate(genome)
                    genome.fitness = result["fitness"]
                    genome.backtest_results = result
            if (i + 1) % 10 == 0:
                print(f"    {i + 1}/{total} evaluated")

    def select_parents(self) -> List[StrategyGenome]:
        """Tournament selection with elite carry-over."""
        sorted_pop = sorted(self.population, key=lambda g: g.fitness, reverse=True)
        parents = sorted_pop[: self.elite_size]
        n_tournament = self.population_size - self.elite_size
        for _ in range(n_tournament):
            tournament = random.sample(
                self.population, min(5, len(self.population))
            )
            winner = max(tournament, key=lambda g: g.fitness)
            parents.append(winner)
        return parents

    def evolve_continuous(
        self,
        max_duration_s: float = 7200,
        no_improvement_limit: int = 12,
        verbose: bool = True,
    ):
        """Run evolution until time limit or stagnation."""
        from evolution.genome import crossover

        start_time = time.time()
        best_fitness_history: List[float] = []
        no_improvement_count = 0

        if verbose:
            print("Starting continuous evolution")
            print(f"  Max duration: {max_duration_s}s")
            print(f"  Pop={self.population_size} elite={self.elite_size} "
                  f"mut={self.mutation_rate} immigrants={self.immigrant_rate}")
            kill_n = self._kill.size if self._kill else 0
            print(f"  Kill archive: {kill_n} neighborhoods tabu"
                  if self._kill else "  Kill archive: off")
            print(f"  No improvement limit: {no_improvement_limit} generations")

        write_activity(
            {
                "phase": "init",
                "generation": 0,
                "population": self.population_size,
                "kill_archive_n": self._kill.size if self._kill else 0,
            }
        )
        self.initialize_population()
        self.evaluate_population()

        best = max(self.population, key=lambda g: g.fitness)
        best_fitness_history.append(best.fitness)
        write_activity(
            {
                "phase": "generation_done",
                "generation": 0,
                "best_fitness": best.fitness,
                "best_genome": best.genome_id,
            }
        )
        if verbose:
            print(f"Generation 0: best fitness = {best.fitness:.2f}")

        gen = 0
        while True:
            gen += 1
            self.generation = gen

            elapsed = time.time() - start_time
            if elapsed > max_duration_s:
                if verbose:
                    print(
                        f"Time limit reached ({elapsed:.0f}s > {max_duration_s}s)"
                    )
                break

            write_activity(
                {
                    "phase": "breeding",
                    "generation": gen,
                    "elapsed_s": round(elapsed, 1),
                }
            )
            parents = self.select_parents()

            next_gen: List[StrategyGenome] = []
            sorted_pop = sorted(
                self.population, key=lambda g: g.fitness, reverse=True
            )
            next_gen.extend(sorted_pop[: self.elite_size])

            n_immigrants = max(1, int(self.population_size * self.immigrant_rate))
            target_bred = self.population_size - n_immigrants

            while len(next_gen) < target_bred:
                if random.random() < self.crossover_rate and len(parents) >= 2:
                    p1, p2 = random.sample(parents, 2)
                    child = crossover(p1, p2)
                    child.backtest_results = None
                    child.fitness = 0.0
                    # If crossover landed in a killed hood, mutate out
                    if self.md_kill_logic(child):
                        child = self._mutate_genome(child, min(1.0, self.mutation_rate * 2))
                else:
                    child = StrategyGenome.from_dict(
                        random.choice(parents).to_dict()
                    )
                    child.genome_id = (
                        f"clone_{child.genome_id}_{random.randint(1000, 9999)}"
                    )
                    child.backtest_results = None
                    child.fitness = 0.0

                if random.random() < self.mutation_rate:
                    child = self._mutate_genome(child, self.mutation_rate)
                # Extra structure mutation boost for exploration
                if random.random() < 0.10:
                    child = self._mutate_genome(child, min(1.0, self.mutation_rate * 2))

                # Final tabu gate
                if self.md_kill_logic(child):
                    child = self._fresh_genome(generation=gen)

                child.generation = gen
                next_gen.append(child)

            # Fresh immigrants keep search broad (tabu-aware)
            while len(next_gen) < self.population_size:
                imm = self._fresh_genome(generation=gen)
                next_gen.append(imm)

            self.population = next_gen[: self.population_size]
            self.evaluate_population()

            current_best = max(self.population, key=lambda g: g.fitness)
            avg_fitness = sum(g.fitness for g in self.population) / len(
                self.population
            )
            self.history.append(
                {
                    "generation": gen,
                    "best_fitness": current_best.fitness,
                    "avg_fitness": avg_fitness,
                    "best_genome_id": current_best.genome_id,
                    "elapsed_s": elapsed,
                }
            )

            if current_best.fitness > best_fitness_history[-1] + 1e-9:
                best_fitness_history.append(current_best.fitness)
                no_improvement_count = 0
                if verbose:
                    print(
                        f"Generation {gen}: NEW BEST = {current_best.fitness:.2f} "
                        f"(avg={avg_fitness:.2f})"
                    )
            else:
                no_improvement_count += 1
                if verbose and gen % 10 == 0:
                    print(
                        f"Generation {gen}: best={current_best.fitness:.2f} "
                        f"avg={avg_fitness:.2f} no_improve={no_improvement_count}"
                    )

            write_activity(
                {
                    "phase": "generation_done",
                    "generation": gen,
                    "best_fitness": current_best.fitness,
                    "avg_fitness": round(avg_fitness, 2),
                    "best_genome": current_best.genome_id,
                    "new_best": no_improvement_count == 0,
                    "no_improve": no_improvement_count,
                    "elapsed_s": round(time.time() - start_time, 1),
                }
            )

            if no_improvement_count >= no_improvement_limit:
                if verbose:
                    print(
                        f"Converged: {no_improvement_limit} generations "
                        f"without improvement"
                    )
                break

        best = max(self.population, key=lambda g: g.fitness)
        elapsed = time.time() - start_time
        write_activity(
            {
                "phase": "done",
                "generation": gen,
                "best_fitness": best.fitness,
                "best_genome": best.genome_id,
                "elapsed_s": round(elapsed, 1),
            }
        )
        if verbose:
            print("\nEvolution complete!")
            print(f"  Duration: {elapsed:.0f}s")
            print(f"  Generations: {gen}")
            print(f"  Best fitness: {best.fitness:.2f}")
            print(f"  Best genome: {best.genome_id}")
            bt = best.backtest_results or {}
            print(
                f"  trades={bt.get('total_trades')} win={bt.get('win_rate')} "
                f"pnl={bt.get('total_pnl')}"
            )
        return best
