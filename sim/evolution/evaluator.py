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
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from evolution.genome import EntryCondition, Filter, ExitRule, StrategyGenome
from layer1.backtest_engine import BacktestEngine, LatencyModel
from layer1.multi_pair import add_cross_pair_features, load_multi_pair
from success_criteria import (
    FEE_RATE_BASE,
    FITNESS_MAX as SC_FITNESS_MAX,
    FITNESS_MIN as SC_FITNESS_MIN,
    EXPLORE_FAMILY_TAX,
    EXPLORE_FAMILY_WINDOW_DAYS,
    EXPLORE_FRONTIER_FRACTION,
    EXPLORE_GRADUATED_TAX,
    FIXED_COST_PER_SIDE_USD,
    LAB_EMBARGO_BARS,
    LAB_MIN_TRADES_FULL,
    LAB_MIN_TRADES_OOS,
    LONG_ONLY,
    MEV_COST_BPS,
    MEV_PROB_SEARCH,
)

ACTIVITY_FILE = Path(__file__).resolve().parent.parent / "logs" / "live_activity.json"

# Vectorized signal precomputation (layer1.fast_signals). Equivalence with the
# legacy per-bar path is enforced by the test suite; FAST_SIGNALS=0 disables.
FAST_SIGNALS_ENABLED = os.environ.get("FAST_SIGNALS", "1") != "0"


def write_activity(state: Dict[str, Any]) -> None:
    try:
        ACTIVITY_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = ACTIVITY_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps({**state, "ts": time.time()}, default=str))
        tmp.replace(ACTIVITY_FILE)
    except Exception:
        pass


class GenomeEvaluator:
    """Evaluates strategy genomes via backtesting.

    Selection goal (what we chase):
      OOS paper profit after fees, with enough trades, without crazy drawdowns.
      Fitness is a ranking score in roughly [-200, +200], NOT dollars and NOT
      annualized Sharpe.

    Features are augmented with cross-pair data (SOL/BTC, SOL/ETH ratios,
    correlations, leading indicators) for free (no API cost).

    Not the goal:
      Inflated Sharpe, win-rate theater, or any metric that can explode to 1e17.
    """

    MIN_TRADES_FULL = LAB_MIN_TRADES_FULL
    MIN_TRADES_OOS = LAB_MIN_TRADES_OOS
    # Hard bounds so one bad metric can never dominate selection/display
    FITNESS_MIN = SC_FITNESS_MIN
    FITNESS_MAX = SC_FITNESS_MAX

    def __init__(
        self,
        features: List[Dict[str, Any]],
        initial_capital: float = 100.0,
        fee_rate: float = FEE_RATE_BASE,  # from success_criteria (Jupiter-measured)
        latency_model: Optional[LatencyModel] = None,
        augment: bool = True,
    ):
        self.features = features
        # Augment with cross-pair features if btc/eth data available.
        # augment=False when features were already augmented (e.g. in
        # parallel workers receiving the parent's augmented features).
        if augment:
            try:
                _, btc_feats, eth_feats = load_multi_pair("SOL/USDC", limit=len(features))
                if btc_feats is not None and eth_feats is not None:
                    self.features = add_cross_pair_features(features, btc_feats, eth_feats)
            except Exception:
                pass  # multi-pair unavailable; use single-pair features
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        # Search uses deterministic expected MEV drag (not coin-flip noise)
        self.latency_model = latency_model or LatencyModel(
            base_latency_s=10.0,
            mev_probability=MEV_PROB_SEARCH,
            mev_cost_bps=MEV_COST_BPS,
            stochastic=False,
        )
        self.engine = BacktestEngine(
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            latency_model=self.latency_model,
            fixed_cost_per_side=FIXED_COST_PER_SIDE_USD,
            long_only=LONG_ONLY,
        )
        self._current_genome: Optional[StrategyGenome] = None
        # Stable OOS/IS slices (computed once so the fast-signal column cache
        # can key on list identity instead of rebuilding per genome).
        # An embargo gap between IS and OOS prevents boundary leakage: rolling
        # features near the split contain IS information.
        n = len(self.features)
        self._split = max(100, (2 * n) // 3)
        self._oos_features = self.features[self._split + LAB_EMBARGO_BARS:]
        if len(self._oos_features) < 80:
            self._oos_features = self.features[n // 2:]
        self._is_features = self.features[: self._split]

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
        oos_features = self._oos_features

        oos = self._run_raw(genome, oos_features)
        if oos.total_trades < self.MIN_TRADES_OOS:
            fitness = -80.0 + oos.total_trades
        else:
            fitness = self._score_result(oos)
            # Consistency: if OOS makes money but IS is a large loss, haircut.
            # If OOS is much weaker than a strong IS, mild haircut (overfit smell).
            is_features = self._is_features
            if len(is_features) >= 80:
                is_res = self._run_raw(genome, is_features)
                if is_res.total_trades >= 10:
                    if oos.total_pnl > 0 and is_res.total_pnl < -abs(oos.total_pnl):
                        fitness *= 0.5
                    elif is_res.total_pnl > 1.0 and oos.total_pnl > 0:
                        ratio = oos.total_pnl / max(is_res.total_pnl, 1e-9)
                        if ratio < 0.3:
                            fitness *= 0.7

        fitness = self._clamp_fitness(fitness)

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

    def _clamp_fitness(self, x: float) -> float:
        if x != x or x in (float("inf"), float("-inf")):  # NaN/inf
            return self.FITNESS_MIN
        return float(max(self.FITNESS_MIN, min(self.FITNESS_MAX, x)))

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
                fixed_cost_per_side=FIXED_COST_PER_SIDE_USD,
                long_only=LONG_ONLY,
            )
        signal_fn = None
        if FAST_SIGNALS_ENABLED:
            try:
                from layer1.fast_signals import build_array_signal_fn
                signal_fn = build_array_signal_fn(genome, features)
            except Exception:
                signal_fn = None  # any surprise -> proven legacy path
        if signal_fn is None:
            signal_fn = self._build_signal_fn(genome)
        return engine.run_backtest(
            strategy_name=genome.genome_id,
            pair="SOL/USDC",
            features=features,
            signal_generator=signal_fn,
            exit_rules=genome.exit_rules,
        )

    def _score_result(self, result) -> float:
        """Rank one window. Primary goal = OOS net PnL after fees.

        Secondary: enough trades, controlled drawdown, mild risk adjust.
        Win rate is intentionally weak (high WR + tiny $ is not the goal).
        """
        if result.total_trades <= 0:
            return -200.0

        pnl = float(result.total_pnl or 0.0)
        trades = int(result.total_trades or 0)
        dd = float(result.max_drawdown or 0.0)
        wr = float(result.win_rate or 0.0)
        # Clamped risk metric from backtest (already in [-5, 5])
        risk = float(result.sharpe_ratio or 0.0)
        risk = max(-5.0, min(5.0, risk))

        if result.profit_factor == float("inf"):
            pf = 3.0
        else:
            pf = max(0.0, min(float(result.profit_factor or 0.0), 3.0))

        # --- Primary: money ---
        # $1 PnL on $100 book ≈ +1.0 fitness unit (readable scale)
        pnl_score = pnl * 1.0

        # --- Sample quality ---
        # Soft bonus up to +15 for getting from 10 → 50+ trades
        # Saturation penalty: trading >50% of bars = degenerate, not skill
        trade_score = min(max(trades - 10, 0) / 40.0, 1.0) * 15.0
        # Heavy penalty if trades exceed 50% of available bars (≈2000 on 4000 bars)
        if trades > 500:
            trade_saturation = min((trades - 500) / 500.0, 3.0) * 10.0
            trade_score -= trade_saturation

        # --- Risk controls ---
        # DD penalty: 10% DD ≈ -10, 30% DD ≈ -45 (convex)
        dd_penalty = (dd * 100.0) + max(0.0, dd - 0.15) * 200.0
        # Mild risk-adjust: at most ±10 from trade-return ratio
        risk_score = risk * 2.0
        # Profit factor mild (cap 3): at most +9
        pf_score = pf * 3.0
        # Win rate almost cosmetic (at most +5) — prevents WR theater
        wr_score = wr * 5.0

        # Losing money must rank below making money even if WR/Sharpe look pretty
        if pnl <= 0:
            score = pnl_score + 0.25 * trade_score + 0.25 * risk_score - dd_penalty
        else:
            score = (
                pnl_score
                + trade_score
                + risk_score
                + pf_score
                + wr_score
                - dd_penalty
            )

        return self._clamp_fitness(score)

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
                # Kelly fraction f* = p - (1-p)/b, with b = avg_win/avg_loss.
                # (Old formula divided by avg_win*avg_loss, giving f*≈65 —
                # i.e. "kelly" silently meant "always max size".)
                win_rate, avg_win, avg_loss = 0.55, 0.01, 0.005
                b = avg_win / max(avg_loss, 1e-9)
                kelly = win_rate - (1.0 - win_rate) / b  # ≈ 0.325
                # Half-Kelly for safety, clamped to [5%, sizing_max]
                size = min(max(kelly * 0.5, 0.05), genome.sizing_max)
            else:
                size = genome.sizing_base

            strength = (
                sum(conditions_met) / len(conditions_met) if conditions_met else 0.5
            )
            return (direction, strength, size)

        return signal_fn


# ---------------------------------------------------------------------------
# Multiprocessing workers (module-level so they are picklable under spawn).
# Each worker process builds one evaluator from the parent's already-augmented
# features and reuses it for every genome it receives.
# ---------------------------------------------------------------------------
_WORKER_EVALUATOR: Optional["GenomeEvaluator"] = None


def _init_eval_worker(
    features: List[Dict[str, Any]],
    initial_capital: float,
    fee_rate: float,
) -> None:
    global _WORKER_EVALUATOR
    _WORKER_EVALUATOR = GenomeEvaluator(
        features,
        initial_capital=initial_capital,
        fee_rate=fee_rate,
        augment=False,
    )


def _eval_worker(genome_dict: Dict[str, Any]) -> Dict[str, Any]:
    genome = StrategyGenome.from_dict(genome_dict)
    return _WORKER_EVALUATOR.evaluate(genome)


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
        seed_genomes: Optional[List[StrategyGenome]] = None,
        n_workers: Optional[int] = None,
        extra_family_tax: Optional[Dict[str, float]] = None,
    ):
        self.features = features
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        # Fraction of each generation that is fresh random immigrants
        self.immigrant_rate = immigrant_rate
        self.use_kill_archive = use_kill_archive
        # Warm-start seeds (e.g. past champions) injected into gen-0 population
        self.seed_genomes = list(seed_genomes or [])

        self.evaluator = GenomeEvaluator(features)

        # Calibrate threshold sampling ranges to the actual (augmented) data
        # so random/mutated conditions land where they can actually flip.
        try:
            from evolution.genome import calibrate_threshold_ranges
            n_cal = calibrate_threshold_ranges(self.evaluator.features)
            if n_cal:
                print(f"  Calibrated threshold ranges for {n_cal} indicators")
        except Exception as e:
            print(f"  Threshold calibration skipped: {e}")

        # Parallel evaluation: default to cpu_count-2 workers (min 1).
        # Override with EVOLUTION_WORKERS env var; <=1 means serial.
        if n_workers is None:
            env_w = os.environ.get("EVOLUTION_WORKERS")
            if env_w is not None:
                try:
                    n_workers = int(env_w)
                except ValueError:
                    n_workers = None
        if n_workers is None:
            n_workers = max(1, (os.cpu_count() or 2) - 2)
        self.n_workers = max(1, int(n_workers))
        self._pool = None

        # Evaluation cache: identical DNA (clones, unchanged elites) never
        # gets re-backtested. Keyed by structural signature.
        self._eval_cache: Dict[Any, Dict[str, Any]] = {}
        self.cache_hits = 0

        self.population: List[StrategyGenome] = []
        self.generation = 0
        self.history: List[Dict[str, Any]] = []
        self.cycle_tag = 0  # set by the runner for strategy-log attribution

        # --- Exploration state (anti-monoculture) ---
        # Recent per-family evaluation counts drive a selection-time fitness
        # tax (diminishing returns); indicator usage drives frontier
        # immigrants toward under-explored features; graduated/champion
        # families get a flat extra tax (they're done).
        self._family_recent: Dict[str, int] = {}
        self._indicator_usage: Dict[str, int] = {}
        self._done_families: set = set()
        # Escalating streak tax from the runner's diversity guard: a family
        # that keeps winning cycle after cycle gets progressively priced out
        self._extra_family_tax: Dict[str, float] = dict(extra_family_tax or {})
        try:
            from evolution.strategy_log import family_counts, indicator_usage
            self._family_recent = family_counts(EXPLORE_FAMILY_WINDOW_DAYS)
            self._indicator_usage = dict(indicator_usage(7.0))
        except Exception:
            pass
        try:
            from evolution.promotion_funnel import done_family_keys
            self._done_families = done_family_keys()
        except Exception:
            pass
        self._log_buffer: List[Dict[str, Any]] = []
        self._kill = None
        self.kill_hits = 0  # how many times we avoided a killed neighborhood
        if use_kill_archive:
            try:
                from evolution.kill_archive import get_archive
                self._kill = get_archive()
            except Exception:
                self._kill = None

    def _fresh_genome(self, generation: int = 0) -> StrategyGenome:
        """Random genome, tabu-aware when kill archive is on.

        A fraction of fresh genomes are 'frontier immigrants': one condition
        is redirected to an under-explored indicator (inverse-frequency
        sampling over recent usage), so coverage gaps get probed instead of
        waiting for luck.
        """
        from evolution.genome import random_genome
        if self._kill is not None:
            g = self._kill.random_genome_clean(generation=generation)
            if self._kill.is_killed(g):
                self.kill_hits += 1
        else:
            g = random_genome(generation=generation)
        if self._indicator_usage and random.random() < EXPLORE_FRONTIER_FRACTION:
            g = self._frontierize(g)
        return g

    def _frontierize(self, genome: StrategyGenome) -> StrategyGenome:
        """Point one entry condition at an under-explored indicator."""
        from evolution.genome import INDICATORS, get_threshold_range
        if not genome.entry_conditions:
            return genome
        # Inverse-frequency weights: never-tried indicators dominate
        weights = [1.0 / (1 + self._indicator_usage.get(i, 0)) for i in INDICATORS]
        pick = random.choices(INDICATORS, weights=weights)[0]
        cond = random.choice(genome.entry_conditions)
        cond.indicator = pick
        lo, hi = get_threshold_range(pick)
        cond.threshold = random.uniform(lo, hi)
        genome.genome_id = f"frontier_{genome.genome_id[-18:]}"
        return genome

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
        """Create initial random population, stratified by strategy type.

        Warm start: past champions (and their mutated variants) are injected
        first, capped at 20% of the population, so each cycle refines known
        good regions instead of always restarting from pure noise.
        """
        from evolution.genome import LOGIC_OPS

        self.population = []
        max_seeds = max(0, int(self.population_size * 0.20))
        n_seeded = 0
        for seed in self.seed_genomes:
            if n_seeded >= max_seeds:
                break
            if self.md_kill_logic(seed):
                continue
            base = StrategyGenome.from_dict(seed.to_dict())
            base.genome_id = f"seed_{seed.genome_id[:24]}_{random.randint(1000, 9999)}"
            base.backtest_results = None
            base.fitness = 0.0
            base.generation = 0
            self.population.append(base)
            n_seeded += 1
            # Two mutated variants per seed to explore its neighborhood
            for _ in range(2):
                if n_seeded >= max_seeds:
                    break
                var = self._mutate_genome(base, self.mutation_rate)
                var.generation = 0
                self.population.append(var)
                n_seeded += 1
        if n_seeded:
            print(f"  Warm start: seeded {n_seeded} genomes from champions")
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

    def _apply_result(self, genome: StrategyGenome, result: Dict[str, Any]) -> None:
        result = dict(result)
        result["genome_id"] = genome.genome_id
        genome.fitness = result["fitness"]
        genome.backtest_results = result

    def _tabu_result(self, genome: StrategyGenome) -> Dict[str, Any]:
        return {
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

    def _get_pool(self):
        """Lazily create the persistent worker pool (spawn-safe)."""
        if self._pool is None and self.n_workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            self._pool = ProcessPoolExecutor(
                max_workers=self.n_workers,
                initializer=_init_eval_worker,
                initargs=(
                    self.evaluator.features,
                    self.evaluator.initial_capital,
                    self.evaluator.fee_rate,
                ),
            )
        return self._pool

    def shutdown_pool(self) -> None:
        if self._pool is not None:
            try:
                self._pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            self._pool = None

    def evaluate_population(self):
        """Evaluate all genomes and assign fitness.

        Order of resolution per genome:
        1. already evaluated (elites)  2. kill-archive tabu
        3. eval cache (identical DNA)  4. backtest (parallel when n_workers>1)
        """
        from evolution.genome import dna_signature

        total = len(self.population)
        kill_n = self._kill.size if self._kill else 0
        print(f"  Evaluating {total} genomes ({self.n_workers} workers)...")

        pending: List[StrategyGenome] = []
        for genome in self.population:
            if genome.backtest_results is not None:
                continue
            if self.md_kill_logic(genome):
                self.kill_hits += 1
                self._apply_result(genome, self._tabu_result(genome))
                continue
            cached = self._eval_cache.get(dna_signature(genome))
            if cached is not None:
                self.cache_hits += 1
                self._apply_result(genome, cached)
                continue
            pending.append(genome)

        write_activity(
            {
                "phase": "evaluate",
                "generation": self.generation,
                "genome_index": total - len(pending),
                "population": total,
                "pending": len(pending),
                "cache_hits": self.cache_hits,
                "kill_archive_n": kill_n,
            }
        )

        def _finish(genome: StrategyGenome, result: Dict[str, Any], done: int):
            self._apply_result(genome, result)
            if len(self._eval_cache) < 50_000:
                self._eval_cache[dna_signature(genome)] = dict(result)
            # Lab notebook: every fresh evaluation becomes a log row, and the
            # in-memory family counter feeds the exploration tax immediately
            try:
                from evolution.strategy_log import genome_family, log_genome
                self._log_buffer.append(log_genome(
                    genome, result, self.cycle_tag, self.generation))
                fam = genome_family(genome)
                self._family_recent[fam] = self._family_recent.get(fam, 0) + 1
                for c in (genome.entry_conditions or []):
                    self._indicator_usage[c.indicator] = (
                        self._indicator_usage.get(c.indicator, 0) + 1)
            except Exception:
                pass
            if done % 20 == 0 or done == len(pending):
                write_activity(
                    {
                        "phase": "evaluate",
                        "generation": self.generation,
                        "genome_index": total - len(pending) + done,
                        "population": total,
                        "current_genome": genome.genome_id,
                        "cache_hits": self.cache_hits,
                        "kill_archive_n": kill_n,
                    }
                )
                print(f"    {done}/{len(pending)} evaluated")

        def _flush_log():
            if self._log_buffer:
                try:
                    from evolution.strategy_log import log_rows
                    log_rows(self._log_buffer)
                except Exception:
                    pass
                self._log_buffer = []

        if pending and self.n_workers > 1:
            try:
                pool = self._get_pool()
                dicts = [g.to_dict() for g in pending]
                for done, (genome, result) in enumerate(
                    zip(pending, pool.map(_eval_worker, dicts, chunksize=4)), 1
                ):
                    _finish(genome, result, done)
                _flush_log()
                return
            except Exception as e:
                print(f"  Parallel evaluation failed ({e}); falling back to serial")
                self.shutdown_pool()

        for done, genome in enumerate(pending, 1):
            if genome.backtest_results is None:
                _finish(genome, self.evaluator.evaluate(genome), done)
        _flush_log()

    def _sel_fitness(self, g: StrategyGenome) -> float:
        """Selection-time fitness: raw fitness minus the exploration tax.

        Families tried many times recently earn less breeding priority
        (diminishing returns); champion/graduated families earn a flat extra
        tax — they're done, re-breeding them adds nothing. Reported fitness
        stays raw everywhere; only SELECTION feels this.
        """
        try:
            from evolution.strategy_log import genome_family
            fam = genome_family(g)
        except Exception:
            return g.fitness
        tax = EXPLORE_FAMILY_TAX * math.log1p(self._family_recent.get(fam, 0))
        if fam in self._done_families:
            tax += EXPLORE_GRADUATED_TAX
        tax += self._extra_family_tax.get(fam, 0.0)
        return g.fitness - tax

    def select_parents(self) -> List[StrategyGenome]:
        """Tournament selection with elite carry-over (exploration-taxed)."""
        sorted_pop = sorted(self.population, key=self._sel_fitness, reverse=True)
        parents = sorted_pop[: self.elite_size]
        n_tournament = self.population_size - self.elite_size
        for _ in range(n_tournament):
            tournament = random.sample(
                self.population, min(5, len(self.population))
            )
            winner = max(tournament, key=self._sel_fitness)
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
            # Elite carry-over is exploration-taxed too: a novel decent family
            # can displace the 900th clone of the reigning one
            sorted_pop = sorted(
                self.population, key=self._sel_fitness, reverse=True
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

        self.shutdown_pool()
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
            print(f"  Eval cache hits: {self.cache_hits}")
            bt = best.backtest_results or {}
            print(
                f"  trades={bt.get('total_trades')} win={bt.get('win_rate')} "
                f"pnl={bt.get('total_pnl')}"
            )
        return best
