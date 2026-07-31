#!/usr/bin/env python3
"""
validate_suite.py
Full validation suite for the trading system.
Runs walk-forward, Monte Carlo, MEV-adjusted, and regime analysis.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

import sys
SIM_DIR = Path.home() / ".hermes" / "trading-bot" / "sim"
sys.path.insert(0, str(SIM_DIR))

from config import LOG_DIR, MODEL_DIR
from layer1.historical_feature_engine import get_historical_features
from layer1.backtest_engine import BacktestEngine, LatencyModel, walk_forward_split
from layer1.strategies import STRATEGIES, list_strategies


class ValidationSuite:
    def __init__(self):
        self.results = {}
        self.timestamp = datetime.now()

    def run_walk_forward(
        self,
        features: list,
        strategy_name: str,
        strategy_fn,
        n_splits: int = 5,
    ) -> Dict[str, Any]:
        """Run walk-forward validation."""
        print(f"\n{'='*60}")
        print(f"Walk-Forward Validation: {strategy_name}")
        print(f"{'='*60}")

        splits = walk_forward_split(features, n_splits)
        split_results = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(splits):
            engine = BacktestEngine(
                initial_capital=100.0,
                fee_rate=0.00022,  # 2.2 bps taker fee (Jupiter measured)
                latency_model=LatencyModel(base_latency_s=10.0, mev_probability=0.3),
            )

            # Test on the test split
            test_features = features[test_start:test_end]
            result = engine.run_backtest(
                strategy_name=f"{strategy_name}_split_{i}",
                pair="SOL/USDC",
                features=test_features,
                signal_generator=strategy_fn,
            )

            split_results.append({
                "split": i,
                "train_range": (train_start, train_end),
                "test_range": (test_start, test_end),
                "trades": result.total_trades,
                "win_rate": result.win_rate,
                "pnl": result.total_pnl,
                "sharpe": result.sharpe_ratio,
                "max_dd": result.max_drawdown,
            })

            print(f"  Split {i}: trades={result.total_trades} pnl={result.total_pnl:.4f} sharpe={result.sharpe_ratio:.2f}")

        # Aggregate
        avg_pnl = np.mean([r["pnl"] for r in split_results])
        avg_sharpe = np.mean([r["sharpe"] for r in split_results if r["sharpe"] != 0])
        avg_win_rate = np.mean([r["win_rate"] for r in split_results])
        total_trades = sum([r["trades"] for r in split_results])

        summary = {
            "strategy": strategy_name,
            "n_splits": len(splits),
            "total_trades": total_trades,
            "avg_pnl": float(avg_pnl),
            "avg_sharpe": float(avg_sharpe) if not np.isnan(avg_sharpe) else 0.0,
            "avg_win_rate": float(avg_win_rate),
            "splits": split_results,
        }

        print(f"\n  Average P&L: {avg_pnl:.4f}")
        print(f"  Average Sharpe: {avg_sharpe:.2f}")
        print(f"  Average Win Rate: {avg_win_rate:.2%}")

        return summary

    def run_monte_carlo(
        self,
        features: list,
        strategy_name: str,
        strategy_fn,
        n_simulations: int = 100,
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation by shuffling trade order."""
        print(f"\n{'='*60}")
        print(f"Monte Carlo Simulation: {strategy_name} ({n_simulations} runs)")
        print(f"{'='*60}")

        # First, run the strategy once to get trades
        engine = BacktestEngine(
            initial_capital=100.0,
            fee_rate=0.00022,  # 2.2 bps taker fee (Jupiter measured)
            latency_model=LatencyModel(base_latency_s=10.0, mev_probability=0.3),
        )

        result = engine.run_backtest(
            strategy_name=strategy_name,
            pair="SOL/USDC",
            features=features,
            signal_generator=strategy_fn,
        )

        if len(result.trades) == 0:
            print("  No trades to simulate")
            return {"strategy": strategy_name, "n_simulations": 0, "results": []}

        # Extract P&L sequence
        original_pnls = [t.net_pnl for t in result.trades]
        original_total = sum(original_pnls)

        # Run Monte Carlo: shuffle trade order
        simulated_totals = []
        for i in range(n_simulations):
            shuffled = np.random.permutation(original_pnls)
            # Recalculate equity curve
            equity = [100.0]
            for pnl in shuffled:
                equity.append(equity[-1] + pnl)
            simulated_totals.append(equity[-1])

        simulated_totals = np.array(simulated_totals)

        # Statistics
        mean_return = np.mean(simulated_totals)
        std_return = np.std(simulated_totals)
        percentile_5 = np.percentile(simulated_totals, 5)
        percentile_95 = np.percentile(simulated_totals, 95)

        # Probability of profit
        prob_profit = np.mean(simulated_totals > 100.0)

        print(f"  Original P&L: {original_total:.4f}")
        print(f"  Mean simulated: {mean_return:.4f}")
        print(f"  Std simulated: {std_return:.4f}")
        print(f"  5th percentile: {percentile_5:.4f}")
        print(f"  95th percentile: {percentile_95:.4f}")
        print(f"  Probability of profit: {prob_profit:.2%}")

        return {
            "strategy": strategy_name,
            "n_simulations": n_simulations,
            "original_pnl": float(original_total),
            "mean_simulated": float(mean_return),
            "std_simulated": float(std_return),
            "percentile_5": float(percentile_5),
            "percentile_95": float(percentile_95),
            "prob_profit": float(prob_profit),
        }

    def run_mev_analysis(
        self,
        features: list,
        strategy_name: str,
        strategy_fn,
    ) -> Dict[str, Any]:
        """Run MEV-adjusted backtest."""
        print(f"\n{'='*60}")
        print(f"MEV-Adjusted Backtest: {strategy_name}")
        print(f"{'='*60}")

        # Test with different MEV probabilities
        mev_levels = [0.0, 0.2, 0.4, 0.6]
        results = []

        for mev_prob in mev_levels:
            engine = BacktestEngine(
                initial_capital=100.0,
                fee_rate=0.00022,  # 2.2 bps taker fee (Jupiter measured)
                latency_model=LatencyModel(
                    base_latency_s=10.0,
                    mev_probability=mev_prob,
                    mev_cost_bps=15.0,
                ),
            )

            result = engine.run_backtest(
                strategy_name=f"{strategy_name}_mev_{mev_prob}",
                pair="SOL/USDC",
                features=features,
                signal_generator=strategy_fn,
            )

            results.append({
                "mev_probability": mev_prob,
                "trades": result.total_trades,
                "pnl": result.total_pnl,
                "win_rate": result.win_rate,
            })

            print(f"  MEV {mev_prob:.0%}: pnl={result.total_pnl:.4f} win={result.win_rate:.1%}")

        # Find break-even MEV level
        break_even = None
        for r in results:
            if r["pnl"] <= 0 and break_even is None and r["mev_probability"] > 0:
                break_even = r["mev_probability"]

        return {
            "strategy": strategy_name,
            "results": results,
            "break_even_mev": break_even,
        }

    def run_regime_analysis(
        self,
        features: list,
        strategy_name: str,
        strategy_fn,
    ) -> Dict[str, Any]:
        """Analyze performance across market regimes."""
        print(f"\n{'='*60}")
        print(f"Regime Analysis: {strategy_name}")
        print(f"{'='*60}")

        # Simple regime detection using volatility
        volatilities = [f["features"]["volatility_1h"] for f in features]
        vol_threshold_high = np.percentile(volatilities, 75)
        vol_threshold_low = np.percentile(volatilities, 25)

        regimes = {
            "low_vol": [],
            "mid_vol": [],
            "high_vol": [],
        }

        for i, f in enumerate(features):
            vol = f["features"]["volatility_1h"]
            if vol < vol_threshold_low:
                regimes["low_vol"].append(i)
            elif vol > vol_threshold_high:
                regimes["high_vol"].append(i)
            else:
                regimes["mid_vol"].append(i)

        results = {}
        for regime_name, indices in regimes.items():
            if len(indices) < 50:
                continue

            regime_features = [features[i] for i in indices]

            engine = BacktestEngine(
                initial_capital=100.0,
                fee_rate=0.00022,  # 2.2 bps taker fee (Jupiter measured)
                latency_model=LatencyModel(base_latency_s=10.0, mev_probability=0.3),
            )

            result = engine.run_backtest(
                strategy_name=f"{strategy_name}_{regime_name}",
                pair="SOL/USDC",
                features=regime_features,
                signal_generator=strategy_fn,
            )

            results[regime_name] = {
                "n_candles": len(indices),
                "trades": result.total_trades,
                "pnl": result.total_pnl,
                "win_rate": result.win_rate,
            }

            print(f"  {regime_name}: trades={result.total_trades} pnl={result.total_pnl:.4f} win={result.win_rate:.1%}")

        return {
            "strategy": strategy_name,
            "regimes": results,
        }

    def run_full_validation(
        self,
        features: list,
        strategy_names: List[str] = None,
    ) -> Dict[str, Any]:
        """Run all validation tests."""
        if strategy_names is None:
            strategy_names = list_strategies()

        all_results = {
            "timestamp": self.timestamp.isoformat(),
            "n_features": len(features),
            "strategies": {},
        }

        for name in strategy_names:
            if name not in STRATEGIES:
                continue

            print(f"\n{'#'*60}")
            print(f"Validating: {name}")
            print(f"{'#'*60}")

            strategy_fn = STRATEGIES[name]

            # Walk-forward
            wf = self.run_walk_forward(features, name, strategy_fn)

            # Monte Carlo
            mc = self.run_monte_carlo(features, name, strategy_fn)

            # MEV analysis
            mev = self.run_mev_analysis(features, name, strategy_fn)

            # Regime analysis
            regime = self.run_regime_analysis(features, name, strategy_fn)

            all_results["strategies"][name] = {
                "walk_forward": wf,
                "monte_carlo": mc,
                "mev_analysis": mev,
                "regime_analysis": regime,
            }

        # Save results
        results_file = LOG_DIR / f"validation_{self.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Validation complete. Results saved to {results_file}")
        print(f"{'='*60}")

        return all_results


def main():
    print("Loading features...")
    features = get_historical_features("SOL/USDC", limit=1000)
    print(f"Loaded {len(features)} features")

    suite = ValidationSuite()
    results = suite.run_full_validation(features)

    # Print summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")

    for name, res in results["strategies"].items():
        wf = res["walk_forward"]
        mc = res["monte_carlo"]
        print(f"\n{name}:")
        print(f"  Walk-forward: avg_pnl={wf['avg_pnl']:.4f} sharpe={wf['avg_sharpe']:.2f}")
        print(f"  Monte Carlo: prob_profit={mc.get('prob_profit', 0):.2%}")
        print(f"  MEV break-even: {res['mev_analysis'].get('break_even_mev', 'N/A')}")


if __name__ == "__main__":
    main()
