"""
walk_forward.py
Out-of-sample validation via rolling train/test windows.
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np

from layer1.backtest_engine import BacktestEngine, LatencyModel
from layer1.historical_feature_engine import get_historical_features


@dataclass
class WalkForwardWindow:
    train_start: int
    train_end: int
    test_start: int
    test_end: int


@dataclass
class WalkForwardResult:
    window_idx: int
    train_period: str
    test_period: str
    train_trades: int
    train_pnl: float
    test_trades: int
    test_pnl: float
    test_win_rate: float
    test_sharpe: float
    test_max_dd: float
    passed: bool


def create_windows(
    features: List[Dict[str, Any]],
    n_windows: int = 3,
    train_ratio: float = 0.67,
) -> List[WalkForwardWindow]:
    """Create rolling walk-forward windows."""
    n = len(features)
    window_size = n // n_windows
    windows = []

    for i in range(n_windows):
        test_start = i * window_size
        test_end = (i + 1) * window_size if i < n_windows - 1 else n

        # Train on all data before test window
        train_end = test_start
        train_start = max(0, train_end - int(window_size * train_ratio / (1 - train_ratio)))

        windows.append(WalkForwardWindow(
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        ))

    return windows


def run_walk_forward(
    genome: Dict[str, Any],
    features: List[Dict[str, Any]],
    n_windows: int = 3,
    initial_capital: float = 100.0,
    fee_rate: float = 0.00022,  # 2.2 bps taker fee (Jupiter measured)
) -> Dict[str, Any]:
    """
    Run walk-forward validation on a strategy genome.

    Returns dict with per-window results and overall pass/fail.
    """
    from evolution.evaluator import GenomeEvaluator

    windows = create_windows(features, n_windows)
    results = []

    for i, w in enumerate(windows):
        train_features = features[w.train_start:w.train_end]
        test_features = features[w.test_start:w.test_end]

        if len(train_features) < 100 or len(test_features) < 50:
            results.append(WalkForwardResult(
                window_idx=i,
                train_period=f"{w.train_start}-{w.train_end}",
                test_period=f"{w.test_start}-{w.test_end}",
                train_trades=0, train_pnl=0.0,
                test_trades=0, test_pnl=0.0,
                test_win_rate=0.0, test_sharpe=0.0, test_max_dd=0.0,
                passed=False,
            ))
            continue

        # Evaluate on train (for reference)
        train_eval = GenomeEvaluator(train_features, initial_capital, fee_rate)
        train_result = train_eval.evaluate(genome)

        # Evaluate on test (the real check)
        test_eval = GenomeEvaluator(test_features, initial_capital, fee_rate)
        test_result = test_eval.evaluate(genome)

        # Pass if test is profitable
        passed = test_result["total_pnl"] > 0 and test_result["total_trades"] >= 3

        results.append(WalkForwardResult(
            window_idx=i,
            train_period=f"{w.train_start}-{w.train_end}",
            test_period=f"{w.test_start}-{w.test_end}",
            train_trades=train_result["total_trades"],
            train_pnl=train_result["total_pnl"],
            test_trades=test_result["total_trades"],
            test_pnl=test_result["total_pnl"],
            test_win_rate=test_result["win_rate"],
            test_sharpe=test_result["sharpe_ratio"],
            test_max_dd=test_result["max_drawdown"],
            passed=passed,
        ))

    # Overall pass: all windows profitable
    all_passed = all(r.passed for r in results)
    avg_test_pnl = np.mean([r.test_pnl for r in results])
    avg_test_sharpe = np.mean([r.test_sharpe for r in results])

    return {
        "layer": "walk_forward",
        "passed": all_passed,
        "n_windows": len(results),
        "windows_passed": sum(1 for r in results if r.passed),
        "avg_test_pnl": avg_test_pnl,
        "avg_test_sharpe": avg_test_sharpe,
        "results": [
            {
                "window": r.window_idx,
                "train_period": r.train_period,
                "test_period": r.test_period,
                "train_trades": r.train_trades,
                "train_pnl": r.train_pnl,
                "test_trades": r.test_trades,
                "test_pnl": r.test_pnl,
                "test_win_rate": r.test_win_rate,
                "test_sharpe": r.test_sharpe,
                "test_max_dd": r.test_max_dd,
                "passed": r.passed,
            }
            for r in results
        ],
    }
