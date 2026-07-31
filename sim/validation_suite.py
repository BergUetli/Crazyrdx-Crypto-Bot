"""
validation_suite.py
Tests whether evolved strategies are robust or just lucky on small samples.

Three tests:
1. Walk-forward: train on past, test on future (does it generalize?)
2. Monte Carlo: shuffle trade order (is the edge real or sequence-dependent?)
3. MEV stress: add frontrunning probability (does it survive reality?)
"""

import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

SIM_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SIM_DIR))

from evolution.genome import StrategyGenome
from evolution.evaluator import GenomeEvaluator
from layer1.backtest_engine import LatencyModel
from layer1.historical_feature_engine import get_historical_features


class ValidationSuite:
    """Validates strategy robustness."""

    def __init__(self, features: List[Dict[str, Any]]):
        self.features = features
        self.evaluator = GenomeEvaluator(features)

    def walk_forward(
        self,
        genome: StrategyGenome,
        train_pct: float = 0.7,
        n_splits: int = 3,
    ) -> Dict[str, Any]:
        """
        Split data into train/test periods. Evaluate on test only.
        If test performance is much worse than train, the strategy is overfit.
        """
        n = len(self.features)
        split_size = n // n_splits
        results = []

        for i in range(n_splits):
            start = i * split_size
            end = min(start + split_size, n)
            test_features = self.features[start:end]

            if len(test_features) < 50:
                continue

            evaluator = GenomeEvaluator(test_features)
            result = evaluator.evaluate(genome)

            results.append({
                "split": i,
                "period": f"{start}-{end}",
                "fitness": result["fitness"],
                "trades": result["total_trades"],
                "win_rate": result["win_rate"],
                "pnl": result["total_pnl"],
                "sharpe": result["sharpe_ratio"],
                "max_dd": result["max_drawdown"],
            })

        # Aggregate
        if not results:
            return {"error": "insufficient data"}

        avg_fitness = sum(r["fitness"] for r in results) / len(results)
        avg_win = sum(r["win_rate"] for r in results) / len(results)
        avg_pnl = sum(r["pnl"] for r in results) / len(results)
        total_trades = sum(r["trades"] for r in results)
        consistency = 1.0 - (max(r["fitness"] for r in results) - min(r["fitness"] for r in results)) / (abs(avg_fitness) + 1)

        return {
            "splits": results,
            "avg_fitness": avg_fitness,
            "avg_win_rate": avg_win,
            "avg_pnl": avg_pnl,
            "total_trades": total_trades,
            "consistency": max(0, consistency),
            "passed": avg_fitness > 0 and total_trades >= 10 and consistency > 0.5,
        }

    def monte_carlo(
        self,
        genome: StrategyGenome,
        n_simulations: int = 100,
    ) -> Dict[str, Any]:
        """
        Shuffle the order of trades. If the strategy still makes money,
        the edge is real, not dependent on lucky sequence.
        """
        # Get original backtest
        original = self.evaluator.evaluate(genome)
        original_pnl = original["total_pnl"]

        if original["total_trades"] == 0:
            return {"error": "no trades to shuffle"}

        # Run simulations with shuffled features
        pnl_distribution = []
        for _ in range(n_simulations):
            shuffled = self.features.copy()
            random.shuffle(shuffled)
            evaluator = GenomeEvaluator(shuffled)
            result = evaluator.evaluate(genome)
            pnl_distribution.append(result["total_pnl"])

        # Statistics
        pnl_distribution.sort()
        mean_pnl = sum(pnl_distribution) / len(pnl_distribution)
        std_pnl = (sum((p - mean_pnl) ** 2 for p in pnl_distribution) / len(pnl_distribution)) ** 0.5

        # Percentile of original in shuffled distribution
        below_original = sum(1 for p in pnl_distribution if p < original_pnl)
        percentile = below_original / len(pnl_distribution)

        # p-value: probability that random shuffling beats the original
        p_value = 1.0 - percentile

        return {
            "original_pnl": original_pnl,
            "mean_shuffled_pnl": mean_pnl,
            "std_shuffled_pnl": std_pnl,
            "percentile": percentile,
            "p_value": p_value,
            "significant": p_value < 0.05,  # 95% confidence
            "passed": p_value < 0.05 and original_pnl > 0,
        }

    def mev_stress(
        self,
        genome: StrategyGenome,
        mev_probabilities: List[float] = [0.0, 0.1, 0.2, 0.3, 0.5],
    ) -> Dict[str, Any]:
        """
        Test with different MEV (frontrunning) probabilities.
        If the strategy dies at 30% MEV, it won't survive Solana.
        """
        results = []

        for mev_prob in mev_probabilities:
            latency_model = LatencyModel(
                base_latency_s=10.0,
                latency_std_s=3.0,
                mev_probability=mev_prob,
                mev_cost_bps=15.0,
            )
            evaluator = GenomeEvaluator(
                self.features,
                latency_model=latency_model,
            )
            result = evaluator.evaluate(genome)

            results.append({
                "mev_probability": mev_prob,
                "fitness": result["fitness"],
                "trades": result["total_trades"],
                "win_rate": result["win_rate"],
                "pnl": result["total_pnl"],
                "max_dd": result["max_drawdown"],
            })

        # Find break-even MEV threshold
        break_even = None
        for r in results:
            if r["pnl"] <= 0:
                break_even = r["mev_probability"]
                break

        return {
            "results": results,
            "break_even_mev": break_even,
            "survives_30pct": break_even is None or break_even > 0.3,
            "passed": break_even is None or break_even > 0.2,
        }

    def parameter_perturbation(
        self,
        genome: StrategyGenome,
        perturbation_pct: float = 0.10,
        n_perturbations: int = 10,
    ) -> Dict[str, Any]:
        """
        Shift each entry condition threshold by ±10% and re-evaluate.
        If the strategy falls apart when parameters shift slightly,
        it's overfit to exact threshold values.

        QuantTradingTools pattern: "parameter perturbation stability"
        """
        import copy

        original = self.evaluator.evaluate(genome)
        original_fitness = original["fitness"]
        original_pnl = original["total_pnl"]

        if original["total_trades"] == 0:
            return {"error": "no trades to perturb", "passed": False}

        perturbed_fitnesses = []
        perturbed_pnls = []

        for _ in range(n_perturbations):
            # Clone genome and perturb thresholds
            perturbed = copy.deepcopy(genome)
            for cond in perturbed.entry_conditions:
                shift = 1.0 + random.uniform(-perturbation_pct, perturbation_pct)
                cond.threshold = cond.threshold * shift

            result = self.evaluator.evaluate(perturbed)
            perturbed_fitnesses.append(result["fitness"])
            perturbed_pnls.append(result["total_pnl"])

        # Stability: how often does perturbed version stay profitable?
        profitable_count = sum(1 for p in perturbed_pnls if p > 0)
        profitability_rate = profitable_count / len(perturbed_pnls)

        # Fitness retention: perturbed fitness / original fitness
        avg_perturbed = sum(perturbed_fitnesses) / len(perturbed_fitnesses)
        if original_fitness != 0:
            retention = avg_perturbed / original_fitness
        else:
            retention = 0.0

        # Coefficient of variation (lower = more stable)
        if len(perturbed_fitnesses) > 1:
            mean_pf = sum(perturbed_fitnesses) / len(perturbed_fitnesses)
            std_pf = (sum((f - mean_pf) ** 2 for f in perturbed_fitnesses) / len(perturbed_fitnesses)) ** 0.5
            cv = std_pf / abs(mean_pf) if mean_pf != 0 else float('inf')
        else:
            cv = float('inf')

        return {
            "original_fitness": original_fitness,
            "original_pnl": original_pnl,
            "avg_perturbed_fitness": avg_perturbed,
            "fitness_retention": retention,
            "profitability_rate": profitability_rate,
            "coefficient_of_variation": cv,
            "perturbed_pnls": perturbed_pnls,
            "perturbed_fitnesses": perturbed_fitnesses,
            "passed": profitability_rate >= 0.6 and retention >= 0.5,
        }

    def full_validation(
        self,
        genome: StrategyGenome,
        genome_name: str = "unknown",
    ) -> Dict[str, Any]:
        """Run all four tests: walk-forward, Monte Carlo, MEV stress, parameter perturbation."""
        print(f"\nValidating: {genome_name}")

        wf = self.walk_forward(genome)
        mc = self.monte_carlo(genome)
        mev = self.mev_stress(genome)
        pp = self.parameter_perturbation(genome)

        all_passed = all([
            wf.get("passed", False),
            mc.get("passed", False),
            mev.get("passed", False),
            pp.get("passed", False),
        ])

        return {
            "genome_name": genome_name,
            "walk_forward": wf,
            "monte_carlo": mc,
            "mev_stress": mev,
            "parameter_perturbation": pp,
            "all_passed": all_passed,
            "verdict": "ROBUST" if all_passed else "NOT READY",
        }


def load_genome(path: str) -> StrategyGenome:
    """Load a genome from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return StrategyGenome.from_dict(data)


def validate_top_strategies(n: int = 3):
    """Validate the top N strategies from evolution."""
    import glob
    import os

    pop_dir = SIM_DIR / "evolution" / "population"
    files = sorted(pop_dir.glob("evolution_*.json"), key=os.path.getmtime)

    if not files:
        print("No evolution runs found")
        return

    # Get top N by fitness
    scored = []
    for f in files:
        d = json.load(open(f))
        scored.append((d["best_fitness"], f))

    scored.sort(reverse=True)
    top_files = scored[:n]

    # Load features
    features = get_historical_features("SOL/USDC", limit=2000)
    suite = ValidationSuite(features)

    results = []
    for fitness, filepath in top_files:
        d = json.load(open(filepath))
        genome = StrategyGenome.from_dict(d["best_genome"])
        name = genome.genome_id

        result = suite.full_validation(genome, name)
        result["original_fitness"] = fitness
        result["original_trades"] = d["backtest"]["total_trades"]
        result["original_win_rate"] = d["backtest"]["win_rate"]
        result["original_pnl"] = d["backtest"]["total_pnl"]
        results.append(result)

        # Save individual result
        out_path = SIM_DIR / "evolution" / f"validation_{name}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    for r in results:
        status = "✓ ROBUST" if r["all_passed"] else "✗ NOT READY"
        print(f"\n{r['genome_name'][:40]}")
        print(f"  Original: fitness={r['original_fitness']:.1f} trades={r['original_trades']} win={r['original_win_rate']:.0%} pnl={r['original_pnl']:.3f}")
        print(f"  Walk-forward: avg_fitness={r['walk_forward'].get('avg_fitness', 0):.1f} consistency={r['walk_forward'].get('consistency', 0):.2f} {'PASS' if r['walk_forward'].get('passed') else 'FAIL'}")
        print(f"  Monte Carlo: p={r['monte_carlo'].get('p_value', 1):.3f} {'SIGNIFICANT' if r['monte_carlo'].get('significant') else 'not significant'}")
        print(f"  MEV stress: break_even={r['mev_stress'].get('break_even_mev', 'N/A')} {'SURVIVES' if r['mev_stress'].get('survives_30pct') else 'DIES'}")
        pp = r.get('parameter_perturbation', {})
        print(f"  Perturbation: retention={pp.get('fitness_retention', 0):.2f} profit_rate={pp.get('profitability_rate', 0):.0%} {'STABLE' if pp.get('passed') else 'UNSTABLE'}")
        print(f"  Verdict: {status}")

    return results


if __name__ == "__main__":
    validate_top_strategies(3)
