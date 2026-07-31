"""
evolution_specialist.py
Specialist population evolution: 4 parallel populations, each focused on
one strategy type. Regime detection filters signals by market condition.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import asdict

import numpy as np

from evolution.genome import StrategyGenome, random_genome
from evolution.evaluator import GenomeEvaluator, write_activity
from layer1.historical_feature_engine_1h import get_historical_features_1h


class SpecialistEvolution:
    """Runs 4 specialist populations in parallel."""

    def __init__(
        self,
        features: List[Dict[str, Any]],
        population_size: int = 100,
        elite_size: int = 10,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.7,
    ):
        self.features = features
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        # 4 specialist populations
        self.populations = {
            "MEANREV": [],
            "BREAKOUT": [],
            "TREND": [],
            "THRESHOLD": [],  # AND/OR
        }

        self.evaluator = GenomeEvaluator(features)
        self.generation = 0
        self.history: List[Dict[str, Any]] = []

    def initialize_populations(self):
        """Create initial populations for each specialist."""
        for pop_name in self.populations:
            self.populations[pop_name] = []
            for _ in range(self.population_size):
                g = random_genome(0)
                # Force specialist type
                if pop_name == "THRESHOLD":
                    g.entry_logic = "AND" if len(self.populations[pop_name]) % 2 == 0 else "OR"
                else:
                    g.entry_logic = pop_name
                self.populations[pop_name].append(g)

    def detect_regime(self, features: List[Dict[str, Any]], idx: int) -> str:
        """Detect market regime: high_vol, low_vol, trending, ranging."""
        if idx < 50:
            return "unknown"

        f = features[idx]["features"]

        # Volatility regime (1h features use volatility_4h and volatility_1d)
        vol_4h = f.get("volatility_4h", 0)
        vol_1d = f.get("volatility_1d", 0)
        vol_ratio = vol_4h / vol_1d if vol_1d > 0 else 1.0

        # Trend strength
        sma5 = f.get("sma_5", 0)
        sma20 = f.get("sma_20", 0)
        trend_strength = abs(sma5 - sma20) / sma20 * 100 if sma20 > 0 else 0

        if vol_ratio > 1.5:
            return "high_vol"
        elif vol_ratio < 0.7:
            return "low_vol"
        elif trend_strength > 2.0:
            return "trending"
        else:
            return "ranging"

    def evaluate_population(self, pop_name: str):
        """Evaluate one specialist population."""
        population = self.populations[pop_name]

        for genome in population:
            # Regime filter: only evaluate if regime matches strategy type
            # Use a sample of recent candles to detect regime
            sample_idx = min(100, len(self.features) - 1)
            regime = self.detect_regime(self.features, sample_idx)

            # Soft regime filter: reduce fitness but don't kill
            regime_penalty = 0.0
            if pop_name == "MEANREV" and regime == "trending":
                regime_penalty = 0.5  # 50% penalty
            elif pop_name == "BREAKOUT" and regime == "ranging":
                regime_penalty = 0.5
            elif pop_name == "TREND" and regime == "ranging":
                regime_penalty = 0.5

            # Evaluate normally
            result = self.evaluator.evaluate(genome)
            genome.fitness = result["fitness"] * (1.0 - regime_penalty)
            genome.backtest_results = result

        # Sort by fitness
        population.sort(key=lambda g: g.fitness, reverse=True)

    def breed_population(self, pop_name: str):
        """Breed next generation for one specialist."""
        population = self.populations[pop_name]
        elite = population[:self.elite_size]

        new_population = elite.copy()

        while len(new_population) < self.population_size:
            # Select parents
            parent1 = np.random.choice(elite)
            parent2 = np.random.choice(elite)

            # Crossover
            if np.random.random() < self.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = StrategyGenome.from_dict(parent1.to_dict())

            # Mutation
            if np.random.random() < self.mutation_rate:
                child = self._mutate(child)

            # Force specialist type
            if pop_name == "THRESHOLD":
                child.entry_logic = "AND" if len(new_population) % 2 == 0 else "OR"
            else:
                child.entry_logic = pop_name

            child.generation = self.generation
            new_population.append(child)

        self.populations[pop_name] = new_population

    def _crossover(self, parent1: StrategyGenome, parent2: StrategyGenome) -> StrategyGenome:
        """Single-point crossover."""
        child = StrategyGenome.from_dict(parent1.to_dict())

        # Crossover entry conditions
        if len(parent1.entry_conditions) > 0 and len(parent2.entry_conditions) > 0:
            point = np.random.randint(0, min(len(parent1.entry_conditions), len(parent2.entry_conditions)))
            child.entry_conditions = parent1.entry_conditions[:point] + parent2.entry_conditions[point:]

        # Crossover filters
        if len(parent1.filters) > 0 and len(parent2.filters) > 0:
            point = np.random.randint(0, min(len(parent1.filters), len(parent2.filters)))
            child.filters = parent1.filters[:point] + parent2.filters[point:]

        # Crossover exit rules
        if len(parent1.exit_rules) > 0 and len(parent2.exit_rules) > 0:
            point = np.random.randint(0, min(len(parent1.exit_rules), len(parent2.exit_rules)))
            child.exit_rules = parent1.exit_rules[:point] + parent2.exit_rules[point:]

        child.genome_id = f"cross_{parent1.genome_id[:8]}_{parent2.genome_id[:8]}_{np.random.randint(1000, 9999)}"
        return child

    def _mutate(self, genome: StrategyGenome) -> StrategyGenome:
        """Mutate a genome."""
        new_genome = StrategyGenome.from_dict(genome.to_dict())

        # Mutate entry conditions
        for cond in new_genome.entry_conditions:
            if np.random.random() < 0.3:
                cond.threshold *= np.random.uniform(0.8, 1.2)

        # Mutate sizing
        if np.random.random() < 0.2:
            new_genome.sizing_base *= np.random.uniform(0.8, 1.2)
            new_genome.sizing_base = min(max(new_genome.sizing_base, 0.05), 0.5)

        new_genome.genome_id = f"mut_{genome.genome_id[:8]}_{np.random.randint(1000, 9999)}"
        return new_genome

    def evolve(self, max_generations: int = 50, verbose: bool = True) -> StrategyGenome:
        """Run evolution across all specialist populations."""
        self.initialize_populations()

        best_overall = None
        best_fitness = -999

        for gen in range(max_generations):
            self.generation = gen

            # Evaluate all populations
            for pop_name in self.populations:
                self.evaluate_population(pop_name)

            # Find best across all populations
            for pop_name, population in self.populations.items():
                if population and population[0].fitness > best_fitness:
                    best_fitness = population[0].fitness
                    best_overall = population[0]

            # Log progress
            if verbose and gen % 10 == 0:
                pop_stats = {name: pop[0].fitness if pop else -999 for name, pop in self.populations.items()}
                print(f"Gen {gen:3d} | best={best_fitness:.1f} | " + " | ".join(f"{k}={v:.1f}" for k, v in pop_stats.items()))

            # Breed next generation
            for pop_name in self.populations:
                self.breed_population(pop_name)

            # Record history
            self.history.append({
                "generation": gen,
                "best_fitness": best_fitness,
                "pop_best": {name: pop[0].fitness if pop else -999 for name, pop in self.populations.items()},
            })

        return best_overall


def run_specialist_evolution(
    features: List[Dict[str, Any]],
    population_size: int = 100,
    max_generations: int = 50,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run specialist evolution and return results."""
    engine = SpecialistEvolution(
        features,
        population_size=population_size,
        elite_size=max(4, population_size // 10),
    )

    best = engine.evolve(max_generations=max_generations, verbose=verbose)

    return {
        "best_genome": best.to_dict() if best else None,
        "best_fitness": best.fitness if best else -999,
        "history": engine.history,
        "populations": {name: len(pop) for name, pop in engine.populations.items()},
    }


if __name__ == "__main__":
    features = get_historical_features_1h("SOL/USDC", limit=2000)
    print(f"Loaded {len(features)} 1h features")

    result = run_specialist_evolution(features, population_size=100, max_generations=50)
    print(f"\nBest fitness: {result['best_fitness']:.1f}")
    print(f"Best genome: {result['best_genome']['genome_id'] if result['best_genome'] else 'None'}")
