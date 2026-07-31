"""
genome.py
Strategy genome representation and mutation/crossover operations.
"""

import json
import random
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple


# Indicators available on the 1h feature engine (must exist in features_1h)
INDICATORS = [
    # Momentum
    "price_roc_1h", "price_roc_4h", "price_roc_1d", "price_roc_3d", "price_roc_1w",
    # Volatility
    "volatility_4h", "volatility_1d", "volatility_3d", "volatility_1w",
    # Range / candle shape
    "hl_range_pct", "hl_range_avg_4h", "hl_range_avg_1d",
    "body_pct", "upper_wick_pct", "lower_wick_pct",
    # Volume / flow
    "volume_roc_4h", "volume_roc_1d", "volume_sma_ratio", "volume_weighted_price",
    "taker_buy_ratio", "taker_buy_roc_4h", "taker_buy_roc_1d", "taker_buy_sma_ratio",
    # Trend
    "sma_5", "sma_20", "sma_50", "price_vs_sma_20", "sma_cross_5_20", "sma_cross_20_50",
    "price_vs_4h_sma", "price_vs_1d_sma",
    # Stats
    "returns_skew_1d", "returns_kurtosis_1d", "autocorrelation_1d",
    # Time
    "hour_of_day_sin", "hour_of_day_cos", "day_of_week", "is_weekend",
    # Lags
    "close_lag_1", "close_lag_2", "close_lag_3", "volume_lag_1",
    "returns_lag_1", "returns_lag_2",
    # Multi-timeframe
    "trend_alignment_1h_4h", "trend_alignment_4h_1d", "trend_alignment_1h_1d", "trend_alignment_all",
    "momentum_divergence_1h_4h", "momentum_divergence_4h_1d", "momentum_divergence_1h_1d",
    "volatility_regime_1h_4h", "volatility_regime_4h_1d", "volatility_regime_1h_1d",
    "volume_confirmation_1h_4h",
    # Model heads (may be zero until TFT inference is wired)
    "tft_prediction", "tft_confidence",
    # External market features (funding, basis, order flow)
    "funding_rate", "funding_rate_8h_avg", "funding_rate_roc", "funding_rate_extreme",
    "cex_dex_basis_bps", "cex_dex_basis_roc_4h", "cex_dex_basis_roc_1d", "cex_dex_basis_extreme",
    "taker_flow_imbalance", "taker_flow_imbalance_4h", "taker_flow_imbalance_roc", "taker_flow_persistence",
    "dex_liquidity_ratio", "funding_basis_divergence", "market_stress_index",
]

# Threshold ranges per indicator type
THRESHOLD_RANGES = {
    "price_roc": (5.0, 200.0),
    "volatility": (10.0, 500.0),
    "hl_range": (5.0, 200.0),
    "body": (1.0, 100.0),
    "wick": (1.0, 50.0),
    "volume": (0.5, 5.0),
    "taker_buy": (0.0, 5.0),
    "sma": (0.01, 10.0),
    "price_vs_sma": (-500.0, 500.0),
    "price_vs_4h": (-500.0, 500.0),
    "price_vs_1d": (-500.0, 500.0),
    "returns": (-5.0, 5.0),
    "time": (0.0, 1.0),
    "lag": (-100.0, 100.0),
    "trend_alignment": (-1.0, 1.0),
    "momentum_divergence": (-100.0, 100.0),
    "volatility_regime": (0.0, 5.0),
    "volume_confirmation": (0.0, 500.0),
    "tft_prediction": (-1.0, 1.0),
    "tft_confidence": (0.0, 1.0),
    "funding_rate": (-0.01, 0.01),
    "funding_rate_8h_avg": (-0.01, 0.01),
    "funding_rate_roc": (-0.005, 0.005),
    "funding_rate_extreme": (0.0, 1.0),
    "cex_dex_basis": (-200.0, 200.0),
    "cex_dex_basis_roc": (-100.0, 100.0),
    "cex_dex_basis_extreme": (0.0, 1.0),
    "taker_flow_imbalance": (-1.0, 1.0),
    "taker_flow_imbalance_4h": (-1.0, 1.0),
    "taker_flow_imbalance_roc": (-2.0, 2.0),
    "taker_flow_persistence": (0.0, 1.0),
    "dex_liquidity_ratio": (0.1, 10.0),
    "funding_basis_divergence": (0.0, 1.0),
    "market_stress_index": (0.0, 1.0),
}

# Categorical options
# RANDOM kept only as optional offline baseline control, never used in selection.
LOGIC_OPS = ["AND", "OR", "MEANREV", "BREAKOUT", "TREND", "TFT"]
BASELINE_LOGIC_OPS = ["RANDOM"]
SIZING_METHODS = ["fixed", "kelly", "volatility_scaled", "equal_weight"]
EXIT_TYPES = ["profit_target", "stop_loss", "time_stop", "trailing_stop", "signal_reversal"]


def get_threshold_range(indicator: str) -> Tuple[float, float]:
    """Get valid threshold range for an indicator."""
    for key, range_tuple in THRESHOLD_RANGES.items():
        if key in indicator:
            return range_tuple
    return (0.0, 100.0)


@dataclass
class EntryCondition:
    indicator: str
    operator: str  # ">", "<", ">=", "<=", "==", "crosses_above", "crosses_below"
    threshold: float


@dataclass
class Filter:
    filter_type: str  # "time_of_day", "day_of_week", "volatility_regime", "trend"
    params: Dict[str, Any]


@dataclass
class ExitRule:
    exit_type: str
    value: float


@dataclass
class StrategyGenome:
    """Complete strategy DNA."""
    # Entry
    entry_conditions: List[EntryCondition] = field(default_factory=list)
    entry_logic: str = "AND"  # AND or OR
    
    # Filters
    filters: List[Filter] = field(default_factory=list)
    
    # Sizing
    sizing_method: str = "fixed"
    sizing_base: float = 0.25
    sizing_max: float = 0.50
    
    # Exit
    exit_rules: List[ExitRule] = field(default_factory=list)
    
    # Metadata
    genome_id: str = ""
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    fitness: float = 0.0
    backtest_results: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyGenome":
        # Reconstruct nested dataclasses
        entry_conds = [EntryCondition(**ec) for ec in data.get("entry_conditions", [])]
        filters = [Filter(**f) for f in data.get("filters", [])]
        exit_rules = [ExitRule(**er) for er in data.get("exit_rules", [])]
        
        return cls(
            entry_conditions=entry_conds,
            entry_logic=data.get("entry_logic", "AND"),
            filters=filters,
            sizing_method=data.get("sizing_method", "fixed"),
            sizing_base=data.get("sizing_base", 0.25),
            sizing_max=data.get("sizing_max", 0.50),
            exit_rules=exit_rules,
            genome_id=data.get("genome_id", ""),
            generation=data.get("generation", 0),
            parent_ids=data.get("parent_ids", []),
            fitness=data.get("fitness", 0.0),
            backtest_results=data.get("backtest_results"),
        )


def random_genome(generation: int = 0) -> StrategyGenome:
    """Generate a random strategy genome.

    Biased toward more trades: 60% chance of OR logic, 60% chance of 1-2 conditions.
    This gives the evolution more trades per strategy to work with.
    """
    # Random entry conditions (1-2, biased toward fewer for more trades)
    n_conditions = random.choices([1, 2, 3], weights=[3, 4, 2])[0]
    conditions = []
    for _ in range(n_conditions):
        indicator = random.choice(INDICATORS)
        min_val, max_val = get_threshold_range(indicator)
        threshold = random.uniform(min_val, max_val)
        operator = random.choice([">", "<", ">=", "<="])
        conditions.append(EntryCondition(indicator, operator, threshold))

    # Selection pool only (no RANDOM lottery tickets)
    entry_logic = random.choices(
        ["AND", "OR", "MEANREV", "BREAKOUT", "TREND", "TFT"],
        weights=[2, 2, 2, 2, 2, 1],
    )[0]

    # Random filters (0-2)
    n_filters = random.randint(0, 2)
    filters = []
    filter_types = ["time_of_day", "day_of_week", "volatility_regime", "trend"]
    for _ in range(n_filters):
        ftype = random.choice(filter_types)
        if ftype == "time_of_day":
            params = {"hours": random.sample(range(24), random.randint(1, 6))}
        elif ftype == "day_of_week":
            params = {"days": random.sample(range(7), random.randint(1, 5))}
        elif ftype == "volatility_regime":
            params = {"min_vol": random.uniform(10, 100), "max_vol": random.uniform(100, 500)}
        else:  # trend
            params = {"sma_period": random.choice([5, 20, 50]), "direction": random.choice(["up", "down"])}
        filters.append(Filter(ftype, params))

    # Random sizing
    sizing_method = random.choice(SIZING_METHODS)
    sizing_base = random.uniform(0.1, 0.5)
    sizing_max = min(sizing_base + random.uniform(0.1, 0.3), 1.0)

    # Random exit rules (1-3)
    n_exits = random.randint(1, 3)
    exit_rules = []
    for _ in range(n_exits):
        etype = random.choice(EXIT_TYPES)
        if etype == "profit_target":
            value = random.uniform(0.001, 0.05)
        elif etype == "stop_loss":
            value = random.uniform(0.001, 0.02)
        elif etype == "time_stop":
            value = random.randint(1, 48)
        elif etype == "trailing_stop":
            value = random.uniform(0.001, 0.03)
        else:  # signal_reversal
            value = random.uniform(0.5, 1.0)
        exit_rules.append(ExitRule(etype, value))

    genome_id = f"gen_{generation}_{random.randint(100000, 999999)}"

    return StrategyGenome(
        entry_conditions=conditions,
        entry_logic=entry_logic,
        filters=filters,
        sizing_method=sizing_method,
        sizing_base=sizing_base,
        sizing_max=sizing_max,
        exit_rules=exit_rules,
        genome_id=genome_id,
        generation=generation,
    )


def mutate(genome: StrategyGenome, mutation_rate: float = 0.1) -> StrategyGenome:
    """Mutate a genome."""
    new_genome = StrategyGenome.from_dict(genome.to_dict())
    new_genome.genome_id = f"mut_{genome.genome_id}_{random.randint(1000, 9999)}"
    new_genome.parent_ids = [genome.genome_id]
    new_genome.generation = genome.generation + 1
    new_genome.fitness = 0.0
    new_genome.backtest_results = None
    
    # Mutate entry conditions
    for i, cond in enumerate(new_genome.entry_conditions):
        if random.random() < mutation_rate:
            # Mutate threshold
            min_val, max_val = get_threshold_range(cond.indicator)
            cond.threshold = random.uniform(min_val, max_val)
        if random.random() < mutation_rate:
            # Mutate operator
            cond.operator = random.choice([">", "<", ">=", "<="])
    
    # Mutate logic within selection pool only
    if random.random() < mutation_rate:
        if new_genome.entry_logic in ("AND", "OR"):
            new_genome.entry_logic = "OR" if new_genome.entry_logic == "AND" else "AND"
        else:
            new_genome.entry_logic = random.choice(LOGIC_OPS)
    
    # Mutate sizing
    if random.random() < mutation_rate:
        new_genome.sizing_base = random.uniform(0.1, 0.5)
    if random.random() < mutation_rate:
        new_genome.sizing_method = random.choice(SIZING_METHODS)
    
    # Mutate exit rules
    for i, rule in enumerate(new_genome.exit_rules):
        if random.random() < mutation_rate:
            if rule.exit_type == "profit_target":
                rule.value = random.uniform(0.001, 0.05)
            elif rule.exit_type == "stop_loss":
                rule.value = random.uniform(0.001, 0.02)
            elif rule.exit_type == "time_stop":
                rule.value = random.randint(1, 48)
    
    # Add/remove conditions
    if random.random() < mutation_rate * 0.5 and len(new_genome.entry_conditions) > 1:
        new_genome.entry_conditions.pop(random.randint(0, len(new_genome.entry_conditions) - 1))
    if random.random() < mutation_rate * 0.5 and len(new_genome.entry_conditions) < 5:
        indicator = random.choice(INDICATORS)
        min_val, max_val = get_threshold_range(indicator)
        new_genome.entry_conditions.append(
            EntryCondition(indicator, random.choice([">", "<"]), random.uniform(min_val, max_val))
        )
    
    return new_genome


def crossover(parent1: StrategyGenome, parent2: StrategyGenome) -> StrategyGenome:
    """Combine two parent genomes."""
    child = StrategyGenome(
        entry_logic=random.choice([parent1.entry_logic, parent2.entry_logic]),
        sizing_method=random.choice([parent1.sizing_method, parent2.sizing_method]),
        sizing_base=(parent1.sizing_base + parent2.sizing_base) / 2,
        sizing_max=max(parent1.sizing_max, parent2.sizing_max),
        genome_id=f"cross_{parent1.genome_id[:8]}_{parent2.genome_id[:8]}_{random.randint(1000, 9999)}",
        generation=max(parent1.generation, parent2.generation) + 1,
        parent_ids=[parent1.genome_id, parent2.genome_id],
    )
    
    # Mix entry conditions
    all_conditions = parent1.entry_conditions + parent2.entry_conditions
    n_child = random.randint(1, min(4, len(all_conditions)))
    child.entry_conditions = random.sample(all_conditions, min(n_child, len(all_conditions)))
    
    # Mix filters
    all_filters = parent1.filters + parent2.filters
    n_filters = random.randint(0, min(3, len(all_filters)))
    child.filters = random.sample(all_filters, min(n_filters, len(all_filters)))
    
    # Mix exit rules
    all_exits = parent1.exit_rules + parent2.exit_rules
    n_exits = random.randint(1, min(3, len(all_exits)))
    child.exit_rules = random.sample(all_exits, min(n_exits, len(all_exits)))
    
    return child
