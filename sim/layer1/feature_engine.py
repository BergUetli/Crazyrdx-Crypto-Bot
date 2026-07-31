"""
feature_engine.py
Computes 40 features from raw quote data for each poll cycle.
"""

import json
import math
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from config import PAIRS, ROLLING_WINDOWS, LAG_PERIODS
from layer1.db import get_quotes_since, get_latest_price, get_spread_history


@dataclass
class FeatureVector:
    ts: float
    pair: str

    # Price features
    price: float
    price_impact_pct: float

    # Spread features
    spread_bps: float
    spread_acceleration: float
    spread_jerk: float
    rolling_spread_mean_30s: float
    rolling_spread_std_30s: float
    rolling_spread_max_30s: float
    rolling_spread_skew_30s: float
    rolling_spread_kurtosis_30s: float

    # Price momentum features
    price_roc_5s: float
    price_roc_30s: float
    price_roc_60s: float
    price_volatility_30s: float
    price_volatility_300s: float

    # Pool depth features (placeholder, filled from pool_snapshots)
    pool_liquidity_usd: float
    pool_depth_ratio: float
    pool_volume_24h: float
    pool_fee_tier: float

    # Cross-market features
    cex_dex_spread_bps: float
    cex_price_lead_ms: float
    cross_pair_correlation: float

    # Network features (placeholder)
    solana_slot_time_ms: float
    network_congestion_score: float
    compute_unit_price: float

    # Temporal features
    time_of_day_sin: float
    time_of_day_cos: float
    day_of_week: int
    seconds_since_last_spike: float
    is_weekend: int

    # Lag features
    spread_bps_lag_1: float
    spread_bps_lag_2: float
    spread_bps_lag_3: float
    price_lag_1: float
    price_lag_2: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class FeatureEngine:
    """Computes feature vectors from raw quote data."""

    def __init__(self, pair: str):
        self.pair = pair
        self.price_history: List[float] = []
        self.spread_history: List[float] = []
        self.ts_history: List[float] = []
        self.last_spike_ts: float = 0.0

    def update(self, ts: float, price: float, price_impact_pct: float):
        """Add a new observation and compute features."""
        self.price_history.append(price)
        self.ts_history.append(ts)

        # Compute spread from previous price
        if len(self.price_history) >= 2:
            prev_price = self.price_history[-2]
            spread_bps = abs(price - prev_price) / prev_price * 10000
        else:
            spread_bps = 0.0

        self.spread_history.append(spread_bps)

        # Track spikes (>50 bps)
        if spread_bps > 50:
            self.last_spike_ts = ts

        # Keep history bounded (last 300 seconds = 60 samples at 5s)
        max_samples = 60
        if len(self.price_history) > max_samples:
            self.price_history = self.price_history[-max_samples:]
            self.spread_history = self.spread_history[-max_samples:]
            self.ts_history = self.ts_history[-max_samples:]

        return self._compute_features(ts, price, price_impact_pct, spread_bps)

    def _compute_features(self, ts: float, price: float,
                          price_impact_pct: float, spread_bps: float) -> FeatureVector:
        """Compute all 40 features."""

        # Spread derivatives
        spread_acc = self._compute_spread_acceleration()
        spread_jerk = self._compute_spread_jerk()

        # Rolling statistics
        rolling_stats = self._compute_rolling_stats()

        # Price momentum
        price_roc = self._compute_price_roc()

        # Volatility
        vol_30s = self._compute_volatility(6)
        vol_300s = self._compute_volatility(60)

        # Pool features (placeholder - would query pool_snapshots DB)
        pool_liquidity = 0.0
        pool_depth_ratio = 0.0
        pool_volume = 0.0
        pool_fee = 0.0

        # Cross-market features (placeholder - would query cex_feeds DB)
        cex_dex_spread = 0.0
        cex_lead = 0.0
        cross_corr = 0.0

        # Network features (placeholder)
        slot_time = 400.0
        congestion = 0.0
        cu_price = 0.0

        # Temporal features
        dt = time.localtime(ts)
        hour = dt.tm_hour + dt.tm_min / 60.0
        time_sin = math.sin(2 * math.pi * hour / 24)
        time_cos = math.cos(2 * math.pi * hour / 24)
        day_of_week = dt.tm_wday
        is_weekend = 1 if day_of_week >= 5 else 0
        seconds_since_spike = ts - self.last_spike_ts if self.last_spike_ts > 0 else 9999.0

        # Lag features
        lag_1 = self.spread_history[-2] if len(self.spread_history) >= 2 else 0.0
        lag_2 = self.spread_history[-3] if len(self.spread_history) >= 3 else 0.0
        lag_3 = self.spread_history[-4] if len(self.spread_history) >= 4 else 0.0
        price_lag_1 = self.price_history[-2] if len(self.price_history) >= 2 else price
        price_lag_2 = self.price_history[-3] if len(self.price_history) >= 3 else price

        return FeatureVector(
            ts=ts,
            pair=self.pair,
            price=price,
            price_impact_pct=price_impact_pct,
            spread_bps=spread_bps,
            spread_acceleration=spread_acc,
            spread_jerk=spread_jerk,
            rolling_spread_mean_30s=rolling_stats.get("mean_30s", 0.0),
            rolling_spread_std_30s=rolling_stats.get("std_30s", 0.0),
            rolling_spread_max_30s=rolling_stats.get("max_30s", 0.0),
            rolling_spread_skew_30s=rolling_stats.get("skew_30s", 0.0),
            rolling_spread_kurtosis_30s=rolling_stats.get("kurtosis_30s", 0.0),
            price_roc_5s=price_roc.get("roc_5s", 0.0),
            price_roc_30s=price_roc.get("roc_30s", 0.0),
            price_roc_60s=price_roc.get("roc_60s", 0.0),
            price_volatility_30s=vol_30s,
            price_volatility_300s=vol_300s,
            pool_liquidity_usd=pool_liquidity,
            pool_depth_ratio=pool_depth_ratio,
            pool_volume_24h=pool_volume,
            pool_fee_tier=pool_fee,
            cex_dex_spread_bps=cex_dex_spread,
            cex_price_lead_ms=cex_lead,
            cross_pair_correlation=cross_corr,
            solana_slot_time_ms=slot_time,
            network_congestion_score=congestion,
            compute_unit_price=cu_price,
            time_of_day_sin=time_sin,
            time_of_day_cos=time_cos,
            day_of_week=day_of_week,
            seconds_since_last_spike=seconds_since_spike,
            is_weekend=is_weekend,
            spread_bps_lag_1=lag_1,
            spread_bps_lag_2=lag_2,
            spread_bps_lag_3=lag_3,
            price_lag_1=price_lag_1,
            price_lag_2=price_lag_2,
        )

    def _compute_spread_acceleration(self) -> float:
        if len(self.spread_history) < 3:
            return 0.0
        s0 = self.spread_history[-3]
        s1 = self.spread_history[-2]
        s2 = self.spread_history[-1]
        return (s2 - s1) - (s1 - s0)

    def _compute_spread_jerk(self) -> float:
        if len(self.spread_history) < 4:
            return 0.0
        s0 = self.spread_history[-4]
        s1 = self.spread_history[-3]
        s2 = self.spread_history[-2]
        s3 = self.spread_history[-1]
        acc1 = (s2 - s1) - (s1 - s0)
        acc2 = (s3 - s2) - (s2 - s1)
        return acc2 - acc1

    def _compute_rolling_stats(self) -> Dict[str, float]:
        if len(self.spread_history) < 6:
            return {}

        # 30s window = 6 samples at 5s intervals
        window = self.spread_history[-6:]
        n = len(window)
        mean = sum(window) / n
        variance = sum((x - mean) ** 2 for x in window) / n
        std = math.sqrt(variance)
        max_val = max(window)

        # Skewness and kurtosis (simplified)
        if std > 0:
            skew = sum((x - mean) ** 3 for x in window) / n / (std ** 3)
            kurt = sum((x - mean) ** 4 for x in window) / n / (std ** 4) - 3
        else:
            skew = 0.0
            kurt = 0.0

        return {
            "mean_30s": mean,
            "std_30s": std,
            "max_30s": max_val,
            "skew_30s": skew,
            "kurtosis_30s": kurt,
        }

    def _compute_price_roc(self) -> Dict[str, float]:
        if len(self.price_history) < 2:
            return {}

        current = self.price_history[-1]

        def roc(periods: int) -> float:
            if len(self.price_history) < periods + 1:
                return 0.0
            past = self.price_history[-periods - 1]
            return (current - past) / past * 10000 if past != 0 else 0.0

        return {
            "roc_5s": roc(1),
            "roc_30s": roc(6),
            "roc_60s": roc(12),
        }

    def _compute_volatility(self, window: int) -> float:
        if len(self.price_history) < window:
            return 0.0

        prices = self.price_history[-window:]
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] != 0:
                ret = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(ret)

        if len(returns) < 2:
            return 0.0

        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
        return math.sqrt(variance) * 10000  # in bps


# Global engines per pair
_engines: Dict[str, FeatureEngine] = {}


def get_engine(pair: str) -> FeatureEngine:
    if pair not in _engines:
        _engines[pair] = FeatureEngine(pair)
    return _engines[pair]


def compute_features_for_quote(quote: Dict[str, Any]) -> FeatureVector:
    """Compute features for a single quote."""
    engine = get_engine(quote["pair"])
    return engine.update(
        ts=quote["ts"],
        price=quote["price"],
        price_impact_pct=quote.get("price_impact_pct", 0.0),
    )
