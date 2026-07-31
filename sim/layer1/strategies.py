"""
strategies.py
Collection of trading strategies to backtest.
Each strategy is a callable that takes (features, idx) and returns
(direction, strength, size_fraction) or None.
"""

from typing import Optional, Tuple, Callable, Dict, Any, List
import numpy as np


# --- Strategy 1: Momentum Breakout ---
def momentum_breakout(features: List[Dict], idx: int) -> Optional[Tuple[str, float, float]]:
    """
    Buy when price breaks above recent high with volume confirmation.
    Works at low latency because breakouts take time to develop.
    """
    if idx < 20:
        return None

    f = features[idx]["features"]
    prev = features[idx-1]["features"]

    # Breakout: price crosses above 20-period high
    recent_high = max(features[i]["features"]["high"] for i in range(idx-20, idx))
    breakout = f["close"] > recent_high and prev["close"] <= recent_high

    # Volume confirmation
    volume_spike = f["volume_sma_ratio"] > 1.5

    # Volatility filter (avoid choppy markets)
    vol_ok = f["volatility_1h"] > 30  # > 30 bps

    if breakout and volume_spike and vol_ok:
        strength = min(f["volume_sma_ratio"] / 3.0, 1.0)
        size = 0.3 + 0.2 * strength  # 30-50% of capital
        return ("long", strength, size)

    return None


# --- Strategy 2: Mean Reversion (Bollinger) ---
def mean_reversion_bollinger(features: List[Dict], idx: int) -> Optional[Tuple[str, float, float]]:
    """
    Buy when price touches lower Bollinger Band with RSI oversold.
    Works at low latency because mean reversion plays out over minutes/hours.
    """
    if idx < 50:
        return None

    f = features[idx]["features"]

    # Bollinger Band position
    sma_20 = f["sma_20"]
    std_20 = f["volatility_1h"] / 10000 * sma_20  # approximate std in price terms

    if std_20 == 0:
        return None

    z_score = (f["close"] - sma_20) / std_20

    # RSI approximation using returns
    returns = [features[i]["features"]["price_roc_5m"] for i in range(max(0, idx-14), idx)]
    if len(returns) < 14:
        return None

    gains = [r for r in returns if r > 0]
    losses = [abs(r) for r in returns if r < 0]
    avg_gain = sum(gains) / len(gains) if gains else 0
    avg_loss = sum(losses) / len(losses) if losses else 0.01

    rsi = 100 - (100 / (1 + avg_gain / avg_loss)) if avg_loss > 0 else 50

    # Oversold + below lower band
    if z_score < -2.0 and rsi < 30:
        strength = min(abs(z_score) / 3.0, 1.0)
        size = 0.2 + 0.2 * strength
        return ("long", strength, size)

    return None


# --- Strategy 3: Volatility Harvesting ---
def volatility_harvesting(features: List[Dict], idx: int) -> Optional[Tuple[str, float, float]]:
    """
    Trade in direction of volatility expansion.
    High volatility = wide spreads = opportunity.
    """
    if idx < 20:
        return None

    f = features[idx]["features"]

    # Volatility expansion
    vol_expanding = f["volatility_15m"] > f["volatility_1h"] * 1.5

    # Price direction
    trend_up = f["price_roc_5m"] > 0

    # Wide range candle
    wide_range = f["hl_range_pct"] > f["hl_range_avg_1h"] * 2

    if vol_expanding and wide_range:
        direction = "long" if trend_up else "short"
        strength = min(f["hl_range_pct"] / 100, 1.0)
        size = 0.25
        return (direction, strength, size)

    return None


# --- Strategy 4: SMA Crossover ---
def sma_crossover(features: List[Dict], idx: int) -> Optional[Tuple[str, float, float]]:
    """
    Classic SMA crossover. Slow but reliable in trending markets.
    """
    if idx < 50:
        return None

    f = features[idx]["features"]
    prev = features[idx-1]["features"]

    # Golden cross / Death cross
    cross_up = f["sma_cross_5_20"] == 1 and prev["sma_cross_5_20"] == -1
    cross_down = f["sma_cross_5_20"] == -1 and prev["sma_cross_5_20"] == 1

    if cross_up:
        return ("long", 0.7, 0.4)
    elif cross_down:
        return ("short", 0.7, 0.4)

    return None


# --- Strategy 5: Volume-Price Divergence ---
def volume_price_divergence(features: List[Dict], idx: int) -> Optional[Tuple[str, float, float]]:
    """
    Price makes new high but volume declines = reversal coming.
    """
    if idx < 10:
        return None

    f = features[idx]["features"]
    prev = features[idx-5]["features"]

    # New high
    price_high = f["close"] > prev["close"]
    # Volume declining
    vol_declining = f["volume_sma_ratio"] < prev["volume_sma_ratio"] * 0.8

    # Overbought
    overbought = f["price_roc_15m"] > 50  # 50 bps in 15m

    if price_high and vol_declining and overbought:
        return ("short", 0.6, 0.25)

    return None


# --- Strategy 6: Time-of-Day Momentum ---
def time_of_day_momentum(features: List[Dict], idx: int) -> Optional[Tuple[str, float, float]]:
    """
    Trade momentum during specific hours (US market open, Asia close).
    """
    if idx < 5:
        return None

    f = features[idx]["features"]

    # Extract hour from sin/cos
    hour_sin = f["hour_of_day_sin"]
    hour_cos = f["hour_of_day_cos"]
    hour = (np.arctan2(hour_sin, hour_cos) / (2 * np.pi) * 24) % 24

    # US market open (14:30-16:30 UTC) has highest volatility
    us_open = 14 <= hour <= 16
    # Asia close (7-9 UTC) also volatile
    asia_close = 7 <= hour <= 9

    if not (us_open or asia_close):
        return None

    # Strong momentum
    momentum = f["price_roc_15m"]
    if abs(momentum) > 30:
        direction = "long" if momentum > 0 else "short"
        strength = min(abs(momentum) / 100, 1.0)
        return (direction, strength, 0.35)

    return None


# --- Strategy 7: Spread Event (simulated arb) ---
def spread_event(features: List[Dict], idx: int) -> Optional[Tuple[str, float, float]]:
    """
    Detect large price moves that create temporary spreads.
    Buy the dip after a sharp drop, sell the rip after a sharp rise.
    """
    if idx < 10:
        return None

    f = features[idx]["features"]

    # Sharp drop with high volume
    sharp_drop = f["price_roc_15m"] < -50  # -50 bps
    volume_spike = f["volume_sma_ratio"] > 2.0

    # Mean reversion likely
    oversold = f["price_vs_sma_20"] < -100  # 1% below SMA

    if sharp_drop and volume_spike and oversold:
        return ("long", 0.8, 0.4)

    # Sharp rise with high volume
    sharp_rise = f["price_roc_15m"] > 50
    overbought = f["price_vs_sma_20"] > 100

    if sharp_rise and volume_spike and overbought:
        return ("short", 0.8, 0.4)

    return None


# --- Strategy Registry ---
STRATEGIES = {
    "momentum_breakout": momentum_breakout,
    "mean_reversion_bollinger": mean_reversion_bollinger,
    "volatility_harvesting": volatility_harvesting,
    "sma_crossover": sma_crossover,
    "volume_price_divergence": volume_price_divergence,
    "time_of_day_momentum": time_of_day_momentum,
    "spread_event": spread_event,
}


def get_strategy(name: str) -> Callable:
    """Get strategy by name."""
    return STRATEGIES.get(name, momentum_breakout)


def list_strategies() -> List[str]:
    """List all strategy names."""
    return list(STRATEGIES.keys())
