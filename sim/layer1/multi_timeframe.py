"""
multi_timeframe.py
Adds 1m, 15m, 1h, 4h features to the 5m base. Computes cross-timeframe
momentum, volatility, and trend alignment.
"""

import sqlite3
import math
from pathlib import Path
from typing import List, Dict, Any, Optional

SIM_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SIM_DIR / "data"
DB_CANDLES = DATA_DIR / "historical_candles.db"
DB_FEATURES = DATA_DIR / "historical_features.db"


def get_candles(pair: str, interval: str, limit: int = 2000) -> List[Dict[str, Any]]:
    """Load candles for a pair. interval param kept for API compat but DB only has 5m."""
    conn = sqlite3.connect(str(DB_CANDLES))
    cursor = conn.execute(
        "SELECT ts, open, high, low, close, volume FROM candles WHERE pair = ? ORDER BY ts DESC LIMIT ?",
        (pair, limit),
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        {"ts": r[0], "open": r[1], "high": r[2], "low": r[3], "close": r[4], "volume": r[5]}
        for r in reversed(rows)
    ]


def resample_candles(candles: List[Dict[str, Any]], target_interval_s: int) -> List[Dict[str, Any]]:
    """Resample candles to a larger timeframe."""
    if not candles:
        return []

    grouped = {}
    for c in candles:
        bucket = c["ts"] // (target_interval_s * 1000) * (target_interval_s * 1000)
        if bucket not in grouped:
            grouped[bucket] = {
                "ts": bucket,
                "open": c["open"],
                "high": c["high"],
                "low": c["low"],
                "close": c["close"],
                "volume": c["volume"],
            }
        else:
            g = grouped[bucket]
            g["high"] = max(g["high"], c["high"])
            g["low"] = min(g["low"], c["low"])
            g["close"] = c["close"]
            g["volume"] += c["volume"]

    return sorted(grouped.values(), key=lambda x: x["ts"])


def compute_timeframe_features(
    candles: List[Dict[str, Any]],
    prefix: str,
) -> Dict[str, float]:
    """Compute features for one timeframe."""
    if len(candles) < 20:
        return {}

    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    volumes = [c["volume"] for c in candles]

    # Momentum
    def roc(n):
        if len(closes) < n + 1:
            return 0.0
        return (closes[-1] - closes[-n]) / closes[-n] * 100

    # Volatility
    def vol(n):
        if len(closes) < n + 1:
            return 0.0
        rets = [(closes[i] - closes[i-1]) / closes[i-1] * 100 for i in range(-n, 0)]
        return (sum((r - sum(rets)/len(rets)) ** 2 for r in rets) / len(rets)) ** 0.5

    # SMA
    def sma(n):
        if len(closes) < n:
            return closes[-1]
        return sum(closes[-n:]) / n

    # Trend
    sma5 = sma(5)
    sma20 = sma(20)
    sma50 = sma(50)

    # Volume
    vol_sma = sma(20)
    vol_ratio = volumes[-1] / vol_sma if vol_sma > 0 else 1.0

    # Range
    hl_range = (highs[-1] - lows[-1]) / closes[-1] * 100 if closes[-1] > 0 else 0

    return {
        f"{prefix}_roc_5": roc(5),
        f"{prefix}_roc_20": roc(20),
        f"{prefix}_volatility_20": vol(20),
        f"{prefix}_sma_5": sma5,
        f"{prefix}_sma_20": sma20,
        f"{prefix}_sma_50": sma50,
        f"{prefix}_price_vs_sma20": (closes[-1] - sma20) / sma20 * 100 if sma20 > 0 else 0,
        f"{prefix}_sma_cross_5_20": (sma5 - sma20) / sma20 * 100 if sma20 > 0 else 0,
        f"{prefix}_sma_cross_20_50": (sma20 - sma50) / sma50 * 100 if sma50 > 0 else 0,
        f"{prefix}_volume_ratio": vol_ratio,
        f"{prefix}_hl_range": hl_range,
    }


def compute_cross_timeframe_features(
    features_1m: Dict[str, float],
    features_5m: Dict[str, float],
    features_15m: Dict[str, float],
    features_1h: Dict[str, float],
    features_4h: Dict[str, float],
) -> Dict[str, float]:
    """Compute features that compare across timeframes."""
    result = {}

    # Trend alignment: are all timeframes trending the same direction?
    def trend_sign(f, p):
        if not f or f"{p}_sma_cross_5_20" not in f:
            return 0
        return 1 if f[f"{p}_sma_cross_5_20"] > 0 else -1

    t1 = trend_sign(features_1m, "1m")
    t5 = trend_sign(features_5m, "5m")
    t15 = trend_sign(features_15m, "15m")
    t1h = trend_sign(features_1h, "1h")
    t4h = trend_sign(features_4h, "4h")

    result["trend_alignment_1m_5m"] = t1 * t5  # 1 if same, -1 if opposite
    result["trend_alignment_5m_15m"] = t5 * t15
    result["trend_alignment_15m_1h"] = t15 * t1h
    result["trend_alignment_1h_4h"] = t1h * t4h
    result["trend_alignment_all"] = t1 * t5 * t15 * t1h * t4h  # 1 if all same

    # Momentum divergence: is short-term momentum leading or lagging long-term?
    m5 = features_5m.get("5m_roc_5", 0) if features_5m else 0
    m15 = features_15m.get("15m_roc_5", 0) if features_15m else 0
    m1h = features_1h.get("1h_roc_5", 0) if features_1h else 0

    result["momentum_divergence_5m_15m"] = m5 - m15
    result["momentum_divergence_15m_1h"] = m15 - m1h
    result["momentum_divergence_5m_1h"] = m5 - m1h

    # Volatility regime: is volatility expanding or contracting across timeframes?
    v5 = features_5m.get("5m_volatility_20", 0) if features_5m else 0
    v15 = features_15m.get("15m_volatility_20", 0) if features_15m else 0
    v1h = features_1h.get("1h_volatility_20", 0) if features_1h else 0
    v4h = features_4h.get("4h_volatility_20", 0) if features_4h else 0

    result["volatility_regime_5m_15m"] = v5 / v15 if v15 > 0 else 1.0
    result["volatility_regime_15m_1h"] = v15 / v1h if v1h > 0 else 1.0
    result["volatility_regime_1h_4h"] = v1h / v4h if v4h > 0 else 1.0
    result["volatility_regime_5m_4h"] = v5 / v4h if v4h > 0 else 1.0

    # Volume confirmation: is volume confirming the trend?
    vol5 = features_5m.get("5m_volume_ratio", 1.0) if features_5m else 1.0
    vol15 = features_15m.get("15m_volume_ratio", 1.0) if features_15m else 1.0
    result["volume_confirmation_5m_15m"] = vol5 * vol15

    # Higher timeframe support/resistance
    close_5m = features_5m.get("5m_sma_20", 0) if features_5m else 0
    sma_1h = features_1h.get("1h_sma_20", 0) if features_1h else 0
    sma_4h = features_4h.get("4h_sma_20", 0) if features_4h else 0

    result["price_vs_1h_sma"] = (close_5m - sma_1h) / sma_1h * 100 if sma_1h > 0 else 0
    result["price_vs_4h_sma"] = (close_5m - sma_4h) / sma_4h * 100 if sma_4h > 0 else 0

    return result


def add_multi_timeframe_features(pair: str = "SOL/USDC", limit: int = 2000) -> int:
    """
    Add multi-timeframe features to the features DB.
    Returns number of feature vectors updated.
    """
    # Load 5m candles (base)
    candles_5m = get_candles(pair, "5m", limit)
    if not candles_5m:
        return 0

    # Resample to other timeframes
    candles_1m = resample_candles(candles_5m, 60)      # 1 minute
    candles_15m = resample_candles(candles_5m, 900)    # 15 minutes
    candles_1h = resample_candles(candles_5m, 3600)    # 1 hour
    candles_4h = resample_candles(candles_5m, 14400)   # 4 hours

    # Compute features for each timeframe
    features_1m = compute_timeframe_features(candles_1m, "1m")
    features_5m = compute_timeframe_features(candles_5m, "5m")
    features_15m = compute_timeframe_features(candles_15m, "15m")
    features_1h = compute_timeframe_features(candles_1h, "1h")
    features_4h = compute_timeframe_features(candles_4h, "4h")

    # Cross-timeframe features
    cross = compute_cross_timeframe_features(
        features_1m, features_5m, features_15m, features_1h, features_4h
    )

    # Merge all
    all_features = {}
    all_features.update(features_1m)
    all_features.update(features_5m)
    all_features.update(features_15m)
    all_features.update(features_1h)
    all_features.update(features_4h)
    all_features.update(cross)

    # Update the features DB
    conn = sqlite3.connect(str(DB_FEATURES))

    # Check if we need to add columns
    cursor = conn.execute("PRAGMA table_info(features)")
    existing_cols = {row[1] for row in cursor.fetchall()}

    new_cols = set(all_features.keys()) - existing_cols
    for col in new_cols:
        conn.execute(f"ALTER TABLE features ADD COLUMN {col} REAL")

    # Update the most recent feature vector
    latest_ts = candles_5m[-1]["ts"]
    set_clause = ", ".join(f"{k} = ?" for k in all_features.keys())
    values = list(all_features.values()) + [latest_ts, pair]

    conn.execute(
        f"UPDATE features SET {set_clause} WHERE ts = ? AND pair = ?",
        values,
    )
    updated = conn.total_changes

    conn.commit()
    conn.close()

    return updated


if __name__ == "__main__":
    n = add_multi_timeframe_features("SOL/USDC")
    print(f"Updated {n} feature vectors with multi-timeframe features")
