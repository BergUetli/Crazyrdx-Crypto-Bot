"""
historical_feature_engine.py
Computes features from historical 5-minute candles for backtesting.
"""

import json
import math
import sqlite3
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from config import DATA_DIR
from layer1.historical_downloader import get_candles, DB_HISTORICAL

DB_HIST_FEATURES = DATA_DIR / "historical_features.db"


def init_hist_features_db():
    conn = sqlite3.connect(str(DB_HIST_FEATURES))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS features (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ts              INTEGER NOT NULL,
            pair            TEXT NOT NULL,
            features_json   TEXT NOT NULL,
            created_at      REAL DEFAULT (strftime('%s','now')),
            UNIQUE(ts, pair)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS features_v2 (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ts              INTEGER NOT NULL,
            pair            TEXT NOT NULL,
            features_json   TEXT NOT NULL,
            created_at      REAL DEFAULT (strftime('%s','now')),
            UNIQUE(ts, pair)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_hist_features_ts ON features(ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_hist_features_pair_ts ON features(pair, ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_hist_features_v2_ts ON features_v2(ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_hist_features_v2_pair_ts ON features_v2(pair, ts)")
    conn.commit()
    conn.close()


@dataclass
class HistoricalFeatureVector:
    ts: int
    pair: str

    # Price features
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    trades: int

    # Taker flow (buy pressure vs sell pressure)
    taker_buy_vol: float
    taker_buy_ratio: float        # taker_buy_vol / volume (0.5 = balanced)
    taker_buy_roc_15m: float      # change in buy ratio over 3 candles
    taker_buy_roc_1h: float       # change over 12 candles
    taker_buy_sma_ratio: float    # current vs 20-candle average

    # Price momentum
    price_roc_5m: float        # rate of change last candle
    price_roc_15m: float       # 3 candles
    price_roc_30m: float       # 6 candles
    price_roc_1h: float        # 12 candles
    price_roc_4h: float        # 48 candles

    # Volatility
    volatility_15m: float      # std of returns, 3 candles
    volatility_1h: float       # 12 candles
    volatility_4h: float       # 48 candles
    volatility_1d: float       # 288 candles

    # Range features
    hl_range_pct: float        # (high-low)/close
    hl_range_avg_15m: float
    hl_range_avg_1h: float
    body_pct: float            # |close-open|/open
    upper_wick_pct: float      # (high-max(open,close))/close
    lower_wick_pct: float      # (min(open,close)-low)/close

    # Volume features
    volume_roc_15m: float
    volume_roc_1h: float
    volume_sma_ratio: float    # volume / SMA(volume, 20)
    volume_weighted_price: float  # quote_volume / volume

    # Trend features
    sma_5: float
    sma_20: float
    sma_50: float
    price_vs_sma_20: float     # (close - sma_20) / sma_20
    sma_cross_5_20: float      # 1 if sma_5 > sma_20, -1 otherwise
    sma_cross_20_50: float

    # Statistical features
    returns_skew_1h: float
    returns_kurtosis_1h: float
    autocorrelation_1h: float  # lag-1 autocorrelation of returns

    # Temporal features
    hour_of_day_sin: float
    hour_of_day_cos: float
    day_of_week: int
    is_weekend: int

    # Lag features (previous candle values)
    close_lag_1: float
    close_lag_2: float
    close_lag_3: float
    volume_lag_1: float
    returns_lag_1: float
    returns_lag_2: float

    # Multi-timeframe trend alignment
    trend_alignment_5m_15m: float   # 1 if same direction, -1 if opposite
    trend_alignment_15m_1h: float
    trend_alignment_1h_4h: float
    trend_alignment_all: float      # 1 if all timeframes agree

    # Multi-timeframe momentum divergence
    momentum_divergence_5m_15m: float
    momentum_divergence_15m_1h: float
    momentum_divergence_5m_1h: float

    # Multi-timeframe volatility regime
    volatility_regime_5m_15m: float
    volatility_regime_15m_1h: float
    volatility_regime_1h_4h: float
    volatility_regime_5m_4h: float

    # Higher timeframe context
    price_vs_1h_sma: float
    price_vs_4h_sma: float
    volume_confirmation_5m_15m: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class HistoricalFeatureEngine:
    """Computes features from a sequence of candles."""

    FEATURE_NAMES = [
        "open", "high", "low", "close", "volume", "quote_volume", "trades",
        "taker_buy_vol", "taker_buy_ratio", "taker_buy_roc_15m", "taker_buy_roc_1h", "taker_buy_sma_ratio",
        "price_roc_5m", "price_roc_15m", "price_roc_30m", "price_roc_1h", "price_roc_4h",
        "volatility_15m", "volatility_1h", "volatility_4h", "volatility_1d",
        "hl_range_pct", "hl_range_avg_15m", "hl_range_avg_1h",
        "body_pct", "upper_wick_pct", "lower_wick_pct",
        "volume_roc_15m", "volume_roc_1h", "volume_sma_ratio", "volume_weighted_price",
        "sma_5", "sma_20", "sma_50", "price_vs_sma_20", "sma_cross_5_20", "sma_cross_20_50",
        "returns_skew_1h", "returns_kurtosis_1h", "autocorrelation_1h",
        "hour_of_day_sin", "hour_of_day_cos", "day_of_week", "is_weekend",
        "close_lag_1", "close_lag_2", "close_lag_3", "volume_lag_1",
        "returns_lag_1", "returns_lag_2",
        "trend_alignment_5m_15m", "trend_alignment_15m_1h", "trend_alignment_1h_4h", "trend_alignment_all",
        "momentum_divergence_5m_15m", "momentum_divergence_15m_1h", "momentum_divergence_5m_1h",
        "volatility_regime_5m_15m", "volatility_regime_15m_1h", "volatility_regime_1h_4h", "volatility_regime_5m_4h",
        "price_vs_1h_sma", "price_vs_4h_sma", "volume_confirmation_5m_15m",
    ]

    def __init__(self, pair: str):
        self.pair = pair
        self.candles: List[Dict[str, Any]] = []
        self.closes: List[float] = []
        self.volumes: List[float] = []
        self.returns: List[float] = []

    def add_candles(self, candles: List[Dict[str, Any]]):
        """Add a batch of candles."""
        for c in candles:
            self.candles.append(c)
            self.closes.append(c["close"])
            self.volumes.append(c["volume"])

            if len(self.closes) >= 2:
                ret = (self.closes[-1] - self.closes[-2]) / self.closes[-2]
                self.returns.append(ret)
            else:
                self.returns.append(0.0)

    def compute(self, idx: int) -> Optional[HistoricalFeatureVector]:
        """
        Compute features for the candle at index idx.
        Needs enough history for all features.
        """
        if idx < 50:  # need at least 50 candles for SMA_50
            return None

        c = self.candles[idx]
        ts = c["ts"]

        # Convert ms to local time for temporal features
        dt = time.localtime(ts / 1000)
        hour = dt.tm_hour + dt.tm_min / 60.0

        # Price ROC
        def roc(periods: int) -> float:
            if idx < periods:
                return 0.0
            past = self.closes[idx - periods]
            if past == 0:
                return 0.0
            return (c["close"] - past) / past * 10000

        # Volatility
        def vol(periods: int) -> float:
            if idx < periods or len(self.returns) < periods:
                return 0.0
            rets = self.returns[idx - periods + 1:idx + 1]
            if len(rets) < 2:
                return 0.0
            return float(np.std(rets)) * 10000

        # Range features
        hl_range = (c["high"] - c["low"]) / c["close"] * 10000 if c["close"] > 0 else 0
        body = abs(c["close"] - c["open"]) / c["open"] * 10000 if c["open"] > 0 else 0
        upper_wick = (c["high"] - max(c["open"], c["close"])) / c["close"] * 10000 if c["close"] > 0 else 0
        lower_wick = (min(c["open"], c["close"]) - c["low"]) / c["close"] * 10000 if c["close"] > 0 else 0

        def avg_hl_range(periods: int) -> float:
            if idx < periods:
                return hl_range
            ranges = []
            for i in range(idx - periods + 1, idx + 1):
                cc = self.candles[i]
                if cc["close"] > 0:
                    ranges.append((cc["high"] - cc["low"]) / cc["close"] * 10000)
            return sum(ranges) / len(ranges) if ranges else hl_range

        # Volume features
        def vol_roc(periods: int) -> float:
            if idx < periods:
                return 0.0
            past = self.volumes[idx - periods]
            if past == 0:
                return 0.0
            return (c["volume"] - past) / past

        vol_sma = sum(self.volumes[max(0, idx-19):idx+1]) / min(20, idx+1)
        vol_ratio = c["volume"] / vol_sma if vol_sma > 0 else 1.0

        # SMA
        def sma(periods: int) -> float:
            if idx < periods - 1:
                return c["close"]
            return sum(self.closes[idx - periods + 1:idx + 1]) / periods

        sma_5 = sma(5)
        sma_20 = sma(20)
        sma_50 = sma(50)

        price_vs_sma = (c["close"] - sma_20) / sma_20 * 10000 if sma_20 > 0 else 0

        # Cross signals
        prev_sma_5 = sma(5) if idx >= 4 else sma_5
        prev_sma_20 = sma(20) if idx >= 19 else sma_20
        cross_5_20 = 1.0 if sma_5 > sma_20 else -1.0
        cross_20_50 = 1.0 if sma_20 > sma_50 else -1.0

        # Statistical features
        def skew_kurt(periods: int) -> Tuple[float, float]:
            if idx < periods or len(self.returns) < periods:
                return 0.0, 0.0
            rets = self.returns[idx - periods + 1:idx + 1]
            if len(rets) < 3:
                return 0.0, 0.0
            mean = sum(rets) / len(rets)
            std = float(np.std(rets))
            if std == 0:
                return 0.0, 0.0
            skew = sum((r - mean) ** 3 for r in rets) / len(rets) / (std ** 3)
            kurt = sum((r - mean) ** 4 for r in rets) / len(rets) / (std ** 4) - 3
            return float(skew), float(kurt)

        skew, kurt = skew_kurt(12)

        # Autocorrelation
        def autocorr(periods: int, lag: int = 1) -> float:
            if idx < periods + lag or len(self.returns) < periods + lag:
                return 0.0
            rets = self.returns[idx - periods + 1:idx + 1]
            if len(rets) < lag + 2:
                return 0.0
            x = rets[:-lag]
            y = rets[lag:]
            if len(x) < 2:
                return 0.0
            # Check for zero variance (causes NaN in corrcoef)
            if np.std(x) == 0 or np.std(y) == 0:
                return 0.0
            return float(np.corrcoef(x, y)[0, 1])

        ac = autocorr(12)

        # Lag features
        close_lag_1 = self.closes[idx-1] if idx >= 1 else c["close"]
        close_lag_2 = self.closes[idx-2] if idx >= 2 else c["close"]
        close_lag_3 = self.closes[idx-3] if idx >= 3 else c["close"]
        vol_lag_1 = self.volumes[idx-1] if idx >= 1 else c["volume"]
        ret_lag_1 = self.returns[idx-1] if idx >= 1 else 0.0
        ret_lag_2 = self.returns[idx-2] if idx >= 2 else 0.0

        # Multi-timeframe features
        def get_sma_at(offset: int, periods: int) -> float:
            """SMA at a specific candle offset."""
            i = idx - offset
            if i < periods - 1:
                return self.closes[i] if i >= 0 else c["close"]
            return sum(self.closes[i - periods + 1:i + 1]) / periods

        def get_roc_at(offset: int, periods: int) -> float:
            """ROC at a specific candle offset."""
            i = idx - offset
            if i < periods:
                return 0.0
            past = self.closes[i - periods]
            if past == 0:
                return 0.0
            return (self.closes[i] - past) / past * 10000

        def get_vol_at(offset: int, periods: int) -> float:
            """Volatility at a specific candle offset."""
            i = idx - offset
            if i < periods or len(self.returns) < i + 1:
                return 0.0
            rets = self.returns[i - periods + 1:i + 1]
            if len(rets) < 2:
                return 0.0
            return float(np.std(rets)) * 10000

        # 15m = 3 candles back, 1h = 12 back, 4h = 48 back
        sma5_15m = get_sma_at(3, 5)
        sma20_15m = get_sma_at(3, 20)
        sma5_1h = get_sma_at(12, 5)
        sma20_1h = get_sma_at(12, 20)
        sma5_4h = get_sma_at(48, 5)
        sma20_4h = get_sma_at(48, 20)

        # Trend alignment
        def trend_sign(sma5, sma20):
            return 1.0 if sma5 > sma20 else -1.0

        t5 = trend_sign(sma_5, sma_20)
        t15 = trend_sign(sma5_15m, sma20_15m)
        t1h = trend_sign(sma5_1h, sma20_1h)
        t4h = trend_sign(sma5_4h, sma20_4h)

        trend_alignment_5m_15m = t5 * t15
        trend_alignment_15m_1h = t15 * t1h
        trend_alignment_1h_4h = t1h * t4h
        trend_alignment_all = t5 * t15 * t1h * t4h

        # Momentum divergence
        roc_5m = roc(1)
        roc_15m = get_roc_at(3, 1)
        roc_1h = get_roc_at(12, 1)

        momentum_divergence_5m_15m = roc_5m - roc_15m
        momentum_divergence_15m_1h = roc_15m - roc_1h
        momentum_divergence_5m_1h = roc_5m - roc_1h

        # Volatility regime
        vol_5m = vol(3)
        vol_15m = get_vol_at(3, 3)
        vol_1h = get_vol_at(12, 3)
        vol_4h = get_vol_at(48, 3)

        volatility_regime_5m_15m = vol_5m / vol_15m if vol_15m > 0 else 1.0
        volatility_regime_15m_1h = vol_15m / vol_1h if vol_1h > 0 else 1.0
        volatility_regime_1h_4h = vol_1h / vol_4h if vol_4h > 0 else 1.0
        volatility_regime_5m_4h = vol_5m / vol_4h if vol_4h > 0 else 1.0

        # Higher timeframe context
        price_vs_1h_sma = (c["close"] - sma20_1h) / sma20_1h * 10000 if sma20_1h > 0 else 0
        price_vs_4h_sma = (c["close"] - sma20_4h) / sma20_4h * 10000 if sma20_4h > 0 else 0

        # Volume confirmation
        vol_ratio_15m = self.volumes[idx-3] / (sum(self.volumes[max(0, idx-22):idx-2]) / 20) if idx >= 22 else 1.0
        volume_confirmation_5m_15m = vol_ratio * vol_ratio_15m

        return HistoricalFeatureVector(
            ts=ts,
            pair=self.pair,
            open=c["open"],
            high=c["high"],
            low=c["low"],
            close=c["close"],
            volume=c["volume"],
            quote_volume=c.get("quote_volume", 0.0),
            trades=c.get("trades", 0),
            price_roc_5m=roc(1),
            price_roc_15m=roc(3),
            price_roc_30m=roc(6),
            price_roc_1h=roc(12),
            price_roc_4h=roc(48),
            volatility_15m=vol(3),
            volatility_1h=vol(12),
            volatility_4h=vol(48),
            volatility_1d=vol(288),
            hl_range_pct=hl_range,
            hl_range_avg_15m=avg_hl_range(3),
            hl_range_avg_1h=avg_hl_range(12),
            body_pct=body,
            upper_wick_pct=upper_wick,
            lower_wick_pct=lower_wick,
            volume_roc_15m=vol_roc(3),
            volume_roc_1h=vol_roc(12),
            volume_sma_ratio=vol_ratio,
            volume_weighted_price=c.get("quote_volume", 0) / c["volume"] if c["volume"] > 0 else c["close"],
            sma_5=sma_5,
            sma_20=sma_20,
            sma_50=sma_50,
            price_vs_sma_20=price_vs_sma,
            sma_cross_5_20=cross_5_20,
            sma_cross_20_50=cross_20_50,
            returns_skew_1h=skew,
            returns_kurtosis_1h=kurt,
            autocorrelation_1h=ac,
            hour_of_day_sin=math.sin(2 * math.pi * hour / 24),
            hour_of_day_cos=math.cos(2 * math.pi * hour / 24),
            day_of_week=dt.tm_wday,
            is_weekend=1 if dt.tm_wday >= 5 else 0,
            close_lag_1=close_lag_1,
            close_lag_2=close_lag_2,
            close_lag_3=close_lag_3,
            volume_lag_1=vol_lag_1,
            returns_lag_1=ret_lag_1,
            returns_lag_2=ret_lag_2,
            trend_alignment_5m_15m=trend_alignment_5m_15m,
            trend_alignment_15m_1h=trend_alignment_15m_1h,
            trend_alignment_1h_4h=trend_alignment_1h_4h,
            trend_alignment_all=trend_alignment_all,
            momentum_divergence_5m_15m=momentum_divergence_5m_15m,
            momentum_divergence_15m_1h=momentum_divergence_15m_1h,
            momentum_divergence_5m_1h=momentum_divergence_5m_1h,
            volatility_regime_5m_15m=volatility_regime_5m_15m,
            volatility_regime_15m_1h=volatility_regime_15m_1h,
            volatility_regime_1h_4h=volatility_regime_1h_4h,
            volatility_regime_5m_4h=volatility_regime_5m_4h,
            price_vs_1h_sma=price_vs_1h_sma,
            price_vs_4h_sma=price_vs_4h_sma,
            volume_confirmation_5m_15m=volume_confirmation_5m_15m,
        )


def compute_all_features(pair: str, save_to_db: bool = True, table: str = "features_v2") -> int:
    """
    Compute features for all candles of a pair.

    Args:
        pair: Trading pair
        save_to_db: Whether to save to DB
        table: Which table to write to (features or features_v2)

    Returns:
        Number of feature vectors computed
    """
    init_hist_features_db()

    candles = get_candles(pair)
    if len(candles) < 51:
        return 0

    engine = HistoricalFeatureEngine(pair)
    engine.add_candles(candles)

    conn = sqlite3.connect(str(DB_HIST_FEATURES))
    count = 0

    for i in range(50, len(candles)):
        fv = engine.compute(i)
        if fv is None:
            continue

        if save_to_db:
            try:
                conn.execute(f"""
                    INSERT OR IGNORE INTO {table} (ts, pair, features_json)
                    VALUES (?, ?, ?)
                """, (fv.ts, fv.pair, fv.to_json()))
                count += 1
            except Exception:
                pass

    conn.commit()
    conn.close()
    return count


def get_historical_features(
    pair: str,
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
    limit: Optional[int] = None,
    table: str = "features_v2",
) -> List[Dict[str, Any]]:
    """Retrieve historical features from DB."""
    conn = sqlite3.connect(str(DB_HIST_FEATURES))
    conn.row_factory = sqlite3.Row

    query = f"SELECT * FROM {table} WHERE pair = ?"
    params = [pair]

    if start_ts:
        query += " AND ts >= ?"
        params.append(start_ts)
    if end_ts:
        query += " AND ts <= ?"
        params.append(end_ts)

    if limit:
        # Most RECENT `limit` rows, in chronological order (see 1h engine).
        query = (
            f"SELECT * FROM ({query} ORDER BY ts DESC LIMIT {int(limit)}) "
            "ORDER BY ts"
        )
    else:
        query += " ORDER BY ts"

    cursor = conn.execute(query, params)
    rows = []
    for r in cursor.fetchall():
        d = dict(r)
        d["features"] = json.loads(d["features_json"])
        rows.append(d)

    conn.close()
    return rows


if __name__ == "__main__":
    # Quick test
    count = compute_all_features("SOL/USDC")
    print(f"Computed {count} feature vectors")
