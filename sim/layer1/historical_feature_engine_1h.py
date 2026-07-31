"""
historical_feature_engine_1h.py
Computes features from historical 1-hour candles for backtesting.

Key differences from 5m:
- 54% of candles move >0.6% (vs 11.8% at 5m) = more usable signals
- Fewer candles (4320 vs 51840) = faster backtesting
- Longer lookback periods adjusted for 1h timeframe
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

DB_HIST_FEATURES_1H = DATA_DIR / "historical_features_1h.db"


def init_hist_features_1h_db():
    conn = sqlite3.connect(str(DB_HIST_FEATURES_1H))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS features_1h (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ts              INTEGER NOT NULL,
            pair            TEXT NOT NULL,
            features_json   TEXT NOT NULL,
            created_at      REAL DEFAULT (strftime('%s','now')),
            UNIQUE(ts, pair)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_hist_features_1h_ts ON features_1h(ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_hist_features_1h_pair_ts ON features_1h(pair, ts)")
    conn.commit()
    conn.close()


@dataclass
class HistoricalFeatureVector1h:
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
    taker_buy_roc_4h: float       # change in buy ratio over 4 candles
    taker_buy_roc_1d: float       # change over 24 candles
    taker_buy_sma_ratio: float    # current vs 20-candle average

    # Price momentum (adjusted for 1h)
    price_roc_1h: float        # rate of change last candle
    price_roc_4h: float        # 4 candles
    price_roc_1d: float        # 24 candles
    price_roc_3d: float        # 72 candles
    price_roc_1w: float        # 168 candles

    # Volatility (adjusted for 1h)
    volatility_4h: float       # 4 candles
    volatility_1d: float       # 24 candles
    volatility_3d: float       # 72 candles
    volatility_1w: float       # 168 candles

    # Range features
    hl_range_pct: float        # (high-low)/close
    hl_range_avg_4h: float
    hl_range_avg_1d: float
    body_pct: float            # |close-open|/open
    upper_wick_pct: float      # (high-max(open,close))/close
    lower_wick_pct: float      # (min(open,close)-low)/close

    # Volume features
    volume_roc_4h: float
    volume_roc_1d: float
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
    returns_skew_1d: float     # 24 candles
    returns_kurtosis_1d: float
    autocorrelation_1d: float  # lag-1 autocorrelation of returns

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

    # Multi-timeframe trend alignment (1h vs 4h vs 1d)
    trend_alignment_1h_4h: float   # 1 if same direction, -1 if opposite
    trend_alignment_4h_1d: float
    trend_alignment_1h_1d: float
    trend_alignment_all: float     # 1 if all timeframes agree

    # Multi-timeframe momentum divergence
    momentum_divergence_1h_4h: float
    momentum_divergence_4h_1d: float
    momentum_divergence_1h_1d: float

    # Multi-timeframe volatility regime
    volatility_regime_1h_4h: float
    volatility_regime_4h_1d: float
    volatility_regime_1h_1d: float

    # Higher timeframe context
    price_vs_4h_sma: float
    price_vs_1d_sma: float
    volume_confirmation_1h_4h: float

    # TFT prediction features (populated if TFT model available)
    tft_prediction: float      # -1 to +1 (bearish to bullish)
    tft_confidence: float      # 0 to 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class HistoricalFeatureEngine1h:
    """Computes features from a sequence of 1h candles."""

    FEATURE_NAMES = [
        "open", "high", "low", "close", "volume", "quote_volume", "trades",
        "taker_buy_vol", "taker_buy_ratio", "taker_buy_roc_4h", "taker_buy_roc_1d", "taker_buy_sma_ratio",
        "price_roc_1h", "price_roc_4h", "price_roc_1d", "price_roc_3d", "price_roc_1w",
        "volatility_4h", "volatility_1d", "volatility_3d", "volatility_1w",
        "hl_range_pct", "hl_range_avg_4h", "hl_range_avg_1d",
        "body_pct", "upper_wick_pct", "lower_wick_pct",
        "volume_roc_4h", "volume_roc_1d", "volume_sma_ratio", "volume_weighted_price",
        "sma_5", "sma_20", "sma_50", "price_vs_sma_20", "sma_cross_5_20", "sma_cross_20_50",
        "returns_skew_1d", "returns_kurtosis_1d", "autocorrelation_1d",
        "hour_of_day_sin", "hour_of_day_cos", "day_of_week", "is_weekend",
        "close_lag_1", "close_lag_2", "close_lag_3", "volume_lag_1",
        "returns_lag_1", "returns_lag_2",
        "trend_alignment_1h_4h", "trend_alignment_4h_1d", "trend_alignment_1h_1d", "trend_alignment_all",
        "momentum_divergence_1h_4h", "momentum_divergence_4h_1d", "momentum_divergence_1h_1d",
        "volatility_regime_1h_4h", "volatility_regime_4h_1d", "volatility_regime_1h_1d",
        "price_vs_4h_sma", "price_vs_1d_sma", "volume_confirmation_1h_4h",
        "tft_prediction", "tft_confidence",
    ]

    def __init__(self, pair: str):
        self.pair = pair
        self.candles: List[Dict[str, Any]] = []
        self.closes: List[float] = []
        self.volumes: List[float] = []
        self.returns: List[float] = []
        self.taker_buys: List[float] = []

    def add_candles(self, candles: List[Dict[str, Any]]):
        """Add a batch of candles."""
        for c in candles:
            self.candles.append(c)
            self.closes.append(c["close"])
            self.volumes.append(c["volume"])
            self.taker_buys.append(c.get("taker_buy_vol", 0))

            if len(self.closes) >= 2:
                ret = (self.closes[-1] - self.closes[-2]) / self.closes[-2]
                self.returns.append(ret)
            else:
                self.returns.append(0.0)

    def compute(self, idx: int) -> Optional[HistoricalFeatureVector1h]:
        """Compute features for the candle at index idx."""
        if idx < 50:  # need at least 50 candles for SMA_50
            return None

        c = self.candles[idx]
        ts = c["ts"]

        # Convert ms to local time for temporal features
        dt = time.localtime(ts / 1000)
        hour = dt.tm_hour + dt.tm_min / 60.0

        # Price ROC (adjusted for 1h)
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

        # Taker flow features
        taker_buy = c.get("taker_buy_vol", 0)
        taker_ratio = taker_buy / c["volume"] if c["volume"] > 0 else 0.5

        def taker_roc(periods: int) -> float:
            if idx < periods:
                return 0.0
            past = self.taker_buys[idx - periods]
            if past == 0:
                return 0.0
            return (taker_buy - past) / past

        taker_sma = sum(self.taker_buys[max(0, idx-19):idx+1]) / min(20, idx+1)
        taker_sma_ratio = taker_buy / taker_sma if taker_sma > 0 else 1.0

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

        skew, kurt = skew_kurt(24)  # 1 day

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
            if np.std(x) == 0 or np.std(y) == 0:
                return 0.0
            return float(np.corrcoef(x, y)[0, 1])

        ac = autocorr(24)

        # Lag features
        close_lag_1 = self.closes[idx-1] if idx >= 1 else c["close"]
        close_lag_2 = self.closes[idx-2] if idx >= 2 else c["close"]
        close_lag_3 = self.closes[idx-3] if idx >= 3 else c["close"]
        vol_lag_1 = self.volumes[idx-1] if idx >= 1 else c["volume"]
        ret_lag_1 = self.returns[idx-1] if idx >= 1 else 0.0
        ret_lag_2 = self.returns[idx-2] if idx >= 2 else 0.0

        # Multi-timeframe features (1h vs 4h vs 1d)
        def get_sma_at(offset: int, periods: int) -> float:
            i = idx - offset
            if i < periods - 1:
                return self.closes[i] if i >= 0 else c["close"]
            return sum(self.closes[i - periods + 1:i + 1]) / periods

        def get_roc_at(offset: int, periods: int) -> float:
            i = idx - offset
            if i < periods:
                return 0.0
            past = self.closes[i - periods]
            if past == 0:
                return 0.0
            return (self.closes[i] - past) / past * 10000

        def get_vol_at(offset: int, periods: int) -> float:
            i = idx - offset
            if i < periods or len(self.returns) < i + 1:
                return 0.0
            rets = self.returns[i - periods + 1:i + 1]
            if len(rets) < 2:
                return 0.0
            return float(np.std(rets)) * 10000

        # 4h = 4 candles back, 1d = 24 candles back
        sma5_4h = get_sma_at(4, 5)
        sma20_4h = get_sma_at(4, 20)
        sma5_1d = get_sma_at(24, 5)
        sma20_1d = get_sma_at(24, 20)

        # Trend alignment
        def trend_sign(sma5, sma20):
            return 1.0 if sma5 > sma20 else -1.0

        t1h = trend_sign(sma_5, sma_20)
        t4h = trend_sign(sma5_4h, sma20_4h)
        t1d = trend_sign(sma5_1d, sma20_1d)

        trend_alignment_1h_4h = t1h * t4h
        trend_alignment_4h_1d = t4h * t1d
        trend_alignment_1h_1d = t1h * t1d
        trend_alignment_all = t1h * t4h * t1d

        # Momentum divergence
        roc_1h = roc(1)
        roc_4h = get_roc_at(4, 1)
        roc_1d = get_roc_at(24, 1)

        momentum_divergence_1h_4h = roc_1h - roc_4h
        momentum_divergence_4h_1d = roc_4h - roc_1d
        momentum_divergence_1h_1d = roc_1h - roc_1d

        # Volatility regime
        vol_1h = vol(4)
        vol_4h = get_vol_at(4, 4)
        vol_1d = get_vol_at(24, 4)

        volatility_regime_1h_4h = vol_1h / vol_4h if vol_4h > 0 else 1.0
        volatility_regime_4h_1d = vol_4h / vol_1d if vol_1d > 0 else 1.0
        volatility_regime_1h_1d = vol_1h / vol_1d if vol_1d > 0 else 1.0

        # Higher timeframe context
        price_vs_4h_sma = (c["close"] - sma20_4h) / sma20_4h * 10000 if sma20_4h > 0 else 0
        price_vs_1d_sma = (c["close"] - sma20_1d) / sma20_1d * 10000 if sma20_1d > 0 else 0

        # Volume confirmation
        vol_ratio_4h = self.volumes[idx-4] / (sum(self.volumes[max(0, idx-23):idx-3]) / 20) if idx >= 23 else 1.0
        volume_confirmation_1h_4h = vol_ratio * vol_ratio_4h

        # TFT prediction (placeholder - will be populated by TFT model if available)
        tft_prediction = 0.0
        tft_confidence = 0.0

        return HistoricalFeatureVector1h(
            ts=ts,
            pair=self.pair,
            open=c["open"],
            high=c["high"],
            low=c["low"],
            close=c["close"],
            volume=c["volume"],
            quote_volume=c.get("quote_volume", 0.0),
            trades=c.get("trades", 0),
            taker_buy_vol=taker_buy,
            taker_buy_ratio=taker_ratio,
            taker_buy_roc_4h=taker_roc(4),
            taker_buy_roc_1d=taker_roc(24),
            taker_buy_sma_ratio=taker_sma_ratio,
            price_roc_1h=roc(1),
            price_roc_4h=roc(4),
            price_roc_1d=roc(24),
            price_roc_3d=roc(72),
            price_roc_1w=roc(168),
            volatility_4h=vol(4),
            volatility_1d=vol(24),
            volatility_3d=vol(72),
            volatility_1w=vol(168),
            hl_range_pct=hl_range,
            hl_range_avg_4h=avg_hl_range(4),
            hl_range_avg_1d=avg_hl_range(24),
            body_pct=body,
            upper_wick_pct=upper_wick,
            lower_wick_pct=lower_wick,
            volume_roc_4h=vol_roc(4),
            volume_roc_1d=vol_roc(24),
            volume_sma_ratio=vol_ratio,
            volume_weighted_price=c.get("quote_volume", 0) / c["volume"] if c["volume"] > 0 else c["close"],
            sma_5=sma_5,
            sma_20=sma_20,
            sma_50=sma_50,
            price_vs_sma_20=price_vs_sma,
            sma_cross_5_20=cross_5_20,
            sma_cross_20_50=cross_20_50,
            returns_skew_1d=skew,
            returns_kurtosis_1d=kurt,
            autocorrelation_1d=ac,
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
            trend_alignment_1h_4h=trend_alignment_1h_4h,
            trend_alignment_4h_1d=trend_alignment_4h_1d,
            trend_alignment_1h_1d=trend_alignment_1h_1d,
            trend_alignment_all=trend_alignment_all,
            momentum_divergence_1h_4h=momentum_divergence_1h_4h,
            momentum_divergence_4h_1d=momentum_divergence_4h_1d,
            momentum_divergence_1h_1d=momentum_divergence_1h_1d,
            volatility_regime_1h_4h=volatility_regime_1h_4h,
            volatility_regime_4h_1d=volatility_regime_4h_1d,
            volatility_regime_1h_1d=volatility_regime_1h_1d,
            price_vs_4h_sma=price_vs_4h_sma,
            price_vs_1d_sma=price_vs_1d_sma,
            volume_confirmation_1h_4h=volume_confirmation_1h_4h,
            tft_prediction=tft_prediction,
            tft_confidence=tft_confidence,
        )


def compute_all_features_1h(pair: str, save_to_db: bool = True) -> int:
    """Compute 1h features for all candles of a pair."""
    init_hist_features_1h_db()

    candles = get_candles(pair, interval="1h")
    if len(candles) < 51:
        return 0

    engine = HistoricalFeatureEngine1h(pair)
    engine.add_candles(candles)

    conn = sqlite3.connect(str(DB_HIST_FEATURES_1H))
    count = 0

    for i in range(50, len(candles)):
        fv = engine.compute(i)
        if fv is None:
            continue

        if save_to_db:
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO features_1h (ts, pair, features_json)
                    VALUES (?, ?, ?)
                """, (fv.ts, fv.pair, fv.to_json()))
                count += 1
            except Exception:
                pass

    conn.commit()
    conn.close()
    return count


def get_historical_features_1h(
    pair: str,
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Retrieve 1h historical features from DB."""
    conn = sqlite3.connect(str(DB_HIST_FEATURES_1H))
    conn.row_factory = sqlite3.Row

    query = "SELECT * FROM features_1h WHERE pair = ?"
    params = [pair]

    if start_ts:
        query += " AND ts >= ?"
        params.append(start_ts)
    if end_ts:
        query += " AND ts <= ?"
        params.append(end_ts)

    query += " ORDER BY ts"

    if limit:
        query += f" LIMIT {limit}"

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
    count = compute_all_features_1h("SOL/USDC")
    print(f"Computed {count} 1h feature vectors")
    if count > 0:
        features = get_historical_features_1h("SOL/USDC", limit=1)
        print(f"Sample has {len(features[0]['features'])} fields")
