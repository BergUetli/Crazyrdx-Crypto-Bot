"""
fast_signals.py
Vectorized signal precomputation for the backtest engine.

The legacy signal path calls a Python closure once per bar, crawling feature
dicts (~3-10µs/bar). Since every signal depends ONLY on that bar's features,
we can precompute the signal for ALL bars at once with numpy, then hand the
engine a closure that just indexes three arrays (~0.2µs/bar).

Semantics are an exact replica of GenomeEvaluator._build_signal_fn — the test
suite proves trade-list equivalence for every strategy family. Any genome or
feature shape this module cannot replicate returns None from
build_array_signal_fn, and callers fall back to the legacy path. The env var
FAST_SIGNALS=0 disables the fast path globally.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np

# LRU of materialized feature columns, keyed by list identity + boundary ts.
# Strong refs, small cap: a process only touches a handful of feature lists
# (full / OOS / IS / funnel slices) at a time.
_COLS_CACHE: "OrderedDict[tuple, Dict[str, np.ndarray]]" = OrderedDict()
_COLS_CACHE_MAX = 16


def get_columns(features: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Materialize feature dicts into float64 column arrays (cached)."""
    key = (id(features), len(features), features[0]["ts"], features[-1]["ts"])
    hit = _COLS_CACHE.get(key)
    if hit is not None:
        _COLS_CACHE.move_to_end(key)
        return hit
    n = len(features)
    # Union of keys from first/middle/last rows (rows are uniform in practice)
    keys = set(features[0]["features"])
    keys |= set(features[n // 2]["features"])
    keys |= set(features[-1]["features"])
    cols: Dict[str, np.ndarray] = {}
    for k in keys:
        cols[k] = np.array(
            [float(f["features"].get(k) or 0.0) for f in features],
            dtype=np.float64,
        )
    _COLS_CACHE[key] = cols
    if len(_COLS_CACHE) > _COLS_CACHE_MAX:
        _COLS_CACHE.popitem(last=False)
    return cols


def _col(
    cols: Dict[str, np.ndarray],
    n: int,
    *names: str,
    default: float = 0.0,
) -> np.ndarray:
    """First existing column among names, else a constant array (mirrors
    chained f.get(a, f.get(b, default)) when rows are uniform)."""
    for name in names:
        c = cols.get(name)
        if c is not None:
            return c
    return np.full(n, float(default))


def _first_threshold(genome, default: float) -> float:
    if genome.entry_conditions:
        return float(genome.entry_conditions[0].threshold)
    return float(default)


def cond_series(cond, cols, n):
    """(values, threshold_value) for one condition — shared by fast and
    legacy paths so invented (derived) conditions stay path-equivalent.

    Plain condition: raw indicator column vs raw threshold.
    Derived condition (combine set): engine-composed series (ratio/diff of
    two indicators) vs a self-calibrating quantile of that series.
    """
    a = _col(cols, n, cond.indicator, default=0.0)
    combine = getattr(cond, "combine", "") or ""
    if not combine:
        return a, float(cond.threshold)
    b = _col(cols, n, getattr(cond, "indicator_b", "") or cond.indicator,
             default=0.0)
    if combine == "ratio":
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.where(np.abs(b) > 1e-12, a / np.where(b == 0, 1.0, b),
                         np.nan)
    else:  # diff
        v = a - b
    q = min(0.95, max(0.05, float(cond.threshold)))
    finite = v[np.isfinite(v)]
    thr = float(np.quantile(finite, q)) if finite.size >= 20 else 0.0
    return v, thr


def _cond_met(cond, cols, n):
    v, t = cond_series(cond, cols, n)
    op = cond.operator
    with np.errstate(invalid="ignore"):
        if op == ">":
            met = v > t
        elif op == "<":
            met = v < t
        elif op == ">=":
            met = v >= t
        elif op == "<=":
            met = v <= t
        else:
            met = np.zeros(n, dtype=bool)
    return np.where(np.isfinite(v), met, False)


def _and_or_masks(genome, cols, n, use_logic: str):
    """AND / OR / KOFN entry mask + per-bar strength (mean of met)."""
    if not genome.entry_conditions:
        return np.zeros(n, dtype=bool), np.zeros(n)
    m = np.vstack([_cond_met(c, cols, n) for c in genome.entry_conditions])
    if use_logic == "AND":
        entry = m.all(axis=0)
    elif use_logic == "KOFN":
        k = min(max(2, int(getattr(genome, "k_of_n", 2) or 2)), m.shape[0])
        entry = m.sum(axis=0) >= k
    else:
        entry = m.any(axis=0)
    strength = m.sum(axis=0) / float(m.shape[0])
    return entry, strength


def _filters_mask(genome, cols, n) -> np.ndarray:
    mask = np.ones(n, dtype=bool)
    for filt in genome.filters:
        if filt.filter_type == "time_of_day":
            hs = _col(cols, n, "hour_of_day_sin")
            hc = _col(cols, n, "hour_of_day_cos")
            hour = (np.arctan2(hs, hc) / (2 * np.pi) * 24) % 24
            hours = filt.params.get("hours", [])
            mask &= np.isin(hour.astype(np.int64), np.array(hours, dtype=np.int64))
        elif filt.filter_type == "day_of_week":
            dow = _col(cols, n, "day_of_week")
            days = filt.params.get("days", [])
            mask &= np.isin(dow.astype(np.int64), np.array(days, dtype=np.int64))
        elif filt.filter_type == "volatility_regime":
            vol = _col(cols, n, "volatility_4h", "volatility_1h", default=0.0)
            mask &= (vol >= float(filt.params.get("min_vol", 0))) & (
                vol <= float(filt.params.get("max_vol", 1000))
            )
        elif filt.filter_type == "trend":
            period = filt.params.get("sma_period", 20)
            sma = _col(cols, n, f"sma_{period}", default=0.0)
            close = _col(cols, n, "close", default=0.0)
            if filt.params.get("direction", "up") == "up":
                mask &= ~(close < sma)
            else:
                mask &= ~(close > sma)
    return mask


def _sizing_array(genome, cols, n) -> np.ndarray:
    base = float(genome.sizing_base)
    smax = float(genome.sizing_max)
    if genome.sizing_method == "volatility_scaled":
        vol = _col(cols, n, "volatility_4h", "volatility_1h", default=50.0)
        return np.minimum(base * (50.0 / np.maximum(vol, 1.0)), smax)
    if genome.sizing_method == "kelly":
        win_rate, avg_win, avg_loss = 0.55, 0.01, 0.005
        b = avg_win / max(avg_loss, 1e-9)
        kelly = win_rate - (1.0 - win_rate) / b
        return np.full(n, min(max(kelly * 0.5, 0.05), smax))
    return np.full(n, base)


def build_array_signal_fn(genome, features: List[Dict[str, Any]]):
    """Precompute (direction, strength, size) for every bar; return an
    array-indexing closure with legacy signal_fn semantics, or None if this
    genome must use the legacy path (e.g. RANDOM baseline)."""
    st = genome.entry_logic
    if st == "RANDOM":
        return None  # stochastic baseline: legacy path only
    n = len(features)
    if n == 0:
        return None
    cols = get_columns(features)
    base = float(genome.sizing_base)

    if st == "MEANREV":
        close = _col(cols, n, "close")
        sma20 = cols.get("sma_20")
        if sma20 is None:
            sma20 = close  # legacy default: f.get("sma_20", close)
        vol = _col(cols, n, "volatility_4h", "volatility_1h", "volatility_1d",
                   default=50.0)
        thr = _first_threshold(genome, 2.0)
        valid = (vol > 0) & (sma20 > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            dev = np.where(valid, (close - sma20) / (vol / 10000.0), 0.0)
        long_m = valid & (dev < -thr)
        short_m = valid & (dev > thr)
        strength = np.minimum(np.abs(dev) / max(thr, 1e-6), 1.0)
        direction = np.where(long_m, 1, np.where(short_m, -1, 0)).astype(np.int8)
        size = base * strength

    elif st == "BREAKOUT":
        hl = _col(cols, n, "hl_range_pct")
        avg = _col(cols, n, "hl_range_avg_4h", "hl_range_avg_1h",
                   "hl_range_avg_1d", default=0.0)
        thr = _first_threshold(genome, 1.5)
        valid = avg > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(valid, hl / np.where(valid, avg, 1.0), 0.0)
        entry = valid & (ratio > thr)
        roc = _col(cols, n, "price_roc_1h", "price_roc_5m")
        direction = np.where(entry, np.where(roc > 0, 1, -1), 0).astype(np.int8)
        strength = np.minimum(ratio / max(thr, 1e-6), 1.0)
        size = base * strength

    elif st == "TREND":
        t_a = _col(cols, n, "trend_alignment_1h_4h", "trend_alignment_5m_15m")
        t_b = _col(cols, n, "trend_alignment_4h_1d", "trend_alignment_15m_1h")
        t_c = _col(cols, n, "trend_alignment_1h_1d", "trend_alignment_1h_4h")
        all_a = _col(cols, n, "trend_alignment_all")
        thr = _first_threshold(genome, 1.0)
        align = (all_a == 1.0) | ((t_a > 0) & (t_b > 0) & (t_c > 0))
        roc = _col(cols, n, "price_roc_4h", "price_roc_15m")
        entry = align & ~(np.abs(roc) < thr)
        direction = np.where(entry, np.where(roc > 0, 1, -1), 0).astype(np.int8)
        strength = np.minimum(np.abs(roc) / max(thr * 2, 1e-6), 1.0)
        size = base * strength

    elif st == "TFT":
        pred = _col(cols, n, "tft_prediction")
        conf = _col(cols, n, "tft_confidence")
        thr = _first_threshold(genome, 0.1)
        active = (np.abs(pred) > 0) | (conf > 0)
        reject = (conf < max(0.05, abs(thr) * 0.01)) & (np.abs(pred) < abs(thr))
        tft_ok = active & ~reject
        tft_strength = np.minimum(np.maximum(np.abs(pred), conf), 1.0)
        tft_dir = np.where(pred >= 0, 1, -1)
        tft_size = base * np.maximum(tft_strength, 0.25)
        # Inactive bars fall through to threshold AND (legacy behavior);
        # note: TFT-active signals bypass filters, the fallthrough does not.
        and_entry, and_strength = _and_or_masks(genome, cols, n, "AND")
        and_entry &= _filters_mask(genome, cols, n)
        roc1 = _col(cols, n, "price_roc_1h", "price_roc_15m")
        and_dir = np.where(roc1 > 0, 1, -1)
        and_size = _sizing_array(genome, cols, n)
        direction = np.where(
            active,
            np.where(tft_ok, tft_dir, 0),
            np.where(and_entry, and_dir, 0),
        ).astype(np.int8)
        strength = np.where(active, tft_strength, and_strength)
        size = np.where(active, tft_size, and_size)

    else:  # AND / OR / KOFN
        entry, strength = _and_or_masks(
            genome, cols, n, st if st in ("AND", "KOFN") else "OR")
        entry &= _filters_mask(genome, cols, n)
        roc1 = _col(cols, n, "price_roc_1h", "price_roc_15m")
        direction = np.where(entry, np.where(roc1 > 0, 1, -1), 0).astype(np.int8)
        size = _sizing_array(genome, cols, n)

    direction = direction.copy()
    direction[: min(50, n)] = 0  # legacy warm-up: no signals before idx 50

    def signal_fn(feats, idx):
        d = direction[idx]
        if d == 0:
            return None
        return ("long" if d == 1 else "short",
                float(strength[idx]), float(size[idx]))

    # Expose the arrays so the engine can jump straight between candidate
    # signal bars instead of walking every bar in Python.
    signal_fn.arrays = (direction, np.asarray(strength, dtype=np.float64),
                        np.asarray(size, dtype=np.float64))
    return signal_fn
