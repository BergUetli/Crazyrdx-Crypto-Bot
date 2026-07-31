"""
multi_pair.py
Load features for SOL, BTC, ETH and compute cross-pair features.

Cheap (free, no API): just joins existing 1h feature tables with
SOL/BTC ratio, cross-correlations, and leading/lagging indicators.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from layer1.historical_feature_engine_1h import get_historical_features_1h

PAIRS = ["SOL/USDC", "BTC/USDC", "ETH/USDC"]


def load_multi_pair(
    pair: str = "SOL/USDC",
    limit: Optional[int] = None,
) -> Tuple[
    List[Dict[str, Any]],
    Optional[List[Dict[str, Any]]],
    Optional[List[Dict[str, Any]]],
]:
    """
    Load features for the primary pair + BTC and ETH for cross-pair context.

    Returns (primary_features, btc_features, eth_features).
    btc_features / eth_features may be None if unavailable.
    """
    primary = get_historical_features_1h(pair, limit=limit)
    btc, eth = None, None
    try:
        btc = get_historical_features_1h("BTC/USDC", limit=limit)
    except Exception:
        pass
    try:
        eth = get_historical_features_1h("ETH/USDC", limit=limit)
    except Exception:
        pass
    return primary, btc, eth


def align_timestamps(
    primary: List[Dict[str, Any]],
    other: List[Dict[str, Any]],
) -> Tuple[List[Optional[Dict[str, Any]]], List[Optional[Dict[str, Any]]]]:
    """
    Align 'other' features to primary timestamps.

    For each primary bar, find the closest other bar at or before the same ts.
    Returns (other_aligned, primary_minus_other_aligned) — both same length as primary.
    """
    if not other:
        return [None] * len(primary), [None] * len(primary)

    ts_map: Dict[int, int] = {}  # ts -> idx in other
    for j, o in enumerate(other):
        ts_map[o["ts"]] = j

    other_ts_sorted = sorted(o["ts"] for o in other)
    aligned: List[Optional[Dict[str, Any]]] = [None] * len(primary)
    lag_map: List[Optional[Dict[str, Any]]] = [None] * len(primary)

    j = 0
    for i, p in enumerate(primary):
        pts = p["ts"]
        # find most recent other ts <= pts
        while j < len(other_ts_sorted) and other_ts_sorted[j] <= pts:
            j += 1
        if j > 0:
            aligned[i] = other[ts_map[other_ts_sorted[j - 1]]]
        # Leading version: next other ts > pts (can be None)
        if j < len(other_ts_sorted):
            lag_map[i] = other[ts_map[other_ts_sorted[j]]]

    return aligned, lag_map


def add_cross_pair_features(
    features: List[Dict[str, Any]],
    btc_features: Optional[List[Dict[str, Any]]] = None,
    eth_features: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Augment primary features with cross-pair fields (SOL/BTC ratio etc.).

    Modifies features in-place and returns them.
    New fields added to each bar's 'features' dict:
      - sol_btc_ratio
      - sol_btc_ratio_roc_4h / _1d
      - sol_btc_corr_1d      (rolling 24-bar correlation)
      - btc_eth_corr_1d
      - btc_leading_sol      (1 if BTC moved >1% in last bar, same direction as SOL now)
      - eth_leading_sol
      - cross_trifecta        (1 if all three moving same direction in trend >0)

    All fields default to 0.0 if btc/eth data is missing at that timestamp.
    """
    for pair_name, other in [("btc", btc_features), ("eth", eth_features)]:
        if not other:
            # Set defaults
            for f in features:
                f["features"][f"sol_{pair_name}_ratio"] = 0.0
                f["features"][f"sol_{pair_name}_ratio_roc_4h"] = 0.0
                f["features"][f"sol_{pair_name}_ratio_roc_1d"] = 0.0
                f["features"][f"sol_{pair_name}_corr_1d"] = 0.0
                f["features"][f"{pair_name}_leading_sol"] = 0.0
            continue

        aligned, _ = align_timestamps(features, other)

        # Compute SOL/BTC ratio series
        ratio_vals: List[float] = []
        for i, f in enumerate(features):
            o = aligned[i]
            sol_close = f["features"].get("close", 0.0)
            btc_close = o["features"].get("close", 0.0) if o else 0.0
            if sol_close > 0 and btc_close > 0:
                ratio = sol_close / btc_close
            else:
                ratio = 0.0 if i > 0 else (sol_close / max(btc_close, 1e-9))
            ratio_vals.append(ratio)
            f["features"][f"sol_{pair_name}_ratio"] = float(ratio)

        # ROC of ratio over 4 and 24 bars
        for i, f in enumerate(features):
            roc_4h = 0.0
            roc_1d = 0.0
            if i >= 4 and ratio_vals[i - 4] != 0:
                roc_4h = (ratio_vals[i] - ratio_vals[i - 4]) / ratio_vals[i - 4] * 10000
            if i >= 24 and ratio_vals[i - 24] != 0:
                roc_1d = (ratio_vals[i] - ratio_vals[i - 24]) / ratio_vals[i - 24] * 10000
            f["features"][f"sol_{pair_name}_ratio_roc_4h"] = float(roc_4h)
            f["features"][f"sol_{pair_name}_ratio_roc_1d"] = float(roc_1d)

        # Rolling 24-bar correlation between SOL and BTC/ETH returns
        rets_sol: List[float] = []
        rets_other: List[float] = []
        for i in range(len(features)):
            sol_c = features[i]["features"].get("close", 0.0)
            sol_p = features[i - 1]["features"].get("close", sol_c) if i > 0 else sol_c
            other_c = (
                aligned[i]["features"].get("close", 0.0)
                if aligned[i]
                else 0.0
            )
            other_p = (
                aligned[i - 1]["features"].get("close", other_c)
                if i > 0 and aligned[i - 1]
                else other_c
            )
            rets_sol.append(
                (sol_c - sol_p) / sol_p * 10000 if sol_p > 0 else 0.0
            )
            rets_other.append(
                (other_c - other_p) / other_p * 10000 if other_p > 0 else 0.0
            )
            corr = 0.0
            if i >= 24:
                s_w = np.array(rets_sol[-24:])
                o_w = np.array(rets_other[-24:])
                if np.std(s_w) > 1e-10 and np.std(o_w) > 1e-10:
                    corr = float(np.corrcoef(s_w, o_w)[0, 1])
            f["features"][f"sol_{pair_name}_corr_1d"] = 0.0 if math.isnan(corr) else float(corr)

        # Leading indicator: did other pair move >1% in same/opposite direction last bar
        for i, f in enumerate(features):
            lead = 0.0
            if i >= 1 and aligned[i - 1]:
                oc = aligned[i - 1]["features"].get("close", 0.0)
                op = (
                    aligned[i - 2]["features"].get("close", oc)
                    if i >= 2 and aligned[i - 2]
                    else oc
                )
                if op > 0:
                    other_ret = (oc - op) / op
                    if abs(other_ret) >= 0.01:
                        sol_ret = features[i]["features"].get("returns_lag_1", 0.0) / 10000.0
                        lead = 1.0 if (other_ret > 0) == (sol_ret > 0) else -1.0
            f["features"][f"{pair_name}_leading_sol"] = float(lead)

    # Cross-trifecta: all three pairs trending same direction?
    if btc_features is not None and eth_features is not None:
        btc_aligned, _ = align_timestamps(features, btc_features)
        eth_aligned, _ = align_timestamps(features, eth_features)
        for i, f in enumerate(features):
            trifecta = 0.0
            sol_dir = 1.0 if (f["features"].get("price_roc_1h", 0.0) or 0.0) > 0 else -1.0
            btc_dir = 0.0
            eth_dir = 0.0
            if btc_aligned[i]:
                btc_roc = btc_aligned[i]["features"].get("price_roc_1h", 0.0) or 0.0
                btc_dir = 1.0 if btc_roc > 0 else -1.0
            if eth_aligned[i]:
                eth_roc = eth_aligned[i]["features"].get("price_roc_1h", 0.0) or 0.0
                eth_dir = 1.0 if eth_roc > 0 else -1.0
            if btc_dir != 0 and eth_dir != 0:
                if sol_dir == btc_dir == eth_dir:
                    trifecta = 1.0
                elif sol_dir != btc_dir or sol_dir != eth_dir:
                    trifecta = -1.0
            f["features"]["cross_trifecta"] = float(trifecta)
    else:
        for f in features:
            f["features"]["cross_trifecta"] = 0.0

    return features