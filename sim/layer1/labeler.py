"""
labeler.py
Retrospective labeling daemon for spread events.
Waits 60 seconds after an event, then labels it based on what happened.
"""

import time
from typing import Optional, Dict, Any, List

from config import SPREAD_THRESHOLDS, LABEL_HORIZONS
from layer1.db import (
    get_unlabeled_events, get_quotes_since, update_spread_event_label,
    insert_spread_event
)


def detect_spread_event(ts: float, pair: str, spread_bps: float, price: float) -> Optional[int]:
    """
    Detect if current spread qualifies as an event worth tracking.
    Returns event_id if created, None otherwise.
    """
    if spread_bps >= SPREAD_THRESHOLDS["exploitable"]:
        return insert_spread_event(ts, pair, spread_bps, price)
    return None


def label_pending_events():
    """
    Label all events that are old enough (60+ seconds).
    Called periodically by the orchestrator.
    """
    cutoff = time.time() - 60  # 60 seconds ago
    pending = get_unlabeled_events(cutoff)

    labeled_count = 0
    for event in pending:
        label = compute_label(event)
        if label:
            update_spread_event_label(
                event_id=event["id"],
                label=label["label"],
                label_ts=time.time(),
                spread_at_5s=label.get("spread_at_5s"),
                spread_at_10s=label.get("spread_at_10s"),
                spread_at_30s=label.get("spread_at_30s"),
                spread_at_60s=label.get("spread_at_60s"),
                max_spread=label.get("max_spread"),
                duration_s=label.get("duration_s"),
            )
            labeled_count += 1

    return labeled_count


def compute_label(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Compute label for a spread event by looking at subsequent prices.
    """
    ts = event["ts"]
    pair = event["pair"]
    initial_spread = event["initial_spread_bps"]
    initial_price = event["initial_price"]

    # Get all quotes after this event
    quotes = get_quotes_since(ts, pair)
    if len(quotes) < 2:
        return None

    # Find spreads at each horizon
    horizons = {}
    max_spread = initial_spread
    duration_s = 0.0

    for horizon_s in LABEL_HORIZONS:
        target_ts = ts + horizon_s * 5  # 5s intervals
        # Find quote closest to target time
        closest = None
        min_diff = float("inf")
        for q in quotes:
            diff = abs(q["ts"] - target_ts)
            if diff < min_diff:
                min_diff = diff
                closest = q

        if closest and min_diff <= 3:  # within 3 seconds of target
            spread = abs(closest["price"] - initial_price) / initial_price * 10000
            horizons[f"spread_at_{horizon_s*5}s"] = spread
            if spread > max_spread:
                max_spread = spread
        else:
            horizons[f"spread_at_{horizon_s*5}s"] = None

    # Compute duration (how long spread stayed above threshold)
    for i, q in enumerate(quotes[1:], 1):
        spread = abs(q["price"] - initial_price) / initial_price * 10000
        if spread >= SPREAD_THRESHOLDS["exploitable"]:
            duration_s = q["ts"] - ts
        else:
            break

    # Determine label
    spread_5s = horizons.get("spread_at_5s")
    spread_30s = horizons.get("spread_at_30s")

    if initial_spread >= SPREAD_THRESHOLDS["high_value"]:
        label = "HIGH_VALUE"
    elif spread_5s is not None and spread_5s < initial_spread * 0.5:
        label = "TRANSIENT"  # collapsed fast
    elif spread_30s is not None and spread_30s >= SPREAD_THRESHOLDS["exploitable"]:
        label = "EXPLOITABLE"  # persisted
    elif spread_5s is not None and spread_5s >= SPREAD_THRESHOLDS["exploitable"]:
        label = "EXPLOITABLE"  # persisted at least 5s
    else:
        label = "NOISE"

    return {
        "label": label,
        "spread_at_5s": spread_5s,
        "spread_at_10s": horizons.get("spread_at_10s"),
        "spread_at_30s": spread_30s,
        "spread_at_60s": horizons.get("spread_at_60s"),
        "max_spread": max_spread,
        "duration_s": duration_s,
    }


async def run_cycle():
    """One labeling cycle."""
    return label_pending_events()
