"""
derivatives_features.py
Join Binance futures derivatives metrics (collected hourly into
sim/data/derivatives.db by derivatives_collector.py) onto the 1h feature
rows the evolution engine consumes.

Design rules:
- Merged at READ time (get_historical_features_1h), never stored back into
  features_1h — stored rows stay pure candle features and backfilled
  derivatives history is picked up automatically on the next load.
- Strictly causal: the value attached at bar ts uses only derivatives rows
  with timestamp <= ts; z-scores use the trailing window ENDING AT THE
  PREVIOUS bar, so a bar never sees its own value in its baseline.
- Bars outside derivatives coverage get float('nan'), NOT 0.0. NaN compares
  False against any threshold in both signal engines (numpy and legacy), so
  a condition on a derivatives feature simply never fires where there is no
  data — it cannot produce fake signals on the ~3 years of candle history
  that predates collection (started 2026-07/08, ~30d Binance retention).
- Scale-free indicators only (rates, ratios, RoCs, z-scores): they transfer
  across assets, so the cross-asset gate stays meaningful.
"""

from __future__ import annotations

import math
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_DIR

DB_DERIVS = DATA_DIR / "derivatives.db"

# Pair traded/validated in the sim -> Binance futures symbol
PAIR_SYMBOL = {
    "SOL/USDC": "SOLUSDT",
    "BTC/USDC": "BTCUSDT",
    "ETH/USDC": "ETHUSDT",
    # Validation-only majors (candles collected, never traded)
    "BNB/USDT": "BNBUSDT",
    "XRP/USDT": "XRPUSDT",
    "DOGE/USDT": "DOGEUSDT",
    "ADA/USDT": "ADAUSDT",
    "AVAX/USDT": "AVAXUSDT",
    "LINK/USDT": "LINKUSDT",
    "LTC/USDT": "LTCUSDT",
}

# The indicator names exposed to the GA. Every feature row that passes
# through attach_derivatives() carries ALL of these keys (value or NaN).
DERIV_INDICATORS = [
    # Own-asset funding (8h cadence, forward-filled onto the 1h grid)
    "d_funding",          # current funding rate (fraction per 8h)
    "d_funding_z_30d",    # funding vs its trailing 30d distribution
    # Own-asset open interest (contracts, made scale-free via RoC)
    "d_oi_roc_4h",
    "d_oi_roc_1d",
    # Own-asset positioning / crowding
    "d_top_ls_ratio",     # top-trader long/short position ratio
    "d_top_ls_z_30d",
    "d_global_ls_ratio",  # global long/short account ratio
    "d_taker_ratio",      # futures taker buy/sell volume ratio
    "d_taker_ratio_roc_4h",
    # Market-wide context from BTC (the market's funding/leverage regime)
    "d_btc_funding",
    "d_btc_oi_roc_4h",
    "d_btc_top_ls_ratio",
]

HOUR = 3600
FFILL_MAX_HOURS = {  # how far a metric may be carried forward
    "funding_rate": 9,          # funding prints every 8h
    "open_interest": 3,         # hourly; bridge short collector gaps only
    "top_ls_position_ratio": 3,
    "global_ls_account_ratio": 3,
    "taker_buy_sell_ratio": 3,
}
Z_WINDOW_H = 720   # 30 days
Z_MIN_OBS = 168    # need at least a week of history before z is defined


def _hour_ts(ts: Any) -> int:
    """Timestamp (seconds OR milliseconds) -> hour-floored seconds.

    Both the candle/feature DB and the derivatives DB store epoch
    milliseconds; synthetic test data uses seconds. Anything above 10^11
    is unambiguously milliseconds until the year 5138."""
    t = int(ts)
    if t > 10 ** 11:
        t //= 1000
    return t // HOUR * HOUR


def _load_metric(conn: sqlite3.Connection, symbol: str,
                 metric: str) -> Dict[int, float]:
    """{hour_ts_seconds: value}."""
    out: Dict[int, float] = {}
    for ts_raw, val in conn.execute(
        "SELECT ts, value FROM derivs WHERE symbol=? AND metric=? ORDER BY ts",
        (symbol, metric),
    ):
        try:
            out[_hour_ts(ts_raw)] = float(val)
        except (TypeError, ValueError):
            continue
    return out


def _ffill_grid(raw: Dict[int, float], hours: List[int],
                max_carry_h: int) -> List[float]:
    """Sample a sparse {ts: value} onto the hour grid, carrying the last
    value forward at most max_carry_h hours. Strictly causal (<= ts)."""
    if not raw:
        return [math.nan] * len(hours)
    keys = sorted(raw)
    out: List[float] = []
    i = -1
    for h in hours:
        while i + 1 < len(keys) and keys[i + 1] <= h:
            i += 1
        if i >= 0 and (h - keys[i]) <= max_carry_h * HOUR:
            out.append(raw[keys[i]])
        else:
            out.append(math.nan)
    return out


def _roc(series: List[float], lag: int) -> List[float]:
    out = []
    for i, v in enumerate(series):
        prev = series[i - lag] if i >= lag else math.nan
        if (v == v and prev == prev and prev != 0.0):
            out.append(v / prev - 1.0)
        else:
            out.append(math.nan)
    return out


def _trailing_z(series: List[float]) -> List[float]:
    """z-score of series[i] against the window ending at i-1 (no self)."""
    out = []
    window: List[float] = []  # running valid values with their index
    idx: List[int] = []
    for i, v in enumerate(series):
        lo = i - Z_WINDOW_H
        while idx and idx[0] < lo:
            idx.pop(0)
            window.pop(0)
        if v == v and len(window) >= Z_MIN_OBS:
            m = sum(window) / len(window)
            var = sum((x - m) ** 2 for x in window) / len(window)
            sd = math.sqrt(var)
            out.append((v - m) / sd if sd > 1e-12 else math.nan)
        else:
            out.append(math.nan)
        if v == v:
            window.append(v)
            idx.append(i)
    return out


def compute_deriv_series(hours: List[int],
                         symbol: str) -> Optional[Dict[str, List[float]]]:
    """All own-asset + BTC-context series aligned to `hours` (ts seconds).

    Returns None if the derivatives DB is unreadable (caller then fills NaN).
    """
    try:
        conn = sqlite3.connect(f"file:{DB_DERIVS}?mode=ro", uri=True,
                               timeout=4.0)
    except sqlite3.Error:
        return None
    try:
        def grid(sym: str, metric: str) -> List[float]:
            return _ffill_grid(_load_metric(conn, sym, metric), hours,
                               FFILL_MAX_HOURS[metric])

        funding = grid(symbol, "funding_rate")
        oi = grid(symbol, "open_interest")
        top_ls = grid(symbol, "top_ls_position_ratio")
        global_ls = grid(symbol, "global_ls_account_ratio")
        taker = grid(symbol, "taker_buy_sell_ratio")
        btc_funding = grid("BTCUSDT", "funding_rate")
        btc_oi = grid("BTCUSDT", "open_interest")
        btc_top_ls = grid("BTCUSDT", "top_ls_position_ratio")
        return {
            "d_funding": funding,
            "d_funding_z_30d": _trailing_z(funding),
            "d_oi_roc_4h": _roc(oi, 4),
            "d_oi_roc_1d": _roc(oi, 24),
            "d_top_ls_ratio": top_ls,
            "d_top_ls_z_30d": _trailing_z(top_ls),
            "d_global_ls_ratio": global_ls,
            "d_taker_ratio": taker,
            "d_taker_ratio_roc_4h": _roc(taker, 4),
            "d_btc_funding": btc_funding,
            "d_btc_oi_roc_4h": _roc(btc_oi, 4),
            "d_btc_top_ls_ratio": btc_top_ls,
        }
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def attach_derivatives(rows: List[Dict[str, Any]], pair: str) -> int:
    """Merge derivatives indicators into feature rows IN PLACE.

    Every row gets every DERIV_INDICATORS key (NaN where uncovered), so
    downstream `.get(k) or 0.0` defaults can never turn "no data" into a
    fake 0.0 reading. Returns the number of rows with real (non-NaN)
    own-asset funding data, as a coverage signal for logs.
    """
    if not rows:
        return 0
    symbol = PAIR_SYMBOL.get(pair)
    hours = [_hour_ts(r["ts"]) for r in rows]
    series = compute_deriv_series(hours, symbol) if symbol else None
    covered = 0
    for i, r in enumerate(rows):
        f = r.get("features")
        if f is None:
            continue
        if series is None:
            for k in DERIV_INDICATORS:
                f[k] = math.nan
        else:
            for k, s in series.items():
                f[k] = s[i]
            if f["d_funding"] == f["d_funding"]:
                covered += 1
    return covered
