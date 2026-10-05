"""
instruments.py
Registry of instruments the engine searches and paper-trades.

Admission rule (2026-10-01): an instrument is tradeable only if it is
executable on our actual venue (Jupiter, Solana spot) at a realistic cost.
Measured $250 round trips on Jupiter vs Binance mid (2026-10-01, single
sample, tracked hourly by execution_probe):

    SOL 1.2 bps | BTC (cbBTC) 0.4 | ETH (Portal) ~0 | JUP ~0 | RAY 4.8
    DOGE 6.1 | AVAX 10.9 then 42 an hour later (volatile, thin pool)
    rejected: JTO 46 | PYTH 21 | RENDER 53 | LINK 44 | BNB 27

Per-side cost = known venue fee (CEX taker fee; 0 on Jupiter) + execution
cost (measured spread/impact x1.5, floor 2.2 bps). Execution cost comes from
the hourly probe's 7-day median once it has enough samples. Majors Jupiter
cannot execute cheaply (XRP, ADA, LTC, LINK, BNB) are a second, paper-only
tier priced on Binance's public order book plus its 0.10% fee.

`pair` is the candle/feature key in historical DBs; `futures` is the Binance
USDT-M symbol for derivatives features.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from success_criteria import FEE_RATE_BASE

DEFAULT_INSTRUMENT = "SOL/USDC"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

INSTRUMENTS: Dict[str, Dict[str, Any]] = {
    "SOL/USDC": {"mint": "So11111111111111111111111111111111111111112",
                 "decimals": 9, "binance": "SOLUSDT", "futures": "SOLUSDT",
                 "rt_bps_measured": 1.2},
    "BTC/USDC": {"mint": "cbbtcf3aa214zXHbiAZQwf4122FBYbraNdFqgw4iMij",
                 "decimals": 8, "binance": "BTCUSDT", "futures": "BTCUSDT",
                 "rt_bps_measured": 0.4},
    "ETH/USDC": {"mint": "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs",
                 "decimals": 8, "binance": "ETHUSDT", "futures": "ETHUSDT",
                 "rt_bps_measured": 0.0},
    "JUP/USDT": {"mint": "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN",
                 "decimals": 6, "binance": "JUPUSDT", "futures": "JUPUSDT",
                 "rt_bps_measured": 0.0},
    "RAY/USDT": {"mint": "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
                 "decimals": 6, "binance": "RAYUSDT", "futures": "RAYSOLUSDT",
                 "rt_bps_measured": 4.8},
    "DOGE/USDT": {"mint": "DoGEV7LASBkQbibMc5k5vKnTZoMg423GpJ5QtJEGfm7R",
                  "decimals": 8, "binance": "DOGEUSDT", "futures": "DOGEUSDT",
                  "rt_bps_measured": 6.1},
    # Two samples on 2026-10-01: 10.9 and 42 bps round trip (thin wrapped
    # pool). The worse one stands until the probe has a 7-day median.
    "AVAX/USDT": {"mint": "avaxGHCq3T7hoxd73oY2KY9hJSTaeMibXvHy5KNzh5D",
                  "decimals": 9, "binance": "AVAXUSDT", "futures": "AVAXUSDT",
                  "rt_bps_measured": 42.0},
    # --- Centralized-exchange tier (2026-10-05, user decision: decide and
    # build). Binance spot: available to Swiss residents, 0.10% maker/taker
    # for regular users (Kraken's entry taker rose to 0.80% in July 2026,
    # which no hourly strategy survives). Measured $250 order-book impact
    # 2026-10-05: XRP 0.3, ADA 1.8, LTC 0.7, LINK 0.4, BNB 0.1 bps one side
    # before the fee. PAPER ONLY: no exchange account exists; going live on
    # this tier needs the user to open and fund one.
    "XRP/USDT": {"venue": "binance", "binance": "XRPUSDT", "futures": "XRPUSDT",
                 "taker_fee_bps": 10.0, "rt_bps_measured": 0.7},
    "ADA/USDT": {"venue": "binance", "binance": "ADAUSDT", "futures": "ADAUSDT",
                 "taker_fee_bps": 10.0, "rt_bps_measured": 3.7},
    "LTC/USDT": {"venue": "binance", "binance": "LTCUSDT", "futures": "LTCUSDT",
                 "taker_fee_bps": 10.0, "rt_bps_measured": 1.4},
    "LINK/USDT": {"venue": "binance", "binance": "LINKUSDT", "futures": "LINKUSDT",
                  "taker_fee_bps": 10.0, "rt_bps_measured": 0.7},
    "BNB/USDT": {"venue": "binance", "binance": "BNBUSDT", "futures": "BNBUSDT",
                 "taker_fee_bps": 10.0, "rt_bps_measured": 0.2},
}

TRADEABLE: List[str] = list(INSTRUMENTS)

# Cost model is measured, not assumed: once the hourly execution probe has
# PROBE_MIN_SAMPLES quotes for an instrument in the last PROBE_WINDOW_DAYS,
# its median one-side cost replaces the registry estimate. An instrument
# whose one-side cost exceeds MAX_ONE_SIDE_BPS leaves the search rotation
# (still probed, so it re-enters automatically when liquidity returns).
PROBE_MIN_SAMPLES = 12
PROBE_WINDOW_DAYS = 7.0
MAX_ONE_SIDE_BPS = 10.0
COST_MARGIN = 1.5


def _probe_db():
    from config import DATA_DIR
    return DATA_DIR / "execution_probe.db"


PROBE_DB = None  # tests may point this at a temp file


def measured_one_side_bps(instrument: str) -> Optional[float]:
    """7-day median one-side cost from real Jupiter quotes, or None when the
    probe has too few samples (or no database)."""
    import sqlite3
    import statistics
    import time
    path = PROBE_DB or _probe_db()
    try:
        if not path.exists():
            return None
        conn = sqlite3.connect(str(path), timeout=10)
        try:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(quotes)")}
            if "instrument" not in cols:
                return None
            vals = [r[0] for r in conn.execute(
                "SELECT cost_bps FROM quotes WHERE instrument=? AND ts>? "
                "AND cost_bps IS NOT NULL",
                (instrument, time.time() - PROBE_WINDOW_DAYS * 86400))]
        finally:
            conn.close()
    except Exception:
        return None
    if len(vals) < PROBE_MIN_SAMPLES:
        return None
    return float(statistics.median(vals))


def venue(instrument: str) -> str:
    meta = INSTRUMENTS.get(instrument or DEFAULT_INSTRUMENT) or {}
    return meta.get("venue", "jupiter")


def venue_fee_rate(instrument: str) -> float:
    """Known, contractual per-side fee (CEX taker fee). Zero on Jupiter,
    where pool fees are already inside every quote."""
    meta = INSTRUMENTS.get(instrument or DEFAULT_INSTRUMENT) or {}
    return float(meta.get("taker_fee_bps", 0.0)) / 1e4


def one_side_bps(instrument: str) -> float:
    """Best current estimate of one-side EXECUTION cost (spread + impact,
    excluding any venue fee): probe median, else registry."""
    m = measured_one_side_bps(instrument)
    if m is not None:
        return m
    meta = INSTRUMENTS.get(instrument or DEFAULT_INSTRUMENT) or {}
    return float(meta.get("rt_bps_measured", 0.0)) / 2.0


def exec_rate(instrument: str) -> float:
    """Uncertain part of the cost: measured execution with a safety margin,
    never below the original Jupiter-measured base."""
    return max(FEE_RATE_BASE,
               COST_MARGIN * max(0.0, one_side_bps(instrument)) / 1e4)


def fee_rate(instrument: str) -> float:
    """Per-side proportional cost used by search, exam and ledger scoring."""
    if (instrument or DEFAULT_INSTRUMENT) not in INSTRUMENTS:
        return FEE_RATE_BASE
    return venue_fee_rate(instrument) + exec_rate(instrument)


def stress_fee_rates(instrument: str) -> List[float]:
    """[base, mid, high] for the exam's fee-stress gate. Only the uncertain
    execution part is stressed (2x / 4x, floored at the historical absolute
    5 / 10 bps); a known venue fee is added on top unchanged. For Jupiter
    instruments this is exactly the previous formula."""
    from success_criteria import FEE_RATE_STRESS_HIGH, FEE_RATE_STRESS_MID
    vf, ex = venue_fee_rate(instrument), exec_rate(instrument)
    return [vf + ex,
            vf + max(FEE_RATE_STRESS_MID, 2.0 * ex),
            vf + max(FEE_RATE_STRESS_HIGH, 4.0 * ex)]


def active_instruments() -> List[str]:
    """Tradeable instruments whose execution cost is currently sane."""
    return [i for i in TRADEABLE if one_side_bps(i) <= MAX_ONE_SIDE_BPS]


def suffix(instrument: str) -> str:
    """Identity suffix for family/kill/DNA keys. Empty for SOL so every key
    recorded before multi-instrument support keeps its meaning."""
    inst = instrument or DEFAULT_INSTRUMENT
    return "" if inst == DEFAULT_INSTRUMENT else f"@{inst}"


def short(instrument: str) -> str:
    return (instrument or DEFAULT_INSTRUMENT).split("/")[0]
