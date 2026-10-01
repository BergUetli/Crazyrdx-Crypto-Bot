"""
instruments.py
Registry of instruments the engine searches and paper-trades.

Admission rule (2026-10-01): an instrument is tradeable only if it is
executable on our actual venue (Jupiter, Solana spot) at a realistic cost.
Measured $250 round trips on Jupiter vs Binance mid (2026-10-01, single
sample, tracked hourly by execution_probe):

    SOL 1.2 bps | BTC (cbBTC) 0.4 | ETH (Portal) ~0 | JUP ~0 | RAY 4.8
    DOGE 6.1 | AVAX 10.9
    rejected: JTO 46 | PYTH 21 | RENDER 53 | LINK 44 | BNB 27

Per-side fee_rate = max(FEE_RATE_BASE, 1.5 x measured per-side cost): the
1.5x margin covers quote variance until the probe has history. Majors that
Jupiter cannot execute cheaply (XRP, ADA, LTC, LINK, BNB) stay
validation-only data until a centralized-exchange venue is decided.

`pair` is the candle/feature key in historical DBs; `futures` is the Binance
USDT-M symbol for derivatives features.
"""

from __future__ import annotations

from typing import Any, Dict, List

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
    "AVAX/USDT": {"mint": "avaxGHCq3T7hoxd73oY2KY9hJSTaeMibXvHy5KNzh5D",
                  "decimals": 9, "binance": "AVAXUSDT", "futures": "AVAXUSDT",
                  "rt_bps_measured": 10.9},
}

TRADEABLE: List[str] = list(INSTRUMENTS)


def fee_rate(instrument: str) -> float:
    """Per-side proportional cost used by search, exam and paper grading."""
    meta = INSTRUMENTS.get(instrument or DEFAULT_INSTRUMENT)
    if not meta:
        return FEE_RATE_BASE
    per_side = float(meta.get("rt_bps_measured", 0.0)) / 2.0 / 1e4
    return max(FEE_RATE_BASE, 1.5 * per_side)


def suffix(instrument: str) -> str:
    """Identity suffix for family/kill/DNA keys. Empty for SOL so every key
    recorded before multi-instrument support keeps its meaning."""
    inst = instrument or DEFAULT_INSTRUMENT
    return "" if inst == DEFAULT_INSTRUMENT else f"@{inst}"


def short(instrument: str) -> str:
    return (instrument or DEFAULT_INSTRUMENT).split("/")[0]
