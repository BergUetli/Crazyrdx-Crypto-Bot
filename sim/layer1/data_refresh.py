#!/usr/bin/env python3
"""
data_refresh.py — hourly market-data refresh (candles + 1h features).

Replaces the legacy Hermes-side refresh loop that silently died on
2026-08-24 and starved the whole system for 8 days. Runs as a LaunchAgent
(hourly), single-shot, self-backfilling: the Binance downloader always
fetches the last N days, so any outage up to that window heals itself.

Refreshes SOL/BTC/ETH 1h candles and recomputes 1h features per pair.
External features (funding/basis) are read by the feature engine from
whatever external data exists — their collectors run separately.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from layer1.historical_downloader import download_candles, init_historical_db
from layer1.historical_feature_engine_1h import compute_all_features_1h

PAIRS_1H = ["SOL/USDC", "BTC/USDC", "ETH/USDC"]


async def _download_all(days: int) -> dict:
    end = int(time.time() * 1000)
    start = end - days * 86400_000
    out = {}
    for pair in PAIRS_1H:
        try:
            out[pair] = await download_candles(pair, start, end, interval="1h")
        except Exception as e:
            out[pair] = f"ERR {e}"
    return out


def main() -> int:
    t0 = time.time()
    init_historical_db()
    counts = asyncio.run(_download_all(days=12))
    feats = {}
    for pair in PAIRS_1H:
        try:
            feats[pair] = compute_all_features_1h(pair)
        except Exception as e:
            feats[pair] = f"ERR {e}"
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] data refresh: "
          f"candles={counts} features={feats} ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
