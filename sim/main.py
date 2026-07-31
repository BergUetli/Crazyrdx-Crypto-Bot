#!/usr/bin/env python3
"""
main.py
Orchestrator for Layer 1 data ingestion.
Runs all fetchers on their schedules, computes features, detects events, labels them.
"""

import asyncio
import json
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

from config import (
    PAIRS, POLL_JUPITER, POLL_POOLS, POLL_CEX, POLL_NETWORK,
    DATA_DIR, LOG_DIR, REPORT_DIR
)
from layer1.db import init_all_dbs, insert_feature_vector
from layer1.jupiter_fetcher import fetch_all_pairs, store_quotes
from layer1.pool_monitor import store_pool_snapshots
from layer1.cex_fetcher import store_cex_feeds
from layer1.feature_engine import compute_features_for_quote
from layer1.labeler import detect_spread_event, label_pending_events


# Graceful shutdown
_shutdown = False


def _signal_handler(signum, frame):
    global _shutdown
    print(f"\n[{datetime.now()}] Shutdown signal received")
    _shutdown = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


class Layer1Orchestrator:
    def __init__(self):
        self.cycle_count = 0
        self.start_time = time.time()
        self.last_jupiter = 0.0
        self.last_pools = 0.0
        self.last_cex = 0.0
        self.last_label = 0.0
        self.stats = {
            "quotes_fetched": 0,
            "pools_fetched": 0,
            "cex_fetched": 0,
            "features_computed": 0,
            "events_detected": 0,
            "events_labeled": 0,
        }

    async def run_jupiter_cycle(self):
        """Fetch Jupiter quotes and compute features."""
        quotes = await fetch_all_pairs()
        if quotes:
            await store_quotes(quotes)
            self.stats["quotes_fetched"] += len(quotes)

            # Compute features for each quote
            for q in quotes:
                fv = compute_features_for_quote({
                    "ts": q.ts,
                    "pair": q.pair,
                    "price": q.price,
                    "price_impact_pct": q.price_impact_pct or 0.0,
                })
                insert_feature_vector(fv.ts, fv.pair, fv.to_json())
                self.stats["features_computed"] += 1

                # Detect spread events
                event_id = detect_spread_event(q.ts, q.pair, fv.spread_bps, q.price)
                if event_id:
                    self.stats["events_detected"] += 1
                    print(f"[{datetime.now()}] EVENT: {q.pair} spread={fv.spread_bps:.1f}bps id={event_id}")

        self.last_jupiter = time.time()
        return len(quotes)

    async def run_pool_cycle(self):
        """Fetch pool snapshots."""
        count = await store_pool_snapshots()
        self.stats["pools_fetched"] += count
        self.last_pools = time.time()
        return count

    async def run_cex_cycle(self):
        """Fetch CEX prices."""
        count = await store_cex_feeds()
        self.stats["cex_fetched"] += count
        self.last_cex = time.time()
        return count

    def run_label_cycle(self):
        """Label pending events."""
        count = label_pending_events()
        self.stats["events_labeled"] += count
        self.last_label = time.time()
        return count

    async def run(self):
        """Main orchestrator loop."""
        print(f"[{datetime.now()}] Layer 1 Orchestrator starting")
        print(f"  Pairs: {list(PAIRS.keys())}")
        print(f"  Jupiter poll: {POLL_JUPITER}s")
        print(f"  Pools poll: {POLL_POOLS}s")
        print(f"  CEX poll: {POLL_CEX}s")
        print(f"  Data dir: {DATA_DIR}")

        # Initialize DBs
        init_all_dbs()
        print(f"[{datetime.now()}] Databases initialized")

        # Initial fetch
        await self.run_jupiter_cycle()
        await self.run_pool_cycle()
        await self.run_cex_cycle()

        print(f"[{datetime.now()}] Initial fetch complete, entering main loop")

        while not _shutdown:
            now = time.time()
            self.cycle_count += 1

            # Jupiter: every 5s
            if now - self.last_jupiter >= POLL_JUPITER:
                try:
                    await self.run_jupiter_cycle()
                except Exception as e:
                    print(f"[{datetime.now()}] Jupiter error: {e}")

            # Pools: every 60s
            if now - self.last_pools >= POLL_POOLS:
                try:
                    await self.run_pool_cycle()
                except Exception as e:
                    print(f"[{datetime.now()}] Pools error: {e}")

            # CEX: every 300s
            if now - self.last_cex >= POLL_CEX:
                try:
                    await self.run_cex_cycle()
                except Exception as e:
                    print(f"[{datetime.now()}] CEX error: {e}")

            # Labeling: every 10s
            if now - self.last_label >= 10:
                try:
                    self.run_label_cycle()
                except Exception as e:
                    print(f"[{datetime.now()}] Label error: {e}")

            # Status print every 60 cycles (~5 min)
            if self.cycle_count % 60 == 0:
                uptime = time.time() - self.start_time
                print(f"[{datetime.now()}] Status | uptime={uptime:.0f}s | "
                      f"quotes={self.stats['quotes_fetched']} | "
                      f"features={self.stats['features_computed']} | "
                      f"events={self.stats['events_detected']} | "
                      f"labeled={self.stats['events_labeled']}")

            await asyncio.sleep(1)

        print(f"[{datetime.now()}] Shutdown complete")
        print(f"  Final stats: {json.dumps(self.stats, indent=2)}")


async def main():
    orchestrator = Layer1Orchestrator()
    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
