#!/usr/bin/env python3
"""
xex_scan.py
Cross-exchange spot arbitrage scanner (research, public data only, no
accounts, no orders). Pulls best bid/ask for USDT pairs on many exchanges,
lists every (buy on A, sell on B) gap that beats both venues' taker fees,
then confirms each candidate against real order-book depth at $250.

Known reasons a "gap" is not money (checked separately, see report):
- deposits/withdrawals suspended on one side (the most common cause),
- the two tickers are different tokens sharing a symbol,
- the depth at the quoted price is a few dollars,
- moving coins between the venues takes minutes to hours.

Usage: python3 sim/research/xex_scan.py [snapshots] [seconds_between]
Writes JSON lines to sim/research/xex_scan_results.jsonl.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import ccxt

EXCHANGES = [
    "binance", "okx", "bybit", "kucoin", "gateio", "mexc", "bitget", "htx",
    "bitmart", "lbank", "xt", "bingx", "coinex", "phemex", "ascendex",
    "probit", "bitrue", "poloniex", "whitebit", "bitfinex", "bitstamp",
    "kraken", "cryptocom", "woo", "digifinex", "bigone", "btse", "exmo",
    "latoken", "hitbtc", "toobit", "hashkey",
]
COINS = ["BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "LTC", "LINK", "AVAX",
         "BNB", "TRX", "TON", "DOT", "ATOM", "NEAR", "APT", "SUI", "ARB",
         "OP", "PEPE", "SHIB", "BCH", "ETC", "XLM", "HBAR"]
SIZE_USD = 250.0
DEFAULT_TAKER = 0.002          # if the exchange publishes no taker fee
OUT = Path(__file__).with_name("xex_scan_results.jsonl")


def _client(name: str):
    try:
        ex = getattr(ccxt, name)({"enableRateLimit": True, "timeout": 15000})
        ex.load_markets()
        return ex
    except Exception:
        return None


def _tickers(ex) -> Dict[str, Dict[str, float]]:
    syms = [f"{c}/USDT" for c in COINS if f"{c}/USDT" in ex.markets
            and ex.markets[f"{c}/USDT"].get("active", True) is not False]
    out: Dict[str, Dict[str, float]] = {}
    try:
        ts = ex.fetch_tickers(syms) if ex.has.get("fetchTickers") else {}
    except Exception:
        ts = {}
    for s in syms:
        t = ts.get(s)
        if t is None:
            try:
                t = ex.fetch_ticker(s)
            except Exception:
                continue
        bid, ask = t.get("bid"), t.get("ask")
        if bid and ask and 0 < bid <= ask:
            taker = ex.markets[s].get("taker") or DEFAULT_TAKER
            out[s.split("/")[0]] = {"bid": float(bid), "ask": float(ask),
                                    "taker": float(taker)}
    return out


def _fill(ex, coin: str, side: str, usd: float) -> Optional[float]:
    """Average price of a $usd market order walking the live book."""
    try:
        ob = ex.fetch_order_book(f"{coin}/USDT", limit=50)
    except Exception:
        return None
    bids, asks = ob.get("bids") or [], ob.get("asks") or []
    # Sanity: a crossed book or non-positive sizes means a broken feed
    # (seen on LATOKEN 2026-10-15: bids 10% above asks, negative amounts)
    if not bids or not asks or float(bids[0][0]) >= float(asks[0][0]):
        return None
    levels = asks if side == "buy" else bids
    got = spent = 0.0
    for lvl in levels:
        p, q = float(lvl[0]), float(lvl[1])
        if p <= 0 or q <= 0:
            return None
        take = min(q, (usd - spent) / p)
        got += take
        spent += take * p
        if spent >= usd - 1e-9:
            return spent / got
    return None  # book too thin for $usd


def snapshot(clients: Dict[str, Any]) -> List[Dict[str, Any]]:
    with ThreadPoolExecutor(max_workers=16) as pool:
        quotes = dict(zip(clients, pool.map(_tickers, clients.values())))
    found = []
    for coin in COINS:
        venues = {n: q[coin] for n, q in quotes.items() if coin in q}
        for a, qa in venues.items():          # buy on a
            for b, qb in venues.items():      # sell on b
                if a == b:
                    continue
                gross = qb["bid"] / qa["ask"] - 1.0
                net = gross - qa["taker"] - qb["taker"]
                if net > 0:
                    found.append({"coin": coin, "buy": a, "sell": b,
                                  "ticker_gross_bps": gross * 1e4,
                                  "ticker_net_bps": net * 1e4,
                                  "fees_bps": (qa["taker"] + qb["taker"]) * 1e4})
    # Depth check at $250 for every ticker-level candidate
    for f in found:
        pa = _fill(clients[f["buy"]], f["coin"], "buy", SIZE_USD)
        pb = _fill(clients[f["sell"]], f["coin"], "sell", SIZE_USD)
        if pa and pb:
            f["depth_net_bps"] = (pb / pa - 1.0) * 1e4 - f["fees_bps"]
        else:
            f["depth_net_bps"] = None
    return sorted(found, key=lambda x: -(x.get("depth_net_bps") or -1e9))


def main() -> int:
    n_snap = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    gap_s = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0
    with ThreadPoolExecutor(max_workers=16) as pool:
        loaded = dict(zip(EXCHANGES, pool.map(_client, EXCHANGES)))
    clients = {k: v for k, v in loaded.items() if v is not None}
    print(f"exchanges reachable: {len(clients)}/{len(EXCHANGES)} "
          f"(failed: {sorted(set(EXCHANGES) - set(clients))})")
    for i in range(n_snap):
        t0 = time.time()
        res = snapshot(clients)
        real = [r for r in res if (r.get("depth_net_bps") or 0) > 0]
        with OUT.open("a") as fh:
            fh.write(json.dumps({"ts": t0, "candidates": res}) + "\n")
        print(f"\nsnapshot {i + 1}: {len(res)} ticker gaps beat fees, "
              f"{len(real)} survive $250 depth ({time.time() - t0:.0f}s)")
        for r in real[:15]:
            print(f"  {r['coin']:5s} buy {r['buy']:10s} sell {r['sell']:10s} "
                  f"net {r['depth_net_bps']:7.1f} bps after fees "
                  f"(fees {r['fees_bps']:.0f})")
        if i < n_snap - 1:
            time.sleep(gap_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
