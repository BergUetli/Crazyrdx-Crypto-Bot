#!/usr/bin/env python3
"""
funding_spread_test.py
Pre-registered test of cross-venue funding-rate arbitrage (research, public
data, no accounts). For each coin: pull ~60 days of funding history from
every venue; walk forward month by month: choose the venue pair with the widest spread
over the PREVIOUS 30 days (short the high-funding venue, long the low
one), then book what that pair earns over the NEXT 30 days. Selection and
evaluation never see the same data, so best-of-many hindsight cannot
flatter it. History length via FUNDING_DAYS (default 60).

Costs: 4 taker legs (open + close on both venues) at 0.06% each = 0.24% of
notional per round trip, charged once per 30-day hold. Yield reported per
year on notional and on capital (capital = 2 x notional: 1x margin on each
venue, no leverage).

Usage: python3 sim/research/funding_spread_test.py
"""

from __future__ import annotations

import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import ccxt

VENUES = ["hyperliquid", "bybit", "okx", "bitget", "mexc", "bingx",
          "phemex", "kucoin", "htx", "woo", "xt", "lbank", "binance"]
COINS = ["BTC", "ETH", "SOL", "XRP", "DOGE", "SUI", "AVAX", "LINK",
         "ADA", "LTC", "BNB", "TON", "DOT", "NEAR", "APT", "ARB"]
DAYS = int(__import__("os").environ.get("FUNDING_DAYS", "60"))
WINDOW = 30
ROUND_TRIP_COST = 4 * 0.0006


def _sym(ex, coin: str) -> Optional[str]:
    for s in (f"{coin}/USDT:USDT", f"{coin}/USDC:USDC"):
        if s in ex.markets:
            return s
    return None


def history(name: str) -> Tuple[str, Dict[str, Dict[int, float]]]:
    """{coin: {day_index: summed funding that day}} for one venue."""
    try:
        ex = getattr(ccxt, name)({"enableRateLimit": True, "timeout": 20000})
        ex.load_markets()
    except Exception:
        return name, {}
    since = int((time.time() - DAYS * 86400) * 1000)
    out: Dict[str, Dict[int, float]] = {}
    for coin in COINS:
        sym = _sym(ex, coin)
        if not sym or not ex.has.get("fetchFundingRateHistory"):
            continue
        rows, cursor = [], since
        try:
            for _ in range(30):
                batch = ex.fetch_funding_rate_history(sym, since=cursor, limit=500)
                if not batch:
                    break
                rows += batch
                last = batch[-1]["timestamp"]
                if last <= cursor or len(batch) < 2:
                    break
                cursor = last + 1
        except Exception:
            pass
        daily: Dict[int, float] = {}
        for r in rows:
            if r.get("fundingRate") is None or r["timestamp"] < since:
                continue
            d = r["timestamp"] // 86_400_000
            daily[d] = daily.get(d, 0.0) + float(r["fundingRate"])
        if len(daily) >= DAYS * 0.8:
            out[coin] = daily
    return name, out


def main() -> int:
    with ThreadPoolExecutor(8) as pool:
        data = dict(pool.map(history, VENUES))
    have = {v: sorted(d) for v, d in data.items() if d}
    print(f"venues with >={int(DAYS * 0.8)} days of history per coin:", have)
    print(f"\nWalk-forward: pick on prior {WINDOW} days, earn on next {WINDOW} "
          f"(cost {ROUND_TRIP_COST:.2%} of notional when the pair changes)")
    total_eval = []
    monthly_all: Dict[int, List[float]] = {}
    for coin in COINS:
        series = {v: d[coin] for v, d in data.items() if coin in d}
        if len(series) < 2:
            continue
        days = sorted(set.intersection(*[set(s) for s in series.values()]))
        n_win = len(days) // WINDOW
        if n_win < 2:
            continue
        windows = [days[i * WINDOW:(i + 1) * WINDOW] for i in range(n_win)]

        def spread(hi: str, lo: str, ds: List[int]) -> float:
            return sum(series[hi][d] - series[lo][d] for d in ds)
        pairs = [(h, l) for h in series for l in series if h != l]
        res = []
        prev = None
        for k in range(1, n_win):
            hi, lo = max(pairs, key=lambda p: spread(p[0], p[1], windows[k - 1]))
            cost = ROUND_TRIP_COST if (hi, lo) != prev else 0.0
            prev = (hi, lo)
            earned = (spread(hi, lo, windows[k]) - cost) * 365 / WINDOW
            res.append((hi, lo, earned))
            monthly_all.setdefault(k, []).append(earned)
        avg = statistics.mean(r[2] for r in res)
        total_eval.append(avg)
        months = " ".join(f"{r[2]:+5.0%}" for r in res)
        pos = sum(1 for r in res if r[2] > 0)
        print(f"  {coin:5s} {len(series):2d} venues | months (yr-rate): {months} | "
              f"avg {avg:+6.1%}/yr notional ({avg / 2:+5.1%} capital), "
              f"{pos}/{len(res)} months positive | last pair: short {res[-1][0]} long {res[-1][1]}")
    if total_eval:
        print(f"\n  average across coins: {statistics.mean(total_eval):+.1%}/yr on "
              f"notional ({statistics.mean(total_eval) / 2:+.1%} on capital)")
        for k in sorted(monthly_all):
            v = monthly_all[k]
            print(f"  month {k}: basket {statistics.mean(v):+6.1%}/yr, "
                  f"{sum(1 for x in v if x > 0)}/{len(v)} coins positive")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
