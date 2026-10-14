#!/usr/bin/env python3
"""
switch_backtests.py
Pre-registered feasibility checks for the two candidate replacement
strategies (switch plan, 2026-10-15). Every parameter below was fixed
BEFORE the first run and must not be tuned afterwards: the point is one
honest look, not another search.

A. Cross-sectional momentum, 12 coins, Binance spot daily closes.
   Rebalance every 7 days; rank by trailing 28-day return; hold the top 3
   equally weighted. Variant B: a top-3 slot whose 28-day return <= 0 sits
   in cash instead. Cost: 11 bps per unit of turnover (Binance 10 bps taker
   + ~1 bp book impact). Benchmarks: BTC buy-and-hold; equal-weight all
   available coins, rebalanced weekly with the same costs.
C. Funding carry: long spot + short perpetual (delta-neutral) on BTC, ETH,
   SOL equally, earning the Binance funding rate. Variant D: only hold
   while the trailing 7-day average funding is positive; each switch costs
   4 legs x 8 bps. Yield reported on notional and on capital (capital =
   2 x notional: spot plus 1x margin, no leverage).

Known biases (stated, not corrected): the 12 coins are 2026 survivors, so
momentum and buy-and-hold are both flattered; Binance funding is a proxy
for a Swiss-accessible perps venue (Binance futures appear unavailable to
Swiss residents). Network only to public Binance endpoints.

Usage: python3 sim/research/switch_backtests.py
"""

from __future__ import annotations

import math
import time
from typing import Dict, List

import httpx

COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX",
         "LINK", "LTC", "JUP", "RAY"]
START = "2021-01-01"
LOOKBACK_D, HOLD_N, REBAL_D = 28, 3, 7
COST_PER_TURNOVER = 0.0011
CARRY_COINS = ["BTC", "ETH", "SOL"]
SWITCH_COST = 4 * 0.0008
SPOT = "https://api.binance.com/api/v3/klines"
FUND = "https://fapi.binance.com/fapi/v1/fundingRate"


def _ms(day: str) -> int:
    return int(time.mktime(time.strptime(day, "%Y-%m-%d")) * 1000)


def daily_closes(client: httpx.Client, coin: str) -> Dict[int, float]:
    out: Dict[int, float] = {}
    start = _ms(START)
    while True:
        r = client.get(SPOT, params={"symbol": f"{coin}USDT", "interval": "1d",
                                     "startTime": start, "limit": 1000})
        rows = r.json() if r.status_code == 200 else []
        if not rows:
            break
        for k in rows:
            out[int(k[0]) // 86_400_000] = float(k[4])
        if len(rows) < 1000:
            break
        start = int(rows[-1][0]) + 86_400_000
    return out


def funding(client: httpx.Client, coin: str) -> List[tuple]:
    out, start = [], _ms(START)
    while True:
        r = client.get(FUND, params={"symbol": f"{coin}USDT",
                                     "startTime": start, "limit": 1000})
        rows = r.json() if r.status_code == 200 else []
        if not rows:
            break
        out += [(int(x["fundingTime"]), float(x["fundingRate"])) for x in rows]
        if len(rows) < 1000:
            break
        start = int(rows[-1]["fundingTime"]) + 1
    return out


def stats(weekly: List[float], label: str, days_per_step: int = 7) -> str:
    if not weekly:
        return f"{label}: no data"
    eq, peak, mdd = 1.0, 1.0, 0.0
    for r in weekly:
        eq *= 1 + r
        peak = max(peak, eq)
        mdd = max(mdd, 1 - eq / peak)
    years = len(weekly) * days_per_step / 365.0
    cagr = eq ** (1 / years) - 1 if eq > 0 else -1.0
    m = sum(weekly) / len(weekly)
    sd = math.sqrt(sum((x - m) ** 2 for x in weekly) / max(len(weekly) - 1, 1))
    sharpe = m / sd * math.sqrt(365.0 / days_per_step) if sd > 0 else 0.0
    return (f"{label:34s} CAGR {cagr:+7.1%}  Sharpe {sharpe:5.2f}  "
            f"maxDD {mdd:6.1%}  x{eq:.2f} over {years:.1f}y")


def momentum(closes: Dict[str, Dict[int, float]]) -> None:
    days = sorted(set().union(*[set(c) for c in closes.values()]))
    reb = days[LOOKBACK_D + 7::REBAL_D]
    res = {"A top3": [], "B top3+cash": [], "EW all coins": [], "BTC hold": []}
    by_year: Dict[str, Dict[int, List[float]]] = {k: {} for k in res}
    prev_w = {k: {} for k in res}
    for t0, t1 in zip(reb, reb[1:]):
        avail = [c for c in COINS if t0 in closes[c] and t1 in closes[c]
                 and (t0 - LOOKBACK_D) in closes[c]]
        if len(avail) < 4:
            continue
        mom = {c: closes[c][t0] / closes[c][t0 - LOOKBACK_D] - 1 for c in avail}
        fwd = {c: closes[c][t1] / closes[c][t0] - 1 for c in avail}
        top = sorted(avail, key=lambda c: -mom[c])[:HOLD_N]
        targets = {
            "A top3": {c: 1 / HOLD_N for c in top},
            "B top3+cash": {c: 1 / HOLD_N for c in top if mom[c] > 0},
            "EW all coins": {c: 1 / len(avail) for c in avail},
            "BTC hold": {"BTC": 1.0} if "BTC" in avail else {},
        }
        year = time.gmtime(t0 * 86400).tm_year
        for k, w in targets.items():
            turnover = sum(abs(w.get(c, 0) - prev_w[k].get(c, 0))
                           for c in set(w) | set(prev_w[k]))
            cost = 0.0 if k == "BTC hold" else turnover * COST_PER_TURNOVER
            r = sum(wt * fwd[c] for c, wt in w.items()) - cost
            res[k].append(r)
            by_year[k].setdefault(year, []).append(r)
            # weights drift with returns until the next rebalance
            grown = {c: wt * (1 + fwd[c]) for c, wt in w.items()}
            tot = sum(grown.values()) + (1 - sum(w.values()))
            prev_w[k] = {c: v / tot for c, v in grown.items()} if tot > 0 else {}
    print("\n=== A/B: cross-sectional momentum (weekly, 12 coins, 11 bps/turnover)")
    for k, v in res.items():
        print("  " + stats(v, k))
    years = sorted(by_year["A top3"])
    print("  per-year return:  " + "  ".join(f"{y}" for y in years))
    for k in res:
        row = []
        for y in years:
            e = 1.0
            for r in by_year[k].get(y, []):
                e *= 1 + r
            row.append(f"{e - 1:+6.0%}")
        print(f"  {k:16s} " + "  ".join(row))


def carry(fund: Dict[str, List[tuple]]) -> None:
    print("\n=== C/D: funding carry, delta-neutral BTC/ETH/SOL (Binance funding)")
    for variant in ("C always on", "D on when 7d avg > 0"):
        per_year: Dict[int, float] = {}
        n_switch = 0
        for coin in CARRY_COINS:
            rows = fund[coin]
            on = False
            for i, (ts, rate) in enumerate(rows):
                y = time.gmtime(ts / 1000).tm_year
                if variant.startswith("C"):
                    want = True
                else:
                    win = [r for _, r in rows[max(0, i - 21):i]]
                    want = bool(win) and sum(win) / len(win) > 0
                if want != on:
                    per_year[y] = per_year.get(y, 0.0) - SWITCH_COST / len(CARRY_COINS)
                    n_switch += 1
                    on = want
                if on:
                    per_year[y] = per_year.get(y, 0.0) + rate / len(CARRY_COINS)
        print(f"  {variant} ({n_switch} switches):")
        for y in sorted(per_year):
            print(f"    {y}: {per_year[y]:+6.1%} on notional, "
                  f"{per_year[y] / 2:+6.1%} on capital")


def main() -> int:
    with httpx.Client(timeout=20) as c:
        closes = {coin: daily_closes(c, coin) for coin in COINS}
        print("daily history from:", {k: time.strftime(
            '%Y-%m', time.gmtime(min(v) * 86400)) for k, v in closes.items() if v})
        momentum(closes)
        fund = {coin: funding(c, coin) for coin in CARRY_COINS}
        carry(fund)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
