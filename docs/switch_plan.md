# Switch plan: what to do after the hourly-rules search

Drafted 2026-10-15 at the user's request, after the exam fixes. Numbers
come from `sim/research/switch_backtests.py` (parameters fixed before the
first run; one honest look, no tuning).

## Why switch

- The forward ledger has read WEAKER for weeks: recently frozen champions
  lose $20-40 per 30 days on unseen data and rank below most random
  strategies.
- No paper book has passed in two cohorts. The September books' profits
  were market drift; the October books, all with stop losses, are losing in
  a falling market.
- With the exam corrected (2026-10-15), none of the 12 champions and none
  of the 83 recent exam passes survive the luck haircut for the ~5 million
  distinct strategy families tried. The hourly-rules search has, in effect,
  answered its own question: no edge after costs at this scale.

## Candidates checked

### A/B. Cross-sectional momentum (12 coins, weekly)

Rank coins by 28-day return each week, hold the top 3 equally; variant B
holds cash instead of any top-3 coin whose 28-day return is negative.
Cost 11 bps per unit of turnover (Binance spot, 0.10% fee).

| Strategy (2021-02 to 2026-10) | CAGR | Sharpe | Max drawdown |
|---|---|---|---|
| A: top 3 | +35.7% | 0.75 | 83% |
| B: top 3, cash filter | +50.8% | 0.87 | 73% |
| Equal-weight all 12 | +30.1% | 0.71 | 83% |
| BTC buy-and-hold | +14.5% | 0.52 | 74% |

By year, B returned +445%, -57%, +299%, +105%, -34%, -19% (2021-2026).

Reading: momentum beat both benchmarks over the full period, consistent
with the published crypto momentum factor, but the edge comes from two
boom years. It lost money in 2025 and 2026, and a 73% drawdown means it is
a high-risk crypto allocation rule, not a steady earner. Survivorship bias
(only coins that still exist in 2026 were tested) flatters it.

### C/D. Funding carry (long spot + short perpetual, BTC/ETH/SOL)

| Year | Always on, yield on capital | Filtered (on when 7d funding > 0) |
|---|---|---|
| 2021 | +16.0% | +15.6% |
| 2022 | -5.5% | -1.6% |
| 2023 | +2.9% | +2.4% |
| 2024 | +6.4% | +5.5% |
| 2025 | +1.7% | -1.7% |
| 2026 to date | +0.5% | -2.1% |

Reading: the carry that paid well in 2021 has been competed away. Recent
yields are below a Swiss savings account, before venue and smart-contract
risk. Not worth building.

## Constraints that apply to any option

- **Venue:** Binance spot serves Swiss residents (0.10% fee); Binance
  futures appear unavailable to them (secondary sources; verify on
  Binance's legal page). Carry would need a decentralized perps venue.
- **Swiss tax:** the private-investor safe harbor (ESTV circular 36)
  expects a 6-month minimum holding period, annual volume under 5x
  holdings, no debt financing, and derivatives only for hedging. The
  current hourly bot fails it, and weekly momentum would too. Failing the
  safe harbor triggers a case-by-case assessment, not automatic
  professional-trader status. Confirm with a tax adviser before trading
  real money with any active strategy.

## Options

1. **Stop** (recommended). Pause the GA search now (it can no longer pass
   its own exam and runs the Mac mini 24/7). Let the 12 running paper books
   finish their terms as a final forward check, keep data collection
   (negligible cost) until Nov 30, then shut everything down and archive
   the repo with a final report.
2. **Momentum as a paper allocation tracker.** Replace the search with the
   pre-registered rule B, paper only, rebalanced weekly (near-zero CPU).
   Honest limit: a weekly rule needs years, not weeks, of forward data to
   prove anything, so this is a way to follow a documented rule, not a
   test that will conclude soon. Real money only after a tax-adviser check
   and only money you can see fall 70%.
3. **Carry:** not recommended (yield gone, venue and tax problems).

## Plug criteria (if anything keeps running)

- Hourly-rules search: plug pulled at the Nov 30 checkpoint unless a paper
  book passes every bar (incl. beat-the-market and the corrected exam).
- Momentum tracker (option 2): review at 6 months; stop if it trails
  equal-weight buy-and-hold over that period or if a drawdown beyond 50%
  would be intolerable with real money.
