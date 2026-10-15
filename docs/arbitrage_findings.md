# Arbitrage findings (2026-10-15)

The user asked to focus on arbitrage and derivatives, on smaller and less
known exchanges. Everything below is measured from public data (no accounts,
no orders). Scripts: `sim/research/xex_scan.py`,
`sim/research/funding_spread_test.py`.

## 1. Spot price gaps across 26 exchanges

25 coins quoted in USDT; best bid/ask from every exchange, every pair
(buy on A, sell on B) checked against both taker fees, then confirmed on the
live order book for a $250 trade. 12 snapshots about 4.5 minutes apart.

- 160-220 gaps per snapshot look profitable on the headline price; almost
  none survive real order-book depth.
- The first snapshot showed 3-10% gaps, all on LATOKEN. Its feed is broken:
  bids above asks and negative quantities. The scanner now rejects crossed
  books.
- After that, depth-confirmed gaps appeared in 3 of 12 snapshots, all small
  (4-20 bps after taker fees), all on smaller venues (Bitrue, Poloniex),
  each gone by the next snapshot.
- Not counted yet and enough to erase most of them: the withdrawal fee to
  move coins back (an XRP withdrawal is ~0.2-1 XRP, about 12-60 bps of a
  $250 trade), transfer time, and needing pre-funded balances on both venues.

Verdict: the big and mid-size exchanges are kept in line by professional
arbitrage bots working in milliseconds. What is left is small, brief, and
on venues with the most counterparty risk. Not viable for a slow bot at our
size.

## 2. Funding-rate spreads between derivatives venues

Hold a long perpetual on the low-funding venue and a short on the
high-funding venue (no market exposure), and collect the difference.
Walk-forward test: each month choose the widest pair on the previous 30
days, book what it earns over the next 30. Costs 0.24% of notional per pair
switch. Capital = 2 x notional (1x margin on each venue).

| History | Venues | Result on capital, per year |
|---|---|---|
| 60 days (1 test month) | 8, incl. XT, Bybit, OKX | +4.7% average; +8 to +13% for SUI, AVAX, DOGE via XT.com |
| 180 days (4 test months) | 5 with long history: Hyperliquid, MEXC, BingX, Phemex, Binance | +1.4% average; best coins ~+3% (DOT, NEAR, SUI, ARB) |

The spread is real and persistent: positive for 9-14 of 15 coins each
month. But it is small, and the larger spreads sit on the riskier venues
(XT.com was hacked in Nov 2024; withdrawals halted). That is a risk premium,
not free money.

Practical limits: whether Swiss residents can trade derivatives on Bybit,
OKX, Phemex or XT is unconfirmed (Binance futures appear closed to them);
perpetuals used for income may count against the tax safe harbor
(derivatives only for hedging); two margin accounts need rebalancing as
prices move.

## 3. Not reachable

Persistent regional premiums (the Korea "kimchi premium" and similar) exist
because of capital controls. Capturing them requires local bank accounts
and citizenship or residency in those countries.

## Bottom line

At $500, the best measured option (cross-venue funding on mainstream
venues, ~1.4-3% a year on capital) is worth about $7-15 a year, and up to
~$50 if we accept a venue like XT. Arbitrage does not change the earlier
conclusion.
