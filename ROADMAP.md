# Roadmap — deferred, evidence-backed work

Written 2026-08-02, before the unattended month. Each item lists WHY it waits.
First action on return: read dashboard section 4 (Chart D verdict + Chart C
wall), then pick from here.

## Waiting on data maturity (~2-3 months of collection)

- **Derivatives features into the feature engine**: funding z-scores/extremes
  across venues, OI level & 1h/24h deltas, OI-price divergence, top-trader и
  global long/short ratios, taker buy/sell ratio (data already collecting
  hourly for 10 majors via `layer1/derivatives_collector.py`). Do NOT
  integrate before enough history spans the search window — features that are
  zero for most bars create degenerate conditions (the exact bug class fixed
  on 2026-08-01). Evidence: funding extremes are contrarian signals (BitMEX
  study; 2022-2025 bottom markers); OI build-ups precede cascades (Amberdata;
  Oct-2025 cascade postmortem).

## Waiting on the ledger verdict (Chart D green)

- **Champion ensemble paper trading**: trade top 3-5 validated champions as a
  risk-parity portfolio instead of one champion. Diversification is the
  cheapest risk-adjusted-return gain; enables larger sizing at same drawdown.
- **Maker execution (Jupiter limit orders)**: converts ~10bps taker+slippage
  round trips into ~0. On thin edges this is the largest single lever.
  Live-stage only.
- **Perps venue (Drift)**: enables real shorts, funding-rate CAPTURE (the
  best-documented absolute-return stream: delta-neutral long-spot/short-perp,
  historically 10-40%/yr market-neutral, decaying), and flipping
  LONG_ONLY=False. ⚠ Swiss tax note: leverage/derivatives beyond hedging are
  explicit professional-trader flags under Kreisschreiben 36 — get tax advice
  BEFORE this step if gains are becoming material.

## Waiting on multi-asset breadth

- **Cross-sectional momentum** across the 10-major basket (rank & rotate;
  published net Sharpe >1.5 for top-20 rotation; Liu-Tsyvinski-Wu momentum
  factor). Needs multi-asset candles+features (spot candles backfill anytime
  from Binance) and multi-book execution. Biggest strategy-family addition.
- **Enforce the cross-asset gate**: after a month of advisory transfer stats
  in funnel_results (gates.cross_asset.would_pass), decide
  `LAB_CROSS_ASSET_ENFORCE = True`.

## Small, safe, whenever

- Explicit `hour_utc` and `is_us_equity_hours` features (evidence: 21:00-23:00
  UTC window and equity-closed sessions carry most BTC returns). Requires a
  clean feature-history recompute — deferred to avoid touching the feature
  pipeline right before unattended running.
- Kelly sizing from realized rolling trade stats instead of static constants.
- External-feature staleness alarm (candle freshness is alarmed; funding/basis
  input staleness is not yet).
- TFT: retire or wire real inference (currently a zeroed placeholder;
  daily report showed 50.8% directional accuracy = coin flip).

## Explicitly rejected

- Memecoin strategy search (manipulation-saturated, no statistical value).
- HFT/latency plays, CEX-DEX sub-second arb (colocation game, not ours).
- Anything manipulative or gray-area: wash trading, spoofing, sandwich MEV.

## Parked 2026-08-12 (evening) — standalone support builds, decision pending

Menu proposed and PARKED by the user for another day; none interfere with the
running measurement month. Ranked: (1) execution-reality probe — hourly
Jupiter quote sampler at $125-250 sizes to build an empirical cost curve vs
the sim's assumed 2.2bps+$0.03+MEV; (2) derivatives signal analyzer — daily
information-coefficient report over the collected funding/OI/positioning data
(feeds the planned feature integration); (3) shadow paper-trader — the
missing PAPER-stage software (champions vs live quotes, own ledger), build
before Chart D verdict so the 30-day paper clock can start instantly;
(4) weekly Hermes digest. Recommendation on record: 1+2 first, then 3.

## Research-informed upgrades (2026-09-02 literature pass)

- **Quality-Diversity archive (MAP-Elites)**: replace/augment the champion
  board with a behavioral grid (trade-frequency x hold-time x exposure
  niches; one elite per cell). Our stratified exam slots + exploration tax
  approximate QD; a formal archive is the principled version and is now the
  standard in automated strategy discovery (incl. LLM-evolution systems like
  MadEvolve/AlphaEvolve-style pipelines).
- **CPCV (combinatorial purged cross-validation)** as a funnel gate: superior
  overfitting rejection vs single OOS split per recent comparative studies;
  moderate compute (engine is fast enough now).
- Both deferred until the current PAPER cohort completes (~Oct 2) to keep
  the measurement window clean.
