# Changelog

All notable changes to this project are documented here.

## 2026-08-02 — Realistic economics: the search now optimizes for executable, cost-surviving strategies

Strategy-level audit found the simulator was rewarding trading styles that
cannot make money live. Three essential fixes:

- **Fixed per-trade costs** (`FIXED_COST_PER_SIDE_USD = 0.03` in
  `success_criteria.py`): every simulated swap now pays a fixed
  network/priority-fee cost in addition to the proportional taker fee. On
  $25–50 positions this is 6–25 bps per side — the thing that makes
  2,000-trade churn strategies structurally unprofitable on a $100 book. The
  vintage-ledger baselines pay the same costs, so comparisons stay fair.
  Evolution will now be pushed toward selective, slower strategies.
- **Long-only simulation** (`LONG_ONLY = True`): Jupiter is a spot venue —
  SOL/USDC cannot be shorted there, yet the backtester was taking short
  trades roughly half the time. Simulated shorts were unexecutable fiction;
  entries are now long-only (short signals still work as exit/reversal
  triggers, which spot can execute). Flip the flag if execution ever moves to
  a perps venue.
- **Legacy champion flush** (`flush_legacy_champions` in
  `promotion_funnel.py`, runs at search startup): all 8 stored "champions"
  carried bug-era 1e17 scores and threshold-bug DNA; they were seeding the
  new warm-start every cycle. They are moved to `champions_legacy.json`
  (kept for history) and warm-start seeding also filters scores > 1e6 as a
  second line of defense.

Suite extended to 56 checks (fixed-cost accounting per trade, long-only
enforcement with a shorts-existed control, flush + idempotency).

## 2026-08-01 (third pass) — Vectorized signal engine

- **New `sim/layer1/fast_signals.py`**: entry signals for every strategy
  family (AND/OR/MEANREV/BREAKOUT/TREND/TFT, filters, all sizing methods) are
  precomputed for all bars at once with numpy, then the backtest engine
  **jumps directly between candidate signal bars** instead of walking every
  bar in Python. Feature columns are materialized once per data window and
  cached (LRU). The exit walk also reads numpy price columns instead of
  crawling dicts, and the vol-targeting window uses plain arithmetic instead
  of `np.mean` (~5µs/call overhead on tiny lists).
- **Equivalence-guaranteed**: the test suite proves bit-identical trade lists
  between the fast and legacy paths across 90 genomes covering every strategy
  family, sizing method, and exit type (incl. signal_reversal), plus 150
  full evaluator comparisons with 0 fitness differences. The stochastic
  RANDOM baseline automatically falls back to the legacy path, as does any
  genome/feature shape the fast path cannot replicate. Kill switch:
  `FAST_SIGNALS=0`.
- **Honest numbers** (synthetic 4000-bar data): ~1.4× single-thread on
  trade-heavy populations (per-trade exit work dominates there), larger on
  sparse-signal genomes where per-bar scanning used to dominate. Combined
  with the worker pool: roughly **5×+ total vs the original serial engine**;
  real-data results on the mini will differ — measure with the benchmark in
  the README.
- Suite is now 46 checks.

## 2026-08-01 (later) — Vintage forward ledger: the honest "is it getting smarter?" measure

- **New `sim/evolution/vintage_ledger.py`.** Every cycle's best genome is
  frozen ("vintage") with the timestamp of the last candle it ever saw. As new
  candles arrive, every vintage is re-backtested ONLY on data newer than its
  freeze point — overfitting cannot help a frozen strategy on candles that did
  not exist when it was frozen, so the trend across vintages is a true
  learning curve. Once per day a control cohort is frozen too: 20 random
  genomes + buy-and-hold + SMA 5/20 cross. Champions are reported as a
  **skill percentile vs same-day randoms on the same future data**
  (regime-proof). State lives in `sim/data/vintage_ledger.db` (gitignored).
- **Runner integration** (`run_broad_evolution.py`): after each cycle the
  champion is frozen and all vintages are forward-scored; failures never kill
  the search loop. Features are also **reloaded from the DB every cycle**, so
  a long-running process now picks up newly downloaded candles (previously it
  ran forever on the data loaded at startup).
- **Dashboard Chart D** ("Forward ledger — the real proof"): weekly champion
  skill percentile vs randoms, with a SMARTER / FLAT / WEAKER / **NO EDGE
  YET** verdict. "NO EDGE YET" fires when champions perform like randoms on
  unseen data — the honest signal to invest in new features rather than more
  search. Expect first points ~3 days after deploy, a trustworthy trend after
  ~4 weeks.
- Tests: suite extended to 42 checks (`sim/test_improvements.py` section 9)
  covering freeze/dedupe, forward scoring, idempotent re-scoring, percentile
  math, weekly cohort summary, and dashboard rendering.

## 2026-08-01 — Engine audit: bug fixes, calibration, parallel search, learning-curve dashboard

Full audit and upgrade of the evolution engine. All changes verified by the new
test suite (`sim/test_improvements.py`, 32 checks, includes two full end-to-end
evolution runs and a full promotion-funnel run on synthetic data).

### Bug fixes

- **Threshold ranges were matched by first substring, corrupting much of the
  search space** (`sim/evolution/genome.py`). Examples of what this caused:
  `volatility_regime_1h_4h` (real values ~0–5) drew thresholds from 10–500;
  `price_vs_sma_20` (±500 scale) drew from 0.01–10; `sol_btc_ratio_roc_4h`
  (±100 scale) drew from 0–0.00001; `hour_of_day_sin`, `day_of_week`,
  `is_weekend` had no range at all and drew from 0–100. Conditions built on
  these were permanently true or permanently false — dead weight the GA kept
  breeding and backtesting. Now: longest-key matching, plus explicit ranges for
  the previously unmatched indicators.
- **Kelly sizing formula was wrong and always hit max size**
  (`sim/evolution/evaluator.py`). The divisor was `avg_win * avg_loss`,
  yielding a Kelly fraction of ~65 (i.e. 6500%), so "kelly" sizing silently
  meant "always sizing_max". Correct formula (`f* = p − (1−p)/b`) with
  half-Kelly safety, clamped to [5%, sizing_max].
- **`limit=N` returned the OLDEST N candles, not the newest**
  (`sim/layer1/historical_feature_engine_1h.py` and
  `historical_feature_engine.py`). Once the feature DB outgrew the limit
  (4000 rows), every search ran on frozen old history and never saw recent
  market data. Now returns the most recent N rows in chronological order.
- **Exit fills ignored gaps** (`sim/layer1/backtest_engine.py`). Stops and
  take-profits always filled at the exact stop price even when price gapped
  past the level between bars; real fills are worse. Now gap-aware: if a bar
  opens beyond the level, the fill is the open.
- **Trailing stop had intrabar look-ahead** (`sim/layer1/backtest_engine.py`).
  The same bar's high was used to raise the peak and the same bar's low to
  trigger the trail. Now the trigger uses the peak from prior bars only.
- **Mutation never touched `trailing_stop` / `signal_reversal` exit values**
  (`sim/evolution/genome.py`). Now mutated like the other exit types.

### Improvements

- **Empirical threshold calibration** (`calibrate_threshold_ranges` in
  `genome.py`, wired into `EvolutionEngine`): at the start of each cycle,
  every indicator's sampling range is set to the 5th–95th percentile of its
  actual observed values in the loaded data (~70 indicators calibrated).
  Random and mutated thresholds are now guaranteed to land where a condition
  can actually flip between true and false. Constant features (e.g. TFT heads
  before inference is wired) are excluded automatically.
- **Parallel population evaluation** (`sim/evolution/evaluator.py`).
  Genome backtests now run across a persistent multiprocess pool
  (default: CPU count − 2 workers, override with `EVOLUTION_WORKERS`,
  `EVOLUTION_WORKERS=1` forces serial). Falls back to serial automatically on
  any pool failure. Measured steady-state speedup on synthetic data: **~3.5×**
  more genome evaluations per second (one-time ~0.7 s pool spawn per cycle).
- **Evaluation cache** (`dna_signature` in `genome.py` + cache in
  `EvolutionEngine`): structurally identical genomes (clones, re-rolled
  duplicates) are never re-backtested within a cycle. Cache capped at 50k
  entries.
- **Champion warm-starting** (`run_broad_evolution.py` + `EvolutionEngine`):
  each cycle seeds up to 20% of the initial population from `champions.json`
  and the last cycle's best genome (plus mutated variants), so search refines
  known-good regions instead of always restarting from pure noise. Seeds that
  fall in kill-archive territory are skipped.
- **Smarter threshold mutation** (`genome.py`): 70% of threshold mutations are
  now a local Gaussian nudge (σ = 10% of the indicator's range) instead of a
  full uniform re-roll; 30% remain re-rolls for exploration.
- **Dashboard: "Is the bot getting smarter?" section** (`sim/dashboard.py`).
  New Chart C tracks, per finished search, how many of the 8 strict-exam gates
  the best candidate cleared (8 = full pass → shortlist). Includes a windowed
  SMARTER / FLAT / WEAKER verdict, the current "wall" (the gate where most
  ideas die), full-pass counts, and kill-archive memory growth — real learning
  signals even while the shortlist is empty. Legacy exam scores from the old
  exploding-Sharpe era (e.g. 1.9e17) are now displayed as
  "legacy (broken old score)" instead of raw garbage. Shortlist renumbered to
  section 5.

### Testing

- New `sim/test_improvements.py`: 32 checks covering every fix above, plus
  two end-to-end evolution runs (parallel and serial) and a full promotion
  funnel run on 1200 bars of synthetic data. All persistent state (kill
  archive, champions, trials, funnel results) is redirected to a temp
  directory during tests so real artifacts are never touched.
