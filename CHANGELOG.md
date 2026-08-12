# Changelog

All notable changes to this project are documented here.

## 2026-08-12 — Reboot-proofing, $500 book, invention grammar v1

- **Reboot incident fixes**: strategy_log.db corrupted by a mid-write reboot
  (writes then failed silently for days) — now WAL + synchronous=NORMAL +
  loud one-time warning; corrupt file quarantined. Runner gained a
  single-instance lock. All three processes (runner, dashboard, derivatives
  collector) now run as LaunchAgents with RunAtLoad+KeepAlive — reboot-proof.
  Dashboard moved to port 8770 (m5-receiver owns 8765 at boot).
- **Book raised to $500** (`BOOK_USD`): fixed swap costs drop from ~15bps to
  ~3bps of position — the viable-strategy space widens ~5x. All dollar bars
  now scale with the book (5%/month standards preserved); search fitness is
  normalized to a $100-equivalent so ranking scales stay comparable; ledger
  baselines use the same book. Ledger re-scores all vintages uniformly under
  current settings, so cross-vintage comparisons remain fair (dollar values
  in past readings scale up ~5x).
- **Invention grammar v1 — the engine now composes its own signals**:
  (a) DERIVED conditions: ratio/diff of any two indicators, thresholded at
  self-calibrating quantiles (~6,900 novel composite signals per combinator,
  none hand-picked); (b) KOFN logic: "at least k of n conditions agree" —
  a decision structure beyond AND/OR. Fully integrated: generation (20% of
  conditions invented), mutation (quantile nudges, component swaps, k
  nudges), crossover, DNA signatures, kill archive, both signal paths with
  PROVEN trade-list equivalence, cross-asset transferability (ratios of
  same-unit series are scale-free). Everything invented faces the same
  9-gate exam and forward ledger.
- **Audit (user-requested "are the positives real?")**: measurement chain
  verified — 0/234 forward scores violate the freeze boundary, random
  controls genuinely trade (median 3), all DBs pass integrity_check. Honest
  recalibration: the strong W31 cohort was dominated by pre-beta-gate
  (demoted) champions riding a rally (med +$2.36/30d vs B&H $4.90-7.41);
  post-beta-gate champions median -$2.19/30d on small early samples. No
  skill demonstrated yet; TOO EARLY stands with better-calibrated hopes.
- Suite now 101 checks.


## 2026-08-05 (later) — Diversity guard + monoculture regression tests

Response to the 40/40 monoculture incident: the test suite now asserts
population diversity as an emergent MULTI-CYCLE property, and the runner
actively enforces it at runtime.

- **Diversity guard** (`run_broad_evolution.py` + `diversity_state.json`):
  every cycle records the winner's family, streak length, and distinct
  winners over the last 20 cycles (printed as `DIVERSITY: ...`). A family
  that wins >=2 consecutive cycles gets an escalating extra selection tax
  (15 x streak) injected into the next cycle — repetition becomes
  mathematically more expensive every cycle until something else wins.
  WARNING printed at streak >=3.
- **Six regression invariants** (suite section [19], now 96 checks):
  multi-cycle winner variety on a shared strategy log; 900-repeat families
  taxed below viability; streak tax prices out repeat winners; exam slots
  span multiple logic families; graduated families get zero exam slots;
  diversity-state streak math.
- Post-change adversarial review: /explore robust to empty DB and hostile
  query params; dedupe sets, funnel logging signature, and datetime imports
  verified.

**Lesson recorded:** the monoculture emerged from mechanism interactions
(warm-start x elites x tournament selection) that each passed unit tests —
population-level invariants over many cycles are now a permanent part of
the quality gate.


## 2026-08-05 — Exploration overhaul: anti-monoculture engine + strategy lab UI

Diagnosis on live data confirmed exploration collapse: the SAME family won
40/40 recent cycles; OR logic held 82% of exam slots (MEANREV/BREAKOUT/TREND:
zero); 902 of 4,483 kills were micro-variants of one neighborhood; one family
re-entered the funnel 43x because passers beyond the capped champion board
were neither champions nor killed.

- **Strategy log** (`sim/evolution/strategy_log.py`, sim/data/strategy_log.db):
  every evaluated genome is recorded (family, indicators, fitness, trades,
  funnel verdict). The search's lab notebook — nothing tried is forgotten.
  45-day retention, pruned daily.
- **Exploration tax**: selection-time fitness penalty
  EXPLORE_FAMILY_TAX*ln(1+recent_tries) per family + flat
  EXPLORE_GRADUATED_TAX for champion/graduated families. Reported fitness
  stays raw; only breeding selection feels it — diminishing returns force
  the GA off exhausted attractors. Tunables in success_criteria.py.
- **Graduated registry** (`graduated_families.json`): families that PASS the
  funnel are remembered and never re-examined (pass-once semantics), ending
  the pass-churn loop.
- **Stratified funnel slots**: the best eligible candidate of EACH logic
  family gets an exam slot before fitness fills the rest — every strategy
  class keeps receiving exam feedback and kill-archive learning.
- **Frontier immigrants**: 30% of fresh genomes point one condition at an
  under-explored indicator (inverse-frequency sampling) — coverage gaps get
  probed deliberately.
- **Complexity ceiling raised**: entry rules up to 6 conditions
  (MAX_CONDITIONS in genome.py), mass still on 2-4.
- **New /explore dashboard page** ("Exploration lab"): per-logic cards
  (tried / distinct families / % viable / % positive / exam pass-fail),
  factor-coverage report with never-tried list, and two-level drill-down to
  the raw per-strategy log. Linked from the main dashboard header.
- Suite now 88 checks.


## 2026-08-02 (fifth pass) — Research-backed upgrades: embargo, cross-asset check, derivatives data

Based on a literature/industry research pass (Liu-Tsyvinski-Wu factors, funding
carry & funding-extreme evidence, Man AHL crypto trend, liquidation/OI
mechanics — see conversation/CHANGELOG history):

- **IS/OOS embargo** (`LAB_EMBARGO_BARS = 24`): out-of-sample windows now
  start 24 bars after the in-sample split in both the evaluator and the
  funnel, so rolling features cannot leak training information across the
  boundary.
- **Advisory cross-asset gate** (funnel Gate 2c): scale-free candidates are
  also backtested on BTC/ETH features; results recorded in gates.cross_asset
  with a would_pass verdict. NEVER blocks promotion yet
  (`LAB_CROSS_ASSET_ENFORCE = False`) — a month of transfer statistics
  accumulates first. Genomes with raw-scale conditions (price/volume levels)
  skip the check as non-transferable.
- **New `sim/layer1/derivatives_collector.py`**: collects free Binance
  futures data (funding rate history, hourly open interest, top-trader and
  global long/short ratios, taker buy/sell ratio) for SOL/BTC/ETH into
  `sim/data/derivatives.db` (gitignored). These are the evidence-backed
  predictor families (funding extremes -> mean reversion, OI build-ups ->
  cascade risk). Binance retains only ~30 days of the /futures/data metrics,
  so this must run on a schedule — every missed month is unrecoverable.
  Verified live: 18/18 feeds. Feature-engine integration is deliberately
  deferred; collection starts now so history exists later.
- None of this resets accumulated learning: kill archive, champions, trial
  counts, and the vintage ledger all persist unchanged.
- Suite now 76 checks.


## 2026-08-02 (fourth pass) — Walk-away hardening: survive an unattended month

Month-scale review for fatal operational flaws before unattended running:

- **Cycle-level crash guard** (`run_broad_evolution.py`): one transient error
  (sqlite lock, network blip, one bad genome) previously killed the process
  and silently ended the month. A failed cycle now logs, waits 60s, and the
  next cycle starts. STOP_EVOLUTION still exits cleanly; worker pools are
  shut down on exception paths so processes never leak.
- **Data-starvation alarm**: if the newest candle is older than 6h, every
  cycle prints a loud WARNING that the downloader may be dead (the search
  keeps running but learns nothing new and the ledger starves).
- **Disk retention**: population files pruned to the newest 400 and funnel
  result files to the newest 1000, every cycle (~36 + ~180 files/day would
  otherwise accumulate unattended).
- **Ledger bloat guards**: forward-scoring is skipped entirely when no new
  candles arrived since the last scoring run; champion freezes are deduped
  against same-DNA-within-24h and capped at 12/day.
- Suite now 69 checks.


## 2026-08-02 (third pass) — Champion revalidation

The 4 champions promoted before the beta gate existed are confirmed
mostly-beta (always-true conditions like `price_vs_1d_sma > -366`); they were
still seeding warm-starts and blocking their families. New
`revalidate_champions()` runs at search startup: every board member is re-run
through the CURRENT 9-gate funnel; failures move to
`champions_unvalidated.json` with the gate they died at, and their families
enter the kill archive (which also stops warm-start from seeding them).
Passers stay. Idempotent. Suite now 66 checks.


## 2026-08-02 (later) — Beta filter: the exam now rejects "long in an up-market" disguised as skill

Post-deploy, 5/5 candidates passed the funnel twice in a row — suspicious right
after costs got harsher. Diagnosis: two holes, both fixed.

- **Benchmark (beta) gate** — new funnel Gate 2b. In a rising market ANY
  long-only strategy shows positive PnL just by being exposed; the funnel
  checked absolute profit only. Now a candidate's OOS profit must beat an
  exposure-matched market benchmark (window return x its own time-in-market x
  its own average position size, minus the same costs) by >=$0.25 and >=25%
  (`LAB_BENCH_MIN_EXCESS_USD`, `LAB_BENCH_EXCESS_FACTOR` in
  success_criteria.py). Matching the benchmark = beta, not edge -> REJECT.
- **Champion re-funnel skip** — warm-started champions re-entered the funnel
  every cycle and got re-promoted (the oscillating promoted=5/5 noise).
  Families already on the champion board are now skipped in candidate
  selection (logged as champion_skipped=N).
- Dashboard: the exam is now 9 gates; Chart C scale, captions, and the
  "wall" labels include the beta filter. Historical gate-depth values shift
  by +1 for gates after OOS (walk_forward onward) — expect a small step in
  old Chart C bars.
- Suite now 62 checks (benchmark math, gate thresholds, flat-market pass,
  champion-skip).

**Note on existing champions:** the 4 champions promoted before this gate
existed were never beta-tested. They stay on the board but should be treated
as unvalidated until the vintage ledger scores them against buy-and-hold on
forward data.


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
