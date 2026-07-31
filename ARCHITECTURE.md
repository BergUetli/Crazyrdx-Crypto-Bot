# Architecture 🏗️

This document explains the internal machinery of the bot. The system is designed to act like a quantitative hedge fund running entirely on a local machine. Instead of relying on human intuition, it uses evolutionary algorithms to systematically build, test, and vet hundreds of thousands of micro-trading strategies.

---

## 1. The Core Loop (How it learns)

The core driver of the bot is the **Broad Explorer** (`sim/run_broad_evolution.py`). It runs in a continuous, infinite loop offline.

1. **Initialization:** The system loads historical 1-hour candle data (e.g., SOL/USDC) and computes *features* (e.g., RSI, Volatility, SMA crosses, multi-timeframe trends).
2. **Breeding (Genetic Algorithm):** It generates 150 random strategies ("Genomes"). 
3. **Simulation:** The `BacktestEngine` runs every strategy against the historical data to see how it would have performed. 
4. **Scoring:** The `GenomeEvaluator` scores them based on a composite fitness function (Sharpe ratio, win rate, total profit, and a penalty for drawdowns/low trade counts).
5. **Evolution:** The worst strategies are discarded. The best (the "elite") are kept. The rest are "bred" via crossover (swapping rules with another good strategy) and mutation (tweaking thresholds slightly). 
6. **Immigration:** To prevent the system from getting stuck on local maximums, 20% of the population every cycle is replaced with brand-new, totally random strategies.

---

## 2. Genomic Structure (What is a strategy?)

A strategy is represented as a **Genome** (`sim/evolution/genome.py`). It is a self-contained set of rules:

* **Entry Logic:** How the indicators combine. 
  * `AND` (all must be true).
  * `OR` (any can be true).
  * `MEANREV` (betting price will revert to the mean after over-extension).
  * `BREAKOUT` (betting price will continue after breaking a volatility range).
  * `TREND` (verifying alignment across 5m, 15m, and 1h intervals before entering).
  * `TFT` (Using Deep Learning / Temporal Fusion Transformer predictions).
* **Entry Conditions:** The actual thresholds (e.g., `price_roc_1h > 5.0`).
* **Sizing Method:** How much capital to risk (e.g., Fixed, Volatility-Scaled, or Kelly Criterion).
* **Exit Rules:** Time stops (max hours to hold), stop-losses, and profit targets.

---

## 3. The Execution physics (Layer 1)

Most crypto bots fail live because they assume 0% fees and instant execution. The `BacktestEngine` (`sim/layer1/backtest_engine.py`) enforces strict physics:

* **Fees:** Assumes a 2.2 bps taker fee per side (based on live Jupiter measurements).
* **Latency Simulation:** We don't assume we get the exact price of the signal. The `LatencyModel` injects a simulated ~10-second delay. Since we use 1h candles, it typically forces execution at the *start of the next candle*.
* **MEV Front-running:** We inject a probability (e.g. 30%) that a searcher bots us. Every time this triggers, slippage is synthetically magnified.

---

## 4. The Promotion Funnel (The Gauntlet)

If a strategy looks highly profitable during the evolutionary search, it is **not** immediately trusted. The search score is known to reward overfitting. Before a strategy is allowed onto the "Shortlist" (the pre-live paper arena), it must pass the `PromotionFunnel` (`sim/evolution/promotion_funnel.py`):

1. **Feasibility Gate:** Does it trade at least 30 times? (If an edge only fires 4 times a year, we don't have statistical confidence in it).
2. **Locked Out-of-Sample (OOS):** We test it on a chronological "holdout" slice of data that the Genetic Algorithm was never allowed to see.
3. **Purged Walk-Forward:** We slice the history into 3-5 chapters. It must remain profitable in the *majority* of the chapters. If it was only profitable in a massive bull run and loses heavily in a crab market, it is rejected.
4. **Parameter Perturbation:** We take the exact genome and randomly nudge all its thresholds by ±10%. If this breaks the strategy's profitability, it is discarded as "curve-fit" to noise.
5. **Fee Stress:** It must survive if exchange fees suddenly jump from 2.2 bps to 5 bps or 10 bps.
6. **Deflated Sharpe Ratio (DSR):** The more strategies we test (the entire trial history across weeks), the higher the bar becomes to prove the strategy is not just a statistical anomaly caused by multiple-testing.

---

## 5. The Kill Archive 

Evolutionary systems often end up re-discovering the same failed ideas infinitely. To solve this, we implemented the `KillArchive` (`sim/evolution/kill_archive.py`).

* When a strategy fails the strict Promotion Funnel, its architectural signature (the exact indicators and rough threshold buckets) is hashed and saved to `killed_dna.json`.
* During future evolutionary breeding, if a newly mutated strategy matches a signature in the Kill Archive, it is instantly given a fitness of `-400` (Tabu) and discarded without wasting CPU cycles backtesting it.

---

## 6. The Dashboard

The system provides a local, human-readable UI at `http://127.0.0.1:8765` (`sim/dashboard.py`). It is explicitly designed to be skeptical:

* It tracks the median trade-count of winning strategies.
* It plots the average fitness to show if the system is converging on better solutions or just randomly generating noise. 
* It exposes the funnel results so you can see exactly *why* a particular top strategy was rejected at the gate. 
