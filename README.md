# Crazyrdx Crypto Bot 📈

An evolutionary, simulation-first Solana DEX algorithmic trading bot designed to discover, validate, and execute short-term crypto trading strategies. 

This is not a high-frequency trading (HFT) "sniper" bot. It focuses on latency-tolerant, seconds-to-minutes market inefficiencies on DEXs (like Jupiter), taking into account real-world slippage, fees, and MEV front-running probabilities.

## 🧠 Core Philosophy
- **Simulation-First:** Strategies must survive a brutal gauntlet of offline paper trading before ever touching live money.
- **Fund-Grade Funnel:** Discovered strategies must pass walk-forward testing (chronological splits), Monte Carlo shuffle tests, fee stress tests, and parameter stability windows.
- **Evolutionary Discovery:** It uses a Genetic Algorithm (GA) to "breed" strategies (AND/OR trees, Breakouts, Mean Reversions, and ML-assisted signals), selecting for robust edges rather than curve-fit historical flukes.
- **No Lottery Tickets:** Strategies are discarded if they don't produce enough trades to be statistically significant (N ≥ 30).

## 🏗️ Architecture

The bot is broken into clear layers:

* **Layer 1: Data & Physics** (`sim/layer1/`)
  - Historical candle downloaders (Binance).
  - Feature engine (momentum, volatility, multi-timeframe regime alignment).
  - `BacktestEngine`: The execution simulator that models 2.2 bps fees, latency, slippage, and MEV stress perfectly.

* **Layer 2: AI & Models** (`sim/layer2/`)
  - Deep Learning hooks (TFT - Temporal Fusion Transformers) providing directional probabilities.

* **Evolution Engine** (`sim/evolution/`)
  - `evaluator.py`: Scores "genomes" (strategies).
  - `promotion_funnel.py`: The strict post-discovery exam. Strategies that pass become *Champions*.
  - `kill_archive.py`: Remembers structural failures so the GA doesn't waste time re-testing known-bad ideas.

* **Dashboard** (`sim/dashboard.py` & `sim/run_broad_evolution.py`)
  - Local HTTP server updating real-time on search progress, showing the median score of strategies, trade frequencies, and paper-trade metrics.
  - "Is the bot getting smarter?" section: tracks how many strict-exam gates each search's best idea clears over time (the learning curve), which gate is the current wall, and kill-archive memory growth.

## ⚡ Engine capabilities (added 2026-08-01)

* **Parallel search:** genome evaluation runs across all CPU cores (persistent worker pool, ~3.5× throughput). Control with `EVOLUTION_WORKERS` (set `1` to force serial).
* **Data-calibrated thresholds:** entry-condition thresholds are sampled from the 5th–95th percentile of each indicator's real distribution, recalibrated every cycle.
* **Evaluation cache:** structurally identical strategies are backtested once, not repeatedly.
* **Champion warm-starting:** each cycle seeds part of its population from past champions instead of restarting from pure noise.
* **Test suite:** `python3 sim/test_improvements.py` verifies the engine end-to-end on synthetic data (no market data or API needed).
* **Vintage forward ledger:** every cycle's champion is frozen and later scored only on candles that arrived after it was created, as a skill percentile vs random strategies frozen the same day. This is the unfakeable "is it getting smarter?" curve (dashboard Chart D).

See [CHANGELOG.md](CHANGELOG.md) for the full list of fixes and improvements.

## 🚀 Quick Start (Local Run)

The primary mode of this bot is offline, meaning *you pay zero API costs* for the ongoing search process.

**1. Create a virtual environment & install requirements:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Start the Broad Evolution Search:**
This runs the continuous genetic algorithm to find and refine edges.
```bash
python3 sim/run_broad_evolution.py
```

**3. View the Dashboard:**
Open a new terminal tab and start the dashboard to monitor progress.
```bash
# To view it on this Mac:
python3 sim/dashboard.py 
# Then visit http://127.0.0.1:8765 in your browser

# To view it from another computer on the same WiFi (e.g. Windows PC):
python3 sim/dashboard.py --lan
# Then visit http://<your-macs-ip>:8765
```
*Note: To find your Mac's IP address, run `ipconfig getifaddr en0` in the terminal. Your home WiFi router acts as a firewall, so this dashboard is securely visible to other devices on your home network, but completely hidden from the public internet and your neighbors.*

## 🛡️ The Promotion Gauntlet (LAB stage)
If the GA finds a strategy that scores well, it enters the `PromotionFunnel`. A strategy only hits the **LAB shortlist** if it passes:
1. **Feasibility:** No "RANDOM" baselines; ≥30 trades; positive PnL; drawdown ≤20% on a $100 book.
2. **Locked OOS:** Profitable on hold-out data it was not ranked on (OOS ≥ ~50% of IS when IS > 0).
3. **Walk-Forward:** Consistent profitability across chronological chapters.
4. **Fee Stress:** Still profitable at 5 bps (and checked at 10 bps).
5. **MEV Stress:** Expected front-run drag; must not break even only under zero MEV.
6. **Parameter shake:** ±10% threshold nudges still mostly profitable.

### What counts as a win on $100?
Encoded in `sim/success_criteria.py` (single source of truth):

| Stage | Bar | Meaning |
|-------|-----|---------|
| **LAB** | ≥30 trades, OOS profit, DD ≤20%, pass funnel | Interesting on history. Not money. |
| **PAPER** | ≥30 days, ≥20 trades, ≥ +$5 net, DD ≤20% | **First real success** before live SOL. |
| **LIVE** | Start ~$25; kill at 20% DD or 5 loss-days | Only after paper pass. |

**Single target:** after 30 days paper-live, net ≥ +$5, max drawdown ≤ $20, ≥ 20 trades, fees included.

**Not a win:** search score alone, high win-rate with tiny $, N<30, one funnel pass without forward paper, clone spam.

## ⚠️ Disclaimer
This code is experimental and intended for research/educational purposes. The financial markets, and crypto DEXs in particular, are extremely volatile. **Never deploy live capital without modifying the execution layer and fully understanding the risks.**