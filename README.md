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

## 🛡️ The Promotion Gauntlet
If the GA finds a strategy that scores well, it enters the `PromotionFunnel`. A strategy only hits the `Shortlist` if it passes:
1. **Feasibility:** No "RANDOM" baselines; must have a solid N-count.
2. **Locked OOS:** It performs equally well on hold-out data it was not trained on.
3. **Walk-Forward:** Consistent profitability across sliced chapters of market history.
4. **Fee Stress:** Must survive execution costs being doubled or quadrupled.
5. **MEV Stress:** Must remain profitable even if front-run 30% of the time.

## ⚠️ Disclaimer
This code is experimental and intended for research/educational purposes. The financial markets, and crypto DEXs in particular, are extremely volatile. **Never deploy live capital without modifying the execution layer and fully understanding the risks.**