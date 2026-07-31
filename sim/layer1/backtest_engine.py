"""
backtest_engine.py
Backtesting engine with realistic latency and slippage modeling.
"""

import json
import math
import sqlite3
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from config import DATA_DIR
from layer1.historical_feature_engine import get_historical_features, DB_HIST_FEATURES

DB_BACKTEST = DATA_DIR / "backtest_results.db"


def init_backtest_db():
    conn = sqlite3.connect(str(DB_BACKTEST))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS backtest_runs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_ts          REAL NOT NULL,
            strategy_name   TEXT NOT NULL,
            pair            TEXT NOT NULL,
            start_ts        INTEGER NOT NULL,
            end_ts          INTEGER NOT NULL,
            latency_s       REAL NOT NULL,
            initial_capital REAL NOT NULL,
            final_capital   REAL NOT NULL,
            total_trades    INTEGER NOT NULL,
            winning_trades  INTEGER NOT NULL,
            total_pnl       REAL NOT NULL,
            sharpe_ratio    REAL,
            max_drawdown    REAL,
            win_rate        REAL,
            profit_factor   REAL,
            params_json     TEXT,
            created_at      REAL DEFAULT (strftime('%s','now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS backtest_trades (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          INTEGER NOT NULL,
            entry_ts        INTEGER NOT NULL,
            exit_ts         INTEGER NOT NULL,
            pair            TEXT NOT NULL,
            direction       TEXT NOT NULL,
            entry_price     REAL NOT NULL,
            exit_price      REAL NOT NULL,
            size_usd        REAL NOT NULL,
            gross_pnl       REAL NOT NULL,
            fee_cost        REAL NOT NULL,
            net_pnl         REAL NOT NULL,
            latency_s       REAL NOT NULL,
            slippage_bps    REAL NOT NULL,
            mev_cost_bps    REAL NOT NULL,
            signal_strength REAL,
            created_at      REAL DEFAULT (strftime('%s','now')),
            FOREIGN KEY (run_id) REFERENCES backtest_runs(id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_bt_runs_strategy ON backtest_runs(strategy_name)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_bt_trades_run ON backtest_trades(run_id)")
    conn.commit()
    conn.close()


@dataclass
class SimulatedTrade:
    entry_ts: int
    exit_ts: int
    pair: str
    direction: str           # "long" or "short"
    entry_price: float
    exit_price: float
    size_usd: float
    gross_pnl: float
    fee_cost: float
    net_pnl: float
    latency_s: float
    slippage_bps: float
    mev_cost_bps: float
    signal_strength: float


@dataclass
class BacktestResult:
    strategy_name: str
    pair: str
    start_ts: int
    end_ts: int
    latency_s: float
    initial_capital: float
    final_capital: float
    total_trades: int
    winning_trades: int
    total_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    trades: List[SimulatedTrade]


class LatencyModel:
    """
    Models execution latency and its impact on price.
    """

    def __init__(
        self,
        base_latency_s: float = 10.0,
        latency_std_s: float = 3.0,
        mev_probability: float = 0.3,
        mev_cost_bps: float = 15.0,
    ):
        self.base_latency = base_latency_s
        self.latency_std = latency_std_s
        self.mev_probability = mev_probability
        self.mev_cost_bps = mev_cost_bps

    def sample_latency(self) -> float:
        """Sample actual latency from distribution."""
        return max(1.0, np.random.normal(self.base_latency, self.latency_std))

    def get_execution_price(
        self,
        signal_price: float,
        future_candles: List[Dict[str, Any]],
        direction: str,
        signal_ts: Optional[int] = None,
    ) -> Tuple[float, float, float]:
        """
        Compute realistic execution price given latency.

        For bars much longer than latency (e.g. 1h bar, 10s latency), fill at
        the next candle close (deterministic, fast). Sub-bar latency walk only
        when future bars are fine-grained enough to matter.

        Returns:
            execution_price, slippage_bps, mev_cost_bps
        """
        if not future_candles:
            return signal_price, 0.0, 0.0

        latency = self.sample_latency()
        first = future_candles[0]
        first_close = (
            first["features"]["close"] if "features" in first else first.get("close", signal_price)
        )

        # Fast path: next-bar fill when latency << bar length
        try:
            if signal_ts is not None:
                bar_ms = first["ts"] - signal_ts
            else:
                bar_ms = 3_600_000  # assume 1h if unknown
            if bar_ms <= 0:
                bar_ms = 3_600_000
            if latency * 1000 < 0.25 * bar_ms:
                execution_price = float(first_close)
            else:
                target_ts = (signal_ts if signal_ts is not None else first["ts"]) + int(
                    latency * 1000
                )
                execution_price = float(signal_price)
                for candle in future_candles:
                    if candle["ts"] >= target_ts:
                        if "features" in candle:
                            execution_price = float(candle["features"]["close"])
                        else:
                            execution_price = float(candle["close"])
                        break
                else:
                    execution_price = float(first_close)
        except Exception:
            execution_price = float(first_close)

        if direction == "long":
            slippage_bps = (execution_price - signal_price) / max(signal_price, 1e-12) * 10000
        else:
            slippage_bps = (signal_price - execution_price) / max(signal_price, 1e-12) * 10000

        mev_cost = 0.0
        if np.random.random() < self.mev_probability:
            mev_cost = self.mev_cost_bps
            if direction == "long":
                execution_price *= 1 + mev_cost / 10000
            else:
                execution_price *= 1 - mev_cost / 10000

        return execution_price, slippage_bps, mev_cost


class BacktestEngine:
    """
    Walk-forward backtesting engine with latency modeling.
    """

    def __init__(
        self,
        initial_capital: float = 100.0,
        fee_rate: float = 0.00022,      # 2.2 bps taker fee per side (Jupiter measured)
        latency_model: Optional[LatencyModel] = None,
        max_position_size: float = 0.5,    # max 50% of capital per trade
        min_signal_strength: float = 0.5,
    ):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.latency_model = latency_model or LatencyModel()
        self.max_position_size = max_position_size
        self.min_signal_strength = min_signal_strength

    def run_backtest(
        self,
        strategy_name: str,
        pair: str,
        features: List[Dict[str, Any]],
        signal_generator,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
        exit_rules: Optional[List[Any]] = None,
    ) -> BacktestResult:
        """
        Run backtest on a sequence of features.

        Args:
            strategy_name: name for this run
            pair: trading pair
            features: list of feature dicts with 'ts' and 'features'
            signal_generator: callable(features, idx) -> (direction, strength, size_fraction) or None
            start_idx: start index in features
            end_idx: end index (default: len(features))
            exit_rules: optional list of ExitRule-like objects
                        (exit_type, value). When omitted, uses a modest
                        default time stop (not a fake fixed 15h forever).
        """
        if end_idx is None:
            end_idx = len(features)

        capital = self.initial_capital
        trades: List[SimulatedTrade] = []
        equity_curve = [capital]
        bar_hours = self._infer_bar_hours(features, start_idx, end_idx)
        rules = self._normalize_exit_rules(exit_rules, bar_hours=bar_hours)

        i = start_idx
        while i < end_idx - 1:
            signal = signal_generator(features, i)

            if signal is None:
                i += 1
                continue

            direction, strength, size_fraction = signal

            if strength < self.min_signal_strength:
                i += 1
                continue

            # Position sizing
            size_usd = capital * min(size_fraction, self.max_position_size)
            if size_usd < 1.0:  # minimum $1 trade
                i += 1
                continue

            current = features[i]
            entry_price = current["features"]["close"]

            # Get future candles for latency simulation
            future = features[i+1:min(i+20, len(features))]  # next 20 candles
            if not future:
                break

            # Simulate execution with latency (next-bar fast path on coarse candles)
            exec_price, slippage, mev_cost = self.latency_model.get_execution_price(
                entry_price, future, direction, signal_ts=current["ts"]
            )

            # Genome-driven exit (TP/SL/time/trail/reversal). No hardcoded 15h.
            exit_idx, exit_price = self._resolve_exit(
                features=features,
                entry_idx=i,
                end_idx=end_idx,
                direction=direction,
                exec_price=exec_price,
                rules=rules,
                signal_generator=signal_generator,
                bar_hours=bar_hours,
            )

            # Calculate P&L
            if direction == "long":
                gross_pnl = (exit_price - exec_price) / exec_price * size_usd
            else:
                gross_pnl = (exec_price - exit_price) / exec_price * size_usd

            # Fees
            fee_cost = size_usd * self.fee_rate * 2  # entry + exit

            # Net P&L
            net_pnl = gross_pnl - fee_cost

            # MEV already accounted in execution price

            trade = SimulatedTrade(
                entry_ts=current["ts"],
                exit_ts=features[exit_idx]["ts"],
                pair=pair,
                direction=direction,
                entry_price=exec_price,
                exit_price=exit_price,
                size_usd=size_usd,
                gross_pnl=gross_pnl,
                fee_cost=fee_cost,
                net_pnl=net_pnl,
                latency_s=self.latency_model.base_latency,
                slippage_bps=slippage,
                mev_cost_bps=mev_cost,
                signal_strength=strength,
            )
            trades.append(trade)

            capital += net_pnl
            equity_curve.append(capital)

            # Move to next opportunity (skip ahead to avoid overlapping trades)
            i = exit_idx + 1

        # Compute metrics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.net_pnl > 0)
        total_pnl = sum(t.net_pnl for t in trades)

        # Sharpe from trade-to-trade equity returns (bar-size independent enough for ranking)
        if len(equity_curve) > 1:
            returns = [(equity_curve[j] - equity_curve[j-1]) / equity_curve[j-1]
                       for j in range(1, len(equity_curve))]
            if returns:
                mean_ret = np.mean(returns)
                std_ret = np.std(returns)
                if std_ret > 0:
                    # Annualize: 105120 5-minute periods per year
                    sharpe = (mean_ret / std_ret) * math.sqrt(105120)
                else:
                    sharpe = 0.0
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        # Max drawdown
        peak = equity_curve[0]
        max_dd = 0.0
        for val in equity_curve:
            if val > peak:
                peak = val
            dd = (peak - val) / peak
            if dd > max_dd:
                max_dd = dd

        # Win rate
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        # Profit factor
        gross_profit = sum(t.net_pnl for t in trades if t.net_pnl > 0)
        gross_loss = abs(sum(t.net_pnl for t in trades if t.net_pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        return BacktestResult(
            strategy_name=strategy_name,
            pair=pair,
            start_ts=features[start_idx]["ts"],
            end_ts=features[end_idx - 1]["ts"],
            latency_s=self.latency_model.base_latency,
            initial_capital=self.initial_capital,
            final_capital=capital,
            total_trades=total_trades,
            winning_trades=winning_trades,
            total_pnl=total_pnl,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            trades=trades,
        )

    # ------------------------------------------------------------------
    # Exit helpers (genome-driven; no hardcoded 15h hold)
    # ------------------------------------------------------------------

    def _infer_bar_hours(
        self,
        features: List[Dict[str, Any]],
        start_idx: int,
        end_idx: int,
    ) -> float:
        """Infer candle length in hours from timestamps (default 1h)."""
        try:
            if end_idx - start_idx >= 2:
                dt_ms = features[start_idx + 1]["ts"] - features[start_idx]["ts"]
                if dt_ms > 0:
                    return max(dt_ms / 3_600_000.0, 1.0 / 60.0)
        except Exception:
            pass
        return 1.0

    def _normalize_exit_rules(
        self,
        exit_rules: Optional[List[Any]],
        bar_hours: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Convert ExitRule objects / dicts into a uniform rule list.

        Supported types:
          - profit_target: value as fraction (0.01 = 1%)
          - stop_loss: value as fraction
          - trailing_stop: value as fraction from peak
          - time_stop: value as bars (preferred) or hours if large
          - signal_reversal: value ignored; exit when opposite signal fires
        """
        rules: List[Dict[str, Any]] = []
        if not exit_rules:
            # Default: modest 4-bar time stop (4h on 1h candles), 1.5% SL
            return [
                {"type": "time_stop", "value": 4.0},
                {"type": "stop_loss", "value": 0.015},
            ]

        for er in exit_rules:
            if hasattr(er, "exit_type"):
                et = str(getattr(er, "exit_type", "") or "")
                val = float(getattr(er, "value", 0) or 0)
            elif isinstance(er, dict):
                et = str(er.get("exit_type") or er.get("type") or "")
                val = float(er.get("value", 0) or 0)
            else:
                continue

            et = et.strip().lower()
            if et == "time_stop":
                # Heuristic: values >= 24 treat as hours; convert to bars
                if val >= 24:
                    bars = max(1.0, round(val / max(bar_hours, 1e-6)))
                elif val > 12 and bar_hours >= 0.9:
                    # ambiguous midrange on 1h data: treat as hours
                    bars = max(1.0, round(val / max(bar_hours, 1e-6)))
                else:
                    bars = max(1.0, round(val))
                # Safety clamp so evolution cannot freeze for hundreds of bars
                bars = float(min(max(bars, 1.0), 168.0))
                rules.append({"type": "time_stop", "value": bars})
            elif et in ("profit_target", "stop_loss", "trailing_stop"):
                # Clamp to sane % band
                v = abs(val)
                if v > 0.25:
                    # maybe bps or percent points entered wrong; try /100
                    if v > 1:
                        v = v / 100.0
                v = float(min(max(v, 0.001), 0.25))
                rules.append({"type": et, "value": v})
            elif et == "signal_reversal":
                rules.append({"type": "signal_reversal", "value": float(val or 0.5)})
            else:
                continue

        if not rules:
            rules = [{"type": "time_stop", "value": 4.0}, {"type": "stop_loss", "value": 0.015}]
        # Always ensure a maximum hold so trades cannot run forever
        if not any(r["type"] == "time_stop" for r in rules):
            rules.append({"type": "time_stop", "value": 24.0})
        return rules

    def _resolve_exit(
        self,
        features: List[Dict[str, Any]],
        entry_idx: int,
        end_idx: int,
        direction: str,
        exec_price: float,
        rules: List[Dict[str, Any]],
        signal_generator,
        bar_hours: float = 1.0,
    ) -> Tuple[int, float]:
        """Walk forward bar-by-bar until an exit rule fires."""
        # Start checking from next bar after entry path settles
        start = min(entry_idx + 1, end_idx - 1)
        last = end_idx - 1
        if start > last:
            return last, features[last]["features"]["close"]

        # max hold from time_stop (largest if multiple)
        time_limits = [int(r["value"]) for r in rules if r["type"] == "time_stop"]
        max_hold = max(time_limits) if time_limits else 24
        max_hold = max(1, min(max_hold, last - entry_idx))

        tp = next((r["value"] for r in rules if r["type"] == "profit_target"), None)
        sl = next((r["value"] for r in rules if r["type"] == "stop_loss"), None)
        trail = next((r["value"] for r in rules if r["type"] == "trailing_stop"), None)
        use_reversal = any(r["type"] == "signal_reversal" for r in rules)

        best_price = exec_price  # peak for longs / trough for shorts
        exit_idx = min(entry_idx + max_hold, last)
        exit_price = features[exit_idx]["features"]["close"]

        for j in range(start, min(entry_idx + max_hold, last) + 1):
            px = features[j]["features"]["close"]
            hi = features[j]["features"].get("high", px)
            lo = features[j]["features"].get("low", px)

            if direction == "long":
                best_price = max(best_price, hi)
                # stop loss
                if sl is not None and lo <= exec_price * (1.0 - sl):
                    return j, exec_price * (1.0 - sl)
                # take profit
                if tp is not None and hi >= exec_price * (1.0 + tp):
                    return j, exec_price * (1.0 + tp)
                # trailing stop from peak
                if trail is not None and lo <= best_price * (1.0 - trail):
                    return j, best_price * (1.0 - trail)
            else:  # short
                best_price = min(best_price, lo)
                if sl is not None and hi >= exec_price * (1.0 + sl):
                    return j, exec_price * (1.0 + sl)
                if tp is not None and lo <= exec_price * (1.0 - tp):
                    return j, exec_price * (1.0 - tp)
                if trail is not None and hi >= best_price * (1.0 + trail):
                    return j, best_price * (1.0 + trail)

            # opposite signal
            if use_reversal and j > start:
                try:
                    sig = signal_generator(features, j)
                except Exception:
                    sig = None
                if sig is not None:
                    opp_dir = sig[0]
                    if opp_dir and opp_dir != direction:
                        return j, px

            exit_idx = j
            exit_price = px

        return exit_idx, exit_price

    def save_result(self, result: BacktestResult) -> int:
        """Save backtest result to DB."""
        init_backtest_db()
        conn = sqlite3.connect(str(DB_BACKTEST))

        cursor = conn.execute("""
            INSERT INTO backtest_runs
            (run_ts, strategy_name, pair, start_ts, end_ts, latency_s,
             initial_capital, final_capital, total_trades, winning_trades,
             total_pnl, sharpe_ratio, max_drawdown, win_rate, profit_factor, params_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            time.time(), result.strategy_name, result.pair,
            result.start_ts, result.end_ts, result.latency_s,
            result.initial_capital, result.final_capital,
            result.total_trades, result.winning_trades,
            result.total_pnl, result.sharpe_ratio,
            result.max_drawdown, result.win_rate, result.profit_factor,
            json.dumps({"latency_model": {
                "base_latency_s": self.latency_model.base_latency,
                "mev_probability": self.latency_model.mev_probability,
                "mev_cost_bps": self.latency_model.mev_cost_bps,
            }})
        ))

        run_id = cursor.lastrowid

        for trade in result.trades:
            conn.execute("""
                INSERT INTO backtest_trades
                (run_id, entry_ts, exit_ts, pair, direction, entry_price, exit_price,
                 size_usd, gross_pnl, fee_cost, net_pnl, latency_s, slippage_bps,
                 mev_cost_bps, signal_strength)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id, trade.entry_ts, trade.exit_ts, trade.pair, trade.direction,
                trade.entry_price, trade.exit_price, trade.size_usd,
                trade.gross_pnl, trade.fee_cost, trade.net_pnl,
                trade.latency_s, trade.slippage_bps, trade.mev_cost_bps,
                trade.signal_strength
            ))

        conn.commit()
        conn.close()
        return run_id


def walk_forward_split(
    features: List[Dict[str, Any]],
    n_splits: int = 5,
) -> List[Tuple[int, int, int, int]]:
    """
    Generate walk-forward train/test splits.

    Returns:
        List of (train_start, train_end, test_start, test_end) tuples
    """
    n = len(features)
    test_size = n // (n_splits + 1)

    splits = []
    for i in range(n_splits):
        train_start = 0
        train_end = test_size * (i + 2)
        test_start = train_end
        test_end = min(test_start + test_size, n)

        if test_end <= test_start:
            break

        splits.append((train_start, train_end, test_start, test_end))

    return splits
