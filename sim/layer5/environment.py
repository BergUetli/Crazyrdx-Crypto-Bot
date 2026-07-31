"""
environment.py
Gym environment for PPO training.
Simulates trading with latency, fees, and risk constraints.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

from layer1.backtest_engine import LatencyModel


class TradingEnvironment(gym.Env):
    """
    Custom Gym environment for trading.

    State: market features + portfolio state + model predictions
    Action: [trade_size, execution_delay, do_nothing, route_preference]
    Reward: net P&L with risk penalties
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        features: List[Dict[str, Any]],
        initial_capital: float = 100.0,
        fee_rate: float = 0.00022,  # 2.2 bps taker fee (Jupiter measured)
        latency_model: Optional[LatencyModel] = None,
        max_position_size: float = 0.5,
        max_daily_loss: float = 20.0,
        max_consecutive_losses: int = 5,
        forecaster=None,      # TFT model for predictions
        classifier=None,      # Opportunity classifier
        risk_assessor=None,   # BNN risk model
    ):
        super().__init__()

        self.features = features
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.latency_model = latency_model or LatencyModel()
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.max_consecutive_losses = max_consecutive_losses

        # Models (optional, for integrated training)
        self.forecaster = forecaster
        self.classifier = classifier
        self.risk_assessor = risk_assessor

        # State space: market features (38) + portfolio (5) + predictions (10)
        n_market = 38
        n_portfolio = 5   # capital, daily_pnl, consecutive_losses, position, last_pnl
        n_predictions = 10  # forecaster(4) + classifier(4) + risk(2)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(n_market + n_portfolio + n_predictions,),
            dtype=np.float32
        )

        # Action space: [size, delay, do_nothing, route]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, -5.0, 0.0]),
            high=np.array([1.0, 30.0, 5.0, 1.0]),
            dtype=np.float32
        )

        # Episode state
        self.current_step = 0
        self.capital = initial_capital
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.position = 0.0  # current position size
        self.last_pnl = 0.0
        self.trades_today = 0
        self.episode_trades = []

        # Feature names for state construction
        self.feature_names = [
            "price_roc_5m", "price_roc_15m", "price_roc_30m", "price_roc_1h", "price_roc_4h",
            "volatility_15m", "volatility_1h", "volatility_4h", "volatility_1d",
            "hl_range_pct", "hl_range_avg_15m", "hl_range_avg_1h",
            "body_pct", "upper_wick_pct", "lower_wick_pct",
            "volume_roc_15m", "volume_roc_1h", "volume_sma_ratio", "volume_weighted_price",
            "sma_5", "sma_20", "sma_50", "price_vs_sma_20", "sma_cross_5_20", "sma_cross_20_50",
            "returns_skew_1h", "returns_kurtosis_1h", "autocorrelation_1h",
            "hour_of_day_sin", "hour_of_day_cos", "day_of_week", "is_weekend",
            "close_lag_1", "close_lag_2", "close_lag_3", "volume_lag_1",
            "returns_lag_1", "returns_lag_2",
        ]

    def _get_market_state(self) -> np.ndarray:
        """Get market features for current step."""
        if self.current_step >= len(self.features):
            return np.zeros(38, dtype=np.float32)

        f = self.features[self.current_step]["features"]
        return np.array([f.get(name, 0.0) for name in self.feature_names], dtype=np.float32)

    def _get_portfolio_state(self) -> np.ndarray:
        """Get portfolio state."""
        return np.array([
            self.capital / self.initial_capital,  # normalized
            self.daily_pnl / self.initial_capital,
            self.consecutive_losses / 10.0,  # normalized
            self.position,
            self.last_pnl / self.initial_capital,
        ], dtype=np.float32)

    def _get_predictions(self) -> np.ndarray:
        """Get model predictions for current state."""
        if self.forecaster is None or self.classifier is None or self.risk_assessor is None:
            return np.zeros(10, dtype=np.float32)

        # Prepare input
        market = self._get_market_state()
        x = torch.FloatTensor(market).unsqueeze(0).unsqueeze(0)  # (1, 1, 38)

        # Forecaster
        with torch.no_grad():
            fc_out = self.forecaster(x)
            fc_pred = fc_out["quantiles"][0, -1, :].cpu().numpy()  # last timestep, 3 quantiles
            fc_cls = fc_out["classification"][0, -1].cpu().numpy()  # last timestep

        # Classifier
        with torch.no_grad():
            cls_out = self.classifier(x)
            cls_probs = torch.softmax(cls_out["classification"], dim=-1)[0].cpu().numpy()
            exploit = cls_out["exploitability"][0].cpu().numpy()

        # Risk assessor
        market_flat = self._get_market_state()
        x_flat = torch.FloatTensor(market_flat).unsqueeze(0)  # (1, 38)
        with torch.no_grad():
            risk_out = self.risk_assessor(x_flat, sample=False)
            adverse = risk_out["adverse_prob"][0].cpu().numpy()
            shortfall = risk_out["shortfall"][0].cpu().numpy()

        return np.concatenate([
            fc_pred,           # 3
            [fc_cls],          # 1
            cls_probs,         # 4
            [exploit],         # 1
            [adverse, shortfall]  # 2
        ]).astype(np.float32)

    def _get_state(self) -> np.ndarray:
        """Get full state vector."""
        market = self._get_market_state()
        portfolio = self._get_portfolio_state()
        predictions = self._get_predictions()
        return np.concatenate([market, portfolio, predictions])

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment to initial state."""
        super().reset(seed=seed)

        self.current_step = 50  # Start after warmup period
        self.capital = self.initial_capital
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.position = 0.0
        self.last_pnl = 0.0
        self.trades_today = 0
        self.episode_trades = []

        return self._get_state(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step."""
        # Parse action
        size_fraction = float(np.clip(action[0], 0.0, self.max_position_size))
        delay_s = float(np.clip(action[1], 0.0, 30.0))
        do_nothing = float(action[2]) > 0.0
        route_pref = float(np.clip(action[3], 0.0, 1.0))

        # Check safety constraints
        if self.daily_pnl < -self.max_daily_loss:
            return self._get_state(), -1.0, True, False, {
                "reason": "daily_loss_limit",
                "capital": self.capital,
                "daily_pnl": self.daily_pnl,
                "trades": self.trades_today,
            }

        if self.consecutive_losses >= self.max_consecutive_losses:
            return self._get_state(), -0.5, True, False, {
                "reason": "consecutive_losses",
                "capital": self.capital,
                "daily_pnl": self.daily_pnl,
                "trades": self.trades_today,
            }

        # Get current market state
        current = self.features[self.current_step]["features"]
        current_price = current["close"]

        # Decision: trade or skip
        reward = 0.0
        trade_executed = False

        if not do_nothing and size_fraction > 0.01:
            # Execute trade
            size_usd = self.capital * size_fraction

            if size_usd >= 1.0:  # minimum trade size
                # Simulate execution with latency
                future = self.features[self.current_step+1:min(self.current_step+20, len(self.features))]
                if len(future) > 0:
                    exec_price, slippage, mev_cost = self.latency_model.get_execution_price(
                        current_price, future, "long"  # simplified: always long for now
                    )

                    # Exit after hold period (1 hour = 12 candles)
                    exit_idx = min(self.current_step + 13, len(self.features) - 1)
                    exit_price = self.features[exit_idx]["features"]["close"]

                    # Calculate P&L
                    gross_pnl = (exit_price - exec_price) / exec_price * size_usd
                    fee_cost = size_usd * self.fee_rate * 2
                    net_pnl = gross_pnl - fee_cost

                    # Update state
                    self.capital += net_pnl
                    self.daily_pnl += net_pnl
                    self.last_pnl = net_pnl
                    self.trades_today += 1
                    trade_executed = True

                    if net_pnl < 0:
                        self.consecutive_losses += 1
                    else:
                        self.consecutive_losses = 0

                    # Reward
                    reward = net_pnl / self.initial_capital  # normalized

                    # Bonus for profitable trades
                    if net_pnl > 0:
                        reward += 0.01

                    self.episode_trades.append({
                        "step": self.current_step,
                        "size": size_usd,
                        "pnl": net_pnl,
                        "slippage": slippage,
                        "mev_cost": mev_cost,
                    })

        # Penalty for doing nothing when there's a clear opportunity
        if do_nothing and self._has_opportunity():
            reward -= 0.001

        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.features) - 20

        # Episode end bonus/penalty
        if done:
            total_return = (self.capital - self.initial_capital) / self.initial_capital
            reward += total_return * 0.1  # small bonus for overall return

        return self._get_state(), reward, done, False, {
            "capital": self.capital,
            "daily_pnl": self.daily_pnl,
            "trades": self.trades_today,
            "trade_executed": trade_executed,
        }

    def _has_opportunity(self) -> bool:
        """Check if current state has a clear opportunity."""
        if self.current_step >= len(self.features):
            return False

        f = self.features[self.current_step]["features"]
        # Simple heuristic: high volatility + strong momentum
        return abs(f.get("price_roc_15m", 0)) > 30 and f.get("volatility_15m", 0) > 50

    def render(self, mode="human"):
        """Render current state."""
        if mode == "human":
            print(f"Step: {self.current_step} | Capital: {self.capital:.2f} | "
                  f"Daily P&L: {self.daily_pnl:.4f} | Trades: {self.trades_today}")


import torch  # noqa: E402
