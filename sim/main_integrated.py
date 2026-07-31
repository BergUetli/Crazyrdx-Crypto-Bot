#!/usr/bin/env python3
"""
main_integrated.py
Full pipeline orchestrator for the trading bot.
Runs all layers in sequence: data -> features -> predictions -> execution -> reporting.
"""

import asyncio
import json
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch

# Add sim dir to path
SIM_DIR = Path.home() / ".hermes" / "trading-bot" / "sim"
sys.path.insert(0, str(SIM_DIR))

from config import LOG_DIR, MODEL_DIR, DATA_DIR
from layer1.historical_downloader import download_all_pairs
from layer1.historical_feature_engine import compute_all_features, get_historical_features
from layer1.backtest_engine import BacktestEngine, LatencyModel
from layer2.tft_trainer import TFTTrainer
from layer3.classifier import OpportunityClassifier
from layer4.bnn_model import BayesianRiskAssessor
from layer5.environment import TradingEnvironment
from layer6.meta_controller import MetaController, LayerMetrics
from cost_tracker.compute_meter import meter, track_cpu
from cost_tracker.cost_model import compute_cost_chf

# Graceful shutdown
_shutdown = False


def _signal_handler(signum, frame):
    global _shutdown
    print(f"\n[{datetime.now()}] Shutdown signal received")
    _shutdown = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


class IntegratedOrchestrator:
    """
    Coordinates all layers of the trading system.
    """

    def __init__(self):
        self.meta_controller = MetaController()
        self.tft_trainer: Optional[TFTTrainer] = None
        self.classifier: Optional[OpportunityClassifier] = None
        self.risk_assessor: Optional[BayesianRiskAssessor] = None
        self.backtest_engine: Optional[BacktestEngine] = None
        self.models_loaded = False

        self.stats = {
            "cycles": 0,
            "predictions_made": 0,
            "trades_executed": 0,
            "alerts_triggered": 0,
            "start_time": time.time(),
        }

    def load_models(self):
        """Load all trained models."""
        print(f"[{datetime.now()}] Loading models...")

        # TFT
        tft_path = MODEL_DIR / "tft_final.pt"
        if tft_path.exists():
            self.tft_trainer = TFTTrainer(
                input_dim=38, hidden_dim=96, seq_len=24, forecast_horizon=12
            )
            self.tft_trainer.load_checkpoint("final")
            print(f"  TFT loaded from {tft_path}")
        else:
            print(f"  TFT not found at {tft_path}")

        # Classifier
        cls_path = MODEL_DIR / "classifier_final.pt"
        if cls_path.exists():
            self.classifier = OpportunityClassifier(input_dim=38, hidden_dim=128)
            checkpoint = torch.load(cls_path, map_location="cpu")
            self.classifier.load_state_dict(checkpoint["model_state_dict"])
            self.classifier.eval()
            print(f"  Classifier loaded from {cls_path}")
        else:
            print(f"  Classifier not found at {cls_path}")

        # Risk assessor
        risk_path = MODEL_DIR / "risk_final.pt"
        if risk_path.exists():
            self.risk_assessor = BayesianRiskAssessor(input_dim=38, hidden_dim=64)
            checkpoint = torch.load(risk_path, map_location="cpu")
            self.risk_assessor.load_state_dict(checkpoint["model_state_dict"])
            self.risk_assessor.eval()
            print(f"  Risk assessor loaded from {risk_path}")
        else:
            print(f"  Risk assessor not found at {risk_path}")

        self.models_loaded = (
            self.tft_trainer is not None and
            self.classifier is not None and
            self.risk_assessor is not None
        )

        if self.models_loaded:
            print(f"[{datetime.now()}] All models loaded successfully")
        else:
            print(f"[{datetime.now()}] WARNING: Some models missing, running in limited mode")

    def run_prediction_cycle(self, features: list) -> Dict[str, Any]:
        """Run one prediction cycle on new data."""
        if not self.models_loaded:
            return {"error": "models_not_loaded"}

        # Prepare input
        market_features = np.array([
            [f["features"].get(name, 0.0) for name in [
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
            ]]
            for f in features[-24:]  # last 24 candles
        ], dtype=np.float32)

        if len(market_features) < 24:
            return {"error": "insufficient_data"}

        x = torch.FloatTensor(market_features).unsqueeze(0)  # (1, 24, 38)

        # TFT prediction
        with torch.no_grad():
            tft_out = self.tft_trainer.model(x)
            tft_pred = tft_out["quantiles"][0, -1, :].cpu().numpy()  # last timestep
            tft_cls = tft_out["classification"][0, -1].cpu().numpy()

        # Classifier prediction
        with torch.no_grad():
            cls_out = self.classifier(x)
            cls_probs = torch.softmax(cls_out["classification"], dim=-1)[0].cpu().numpy()
            cls_exploit = cls_out["exploitability"][0].cpu().numpy()
            cls_duration = cls_out["duration"][0].cpu().numpy()
            cls_size = cls_out["size"][0].cpu().numpy()

        # Risk assessment
        x_flat = torch.FloatTensor(market_features[-1]).unsqueeze(0)  # (1, 38)
        with torch.no_grad():
            risk_out = self.risk_assessor(x_flat, sample=False)
            adverse_prob = risk_out["adverse_prob"][0].cpu().numpy()
            shortfall = risk_out["shortfall"][0].cpu().numpy()
            drawdown = risk_out["drawdown"][0].cpu().numpy()

        # Combine predictions
        prediction = {
            "timestamp": time.time(),
            "tft": {
                "price_pred": float(tft_pred[1]),  # P50
                "price_pred_lower": float(tft_pred[0]),  # P10
                "price_pred_upper": float(tft_pred[2]),  # P90
                "direction_prob": float(tft_cls),
            },
            "classifier": {
                "class_probs": cls_probs.tolist(),
                "class_idx": int(np.argmax(cls_probs)),
                "class_name": ["NOISE", "TRANSIENT", "EXPLOITABLE", "HIGH_VALUE"][int(np.argmax(cls_probs))],
                "exploitability": float(cls_exploit),
                "duration_s": float(cls_duration),
                "size_fraction": float(cls_size),
            },
            "risk": {
                "adverse_prob": float(adverse_prob),
                "shortfall": float(shortfall),
                "drawdown": float(drawdown),
            },
        }

        self.stats["predictions_made"] += 1

        # Log to meta-controller
        self.meta_controller.log_metrics(LayerMetrics(
            layer_name="tft",
            timestamp=time.time(),
            custom_metrics={
                "direction_prob": float(tft_cls),
                "pred_range": float(tft_pred[2] - tft_pred[0]),
            }
        ))

        return prediction

    def run_backtest_cycle(self, features: list) -> Dict[str, Any]:
        """Run backtest on historical data."""
        if self.backtest_engine is None:
            self.backtest_engine = BacktestEngine(
                initial_capital=100.0,
                fee_rate=0.00022,  # 2.2 bps taker fee (Jupiter measured)
                latency_model=LatencyModel(base_latency_s=10.0, mev_probability=0.3),
            )

        # Simple strategy using model predictions
        def model_strategy(features, idx):
            if not self.models_loaded or idx < 24:
                return None

            # Get predictions for this window
            window = features[max(0, idx-24):idx]
            pred = self.run_prediction_cycle(window)

            if "error" in pred:
                return None

            # Decision logic
            if pred["classifier"]["class_name"] in ["EXPLOITABLE", "HIGH_VALUE"]:
                if pred["risk"]["adverse_prob"] < 0.3:  # low risk
                    direction = "long" if pred["tft"]["direction_prob"] > 0.5 else "short"
                    strength = pred["classifier"]["exploitability"]
                    size = pred["classifier"]["size_fraction"]
                    return (direction, strength, size)

            return None

        result = self.backtest_engine.run_backtest(
            strategy_name="model_integrated",
            pair="SOL/USDC",
            features=features,
            signal_generator=model_strategy,
        )

        return {
            "strategy": "model_integrated",
            "trades": result.total_trades,
            "win_rate": result.win_rate,
            "total_pnl": result.total_pnl,
            "sharpe": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
        }

    def run_validation(self):
        """Run full validation suite."""
        print(f"[{datetime.now()}] Running validation suite...")

        # Load features
        features = get_historical_features("SOL/USDC", limit=1000)
        print(f"  Loaded {len(features)} features for validation")

        # Run backtest
        bt_result = self.run_backtest_cycle(features)
        print(f"  Backtest: {bt_result}")

        # Check drift
        if len(features) > 100:
            recent_data = np.array([f["features"]["close"] for f in features[-100:]])
            self.meta_controller.register_drift_detector("price_stream", recent_data)

            # Simulate drift check on older data
            older_data = np.array([f["features"]["close"] for f in features[:100]])
            alert = self.meta_controller.check_drift("price_stream", older_data)
            if alert:
                print(f"  Drift alert: {alert.message}")
                self.stats["alerts_triggered"] += 1

        # Cost check
        usage = meter.get_summary()
        cost_data = compute_cost_chf(usage)
        cost_alert = self.meta_controller.check_cost_ceiling(cost_data.get("total_chf", 0))
        if cost_alert:
            print(f"  Cost alert: {cost_alert.message}")
            self.stats["alerts_triggered"] += 1

        # Save meta-controller report
        report_path = LOG_DIR / f"meta_controller_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.meta_controller.save_report(report_path)
        print(f"  Meta-controller report saved to {report_path}")

        return {
            "backtest": bt_result,
            "alerts": len(self.meta_controller.alerts),
            "cost_chf": cost_data.get("total_chf", 0),
        }

    async def run(self):
        """Main orchestrator loop."""
        print(f"[{datetime.now()}] Integrated Orchestrator starting")
        print(f"  Models dir: {MODEL_DIR}")
        print(f"  Data dir: {DATA_DIR}")
        print(f"  Log dir: {LOG_DIR}")

        # Load models
        self.load_models()

        # Run validation
        validation = self.run_validation()
        print(f"[{datetime.now()}] Validation complete: {validation}")

        # Main loop (for live mode, would poll Jupiter here)
        print(f"[{datetime.now()}] Entering main loop (Ctrl+C to stop)")

        while not _shutdown:
            self.stats["cycles"] += 1

            # Status print every 60 cycles
            if self.stats["cycles"] % 60 == 0:
                uptime = time.time() - self.stats["start_time"]
                print(f"[{datetime.now()}] Status | uptime={uptime:.0f}s | "
                      f"cycles={self.stats['cycles']} | "
                      f"predictions={self.stats['predictions_made']} | "
                      f"alerts={self.stats['alerts_triggered']}")

            await asyncio.sleep(1)

        print(f"[{datetime.now()}] Shutdown complete")
        print(f"  Final stats: {json.dumps(self.stats, indent=2)}")


async def main():
    orchestrator = IntegratedOrchestrator()
    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
