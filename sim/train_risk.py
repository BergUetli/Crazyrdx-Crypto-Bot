#!/usr/bin/env python3
"""train_risk.py - Trains the Bayesian risk assessor."""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

SIM_DIR = Path.home() / ".hermes" / "trading-bot" / "sim"
sys.path.insert(0, str(SIM_DIR))

from config import MODEL_DIR
from layer4.bnn_model import BayesianRiskAssessor, ELBOLoss
from layer1.historical_feature_engine import get_historical_features

LOG_DIR = SIM_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


class RiskDataset(Dataset):
    def __init__(self, features, adverse, shortfall, drawdown):
        self.features = torch.FloatTensor(features)
        self.adverse = torch.FloatTensor(adverse)
        self.shortfall = torch.FloatTensor(shortfall)
        self.drawdown = torch.FloatTensor(drawdown)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "features": self.features[idx],
            "adverse_label": self.adverse[idx],
            "shortfall": self.shortfall[idx],
            "drawdown": self.drawdown[idx],
        }


def generate_risk_labels(features: list) -> tuple:
    """Generate risk labels from price data."""
    X, adverse, shortfall, drawdown = [], [], [], []
    for i in range(50, len(features) - 12):
        f = features[i]["features"]
        future = features[i+12]["features"]
        current = f["close"]
        future_price = future["close"]
        ret = (future_price - current) / current if current > 0 else 0
        # Adverse move: > 0.5% against us
        adv = 1.0 if abs(ret) > 0.005 else 0.0
        # Shortfall: average loss in worst 10%
        sf = abs(min(ret, 0)) * 100
        # Drawdown: max drop from peak
        peak = max(features[j]["features"]["close"] for j in range(i, min(i+12, len(features))))
        dd = (peak - future_price) / peak * 100 if peak > 0 else 0
        X.append([f.get(name, 0.0) for name in [
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
        ]])
        adverse.append(adv)
        shortfall.append(sf)
        drawdown.append(dd)
    return np.array(X), np.array(adverse), np.array(shortfall), np.array(drawdown)


def main():
    log_file = LOG_DIR / f"risk_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    def log(msg):
        print(msg)
        with open(log_file, "a") as f:
            f.write(f"[{datetime.now()}] {msg}\n")

    log("=" * 60)
    log("Risk Assessor Training Started")
    log("=" * 60)

    features = get_historical_features("SOL/USDC", limit=1000)
    log(f"Loaded {len(features)} features")

    X, adverse, shortfall, drawdown = generate_risk_labels(features)
    log(f"Generated {len(X)} samples")
    log(f"Adverse rate: {adverse.mean():.1%}")

    dataset = RiskDataset(X, adverse, shortfall, drawdown)
    n = len(dataset)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, [n_train, n_val, n - n_train - n_val])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log(f"Using device: {device}")

    model = BayesianRiskAssessor(input_dim=38, hidden_dim=64).to(device)
    criterion = ELBOLoss(kl_weight=1e-5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_val = float("inf")
    for epoch in range(50):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x = batch["features"].to(device)
            targets = {
                "adverse_label": batch["adverse_label"].to(device),
                "shortfall": batch["shortfall"].to(device),
                "drawdown": batch["drawdown"].to(device),
            }
            optimizer.zero_grad()
            out = model(x, sample=True)
            loss, _ = criterion(out, targets, model.kl_divergence())
            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["features"].to(device)
                targets = {
                    "adverse_label": batch["adverse_label"].to(device),
                    "shortfall": batch["shortfall"].to(device),
                    "drawdown": batch["drawdown"].to(device),
                }
                out = model(x, sample=False)
                loss, _ = criterion(out, targets, model.kl_divergence())
                if not torch.isnan(loss):
                    val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        if (epoch + 1) % 10 == 0:
            log(f"Epoch {epoch+1:3d} | train={train_loss:.6f} | val={val_loss:.6f}")

        if val_loss < best_val and not np.isnan(val_loss):
            best_val = val_loss
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict()}, MODEL_DIR / "risk_best.pt")

    # Save final
    torch.save({"model_state_dict": model.state_dict()}, MODEL_DIR / "risk_final.pt")
    log(f"Model saved to {MODEL_DIR / 'risk_final.pt'}")
    log("=" * 60)


if __name__ == "__main__":
    main()
