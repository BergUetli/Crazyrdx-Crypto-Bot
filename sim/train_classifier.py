#!/usr/bin/env python3
"""
train_classifier.py
Trains the opportunity classifier on labeled spread events.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import sys
SIM_DIR = Path.home() / ".hermes" / "trading-bot" / "sim"
sys.path.insert(0, str(SIM_DIR))

from config import MODEL_DIR, DATA_DIR
from layer3.classifier import OpportunityClassifier, MultiTaskLoss
from layer1.historical_feature_engine import get_historical_features

LOG_DIR = SIM_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


class ClassifierDataset(Dataset):
    """Dataset for classifier training."""

    def __init__(self, sequences, labels, exploitability, duration, size):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        self.exploitability = torch.FloatTensor(exploitability)
        self.duration = torch.FloatTensor(duration)
        self.size = torch.FloatTensor(size)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            "sequence": self.sequences[idx],
            "class_idx": self.labels[idx],
            "exploitability": self.exploitability[idx],
            "duration": self.duration[idx],
            "size": self.size[idx],
        }


def generate_labels_from_features(features: list) -> tuple:
    """
    Generate labels from feature vectors.
    Uses price movement to classify opportunities.
    """
    sequences = []
    labels = []
    exploitability = []
    duration = []
    size = []

    for i in range(24, len(features) - 12):
        # Input: last 24 candles
        window = features[i-24:i]
        seq = [[f["features"].get(name, 0.0) for name in [
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
        ]] for f in window]

        # Label: based on future price movement
        current_price = features[i]["features"]["close"]
        future_price = features[i+12]["features"]["close"]  # 1h ahead

        if current_price == 0:
            continue

        future_return = (future_price - current_price) / current_price

        # Classify
        abs_return = abs(future_return)
        if abs_return < 0.001:  # < 0.1%
            label = 0  # NOISE
            exploit = 0.1
            dur = 0.0
            sz = 0.0
        elif abs_return < 0.003:  # 0.1-0.3%
            label = 1  # TRANSIENT
            exploit = 0.3
            dur = 15.0
            sz = 0.1
        elif abs_return < 0.01:  # 0.3-1%
            label = 2  # EXPLOITABLE
            exploit = 0.7
            dur = 45.0
            sz = 0.3
        else:  # > 1%
            label = 3  # HIGH_VALUE
            exploit = 0.9
            dur = 90.0
            sz = 0.5

        sequences.append(seq)
        labels.append(label)
        exploitability.append(exploit)
        duration.append(dur)
        size.append(sz)

    return (
        np.array(sequences, dtype=np.float32),
        np.array(labels, dtype=np.int64),
        np.array(exploitability, dtype=np.float32),
        np.array(duration, dtype=np.float32),
        np.array(size, dtype=np.float32),
    )


def main():
    log_file = LOG_DIR / f"classifier_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    def log(msg):
        print(msg)
        with open(log_file, "a") as f:
            f.write(f"[{datetime.now()}] {msg}\n")

    log("=" * 60)
    log("Classifier Training Started")
    log("=" * 60)

    # Load features
    log("Loading features...")
    features = get_historical_features("SOL/USDC", limit=1000)
    log(f"Loaded {len(features)} features")

    # Generate labels
    log("Generating labels...")
    sequences, labels, exploitability, duration, size = generate_labels_from_features(features)
    log(f"Generated {len(sequences)} samples")
    log(f"Label distribution: {np.bincount(labels)}")

    # Create dataset
    dataset = ClassifierDataset(sequences, labels, exploitability, duration, size)

    # Split
    n = len(dataset)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val, n - n_train - n_val]
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    log(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Model
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log(f"Using device: {device}")

    model = OpportunityClassifier(input_dim=38, hidden_dim=128).to(device)
    criterion = MultiTaskLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0

    for epoch in range(50):
        # Train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            seq = batch["sequence"].to(device)
            targets = {
                "class_idx": batch["class_idx"].to(device),
                "exploitability": batch["exploitability"].to(device),
                "duration": batch["duration"].to(device),
                "size": batch["size"].to(device),
            }

            optimizer.zero_grad()
            outputs = model(seq)
            loss, _ = criterion(outputs, targets)

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            train_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                seq = batch["sequence"].to(device)
                targets = {
                    "class_idx": batch["class_idx"].to(device),
                    "exploitability": batch["exploitability"].to(device),
                    "duration": batch["duration"].to(device),
                    "size": batch["size"].to(device),
                }

                outputs = model(seq)
                loss, _ = criterion(outputs, targets)

                if not torch.isnan(loss):
                    val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        if (epoch + 1) % 10 == 0:
            log(f"Epoch {epoch+1:3d} | train={train_loss:.6f} | val={val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss and not np.isnan(val_loss):
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
            }, MODEL_DIR / "classifier_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log(f"Early stopping at epoch {epoch+1}")
                break

    # Evaluate on test set
    log("Evaluating on test set...")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            seq = batch["sequence"].to(device)
            outputs = model(seq)
            preds = torch.argmax(outputs["classification"], dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["class_idx"].numpy())

    # Metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = np.mean(all_preds == all_labels)

    # Per-class metrics
    class_names = ["NOISE", "TRANSIENT", "EXPLOITABLE", "HIGH_VALUE"]
    for i, name in enumerate(class_names):
        mask = all_labels == i
        if mask.sum() > 0:
            class_acc = np.mean(all_preds[mask] == all_labels[mask])
            log(f"  {name}: {class_acc:.2%} ({mask.sum()} samples)")

    log(f"  Overall accuracy: {accuracy:.2%}")

    # Save final model
    torch.save({
        "model_state_dict": model.state_dict(),
    }, MODEL_DIR / "classifier_final.pt")

    log(f"Model saved to {MODEL_DIR / 'classifier_final.pt'}")
    log("=" * 60)
    log("Classifier training complete")
    log("=" * 60)


if __name__ == "__main__":
    main()
