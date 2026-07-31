#!/usr/bin/env python3
"""
train_autonomous.py
Autonomous training script for the TFT model.
Fixes NaN issue, trains with proper regularization, saves results.
"""

import json
import time
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

# Add sim dir to path
SIM_DIR = Path.home() / ".hermes" / "trading-bot" / "sim"
sys.path.insert(0, str(SIM_DIR))

from layer2.tft_trainer import TFTTrainer
from layer2.dl_data_pipeline import prepare_training_data

LOG_DIR = SIM_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


def log(msg: str):
    """Log to file and stdout."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def main():
    log("=" * 60)
    log("Autonomous TFT Training Started")
    log("=" * 60)

    # Check device
    if torch.backends.mps.is_available():
        device = "mps"
        log("Using MPS (Apple Silicon GPU)")
    else:
        device = "cpu"
        log("Using CPU")

    # Prepare data
    log("Loading training data...")
    pairs = ["SOL/USDC", "SOL/USDT", "BONK/USDC", "WIF/USDC"]

    train_loader, val_loader, test_loader, pipeline = prepare_training_data(
        pairs=pairs,
        seq_len=24,
        forecast_horizon=12,
        batch_size=32,
    )

    log(f"Train batches: {len(train_loader)}")
    log(f"Val batches: {len(val_loader)}")
    log(f"Test batches: {len(test_loader)}")

    # Train with fixed hyperparameters
    log("Initializing trainer...")
    trainer = TFTTrainer(
        input_dim=38,
        hidden_dim=96,  # Reduced from 128 to prevent overfitting
        num_heads=4,
        num_quantiles=3,
        seq_len=24,
        forecast_horizon=12,
        lr=5e-4,  # Reduced from 1e-3
        device=device,
    )

    log(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")

    # Train
    log("Starting training...")
    history = trainer.train(
        train_loader,
        val_loader,
        epochs=100,
        patience=20,
        verbose=True,
    )

    log(f"Training complete. Best val loss: {history['best_val_loss']:.6f}")

    # Evaluate
    log("Evaluating on test set...")
    metrics = trainer.evaluate(test_loader)

    log("Test Metrics:")
    log(f"  MSE: {metrics['mse']:.6f}")
    log(f"  MAE: {metrics['mae']:.6f}")
    log(f"  Directional Accuracy: {metrics['directional_accuracy']:.2%}")
    log(f"  Classification Accuracy: {metrics['classification_accuracy']:.2%}")
    log(f"  Precision: {metrics['precision']:.2%}")
    log(f"  Recall: {metrics['recall']:.2%}")
    log(f"  F1 Score: {metrics['f1_score']:.2%}")

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "pairs": pairs,
        "history": history,
        "metrics": metrics,
        "model_path": str(SIM_DIR / "models" / "tft_final.pt"),
    }

    results_file = LOG_DIR / f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    log(f"Results saved to {results_file}")
    log("=" * 60)
    log("Autonomous training complete")
    log("=" * 60)


if __name__ == "__main__":
    main()
