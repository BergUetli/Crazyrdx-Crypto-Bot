"""
tft_trainer_1h.py
Training pipeline for TFT on 1h features (67 fields).
"""

import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from config import MODEL_DIR, DATA_DIR
from layer2.tft_model import TemporalFusionTransformer, CombinedLoss
from layer1.historical_feature_engine_1h import get_historical_features_1h


class Dataset1h(Dataset):
    """Dataset for 1h TFT training."""

    def __init__(
        self,
        features: List[Dict[str, Any]],
        seq_len: int = 24,
        forecast_horizon: int = 12,
    ):
        self.features = features
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon

        # Extract close prices for target
        self.closes = [f["features"]["close"] for f in features]

        # Feature names (67 fields)
        self.feature_names = list(features[0]["features"].keys())

    def __len__(self):
        return len(self.features) - self.seq_len - self.forecast_horizon

    def __getitem__(self, idx):
        # Input sequence: seq_len candles of features
        seq_features = []
        for i in range(idx, idx + self.seq_len):
            f = self.features[i]["features"]
            # Convert to list in consistent order, converting strings to float
            row = []
            for name in self.feature_names:
                val = f[name]
                if isinstance(val, str):
                    # Convert string values to float (e.g., "long" -> 1.0, "short" -> -1.0)
                    val = 1.0 if val == "long" else -1.0 if val == "short" else 0.0
                row.append(float(val))
            seq_features.append(row)

        # Target: future returns over forecast_horizon
        current_close = self.closes[idx + self.seq_len - 1]
        future_returns = []
        for h in range(1, self.forecast_horizon + 1):
            future_close = self.closes[idx + self.seq_len - 1 + h]
            ret = (future_close - current_close) / current_close
            future_returns.append(ret)

        return {
            "sequence": torch.tensor(seq_features, dtype=torch.float32),
            "target": torch.tensor(future_returns, dtype=torch.float32),
        }


def prepare_1h_training_data(
    pairs: List[str],
    seq_len: int = 24,
    forecast_horizon: int = 12,
    batch_size: int = 32,
    train_pct: float = 0.7,
    val_pct: float = 0.15,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """Prepare 1h training data loaders."""

    all_features = []
    for pair in pairs:
        features = get_historical_features_1h(pair, limit=2000)
        all_features.extend(features)

    # Sort by timestamp
    all_features.sort(key=lambda x: x["ts"])

    # Split into train/val/test
    n = len(all_features)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))

    train_features = all_features[:train_end]
    val_features = all_features[train_end:val_end]
    test_features = all_features[val_end:]

    # Create datasets
    train_ds = Dataset1h(train_features, seq_len, forecast_horizon)
    val_ds = Dataset1h(val_features, seq_len, forecast_horizon)
    test_ds = Dataset1h(test_features, seq_len, forecast_horizon)

    # Create loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    input_dim = len(all_features[0]["features"])

    return train_loader, val_loader, test_loader, input_dim


class TFTTrainer1h:
    def __init__(
        self,
        input_dim: int = 67,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_quantiles: int = 3,
        seq_len: int = 24,
        forecast_horizon: int = 12,
        lr: float = 1e-3,
        device: Optional[str] = None,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon

        # Device selection
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Model
        self.model = TemporalFusionTransformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_quantiles=num_quantiles,
            seq_len=seq_len,
            forecast_horizon=forecast_horizon,
        ).to(self.device)

        # Loss and optimizer
        self.criterion = CombinedLoss(self.model.quantiles).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        self.grad_clip = 0.5
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            seq = batch["sequence"].to(self.device)
            target = batch["target"].to(self.device)

            self.optimizer.zero_grad()
            output = self.model(seq)
            loss, _ = self.criterion(output, target)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def validate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                seq = batch["sequence"].to(self.device)
                target = batch["target"].to(self.device)

                output = self.model(seq)
                loss, _ = self.criterion(output, target)

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        patience: int = 15,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.scheduler.step(val_loss)

            if verbose and (epoch + 1) % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1:3d} | train={train_loss:.6f} | val={val_loss:.6f} | lr={current_lr:.2e}")

            if not torch.isnan(torch.tensor(val_loss)) and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint("best_1h")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "epochs_trained": len(self.train_losses),
        }

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_classifications = []

        with torch.no_grad():
            for batch in test_loader:
                seq = batch["sequence"].to(self.device)
                target = batch["target"].to(self.device)

                output = self.model(seq)
                pred = output["quantiles"][:, :, 1]  # P50
                cls = output["classification"]

                pred_1h = pred[:, -1]
                cls_1h = cls[:, -1]

                all_predictions.append(pred_1h.cpu().numpy())
                all_targets.append(target[:, -1].cpu().numpy())
                all_classifications.append(cls_1h.cpu().numpy())

        predictions = np.concatenate(all_predictions)
        targets = np.concatenate(all_targets)
        classifications = np.concatenate(all_classifications)

        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))

        pred_direction = (predictions > 0).astype(int)
        true_direction = (targets > 0).astype(int)
        directional_accuracy = np.mean(pred_direction == true_direction)

        cls_binary = (classifications > 0.5).astype(int)
        cls_accuracy = np.mean(cls_binary == true_direction)

        true_pos = np.sum((cls_binary == 1) & (true_direction == 1))
        false_pos = np.sum((cls_binary == 1) & (true_direction == 0))
        false_neg = np.sum((cls_binary == 0) & (true_direction == 1))

        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "mse": float(mse),
            "mae": float(mae),
            "directional_accuracy": float(directional_accuracy),
            "classification_accuracy": float(cls_accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        }

    def save_checkpoint(self, name: str = "best_1h"):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = MODEL_DIR / f"tft_{name}.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "seq_len": self.seq_len,
            "forecast_horizon": self.forecast_horizon,
            "best_val_loss": self.best_val_loss,
        }, path)

    def load_checkpoint(self, name: str = "best_1h"):
        path = MODEL_DIR / f"tft_{name}.pt"
        checkpoint = torch.load(path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.model = self.model.to(self.device)


def train_tft_1h(
    pairs: List[str],
    seq_len: int = 24,
    forecast_horizon: int = 12,
    hidden_dim: int = 128,
    batch_size: int = 32,
    epochs: int = 100,
    lr: float = 1e-3,
    verbose: bool = True,
) -> Dict[str, Any]:
    """High-level training function for 1h TFT."""
    print(f"Training 1h TFT on {pairs}")
    print(f"  seq_len={seq_len}, forecast_horizon={forecast_horizon}, hidden_dim={hidden_dim}")

    train_loader, val_loader, test_loader, input_dim = prepare_1h_training_data(
        pairs=pairs,
        seq_len=seq_len,
        forecast_horizon=forecast_horizon,
        batch_size=batch_size,
    )

    trainer = TFTTrainer1h(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        seq_len=seq_len,
        forecast_horizon=forecast_horizon,
        lr=lr,
    )

    print(f"  device={trainer.device}")
    print(f"  parameters={sum(p.numel() for p in trainer.model.parameters()):,}")

    history = trainer.train(train_loader, val_loader, epochs=epochs, verbose=verbose)

    print("\nEvaluating on test set...")
    metrics = trainer.evaluate(test_loader)

    print(f"\nTest Metrics:")
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
    print(f"  Directional Accuracy: {metrics['directional_accuracy']:.2%}")
    print(f"  Classification Accuracy: {metrics['classification_accuracy']:.2%}")
    print(f"  Precision: {metrics['precision']:.2%}")
    print(f"  Recall: {metrics['recall']:.2%}")
    print(f"  F1 Score: {metrics['f1_score']:.2%}")

    trainer.save_checkpoint("final_1h")

    return {
        "history": history,
        "metrics": metrics,
        "trainer": trainer,
    }


if __name__ == "__main__":
    result = train_tft_1h(
        pairs=["SOL/USDC"],
        epochs=50,
        verbose=True,
    )
