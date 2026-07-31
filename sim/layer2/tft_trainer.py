"""
tft_trainer.py
Training pipeline for the Temporal Fusion Transformer.
"""

import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import MODEL_DIR, DATA_DIR
from layer2.tft_model import TemporalFusionTransformer, CombinedLoss
from layer2.dl_data_pipeline import prepare_training_data


class TFTTrainer:
    def __init__(
        self,
        input_dim: int = 38,
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

        # Gradient clipping value
        self.grad_clip = 0.5

        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")

    def train_epoch(self, dataloader: DataLoader) -> float:
        """One training epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            seq = batch["sequence"].to(self.device)
            target = batch["target"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(seq)

            # Compute loss (use the forecast horizon target)
            # Target shape: (batch, 4) for [5m, 15m, 30m, 1h]
            # We want the 1h prediction (index 3)
            # Expand to (batch, horizon) for quantile loss
            horizon_target = target[:, 3:4].expand(-1, self.forecast_horizon)
            loss, _ = self.criterion(output, horizon_target)

            # Skip NaN losses
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def validate(self, dataloader: DataLoader) -> float:
        """Validation pass."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                seq = batch["sequence"].to(self.device)
                target = batch["target"].to(self.device)

                output = self.model(seq)
                horizon_target = target[:, 3:4].expand(-1, self.forecast_horizon)
                loss, _ = self.criterion(output, horizon_target)

                # Skip NaN losses
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
        """Full training loop with early stopping."""
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            if verbose and (epoch + 1) % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1:3d} | train={train_loss:.6f} | val={val_loss:.6f} | lr={current_lr:.2e}")

            # Early stopping (skip if val_loss is NaN)
            if not torch.isnan(torch.tensor(val_loss)) and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint("best")
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
        """Evaluate on test set."""
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_classifications = []

        with torch.no_grad():
            for batch in test_loader:
                seq = batch["sequence"].to(self.device)
                target = batch["target"].to(self.device)

                output = self.model(seq)
                pred = output["quantiles"][:, :, 1]  # P50 (median)
                cls = output["classification"]

                # For evaluation, compare 1h prediction (last timestep) with 1h target
                pred_1h = pred[:, -1]  # (batch,)
                cls_1h = cls[:, -1]    # (batch,)

                all_predictions.append(pred_1h.cpu().numpy())
                all_targets.append(target[:, 3].cpu().numpy())  # 1h target
                all_classifications.append(cls_1h.cpu().numpy())

        predictions = np.concatenate(all_predictions)
        targets = np.concatenate(all_targets)
        classifications = np.concatenate(all_classifications)

        # Regression metrics
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))

        # Directional accuracy
        pred_direction = (predictions > 0).astype(int)
        true_direction = (targets > 0).astype(int)
        directional_accuracy = np.mean(pred_direction == true_direction)

        # Classification metrics (using classification head)
        cls_binary = (classifications > 0.5).astype(int)
        cls_accuracy = np.mean(cls_binary == true_direction)

        # Precision/Recall for positive predictions
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

    def save_checkpoint(self, name: str = "best"):
        """Save model checkpoint."""
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

    def load_checkpoint(self, name: str = "best"):
        """Load model checkpoint."""
        path = MODEL_DIR / f"tft_{name}.pt"
        # Load to CPU first to avoid device mismatch
        checkpoint = torch.load(path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        # Move to correct device after loading
        self.model = self.model.to(self.device)


def train_tft(
    pairs: List[str],
    seq_len: int = 24,
    forecast_horizon: int = 12,
    hidden_dim: int = 128,
    batch_size: int = 32,
    epochs: int = 100,
    lr: float = 1e-3,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    High-level training function for TFT.
    """
    print(f"Training TFT on {pairs}")
    print(f"  seq_len={seq_len}, forecast_horizon={forecast_horizon}, hidden_dim={hidden_dim}")

    # Prepare data
    train_loader, val_loader, test_loader, pipeline = prepare_training_data(
        pairs=pairs,
        seq_len=seq_len,
        forecast_horizon=forecast_horizon,
        batch_size=batch_size,
    )

    # Train
    trainer = TFTTrainer(
        input_dim=38,
        hidden_dim=hidden_dim,
        seq_len=seq_len,
        forecast_horizon=forecast_horizon,
        lr=lr,
    )

    print(f"  device={trainer.device}")
    print(f"  parameters={sum(p.numel() for p in trainer.model.parameters()):,}")

    history = trainer.train(train_loader, val_loader, epochs=epochs, verbose=verbose)

    # Evaluate
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

    # Save final model
    trainer.save_checkpoint("final")

    return {
        "history": history,
        "metrics": metrics,
        "trainer": trainer,
    }
