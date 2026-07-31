"""
train.py
Training pipeline for the feature autoencoder.
"""

import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from config import MODEL_DIR
from layer2.autoencoder import get_model, vae_loss
from layer2.dataset import create_dataloader, normalize_features, load_features_from_db


class AutoencoderTrainer:
    def __init__(
        self,
        model_type: str = "standard",
        input_dim: int = 35,
        latent_dim: int = 16,
        lr: float = 1e-3,
        device: Optional[str] = None,
    ):
        self.model_type = model_type
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lr = lr

        # Device selection: MPS > CPU
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = get_model(model_type, input_dim, latent_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")

    def train_epoch(self, dataloader: DataLoader) -> float:
        """One training epoch."""
        self.model.train()
        total_loss = 0.0

        for batch in dataloader:
            batch = batch.to(self.device)

            self.optimizer.zero_grad()

            if self.model_type == "vae":
                recon, mu, logvar = self.model(batch)
                loss, recon_loss, kl_div = vae_loss(recon, batch, mu, logvar, beta=0.1)
            else:
                recon, _ = self.model(batch)
                loss = self.criterion(recon, batch)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def validate(self, dataloader: DataLoader) -> float:
        """Validation pass."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)

                if self.model_type == "vae":
                    recon, mu, logvar = self.model(batch)
                    loss, _, _ = vae_loss(recon, batch, mu, logvar, beta=0.1)
                else:
                    recon, _ = self.model(batch)
                    loss = self.criterion(recon, batch)

                total_loss += loss.item()

        return total_loss / len(dataloader)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        patience: int = 10,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Full training loop with early stopping."""
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                # Save best model
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

    def save_checkpoint(self, name: str = "best"):
        """Save model checkpoint."""
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = MODEL_DIR / f"autoencoder_{self.model_type}_{name}.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_type": self.model_type,
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "best_val_loss": self.best_val_loss,
        }, path)

    def load_checkpoint(self, name: str = "best"):
        """Load model checkpoint."""
        path = MODEL_DIR / f"autoencoder_{self.model_type}_{name}.pt"
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))


def prepare_data(
    pair: Optional[str] = None,
    since_ts: float = 0,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.15,
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Prepare train/val/test dataloaders.

    Returns:
        train_loader, val_loader, test_loader, norm_stats
    """
    features, _ = load_features_from_db(pair, since_ts)

    if len(features) < 100:
        raise ValueError(f"Need at least 100 samples, got {len(features)}")

    # Normalize
    normalized, norm_stats = normalize_features(features)

    # Split
    n = len(normalized)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_val - n_test

    # Time-series split: train on oldest, test on newest
    train_data = normalized[:n_train]
    val_data = normalized[n_train:n_train + n_val]
    test_data = normalized[n_train + n_val:]

    # Create datasets
    from layer2.dataset import FeatureDataset

    train_dataset = FeatureDataset(train_data)
    val_dataset = FeatureDataset(val_data)
    test_dataset = FeatureDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, norm_stats


def train_autoencoder(
    model_type: str = "standard",
    pair: Optional[str] = None,
    since_ts: float = 0,
    latent_dim: int = 16,
    batch_size: int = 32,
    epochs: int = 100,
    lr: float = 1e-3,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    High-level training function.
    """
    print(f"Training {model_type} autoencoder on {pair or 'all pairs'}")
    print(f"  latent_dim={latent_dim}, batch_size={batch_size}, epochs={epochs}")

    # Prepare data
    train_loader, val_loader, test_loader, norm_stats = prepare_data(
        pair=pair, since_ts=since_ts, batch_size=batch_size
    )

    print(f"  train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")

    # Train
    trainer = AutoencoderTrainer(
        model_type=model_type,
        input_dim=35,
        latent_dim=latent_dim,
        lr=lr,
    )

    print(f"  device={trainer.device}")

    history = trainer.train(train_loader, val_loader, epochs=epochs, verbose=verbose)

    # Evaluate on test set
    test_loss = trainer.validate(test_loader)
    history["test_loss"] = test_loss

    print(f"  test_loss={test_loss:.6f}")
    print(f"  best_val_loss={history['best_val_loss']:.6f}")

    # Save normalization stats
    stats_path = MODEL_DIR / f"autoencoder_{model_type}_norm_stats.json"
    with open(stats_path, "w") as f:
        json.dump(norm_stats, f)

    # Save final model
    trainer.save_checkpoint("final")

    return history
