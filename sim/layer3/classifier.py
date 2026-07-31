"""
classifier.py
Multi-task neural network for opportunity classification.
Classifies spread events into: NOISE, TRANSIENT, EXPLOITABLE, HIGH_VALUE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional


class SharedEncoder(nn.Module):
    """Shared encoder for all task heads."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, (h_n, c_n) = self.lstm(x)
        # Use last hidden state from both directions
        hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (batch, hidden_dim * 2)
        return self.layer_norm(self.dropout(hidden))


class OpportunityClassifier(nn.Module):
    """
    Multi-task classifier for arbitrage opportunities.

    Tasks:
    1. Classification: NOISE, TRANSIENT, EXPLOITABLE, HIGH_VALUE
    2. Regression: exploitability score (0-1)
    3. Regression: expected duration (seconds)
    4. Regression: recommended size (fraction of capital)
    """

    NUM_CLASSES = 4

    def __init__(
        self,
        input_dim: int = 38,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Shared encoder
        self.encoder = SharedEncoder(input_dim, hidden_dim, dropout)

        # Task heads
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.NUM_CLASSES),
        )

        self.exploitability_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.duration_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.ReLU(),  # duration >= 0
        )

        self.size_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # 0-1 fraction
        )

    def forward(self, x) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            dict with classification logits, exploitability, duration, size
        """
        shared = self.encoder(x)

        return {
            "classification": self.classification_head(shared),
            "exploitability": self.exploitability_head(shared).squeeze(-1),
            "duration": self.duration_head(shared).squeeze(-1),
            "size": self.size_head(shared).squeeze(-1),
        }

    def predict(self, x) -> Dict[str, Any]:
        """Get human-readable predictions."""
        self.eval()
        with torch.no_grad():
            out = self.forward(x)

            class_probs = F.softmax(out["classification"], dim=-1)
            class_idx = torch.argmax(class_probs, dim=-1)

            class_names = ["NOISE", "TRANSIENT", "EXPLOITABLE", "HIGH_VALUE"]

            return {
                "class": [class_names[i] for i in class_idx.cpu().numpy()],
                "class_probs": class_probs.cpu().numpy(),
                "exploitability": out["exploitability"].cpu().numpy(),
                "duration_s": out["duration"].cpu().numpy(),
                "size_fraction": out["size"].cpu().numpy(),
            }


class MultiTaskLoss(nn.Module):
    """Combined loss for all tasks."""

    def __init__(
        self,
        classification_weight: float = 1.0,
        exploitability_weight: float = 0.5,
        duration_weight: float = 0.3,
        size_weight: float = 0.2,
    ):
        super().__init__()
        self.weights = {
            "classification": classification_weight,
            "exploitability": exploitability_weight,
            "duration": duration_weight,
            "size": size_weight,
        }
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            predictions: model output dict
            targets: dict with class_idx, exploitability, duration, size
        """
        # Classification loss
        c_loss = self.ce_loss(predictions["classification"], targets["class_idx"])

        # Regression losses
        e_loss = self.mse_loss(predictions["exploitability"], targets["exploitability"])
        d_loss = self.mse_loss(predictions["duration"], targets["duration"])
        s_loss = self.mse_loss(predictions["size"], targets["size"])

        total = (
            self.weights["classification"] * c_loss +
            self.weights["exploitability"] * e_loss +
            self.weights["duration"] * d_loss +
            self.weights["size"] * s_loss
        )

        return total, {
            "classification_loss": c_loss.item(),
            "exploitability_loss": e_loss.item(),
            "duration_loss": d_loss.item(),
            "size_loss": s_loss.item(),
            "total_loss": total.item(),
        }
