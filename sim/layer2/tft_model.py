"""
tft_model.py
Temporal Fusion Transformer for time series forecasting.
Simplified implementation optimized for M-series Mac.
"""

import math
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedResidualNetwork(nn.Module):
    """GRN: Gated Residual Network from TFT paper."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_dim = context_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()

        if context_dim is not None:
            self.context_fc = nn.Linear(context_dim, hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

        # Projection from hidden to output dimension
        if hidden_dim != output_dim:
            self.output_proj = nn.Linear(hidden_dim, output_dim)
        else:
            self.output_proj = nn.Identity()

        if input_dim != output_dim:
            self.skip = nn.Linear(input_dim, output_dim)
        else:
            self.skip = nn.Identity()

    def forward(self, x, context: Optional[torch.Tensor] = None):
        residual = self.skip(x)

        x = self.fc1(x)
        if context is not None and self.context_dim is not None:
            x = x + self.context_fc(context)
        x = self.elu(x)
        x = self.fc2(x)
        x = self.dropout(x)

        # Gating mechanism
        gate_out = torch.sigmoid(self.gate(x))
        x = gate_out * x

        # Project to output dimension
        x = self.output_proj(x)

        return self.layer_norm(x + residual)


class VariableSelectionNetwork(nn.Module):
    """VSN: Learns which variables are important per timestep."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_vars: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_vars = num_vars
        self.hidden_dim = hidden_dim

        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(1, hidden_dim, hidden_dim, dropout)
            for _ in range(num_vars)
        ])

        self.selection_grn = GatedResidualNetwork(
            num_vars, hidden_dim, num_vars, dropout
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, num_vars)
        Returns:
            selected: (batch, seq_len, hidden_dim)
            weights: (batch, seq_len, num_vars)
        """
        batch, seq_len, num_vars = x.shape

        # Process each variable independently
        var_outputs = []
        for i in range(num_vars):
            var_input = x[:, :, i:i+1]  # (batch, seq_len, 1)
            var_out = self.var_grns[i](var_input)  # (batch, seq_len, hidden_dim)
            var_outputs.append(var_out)

        var_outputs = torch.stack(var_outputs, dim=2)  # (batch, seq_len, num_vars, hidden_dim)

        # Compute selection weights
        flat = x.reshape(batch, seq_len, -1)  # (batch, seq_len, num_vars)
        weights = self.selection_grn(flat)  # (batch, seq_len, num_vars)
        weights = F.softmax(weights, dim=-1)  # (batch, seq_len, num_vars)

        # Weighted sum
        weights = weights.unsqueeze(-1)  # (batch, seq_len, num_vars, 1)
        selected = (var_outputs * weights).sum(dim=2)  # (batch, seq_len, hidden_dim)

        return selected, weights.squeeze(-1)


class TemporalFusionTransformer(nn.Module):
    """
    Simplified TFT for price forecasting.

    Architecture:
    1. Variable Selection Network (VSN)
    2. LSTM Encoder-Decoder
    3. Multi-Head Attention
    4. Position-wise Feedforward
    5. Quantile Output Heads
    """

    def __init__(
        self,
        input_dim: int,           # Number of input features
        hidden_dim: int = 128,    # Hidden dimension
        num_heads: int = 4,       # Attention heads
        num_quantiles: int = 3,   # P10, P50, P90
        dropout: float = 0.1,
        seq_len: int = 24,        # Input sequence length
        forecast_horizon: int = 12,  # Output sequence length
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_quantiles = num_quantiles
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon

        # Variable Selection
        self.vsn = VariableSelectionNetwork(
            input_dim=1,  # Each variable processed independently
            hidden_dim=hidden_dim,
            num_vars=input_dim,
            dropout=dropout,
        )

        # LSTM Encoder-Decoder
        self.encoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, batch_first=True, bidirectional=False
        )
        self.decoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, batch_first=True, bidirectional=False
        )

        # Multi-Head Attention
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Position-wise Feedforward
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Output heads for quantile regression
        self.quantile_heads = nn.ModuleList([
            nn.Linear(hidden_dim, forecast_horizon)
            for _ in range(num_quantiles)
        ])

        # Classification head (probability of positive return)
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, forecast_horizon),
            nn.Sigmoid(),
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Quantiles to predict
        self.register_buffer(
            "quantiles",
            torch.tensor([0.1, 0.5, 0.9][:num_quantiles])
        )

    def forward(
        self,
        x: torch.Tensor,  # (batch, seq_len, input_dim)
    ) -> Dict[str, torch.Tensor]:
        batch_size = x.size(0)

        # 1. Variable Selection
        selected, var_weights = self.vsn(x)  # (batch, seq_len, hidden_dim)

        # 2. LSTM Encoder
        encoder_out, (h_n, c_n) = self.encoder_lstm(selected)

        # 3. LSTM Decoder (use last hidden state to initialize)
        # For simplicity, we decode the full sequence
        decoder_out, _ = self.decoder_lstm(selected, (h_n, c_n))

        # 4. Self-Attention
        attn_out, attn_weights = self.attention(
            decoder_out, decoder_out, decoder_out
        )

        # 5. Position-wise Feedforward
        ff_out = self.feedforward(attn_out)
        ff_out = self.dropout(ff_out)
        ff_out = self.layer_norm(ff_out + attn_out)

        # 6. Global pooling (take last timestep for forecasting)
        pooled = ff_out[:, -1, :]  # (batch, hidden_dim)

        # 7. Output heads
        quantile_outputs = []
        for head in self.quantile_heads:
            q_out = head(pooled)  # (batch, forecast_horizon)
            quantile_outputs.append(q_out)

        quantile_outputs = torch.stack(quantile_outputs, dim=-1)  # (batch, forecast_horizon, num_quantiles)

        # Classification (probability of positive return)
        classification = self.classification_head(pooled)  # (batch, forecast_horizon)

        return {
            "quantiles": quantile_outputs,
            "classification": classification,
            "variable_weights": var_weights,
            "attention_weights": attn_weights,
        }

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get point prediction (median quantile)."""
        self.eval()
        with torch.no_grad():
            out = self.forward(x)
            # Return P50 (median)
            return out["quantiles"][:, :, 1]


class QuantileLoss(nn.Module):
    """Pinball loss for quantile regression."""

    def __init__(self, quantiles: torch.Tensor):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (batch, horizon, num_quantiles)
            targets: (batch, horizon)
        """
        batch, horizon, num_quantiles = predictions.shape
        targets = targets.unsqueeze(-1).expand(-1, -1, num_quantiles)

        errors = targets - predictions

        # Clamp errors to prevent NaN from extreme values
        errors = torch.clamp(errors, -10.0, 10.0)

        losses = torch.max(
            (self.quantiles - 1) * errors,
            self.quantiles * errors
        )

        return losses.mean()


class CombinedLoss(nn.Module):
    """Combined quantile + classification loss."""

    def __init__(self, quantiles: torch.Tensor, classification_weight: float = 0.3):
        super().__init__()
        self.quantile_loss = QuantileLoss(quantiles)
        self.classification_loss = nn.BCELoss()
        self.classification_weight = classification_weight

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            predictions: model output dict
            targets: (batch, horizon) future returns
        """
        # Quantile loss
        q_loss = self.quantile_loss(predictions["quantiles"], targets)

        # Classification loss (predict sign of return)
        target_signs = (targets > 0).float()
        c_loss = self.classification_loss(predictions["classification"], target_signs)

        total = q_loss + self.classification_weight * c_loss

        return total, {
            "quantile_loss": q_loss.item(),
            "classification_loss": c_loss.item(),
            "total_loss": total.item(),
        }
