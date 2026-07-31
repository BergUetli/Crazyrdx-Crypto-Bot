"""
bnn_model.py
Bayesian Neural Network for risk assessment.
Estimates uncertainty in predictions using variational inference.
"""

import math
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianLinear(nn.Module):
    """
    Bayesian linear layer with weight uncertainty.
    Uses local reparameterization trick for efficient sampling.
    """

    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight parameters: mean and log variance
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_logvar = nn.Parameter(torch.randn(out_features, in_features) * 0.1 - 5)

        # Bias parameters
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_logvar = nn.Parameter(torch.zeros(out_features) - 5)

        # Prior
        self.prior_std = prior_std

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if self.training and sample:
            # Sample weights from posterior
            weight_std = torch.exp(0.5 * self.weight_logvar)
            bias_std = torch.exp(0.5 * self.bias_logvar)

            weight = self.weight_mu + weight_std * torch.randn_like(weight_std)
            bias = self.bias_mu + bias_std * torch.randn_like(bias_std)
        else:
            # Use mean weights
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence between posterior and prior."""
        # KL for weights
        weight_std = torch.exp(0.5 * self.weight_logvar)
        kl_weight = 0.5 * torch.sum(
            (self.weight_mu / self.prior_std) ** 2 +
            (weight_std / self.prior_std) ** 2 -
            1 - 2 * torch.log(weight_std / self.prior_std)
        )

        # KL for bias
        bias_std = torch.exp(0.5 * self.bias_logvar)
        kl_bias = 0.5 * torch.sum(
            (self.bias_mu / self.prior_std) ** 2 +
            (bias_std / self.prior_std) ** 2 -
            1 - 2 * torch.log(bias_std / self.prior_std)
        )

        return kl_weight + kl_bias


class BayesianRiskAssessor(nn.Module):
    """
    Bayesian NN for risk assessment.

    Outputs:
    1. P(adverse_move > threshold)
    2. Expected shortfall (average loss in worst 10%)
    3. Max drawdown estimate
    """

    def __init__(
        self,
        input_dim: int = 38,
        hidden_dim: int = 64,
        prior_std: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )

        # Bayesian layers
        self.bayes1 = BayesianLinear(hidden_dim, hidden_dim, prior_std)
        self.bayes2 = BayesianLinear(hidden_dim, hidden_dim // 2, prior_std)

        # Output heads
        self.adverse_head = nn.Linear(hidden_dim // 2, 1)
        self.shortfall_head = nn.Linear(hidden_dim // 2, 1)
        self.drawdown_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x: torch.Tensor, sample: bool = True) -> Dict[str, torch.Tensor]:
        # Encode
        h = self.encoder(x)

        # Bayesian layers
        h = F.relu(self.bayes1(h, sample))
        h = F.relu(self.bayes2(h, sample))

        # Outputs
        adverse_logit = self.adverse_head(h).squeeze(-1)
        adverse_prob = torch.sigmoid(adverse_logit)

        shortfall = F.softplus(self.shortfall_head(h)).squeeze(-1)  # >= 0
        drawdown = F.softplus(self.drawdown_head(h)).squeeze(-1)  # >= 0

        return {
            "adverse_prob": adverse_prob,
            "adverse_logit": adverse_logit,
            "shortfall": shortfall,
            "drawdown": drawdown,
        }

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 100,
    ) -> Dict[str, Any]:
        """
        Monte Carlo sampling for uncertainty estimation.
        Returns mean and std of predictions.
        """
        self.train()  # Enable sampling

        adverse_samples = []
        shortfall_samples = []
        drawdown_samples = []

        with torch.no_grad():
            for _ in range(n_samples):
                out = self.forward(x, sample=True)
                adverse_samples.append(out["adverse_prob"])
                shortfall_samples.append(out["shortfall"])
                drawdown_samples.append(out["drawdown"])

        adverse_samples = torch.stack(adverse_samples)
        shortfall_samples = torch.stack(shortfall_samples)
        drawdown_samples = torch.stack(drawdown_samples)

        return {
            "adverse_prob_mean": adverse_samples.mean(dim=0).cpu().numpy(),
            "adverse_prob_std": adverse_samples.std(dim=0).cpu().numpy(),
            "shortfall_mean": shortfall_samples.mean(dim=0).cpu().numpy(),
            "shortfall_std": shortfall_samples.std(dim=0).cpu().numpy(),
            "drawdown_mean": drawdown_samples.mean(dim=0).cpu().numpy(),
            "drawdown_std": drawdown_samples.std(dim=0).cpu().numpy(),
        }

    def kl_divergence(self) -> torch.Tensor:
        """Total KL divergence for all Bayesian layers."""
        return self.bayes1.kl_divergence() + self.bayes2.kl_divergence()


class ELBOLoss(nn.Module):
    """
    Evidence Lower Bound (ELBO) loss for Bayesian NN.
    ELBO = likelihood - KL_divergence
    We minimize: -ELBO = KL_divergence - likelihood
    """

    def __init__(self, kl_weight: float = 1e-4):
        super().__init__()
        self.kl_weight = kl_weight
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        kl_divergence: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            predictions: model output
            targets: dict with adverse_label, shortfall, drawdown
            kl_divergence: KL divergence from model
        """
        # Likelihood terms
        adverse_loss = self.bce_loss(predictions["adverse_prob"], targets["adverse_label"])
        shortfall_loss = self.mse_loss(predictions["shortfall"], targets["shortfall"])
        drawdown_loss = self.mse_loss(predictions["drawdown"], targets["drawdown"])

        likelihood = adverse_loss + shortfall_loss + drawdown_loss

        # ELBO
        elbo = likelihood + self.kl_weight * kl_divergence

        return elbo, {
            "adverse_loss": adverse_loss.item(),
            "shortfall_loss": shortfall_loss.item(),
            "drawdown_loss": drawdown_loss.item(),
            "kl_divergence": kl_divergence.item(),
            "total_loss": elbo.item(),
        }
