"""
autoencoder.py
Feature encoder: compresses 35 raw features into 16-dim latent representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAutoencoder(nn.Module):
    """
    Autoencoder for feature compression.
    Input: 35 features (excluding ts, pair)
    Latent: 16 dimensions
    """

    def __init__(self, input_dim: int = 35, latent_dim: int = 16):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),

            nn.Linear(32, 24),
            nn.ReLU(),
            nn.BatchNorm1d(24),

            nn.Linear(24, latent_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 24),
            nn.ReLU(),
            nn.BatchNorm1d(24),

            nn.Linear(24, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),

            nn.Linear(32, input_dim),
        )

    def forward(self, x):
        """Full pass: encode then decode."""
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction, latent

    def encode(self, x):
        """Compress input to latent representation."""
        return self.encoder(x)

    def decode(self, latent):
        """Reconstruct input from latent representation."""
        return self.decoder(latent)

    def get_latent(self, x):
        """Inference mode: just get the latent vector."""
        self.eval()
        with torch.no_grad():
            return self.encode(x)


class FeatureAutoencoderV2(nn.Module):
    """
    Variational Autoencoder version for probabilistic latent space.
    Better for uncertainty quantification downstream.
    """

    def __init__(self, input_dim: int = 35, latent_dim: int = 16):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_shared = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 24),
            nn.ReLU(),
            nn.BatchNorm1d(24),
        )

        self.fc_mu = nn.Linear(24, latent_dim)
        self.fc_logvar = nn.Linear(24, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 24),
            nn.ReLU(),
            nn.BatchNorm1d(24),
            nn.Linear(24, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, input_dim),
        )

    def encode(self, x):
        h = self.encoder_shared(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

    def get_latent(self, x, sample: bool = False):
        """Get latent vector. If sample=True, sample from distribution."""
        self.eval()
        with torch.no_grad():
            mu, logvar = self.encode(x)
            if sample:
                return self.reparameterize(mu, logvar)
            return mu


def vae_loss(recon_x, x, mu, logvar, beta: float = 1.0):
    """
    VAE loss = reconstruction loss + KL divergence.
    beta controls the strength of the KL term (beta-VAE).
    """
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_div, recon_loss, kl_div


def get_model(model_type: str = "standard", input_dim: int = 35, latent_dim: int = 16):
    """Factory function."""
    if model_type == "vae":
        return FeatureAutoencoderV2(input_dim, latent_dim)
    return FeatureAutoencoder(input_dim, latent_dim)
