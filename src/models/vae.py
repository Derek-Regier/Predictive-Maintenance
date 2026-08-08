"""
src/models/vae.py

Variational Autoencoder used for unsupervised engine health monitoring.

Trained exclusively on healthy engine windows (RUL > threshold — see
train_vae.py). At inference, deviations between an engine's current
encoded distribution and the learned "healthy" reference distribution
become information-geometric health indices (KL / JS / Wasserstein —
see src/health/geometry.py) plus a reconstruction-error-based drift flag.

Architecture
------------
Encoder : LSTM -> (mu, log_sigma^2), same pooling convention as the
          predictive backbones (src/models/backbones/*.py): the hidden
          state of the final LSTM layer at the last timestep.
Decoder : z -> initial hidden state -> LSTM unrolled over seq_length
          timesteps with a zero-valued input sequence (non-autoregressive
          — simpler and more stable to train than feeding predictions
          back in as the next input).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEEncoder(nn.Module):
    """LSTM encoder producing (mu, logvar) of the approximate posterior q(z|x)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (batch, seq_len, input_dim) -> mu, logvar: (batch, latent_dim)."""
        _, (hn, _) = self.lstm(x)
        h_last = hn[-1]  # final layer's hidden state — matches backbone .encode()
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        return mu, logvar


class VAEDecoder(nn.Module):
    """
    LSTM decoder reconstructing a sequence from a latent vector z.

    Non-autoregressive: the LSTM is unrolled over a zero-valued input
    sequence, so all reconstruction information must flow through the
    hidden state seeded from z.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        seq_length: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_length = seq_length

        # Project z into an initial hidden state for every LSTM layer at once
        self.fc_init = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.lstm = nn.LSTM(
            input_size=1,  # dummy scalar zero-input at every timestep
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (batch, latent_dim) -> reconstruction: (batch, seq_length, output_dim)."""
        batch_size = z.size(0)

        h0 = self.fc_init(z)  # (batch, hidden_dim * num_layers)
        h0 = (
            h0.view(batch_size, self.num_layers, self.hidden_dim)
            .transpose(0, 1)
            .contiguous()
        )  # (num_layers, batch, hidden_dim) — matches PyTorch LSTM hidden-state shape
        c0 = torch.zeros_like(h0)

        dummy_input = torch.zeros(batch_size, self.seq_length, 1, device=z.device)
        out, _ = self.lstm(dummy_input, (h0, c0))  # (batch, seq_length, hidden_dim)
        return self.out(out)  # (batch, seq_length, output_dim)


class VAE(nn.Module):
    """Full sequence VAE: encoder + reparameterization + decoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        seq_length: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.seq_length = seq_length

        self.encoder = VAEEncoder(input_dim, hidden_dim, latent_dim, num_layers, dropout)
        self.decoder = VAEDecoder(
            latent_dim,
            hidden_dim,
            output_dim=input_dim,
            seq_length=seq_length,
            num_layers=num_layers,
            dropout=dropout,
        )

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: (batch, seq_len, input_dim) -> (x_hat, mu, logvar)."""
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Deterministic encode for inference / health monitoring — no sampling.
        Returns (mu, logvar), both (batch, latent_dim). This is what
        health_monitor.py and the healthy-reference builder should call,
        not forward(), so results are reproducible across runs.
        """
        return self.encoder(x)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-example MSE reconstruction error using the deterministic mu
        (no sampling noise) — used for drift flagging in health_monitor.py.
        Returns (batch,).
        """
        mu, _ = self.encode(x)
        x_hat = self.decoder(mu)
        return F.mse_loss(x_hat, x, reduction="none").mean(dim=(1, 2))


def vae_loss(
    x:         torch.Tensor,
    x_hat:     torch.Tensor,
    mu:        torch.Tensor,
    logvar:    torch.Tensor,
    beta:      float = 1.0,
    free_bits: float = 0.1,   # minimum nats per latent dimension
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon_loss = F.mse_loss(x_hat, x, reduction="mean")

    # KL per dimension: shape (batch, latent_dim)
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

    # Clamp each dimension to free_bits minimum — prevents any dimension
    # from collapsing to 0 while still penalising dimensions that exceed
    # their minimum. Sum over dimensions, mean over batch.
    kl_loss = torch.clamp(kl_per_dim, min=free_bits).sum(dim=-1).mean()

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss