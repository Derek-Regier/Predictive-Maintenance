"""
src/models/vae.py

Variational Autoencoder used for unsupervised engine health monitoring.

Trained exclusively on healthy engine windows (RUL > threshold — see
train_vae.py). At inference, deviations between an engine's current
encoded distribution and the learned "healthy" reference distribution
become information-geometric health indices (Mahalanobis / KL / JS /
Bures-Wasserstein / Fisher-Rao — see src/health/geometry.py) plus a
reconstruction-error-based drift flag.

Architecture
------------
Encoder : LSTM -> (mu, log_sigma^2), same pooling convention as the
          predictive backbones (src/models/backbones/*.py): the hidden
          state of the final LSTM layer at the last timestep.
Decoder : z -> initial hidden state -> LSTM unrolled over seq_length
          timesteps. The decoder input at each step is a normalised
          position ramp t/(T-1) rather than a constant zero, so the
          decoder knows *which* timestep it is generating; z is also
          concatenated at every timestep so its influence does not
          decay across a long unroll.

WHAT CHANGED IN THIS REVISION (and why)
---------------------------------------
1. `vae_loss` reduction mismatch — THE bug behind posterior collapse.
   The old version used `F.mse_loss(..., reduction="mean")`, which
   divides by (batch x seq_length x input_dim), while the KL term was
   only divided by batch. Minimising

       recon_mean + beta * kl

   is algebraically identical to minimising

       recon_sum + (beta * seq_length * input_dim) * kl

   so with seq_length=30 and input_dim=24 a nominal beta=1.0 was
   really beta ~ 720. No warm-up schedule or free-bits floor survives
   that. Recon is now summed over sequence and feature dimensions and
   averaged over batch, matching the KL's reduction, so beta=1.0 is
   the true ELBO and sensible search ranges are roughly [0.01, 1.0].

2. Free bits applied to the BATCH MEAN per dimension, not per example.
   Clamping each (example, dim) value upward biases the estimate: you
   pay full price for above-floor examples and get a free ride on
   below-floor ones, which inflates the reported KL. The standard
   formulation (Kingma et al. 2016) averages over the batch first,
   then clamps per dimension.

3. Decoder gets z at every timestep plus a position ramp. The old
   decoder saw only zeros and had to carry all reconstruction
   information through h0 across 30 LSTM steps. That is a weak path,
   and a decoder that cannot use z is another route to a dead latent
   space. (Note this is the opposite of the usual advice to *weaken*
   the decoder — that advice targets AUTOREGRESSIVE decoders which can
   cheat by modelling p(x_t | x_<t). This decoder never sees x, so it
   cannot cheat, and making it better at using z is strictly good.)

4. `posterior_diagnostics()` added — computes per-dimension KL and
   active-unit count so training can report whether the latent space
   is actually alive instead of inferring it from val loss.
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

        # Clamp logvar to a numerically safe band. Without this, an
        # unconstrained logvar head can run away to +-30 during early
        # training, where exp(0.5 * logvar) either overflows or
        # underflows to exactly 0 — and a sigma of exactly 0 makes every
        # downstream information-geometric quantity (which divides by
        # sigma or takes log sigma) either infinite or NaN.
        # [-6, 2] corresponds to sigma in roughly [0.05, 2.7].
        logvar = torch.clamp(self.fc_logvar(h_last), min=-6.0, max=2.0)
        return mu, logvar


class VAEDecoder(nn.Module):
    """
    LSTM decoder reconstructing a sequence from a latent vector z.

    Non-autoregressive: the decoder never sees the true x. At each
    timestep its input is [z, t/(T-1)] — the latent vector repeated,
    concatenated with a normalised position ramp.

    Feeding z at every step (rather than only through h0) matters
    because an LSTM unrolled over 30 steps will progressively forget a
    signal injected only at initialisation. The position ramp gives the
    decoder an explicit notion of where in the window it is, so it does
    not have to manufacture time-variation purely from its own
    recurrent dynamics.
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
        self.latent_dim = latent_dim

        # Project z into an initial hidden state for every LSTM layer at once
        self.fc_init = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.lstm = nn.LSTM(
            input_size=latent_dim + 1,  # z repeated + position ramp
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out = nn.Linear(hidden_dim, output_dim)

        # Position ramp is fixed for a given seq_length, so build it once and
        # register it as a buffer. Buffers move with .to(DEVICE) alongside
        # parameters but are not optimised and do not appear in gradients.
        # persistent=False keeps it out of state_dict, so checkpoints written
        # before this change still load without a key-mismatch error.
        ramp = torch.linspace(0.0, 1.0, steps=seq_length).view(1, seq_length, 1)
        self.register_buffer("pos_ramp", ramp, persistent=False)

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

        # Repeat z across the time axis and append the position ramp.
        z_seq = z.unsqueeze(1).expand(batch_size, self.seq_length, self.latent_dim)
        ramp = self.pos_ramp.expand(batch_size, self.seq_length, 1)
        dec_input = torch.cat([z_seq, ramp], dim=-1)

        out, _ = self.lstm(dec_input, (h0, c0))  # (batch, seq_length, hidden_dim)
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

        NOTE: this stays a MEAN over (seq_length, input_dim), unlike the
        training loss which now sums. That is deliberate — this value is
        a human-readable per-element error used for thresholding and
        dashboard display, not a term in an ELBO, so keeping it on a
        per-element scale means the numbers remain comparable to the old
        artefacts and independent of window size.
        """
        mu, _ = self.encode(x)
        x_hat = self.decoder(mu)
        return F.mse_loss(x_hat, x, reduction="none").mean(dim=(1, 2))


def vae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    free_bits: float = 0.05,   # minimum nats per latent dimension
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    beta-VAE loss with free bits, on a CONSISTENT reduction scale.

    Both terms are now "sum over the dimensions of the object, mean over
    batch":

        recon : sum over (seq_length, input_dim), mean over batch
        kl    : sum over latent_dim,             mean over batch

    which is what the ELBO actually says. Previously recon was meaned
    over (seq_length, input_dim) as well, silently multiplying the
    effective beta by seq_length * input_dim.

    Because the loss is now on a per-window rather than per-element
    scale, the absolute numbers printed during training will be far
    larger than before (hundreds or thousands rather than ~0.1). That is
    expected — do not compare them to old runs. Compare KL and the
    active-unit count instead.

    free_bits: each latent dimension is allowed `free_bits` nats of KL
    "for free". Applied to the batch mean per dimension so the estimate
    is unbiased. Set to 0.0 to disable.
    """
    # Sum over sequence and feature dims -> (batch,); then mean over batch.
    recon_loss = F.mse_loss(x_hat, x, reduction="none").sum(dim=(1, 2)).mean()

    # Analytic KL(q(z|x) || N(0,I)) per dimension: shape (batch, latent_dim)
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

    # Average over the batch FIRST, then apply the free-bits floor per
    # dimension. clamp() passes zero gradient below the floor, so a
    # dimension sitting under its allowance is left alone; dimensions
    # above it are penalised normally.
    kl_mean_per_dim = kl_per_dim.mean(dim=0)                  # (latent_dim,)
    kl_loss = torch.clamp(kl_mean_per_dim, min=free_bits).sum()

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


@torch.no_grad()
def posterior_diagnostics(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    active_threshold: float = 1e-2,
) -> dict:
    """
    Measure whether the latent space is alive, given encodings of a full
    (validation) set.

    Two standard diagnostics:

    per_dim_kl
        Mean KL(q(z_j|x) || N(0,1)) for each latent dimension j. A
        dimension whose KL sits at ~0 is carrying no information: the
        encoder outputs the prior regardless of input. Posterior
        collapse is the case where most or all dimensions look like this.

    active_units
        Burda et al. (2016): dimension j is "active" if
        Var_x(E[z_j | x]) > threshold, i.e. the posterior MEAN actually
        moves as the input changes. This is the more honest test,
        because a dimension can have non-zero KL from a constant offset
        while still ignoring the input entirely — which is exactly the
        failure mode the original FD001 run hit (||mu|| pinned at 22.75
        with variation only in the fourth decimal place).

    Returns a dict of plain Python types, ready for the YAML registry.
    """
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (N, latent_dim)
    per_dim_kl = kl_per_dim.mean(dim=0)                          # (latent_dim,)

    mu_var = mu.var(dim=0, unbiased=True)                        # (latent_dim,)
    active_mask = mu_var > active_threshold

    return {
        "per_dim_kl": [float(v) for v in per_dim_kl.cpu()],
        "per_dim_mu_var": [float(v) for v in mu_var.cpu()],
        "active_units": int(active_mask.sum().item()),
        "latent_dim": int(mu.shape[1]),
        "active_fraction": float(active_mask.float().mean().item()),
        "total_kl": float(per_dim_kl.sum().item()),
        "mu_norm_mean": float(mu.norm(dim=1).mean().item()),
        "mu_norm_std": float(mu.norm(dim=1).std().item()),
        "sigma_mean": float(torch.exp(0.5 * logvar).mean().item()),
        "active_threshold": float(active_threshold),
    }