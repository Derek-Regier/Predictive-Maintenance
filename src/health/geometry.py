"""
src/health/geometry.py

Information-geometric distance functions between two diagonal-covariance
Gaussians in the VAE latent space: the engine's current encoded
distribution (mu, sigma) and its healthy reference distribution
(mu_ref, sigma_ref) for the same operating cluster.

All three functions are pure NumPy (no PyTorch dependency), so they're
usable from health_monitor.py, the Streamlit dashboard, and tests without
pulling in a model or a GPU.

Shapes
------
Every function accepts either:
  - 1D arrays of shape (latent_dim,)                — single comparison
  - 2D arrays of shape (batch, latent_dim)           — batched comparison
and sums (KL/JS) or sums (Wasserstein) over the latent dimension, returning
either a scalar (1D input) or an array of shape (batch,) (2D input).

sigma, not variance, is expected as input throughout — health_monitor.py
should convert the VAE's logvar via `sigma = np.exp(0.5 * logvar)` before
calling these.
"""

from __future__ import annotations

import numpy as np

# Floor to prevent division-by-zero / log(0) if a reference cluster's sigma
# collapses to ~0 (e.g. a very small or unusually uniform healthy cluster).
_EPS = 1e-8

def kl_divergence(
    mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray
) -> np.ndarray | float:
    """
    KL(N(mu1, sigma1^2) || N(mu2, sigma2^2)), summed over the latent
    dimension. Asymmetric: intended usage is current -> healthy, i.e.
    "how surprised is the healthy reference distribution by what we're
    seeing now."
    """
    sigma1 = np.clip(sigma1, _EPS, None)
    sigma2 = np.clip(sigma2, _EPS, None)

    per_dim = (np.log(sigma2 / sigma1) + (sigma1**2 + (mu1 - mu2) ** 2) / (2 * sigma2**2)- 0.5)
    return np.sum(per_dim, axis=-1)


def js_divergence(
    mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray
) -> np.ndarray | float:
    """
    Jensen-Shannon divergence: symmetric, bounded in [0, log(2)] per
    dimension summed over latent_dim. More stable than KL alone when
    either distribution's sigma might be small, since it doesn't blow up
    as sigma1 -> 0 or sigma2 -> 0 the way a raw KL ratio can.
    """
    mu_m = (mu1 + mu2) / 2
    sigma_m = np.sqrt((sigma1**2 + sigma2**2) / 2)

    return 0.5 * kl_divergence(mu1, sigma1, mu_m, sigma_m) + 0.5 * kl_divergence(
        mu2, sigma2, mu_m, sigma_m
    )


def wasserstein2(
    mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray
) -> np.ndarray | float:
    """
    Squared 2-Wasserstein distance, closed form for diagonal Gaussians,
    summed over the latent dimension. Symmetric and doesn't involve any
    division, so it's the most numerically robust of the three — useful
    as a sanity check against KL/JS if either looks unstable.
    """
    return np.sum((mu1 - mu2) ** 2 + (sigma1 - sigma2) ** 2, axis=-1)


def all_distances(
    mu: np.ndarray, sigma: np.ndarray, mu_ref: np.ndarray, sigma_ref: np.ndarray
) -> dict[str, np.ndarray | float]:
    """
    Convenience wrapper computing all three distances at once — this is
    what health_monitor.py should call per engine-timestep (or per batch)
    rather than calling each function separately.

    mu/sigma : current encoded distribution, shape (latent_dim,) or (batch, latent_dim)
    mu_ref/sigma_ref : healthy reference for this engine's operating cluster, same shape

    Returns a dict with keys "kl_div", "js_div", "wasserstein" — each a
    scalar or (batch,) array matching the input shape.
    """
    return {
        "kl_div": kl_divergence(mu, sigma, mu_ref, sigma_ref),
        "js_div": js_divergence(mu, sigma, mu_ref, sigma_ref),
        "wasserstein": wasserstein2(mu, sigma, mu_ref, sigma_ref),
    }