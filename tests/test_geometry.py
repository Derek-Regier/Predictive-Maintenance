"""
tests/test_geometry.py

Tests for the information-geometric distance functions in src/health/geometry.py.

These are the most important tests in the suite because:
  1. KL, JS, and Wasserstein have easy-to-miss sign errors in their
     closed-form implementations.
  2. The health monitor and dashboard consume these values directly —
     a wrong sign would flip the health index direction.
  3. They require no model files or data, so they run in any environment.

All tests use simple 1D or low-dimensional Gaussian inputs with known
analytical answers, making failures easy to diagnose.
"""

import numpy as np
import pytest

from geometry import all_distances, js_divergence, kl_divergence, wasserstein2


# ─────────────────────────────────────────────────────────────────────────────
# IDENTITY PROPERTY — distance(P, P) should be zero (or near-zero)
# This is the most fundamental correctness check. If a model's current
# encoding equals the healthy reference, health indices should be 0.
# ─────────────────────────────────────────────────────────────────────────────

def test_kl_same_distribution_is_zero():
    """KL(P || P) = 0 for any distribution P."""
    mu    = np.array([1.0, 2.0, -1.0])
    sigma = np.array([0.5, 1.0,  2.0])
    result = kl_divergence(mu, sigma, mu, sigma)
    assert abs(result) < 1e-9, f"KL(P||P) should be 0, got {result}"


def test_js_same_distribution_is_zero():
    """JS(P, P) = 0 for any distribution P."""
    mu    = np.array([0.0, 5.0])
    sigma = np.array([1.0, 0.5])
    result = js_divergence(mu, sigma, mu, sigma)
    assert abs(result) < 1e-9, f"JS(P,P) should be 0, got {result}"


def test_wasserstein_same_distribution_is_zero():
    """W₂(P, P) = 0 for any distribution P."""
    mu    = np.array([3.0, -2.0, 0.0])
    sigma = np.array([1.0,  0.5, 2.0])
    result = wasserstein2(mu, sigma, mu, sigma)
    assert result == 0.0, f"W2(P,P) should be exactly 0, got {result}"


# ─────────────────────────────────────────────────────────────────────────────
# NON-NEGATIVITY — all distances should be ≥ 0
# A negative health index would be physically meaningless and would break
# the dashboard's normalisation and drift threshold logic.
# ─────────────────────────────────────────────────────────────────────────────

def test_all_distances_non_negative():
    """KL, JS, and Wasserstein must all be ≥ 0 for any input pair."""
    mu1    = np.array([2.0, -1.0,  0.5])
    sigma1 = np.array([1.0,  0.5,  2.0])
    mu2    = np.array([0.0,  1.0, -0.5])
    sigma2 = np.array([2.0,  1.0,  0.5])

    distances = all_distances(mu1, sigma1, mu2, sigma2)

    for metric_name, value in distances.items():
        assert np.all(value >= 0), \
            f"{metric_name} returned a negative value: {value}"


# ─────────────────────────────────────────────────────────────────────────────
# MONOTONICITY — distances should increase as distributions diverge
# This validates the health monitoring intuition: as an engine degrades and
# its encoding moves away from the healthy reference, all three indices rise.
# ─────────────────────────────────────────────────────────────────────────────

def test_wasserstein_increases_with_mean_shift():
    """
    An engine far from healthy (large mean shift) should have a higher
    Wasserstein distance than one close to healthy (small mean shift).
    """
    mu_ref    = np.zeros(4)
    sigma_ref = np.ones(4)

    # Small shift — engine slightly degraded
    w_close = wasserstein2(np.full(4, 0.5), sigma_ref, mu_ref, sigma_ref)
    # Large shift — engine heavily degraded
    w_far   = wasserstein2(np.full(4, 5.0), sigma_ref, mu_ref, sigma_ref)

    assert w_far > w_close, \
        "Wasserstein should increase as the mean moves further from the reference"


def test_kl_increases_with_mean_shift():
    """KL divergence should be larger when means are further apart."""
    sigma  = np.array([1.0])
    mu_ref = np.array([0.0])

    kl_1 = kl_divergence(np.array([1.0]), sigma, mu_ref, sigma)
    kl_5 = kl_divergence(np.array([5.0]), sigma, mu_ref, sigma)

    assert kl_5 > kl_1, \
        "KL should increase as the current mean moves further from the reference"


def test_js_increases_with_mean_shift():
    """JS divergence should be larger when distributions are further apart."""
    sigma  = np.array([1.0, 1.0])
    mu_ref = np.zeros(2)

    js_close = js_divergence(np.full(2, 0.1), sigma, mu_ref, sigma)
    js_far   = js_divergence(np.full(2, 5.0), sigma, mu_ref, sigma)

    assert js_far > js_close, \
        "JS should increase as distributions diverge"


# ─────────────────────────────────────────────────────────────────────────────
# SYMMETRY — Wasserstein and JS are symmetric; KL is not
# JS is designed to fix KL's asymmetry, making it a proper metric.
# Wasserstein is also symmetric by its optimal-transport definition.
# Checking these properties catches transposed argument bugs.
# ─────────────────────────────────────────────────────────────────────────────

def test_wasserstein_is_symmetric():
    """W₂(P, Q) should equal W₂(Q, P) for any P, Q."""
    mu1, sigma1 = np.array([1.0, 2.0]), np.array([0.5, 1.5])
    mu2, sigma2 = np.array([3.0, 0.0]), np.array([1.0, 0.5])

    w_pq = wasserstein2(mu1, sigma1, mu2, sigma2)
    w_qp = wasserstein2(mu2, sigma2, mu1, sigma1)

    assert abs(w_pq - w_qp) < 1e-9, \
        f"Wasserstein should be symmetric: W(P,Q)={w_pq} ≠ W(Q,P)={w_qp}"


def test_js_is_symmetric():
    """JS(P, Q) should equal JS(Q, P)."""
    mu1, sigma1 = np.array([0.0]), np.array([1.0])
    mu2, sigma2 = np.array([3.0]), np.array([2.0])

    js_pq = js_divergence(mu1, sigma1, mu2, sigma2)
    js_qp = js_divergence(mu2, sigma2, mu1, sigma1)

    assert abs(js_pq - js_qp) < 1e-9, \
        f"JS should be symmetric: JS(P,Q)={js_pq} ≠ JS(Q,P)={js_qp}"


def test_kl_is_asymmetric():
    """
    KL(P || Q) ≠ KL(Q || P) in general — this asymmetry is intentional.
    KL measures how surprised Q is by samples from P, which is directional.
    We verify this property holds so no future refactor accidentally
    symmetrises it.
    """
    mu1, sigma1 = np.array([0.0]), np.array([1.0])
    mu2, sigma2 = np.array([3.0]), np.array([0.5])   # different mean AND sigma

    kl_pq = kl_divergence(mu1, sigma1, mu2, sigma2)
    kl_qp = kl_divergence(mu2, sigma2, mu1, sigma1)

    assert abs(kl_pq - kl_qp) > 0.01, \
        "KL should be asymmetric — if this fails, the implementation may be wrong"


# ─────────────────────────────────────────────────────────────────────────────
# all_distances() INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

def test_all_distances_returns_correct_keys():
    """all_distances() should return exactly the three expected metric names."""
    mu, sigma = np.array([1.0, 0.0]), np.array([1.0, 1.0])
    result = all_distances(mu, sigma, mu + 1, sigma)
    assert set(result.keys()) == {"kl_div", "js_div", "wasserstein"}, \
        f"Unexpected keys: {set(result.keys())}"


def test_all_distances_1d_input():
    """all_distances() should work with single-dimensional latent spaces."""
    result = all_distances(
        np.array([2.0]), np.array([1.0]),
        np.array([0.0]), np.array([1.0]),
    )
    for key, val in result.items():
        assert np.isfinite(val), f"{key} returned non-finite value: {val}"


# ─────────────────────────────────────────────────────────────────────────────
# BATCHED INPUT — 2D arrays (batch, latent_dim)
# health_monitor.py calls these functions with batched arrays for efficiency.
# The output should have shape (batch,) matching the input batch dimension.
# ─────────────────────────────────────────────────────────────────────────────

def test_batched_wasserstein_output_shape():
    """
    When inputs are 2D (batch, latent_dim), Wasserstein should return
    a 1D array of shape (batch,) — one distance per sequence.
    """
    batch_size = 5
    latent_dim = 4

    mu_batch    = np.random.randn(batch_size, latent_dim)
    sigma_batch = np.abs(np.random.randn(batch_size, latent_dim)) + 0.1
    mu_ref      = np.zeros(latent_dim)
    sigma_ref   = np.ones(latent_dim)

    result = wasserstein2(mu_batch, sigma_batch, mu_ref, sigma_ref)

    assert hasattr(result, "__len__"), "Batched input should produce an array output"
    assert len(result) == batch_size, \
        f"Expected {batch_size} distances, got {len(result)}"


def test_batched_all_distances_shapes():
    """all_distances() with batched input should return (batch,) arrays."""
    batch_size  = 8
    latent_dim  = 3
    mu_batch    = np.random.randn(batch_size, latent_dim)
    sigma_batch = np.abs(np.random.randn(batch_size, latent_dim)) + 0.1
    mu_ref      = np.zeros(latent_dim)
    sigma_ref   = np.ones(latent_dim)

    result = all_distances(mu_batch, sigma_batch, mu_ref, sigma_ref)

    for key, vals in result.items():
        assert len(vals) == batch_size, \
            f"{key}: expected shape ({batch_size},), got len={len(vals)}"
        assert np.all(vals >= 0), f"{key} has negative values in batch mode"