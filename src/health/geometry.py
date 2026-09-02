"""
src/health/geometry.py

Information-geometric distance functions on the manifold of Gaussian
distributions, used to compare an engine's current encoded distribution
in VAE latent space against a healthy reference distribution.

The module has two tiers.

TIER 1 — diagonal, distribution-to-distribution
    kl_divergence, js_divergence, wasserstein2, fisher_rao
    Compare N(mu1, diag(sigma1^2)) with N(mu2, diag(sigma2^2)).
    Pure elementwise NumPy, batched over a leading axis.

TIER 2 — full covariance, point-or-distribution to POPULATION
    mahalanobis, kl_divergence_full, bures_wasserstein
    Compare against a reference with a full covariance matrix
    Sigma_ref, estimated as the covariance of healthy encodings across
    the fleet.

WHY TIER 2 EXISTS
-----------------
The original design compared a window's posterior N(mu_i, sigma_i)
against a reference built as the *average posterior* of healthy
windows. Those are both narrow distributions around nearly the same
point, so every divergence between them came out at ~1e-5 and the
metrics were effectively measuring nothing.

The right reference for "is this engine abnormal" is not the average
posterior width — it is the SPREAD OF HEALTHY ENCODINGS ACROSS THE
FLEET, i.e. Sigma_ref = Cov(mu | healthy). That covariance says which
directions in latent space healthy engines naturally vary along.
Mahalanobis distance under it whitens out the benign directions and
amplifies the ones healthy engines never move in — which is exactly
what you want when the degradation signal is a small perturbation
riding on a large constant offset.

ON "INFORMATION GEOMETRY"
-------------------------
KL, JS and Wasserstein are divergences: they measure discrepancy but
are not Riemannian distances (KL is not even symmetric). Fisher-Rao IS
the geodesic distance induced by the Fisher information metric on the
statistical manifold, so it is the one that makes the
information-geometry framing literal rather than decorative. It is also
the same metric NGBoost uses to define its natural gradient — the
predictor and the health monitor end up working in the same geometry
for different reasons, which is worth being able to say out loud.

SHAPES
------
Tier 1 accepts either 1D (latent_dim,) or 2D (batch, latent_dim) arrays
and reduces over the last axis, returning a scalar or (batch,).

Tier 2 accepts mu of shape (latent_dim,) or (batch, latent_dim), and
covariance of shape (latent_dim, latent_dim). Returns a scalar or
(batch,).

sigma, not variance, is expected as input throughout the Tier 1
functions — health_monitor.py converts the VAE's logvar via
`sigma = np.exp(0.5 * logvar)` before calling these.
"""

from __future__ import annotations

import numpy as np

# Floor to prevent division-by-zero / log(0) if a reference cluster's sigma
# collapses to ~0 (e.g. a very small or unusually uniform healthy cluster).
_EPS = 1e-8


# ==========================================================================
# TIER 1 — diagonal Gaussian divergences
# ==========================================================================

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

    per_dim = (np.log(sigma2 / sigma1) + (sigma1**2 + (mu1 - mu2) ** 2) / (2 * sigma2**2) - 0.5)
    return np.sum(per_dim, axis=-1)


def js_divergence(
    mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray
) -> np.ndarray | float:
    """
    Jensen-Shannon divergence: symmetric, bounded in [0, log(2)] per
    dimension summed over latent_dim. More stable than KL alone when
    either distribution's sigma might be small, since it doesn't blow up
    as sigma1 -> 0 or sigma2 -> 0 the way a raw KL ratio can.

    Note this is the Gaussian-approximated JS: the true mixture
    0.5*(p+q) is not Gaussian, so we substitute the moment-matched
    Gaussian N(mu_m, sigma_m). Standard practice, but worth knowing it
    is an approximation if asked.
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


def fisher_rao(
    mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray
) -> np.ndarray | float:
    """
    Fisher-Rao geodesic distance between two diagonal Gaussians.

    For a single univariate Gaussian family, the Fisher information
    metric turns the (mu, sigma) upper half-plane into a hyperbolic
    space of curvature -1/2. The geodesic distance has a closed form:

        d(theta1, theta2) = sqrt(2) * arccosh(1 + delta)

        delta = [ (mu1 - mu2)^2 + 2*(sigma1 - sigma2)^2 ]
                / ( 4 * sigma1 * sigma2 )

    For a diagonal (product) family the manifold is a product of these
    half-planes, and the geodesic distance is the L2 combination of the
    per-dimension distances:

        d_total = sqrt( sum_j d_j^2 )

    Unlike KL/JS/W2 this is a true metric: symmetric, non-negative, and
    satisfies the triangle inequality. It is the actual distance along
    the manifold rather than a discrepancy measure evaluated at two
    points, which is what makes the "information-geometric" label
    accurate.

    Numerical notes
    ---------------
    - arccosh is only defined for arguments >= 1. Analytically
      1 + delta >= 1 always, but floating-point subtraction of two
      nearly identical parameter vectors can produce 1 - 1e-16, so the
      argument is clipped.
    - sigma is floored at _EPS because delta divides by sigma1*sigma2.
      This is the one place a genuinely collapsed posterior still bites:
      with sigma -> 0 the hyperbolic metric stretches distances without
      bound. If the ELBO fix leaves sigma healthy (~0.1 to 1) this is a
      non-issue; if not, prefer the Tier 2 metrics.
    """
    sigma1 = np.clip(sigma1, _EPS, None)
    sigma2 = np.clip(sigma2, _EPS, None)

    delta = ((mu1 - mu2) ** 2 + 2.0 * (sigma1 - sigma2) ** 2) / (4.0 * sigma1 * sigma2)
    per_dim = np.sqrt(2.0) * np.arccosh(np.clip(1.0 + delta, 1.0, None))

    return np.sqrt(np.sum(per_dim**2, axis=-1))


# ==========================================================================
# TIER 2 — full-covariance reference
# ==========================================================================

def _as_2d(mu: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Promote a 1D (latent_dim,) input to (1, latent_dim) so the same code
    path handles single and batched calls. Returns (array_2d, was_1d)
    so the caller can squeeze the result back to a scalar.
    """
    mu = np.asarray(mu, dtype=np.float64)
    if mu.ndim == 1:
        return mu[None, :], True
    return mu, False


def regularise_covariance(cov: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """
    Add a small multiple of the identity to a covariance matrix so it is
    safely invertible.

    The ridge is scaled RELATIVE to the matrix: ridge * trace(cov)/d.
    A fixed absolute ridge would be either negligible or dominant
    depending on how large the latent activations happen to be, and we
    have no control over that — with the constant-offset failure mode
    the encoder's scale was ~22, while a well-trained posterior sits
    near unit scale.

    Call this once when building the reference, not per query.
    """
    cov = np.asarray(cov, dtype=np.float64)
    d = cov.shape[0]
    scale = np.trace(cov) / max(d, 1)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return cov + ridge * scale * np.eye(d)


def mahalanobis(
    mu: np.ndarray, mu_ref: np.ndarray, cov_ref_inv: np.ndarray
) -> np.ndarray | float:
    """
    Mahalanobis distance of an encoding from the healthy population.

        d_M(x) = sqrt( (x - mu_ref)^T Sigma_ref^-1 (x - mu_ref) )

    This is the single most useful health index in this module, and the
    reason is geometric rather than statistical. Sigma_ref^-1 whitens
    the latent space by the healthy fleet's own variation: movement
    along a direction healthy engines routinely explore costs little,
    while the same magnitude of movement along a direction they never
    explore costs a lot. Plain Euclidean distance treats both equally
    and therefore drowns the degradation signal in benign variation.

    Under a Gaussian model d_M^2 is chi-squared with latent_dim degrees
    of freedom, which gives a principled way to turn the raw distance
    into a calibrated alarm threshold. In practice the empirical
    healthy quantiles (computed in train_vae.py on healthy validation
    windows) are used instead, because the encodings are not exactly
    Gaussian.

    Parameters
    ----------
    mu          : (latent_dim,) or (batch, latent_dim) — current encodings
    mu_ref      : (latent_dim,) — healthy population mean
    cov_ref_inv : (latent_dim, latent_dim) — INVERSE of the healthy covariance
                  (precomputed once; inverting per call would be wasteful)
    """
    mu2d, was_1d = _as_2d(mu)
    delta = mu2d - np.asarray(mu_ref, dtype=np.float64)[None, :]

    # einsum computes the quadratic form for every row without ever
    # materialising a (batch, batch) matrix, which a naive
    # delta @ cov_inv @ delta.T would do.
    quad = np.einsum("ij,jk,ik->i", delta, cov_ref_inv, delta)

    # Tiny negatives can appear from floating-point error when the point
    # sits essentially on the reference mean.
    out = np.sqrt(np.clip(quad, 0.0, None))
    return float(out[0]) if was_1d else out


def kl_divergence_full(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu_ref: np.ndarray,
    cov_ref_inv: np.ndarray,
    cov_ref_logdet: float,
) -> np.ndarray | float:
    """
    KL( N(mu1, diag(sigma1^2)) || N(mu_ref, Sigma_ref) ), summed over
    the latent dimension.

        KL = 0.5 * [ tr(S_ref^-1 S_1)
                     + (mu_ref - mu1)^T S_ref^-1 (mu_ref - mu1)
                     - d
                     + ln det S_ref - ln det S_1 ]

    The query distribution keeps its diagonal posterior covariance
    (that IS what the VAE gives us per window) while the reference
    carries the full healthy population covariance. Two structurally
    different objects, which is correct: one is "the encoder's
    uncertainty about this window", the other is "the region healthy
    engines occupy".

    Because Sigma_ref is fixed, tr(S_ref^-1 S_1) collapses to a dot
    product of the diagonal of S_ref^-1 with sigma1^2, and ln det S_1 is
    just 2 * sum(log sigma1). Both are O(d) per sample rather than
    O(d^3).

    cov_ref_logdet is passed in precomputed (via np.linalg.slogdet) —
    computing it per call is both wasteful and a numerical hazard, since
    a direct det() on a 16x16 covariance underflows readily.
    """
    mu2d, was_1d = _as_2d(mu1)
    sigma2d, _ = _as_2d(sigma1)
    sigma2d = np.clip(sigma2d, _EPS, None)

    d = mu2d.shape[1]
    var1 = sigma2d**2                                    # (batch, d)

    # tr(Sigma_ref^-1 @ diag(var1)) = sum_j (Sigma_ref^-1)_jj * var1_j
    trace_term = var1 @ np.diag(cov_ref_inv)             # (batch,)

    delta = mu2d - np.asarray(mu_ref, dtype=np.float64)[None, :]
    quad_term = np.einsum("ij,jk,ik->i", delta, cov_ref_inv, delta)

    logdet1 = np.sum(np.log(var1), axis=-1)              # ln det diag(var1)

    out = 0.5 * (trace_term + quad_term - d + cov_ref_logdet - logdet1)
    return float(out[0]) if was_1d else out


def js_divergence_full(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu_ref: np.ndarray,
    cov_ref: np.ndarray,
    cov_ref_inv: np.ndarray,
    cov_ref_logdet: float,
) -> np.ndarray | float:
    """
    Symmetrised (Jensen-Shannon) version of kl_divergence_full, using
    the moment-matched Gaussian midpoint

        Sigma_m = (diag(sigma1^2) + Sigma_ref) / 2
        mu_m    = (mu1 + mu_ref) / 2

    Sigma_m differs per sample because sigma1 does, so this needs one
    Cholesky per sample. That is O(batch * d^3) with d=16 — a few
    hundred milliseconds for tens of thousands of windows, acceptable
    for an offline batch job but not something to put in a hot loop.

    Returned as a symmetric non-negative quantity, useful when you want
    a divergence that does not privilege a direction. Prefer
    kl_divergence_full when the asymmetry is meaningful (it usually is:
    "how surprising is this window under the healthy model" is a
    directional question).
    """
    mu2d, was_1d = _as_2d(mu1)
    sigma2d, _ = _as_2d(sigma1)
    sigma2d = np.clip(sigma2d, _EPS, None)

    n, d = mu2d.shape
    mu_ref = np.asarray(mu_ref, dtype=np.float64)
    cov_ref = np.asarray(cov_ref, dtype=np.float64)

    var1 = sigma2d**2
    mu_m = 0.5 * (mu2d + mu_ref[None, :])

    # (batch, d, d): diagonal query covariance averaged with the full reference
    cov_m = 0.5 * (np.einsum("ij,jk->ijk", var1, np.eye(d)) + cov_ref[None, :, :])

    # Batched inverse and log-determinant. np.linalg handles a leading
    # batch axis natively; slogdet is used rather than log(det(...))
    # because a raw determinant of a small-scale covariance underflows.
    cov_m_inv = np.linalg.inv(cov_m)
    _, logdet_m = np.linalg.slogdet(cov_m)

    # --- KL( N(mu1, diag(var1)) || N(mu_m, cov_m) )
    trace_a = np.einsum("ijj,ij->i", cov_m_inv, var1)
    delta_a = mu2d - mu_m
    quad_a = np.einsum("ij,ijk,ik->i", delta_a, cov_m_inv, delta_a)
    logdet_1 = np.sum(np.log(var1), axis=-1)
    kl_a = 0.5 * (trace_a + quad_a - d + logdet_m - logdet_1)

    # --- KL( N(mu_ref, cov_ref) || N(mu_m, cov_m) )
    trace_b = np.einsum("ijk,kj->i", cov_m_inv, cov_ref)
    delta_b = mu_ref[None, :] - mu_m
    quad_b = np.einsum("ij,ijk,ik->i", delta_b, cov_m_inv, delta_b)
    kl_b = 0.5 * (trace_b + quad_b - d + logdet_m - cov_ref_logdet)

    out = 0.5 * kl_a + 0.5 * kl_b
    return float(out[0]) if was_1d else out


def bures_wasserstein(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu_ref: np.ndarray,
    cov_ref_sqrt: np.ndarray,
) -> np.ndarray | float:
    """
    Squared 2-Wasserstein distance between N(mu1, diag(sigma1^2)) and
    N(mu_ref, Sigma_ref), full-covariance closed form:

        W2^2 = ||mu1 - mu_ref||^2
               + tr(S1) + tr(S_ref) - 2*tr( (S_ref^1/2 S1 S_ref^1/2)^1/2 )

    The trailing term is the Bures metric on positive-definite matrices.
    It is the covariance-aware generalisation of the (sigma1 - sigma2)^2
    term in the diagonal `wasserstein2` above — that version implicitly
    assumes the two covariances are simultaneously diagonalisable, which
    stops being true the moment the reference has off-diagonal
    structure. Healthy latent encodings are correlated across
    dimensions, so they do.

    Involves no division and no logarithm, which makes it the most
    numerically forgiving of the Tier 2 metrics — the right thing to
    reach for if KL or Fisher-Rao start returning nonsense.

    cov_ref_sqrt is the precomputed symmetric PSD square root of
    Sigma_ref (via eigendecomposition, in build_reference_matrices).
    """
    mu2d, was_1d = _as_2d(mu1)
    sigma2d, _ = _as_2d(sigma1)
    sigma2d = np.clip(sigma2d, _EPS, None)

    mu_ref = np.asarray(mu_ref, dtype=np.float64)
    S = np.asarray(cov_ref_sqrt, dtype=np.float64)

    var1 = sigma2d**2                                    # (batch, d)

    mean_term = np.sum((mu2d - mu_ref[None, :]) ** 2, axis=-1)
    trace_1 = np.sum(var1, axis=-1)
    trace_ref = np.sum(S @ S * np.eye(S.shape[0]))       # tr(Sigma_ref)

    # A_i = S @ diag(var1_i) @ S, built without forming diag() per sample.
    # Chunked because this is (batch, d, d) in memory and FD004 can have
    # tens of thousands of windows.
    n = mu2d.shape[0]
    cross = np.empty(n, dtype=np.float64)
    chunk = 4096
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        A = np.einsum("jm,im,mk->ijk", S, var1[start:stop], S)
        # A is symmetric PSD by construction, so eigvalsh is valid and
        # far cheaper than a general eig. tr(A^1/2) = sum of sqrt of
        # eigenvalues — no need to form the matrix square root itself.
        eigvals = np.linalg.eigvalsh(A)
        cross[start:stop] = np.sqrt(np.clip(eigvals, 0.0, None)).sum(axis=-1)

    out = mean_term + trace_1 + trace_ref - 2.0 * cross
    # W2^2 is non-negative analytically; clip the floating-point dust.
    out = np.clip(out, 0.0, None)
    return float(out[0]) if was_1d else out


# ==========================================================================
# Reference construction helper
# ==========================================================================

def build_reference_matrices(
    mu_healthy: np.ndarray,
    sigma_healthy: np.ndarray,
    ridge: float = 1e-6,
) -> dict:
    """
    Turn a set of healthy encodings into everything the Tier 2 functions
    need, precomputed once.

    mu_healthy    : (n_healthy, latent_dim) — posterior means of healthy windows
    sigma_healthy : (n_healthy, latent_dim) — posterior sigmas of the same

    Returns a dict with:
        mu       (d,)     population mean of healthy encodings
        sigma    (d,)     mean posterior sigma — kept for the Tier 1 /
                          backward-compatible diagonal metrics
        cov      (d, d)   POPULATION covariance of healthy mu, ridged
        cov_inv  (d, d)   its inverse
        cov_sqrt (d, d)   its symmetric PSD square root (for Bures)
        cov_logdet float  log|cov| via slogdet
        n        int      number of healthy windows the estimate is built from

    The distinction that matters: `sigma` is the encoder's average
    uncertainty about an individual window, `cov` is how much healthy
    windows differ from each other. The original implementation used the
    former where the latter was needed, which is why every distance came
    out at ~1e-5.

    A caveat worth being ready for: cov is estimated from n_healthy
    windows in d dimensions, and those windows are heavily
    autocorrelated (consecutive sliding windows overlap by
    seq_length - 1 timesteps). The effective sample size is therefore
    much smaller than n_healthy, so the covariance is noisier than the
    raw count suggests. The ridge handles conditioning; if d ever grows
    past ~32, switch to Ledoit-Wolf shrinkage.
    """
    mu_healthy = np.asarray(mu_healthy, dtype=np.float64)
    sigma_healthy = np.asarray(sigma_healthy, dtype=np.float64)

    if mu_healthy.ndim != 2:
        raise ValueError(f"mu_healthy must be 2D (n, d), got {mu_healthy.shape}")

    n, d = mu_healthy.shape
    mu_bar = mu_healthy.mean(axis=0)

    if n > d:
        cov = np.cov(mu_healthy, rowvar=False)
    else:
        # Not enough samples for a full covariance — fall back to a
        # diagonal estimate rather than returning a singular matrix.
        cov = np.diag(mu_healthy.var(axis=0) + _EPS)
    cov = np.atleast_2d(cov)
    cov = regularise_covariance(cov, ridge=ridge)

    cov_inv = np.linalg.inv(cov)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("Healthy covariance is not positive definite after ridging.")

    # Symmetric PSD square root via eigendecomposition. scipy.linalg.sqrtm
    # would also work but returns a complex array that then needs its
    # imaginary part discarded, and it is much slower.
    evals, evecs = np.linalg.eigh(cov)
    cov_sqrt = evecs @ np.diag(np.sqrt(np.clip(evals, 0.0, None))) @ evecs.T

    return {
        "mu": mu_bar.astype(np.float64),
        "sigma": sigma_healthy.mean(axis=0).astype(np.float64),
        "cov": cov,
        "cov_inv": cov_inv,
        "cov_sqrt": cov_sqrt,
        "cov_logdet": float(logdet),
        "n": int(n),
    }


# ==========================================================================
# Convenience wrappers
# ==========================================================================

def all_distances(
    mu: np.ndarray, sigma: np.ndarray, mu_ref: np.ndarray, sigma_ref: np.ndarray
) -> dict[str, np.ndarray | float]:
    """
    Diagonal-reference wrapper — UNCHANGED signature and behaviour, kept
    so existing tests and any older calling code keep working.

    New code should prefer all_distances_full(), which uses the healthy
    population covariance rather than an averaged posterior sigma.

    Returns keys "kl_div", "js_div", "wasserstein", "fisher_rao".
    """
    return {
        "kl_div": kl_divergence(mu, sigma, mu_ref, sigma_ref),
        "js_div": js_divergence(mu, sigma, mu_ref, sigma_ref),
        "wasserstein": wasserstein2(mu, sigma, mu_ref, sigma_ref),
        "fisher_rao": fisher_rao(mu, sigma, mu_ref, sigma_ref),
    }


def all_distances_full(
    mu: np.ndarray,
    sigma: np.ndarray,
    ref: dict,
    include_js: bool = True,
) -> dict[str, np.ndarray | float]:
    """
    Full-covariance wrapper — the one health_monitor.py should call.

    mu, sigma : (batch, latent_dim) current encodings and posterior sigmas
    ref       : the dict returned by build_reference_matrices()
    include_js: js_divergence_full costs a batched matrix inverse per
                sample; set False to skip it on very large datasets.

    Returns:
        mahalanobis  point-to-population distance under Sigma_ref^-1
        kl_div       KL(current posterior || healthy population)
        js_div       symmetrised version of the above (if include_js)
        wasserstein  squared Bures-Wasserstein distance
        fisher_rao   geodesic distance on the Gaussian manifold, using
                     the DIAGONAL of Sigma_ref as the reference spread
                     (Fisher-Rao has no tractable closed form for full
                     covariance Gaussians in dimension > 1 — this is the
                     honest approximation, and it is why the diagonal
                     Tier 1 function is the one used here)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    out: dict[str, np.ndarray | float] = {
        "mahalanobis": mahalanobis(mu, ref["mu"], ref["cov_inv"]),
        "kl_div": kl_divergence_full(
            mu, sigma, ref["mu"], ref["cov_inv"], ref["cov_logdet"]
        ),
        "wasserstein": bures_wasserstein(mu, sigma, ref["mu"], ref["cov_sqrt"]),
        "fisher_rao": fisher_rao(
            mu, sigma, ref["mu"], np.sqrt(np.clip(np.diag(ref["cov"]), _EPS, None))
        ),
    }

    if include_js:
        out["js_div"] = js_divergence_full(
            mu, sigma, ref["mu"], ref["cov"], ref["cov_inv"], ref["cov_logdet"]
        )

    return out