"""
Numba JIT-compiled fused kernels for OED likelihood computations.

These kernels avoid materializing full 3D (nobs, ninner, nouter) arrays
by looping over nobs and computing element-wise. This dramatically
reduces memory allocation and improves cache locality.

All functions operate on raw NumPy arrays (not backend-wrapped).
The dispatch layer in dispatch.py handles the backend conversion.

Note: This module requires numba. If numba is not available, importing
this module will raise ImportError, which dispatch.py handles gracefully.
"""
import math

import numpy as np
from numba import njit, prange
from numba.core.types import optional, Array as NumbaArray  # noqa: F401


@njit(cache=True, parallel=True)
def logpdf_matrix_numba(
    shapes: np.ndarray,
    obs: np.ndarray,
    base_variances: np.ndarray,
    design_weights: np.ndarray,
) -> np.ndarray:
    """Compute log-likelihood matrix without 3D intermediate arrays.

    Loops over nobs (sequential), parallelizes over ninner with prange.
    Accumulates the squared Mahalanobis distance into (ninner, nouter).

    Parameters
    ----------
    shapes : np.ndarray
        Model outputs. Shape: (nobs, ninner)
    obs : np.ndarray
        Observations. Shape: (nobs, nouter)
    base_variances : np.ndarray
        Base noise variances. Shape: (nobs,)
    design_weights : np.ndarray
        Design weights. Shape: (nobs, 1)

    Returns
    -------
    np.ndarray
        Log-likelihood matrix. Shape: (ninner, nouter)
    """
    nobs = shapes.shape[0]
    ninner = shapes.shape[1]
    nouter = obs.shape[1]
    inv_var = design_weights[:, 0] / base_variances
    result = np.zeros((ninner, nouter))

    for k in range(nobs):
        s = inv_var[k]
        for i in prange(ninner):
            for j in range(nouter):
                r = obs[k, j] - shapes[k, i]
                result[i, j] += s * r * r

    # Compute log normalization
    log_det = 0.0
    for k in range(nobs):
        log_det += math.log(base_variances[k]) - math.log(
            design_weights[k, 0]
        )
    log_norm = -0.5 * nobs * math.log(2.0 * math.pi) - 0.5 * log_det

    for i in prange(ninner):
        for j in range(nouter):
            result[i, j] = log_norm - 0.5 * result[i, j]

    return result


@njit(cache=True, parallel=True)
def jacobian_matrix_numba(
    shapes: np.ndarray,
    obs: np.ndarray,
    latent_samples: np.ndarray,
    base_variances: np.ndarray,
    design_weights: np.ndarray,
    has_latent: bool,
) -> np.ndarray:
    """Compute Jacobian of log-likelihood matrix without 3D intermediates.

    Note: The output is still (ninner, nouter, nobs) because callers
    may need the full Jacobian tensor. The savings come from avoiding
    intermediate 3D arrays during computation.

    Parameters
    ----------
    shapes : np.ndarray
        Model outputs. Shape: (nobs, ninner)
    obs : np.ndarray
        Observations. Shape: (nobs, nouter)
    latent_samples : np.ndarray
        Latent samples. Shape: (nobs, nouter). Ignored if has_latent=False.
    base_variances : np.ndarray
        Base noise variances. Shape: (nobs,)
    design_weights : np.ndarray
        Design weights. Shape: (nobs, 1)
    has_latent : bool
        Whether to include the reparameterization term.

    Returns
    -------
    np.ndarray
        Jacobian tensor. Shape: (ninner, nouter, nobs)
    """
    nobs = shapes.shape[0]
    ninner = shapes.shape[1]
    nouter = obs.shape[1]
    result = np.zeros((ninner, nouter, nobs))

    for k in range(nobs):
        var_k = base_variances[k]
        w_k = design_weights[k, 0]
        det_term = 0.5 / w_k
        sqrt_var_w = 0.0
        if has_latent:
            sqrt_var_w = math.sqrt(var_k) * math.sqrt(w_k)

        for i in prange(ninner):
            for j in range(nouter):
                r = obs[k, j] - shapes[k, i]
                val = -0.5 * r * r / var_k + det_term
                if has_latent:
                    val += 0.5 * r * latent_samples[k, j] / sqrt_var_w
                result[i, j, k] = val

    return result


@njit(cache=True, parallel=True)
def fused_evidence_jacobian_numba(
    shapes: np.ndarray,
    obs: np.ndarray,
    latent_samples: np.ndarray,
    base_variances: np.ndarray,
    design_weights: np.ndarray,
    quad_weighted_like: np.ndarray,
    has_latent: bool,
) -> np.ndarray:
    """Fused evidence jacobian without 3D materialization.

    Computes:
        result[j, k] = sum_i quad_weighted_like[i, j] * d/dw_k loglike[i, j]

    This fuses jacobian_matrix() + einsum("io,iok->ok") into a single pass,
    avoiding the (ninner, nouter, nobs) 3D tensor entirely.

    Loop structure (race-condition-free):
        - Outer: k in range(nobs) — sequential, ~50 iterations
        - Middle: j in prange(nouter) — parallel, each thread owns result[j,:]
        - Inner: i in range(ninner) — sequential reduction into scalar acc

    No race: each prange thread j writes to unique result[j, k].
    The i-reduction is a scalar accumulation, deterministic within each thread.

    Parameters
    ----------
    shapes : np.ndarray
        Model outputs. Shape: (nobs, ninner)
    obs : np.ndarray
        Observations. Shape: (nobs, nouter)
    latent_samples : np.ndarray
        Latent samples. Shape: (nobs, nouter). Ignored if has_latent=False.
    base_variances : np.ndarray
        Base noise variances. Shape: (nobs,)
    design_weights : np.ndarray
        Design weights. Shape: (nobs, 1)
    quad_weighted_like : np.ndarray
        Quadrature-weighted likelihoods. Shape: (ninner, nouter)
    has_latent : bool
        Whether to include the reparameterization term.

    Returns
    -------
    np.ndarray
        Evidence jacobian. Shape: (nouter, nobs)
    """
    nobs = shapes.shape[0]
    ninner = shapes.shape[1]
    nouter = obs.shape[1]
    result = np.zeros((nouter, nobs))

    for k in range(nobs):
        var_k = base_variances[k]
        w_k = design_weights[k, 0]
        det_term = 0.5 / w_k
        sqrt_var_w = 0.0
        if has_latent:
            sqrt_var_w = math.sqrt(var_k) * math.sqrt(w_k)

        for j in prange(nouter):
            acc = 0.0
            for i in range(ninner):
                r = obs[k, j] - shapes[k, i]
                jac_val = -0.5 * r * r / var_k + det_term
                if has_latent:
                    jac_val += (
                        0.5 * r * latent_samples[k, j] / sqrt_var_w
                    )
                acc += quad_weighted_like[i, j] * jac_val
            result[j, k] += acc

    return result
