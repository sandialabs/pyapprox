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
from numba.core.types import optional  # noqa: F401


@njit(cache=True, parallel=True, fastmath=True)
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
        log_det += math.log(base_variances[k]) - math.log(design_weights[k, 0])
    log_norm = -0.5 * nobs * math.log(2.0 * math.pi) - 0.5 * log_det

    for i in prange(ninner):
        for j in range(nouter):
            result[i, j] = log_norm - 0.5 * result[i, j]

    return result


@njit(cache=True, parallel=True, fastmath=True)
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


@njit(cache=True, parallel=True, fastmath=True)
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
                    jac_val += 0.5 * r * latent_samples[k, j] / sqrt_var_w
                acc += quad_weighted_like[i, j] * jac_val
            result[j, k] += acc

    return result


@njit(cache=True, parallel=True, fastmath=True)
def fused_weighted_jacobian_numba(
    shapes_ik: np.ndarray,
    obs_jk: np.ndarray,
    latent_jk: np.ndarray,
    base_variances: np.ndarray,
    design_weights: np.ndarray,
    qwl_ratio: np.ndarray,
    weights_a_qi: np.ndarray,
    weights_b_qi: np.ndarray,
    has_latent: bool,
):
    """Fused jacobian contractions with two arbitrary per-inner weight
    matrices — avoids the (ninner, nouter, nobs) intermediate.

    Computes the first term of the normalized-like jacobian contracted
    with two independent weight matrices ``W_a[i, q]`` and ``W_b[i, q]``:

        part_a[q, j, k] = sum_i W_a[i, q] * qwl_ratio[i, j] * Jlog[i, j, k]
        part_b[q, j, k] = sum_i W_b[i, q] * qwl_ratio[i, j] * Jlog[i, j, k]

    where ``Jlog[i, j, k] = d/dw_k loglike(obs_j | shape_i)`` is
    reconstructed element-wise inside the inner loop.

    The outputs are 3D ``(npred, nouter, nobs)`` — small (e.g.
    20×1000×50 ≈ 8 MB). What this kernel avoids is the
    ``(ninner, nouter, nobs)`` intermediate ``normalized_like_jac``
    tensor — at typical sizes that is ~400 MB and dominates allocation
    and bandwidth.

    The full contractions (with the second term restored) are obtained
    by the caller via:

        full_a = part_a - M_a[:, :, None] * evid_jac[None, :, :]
        full_b = part_b - M_b[:, :, None] * evid_jac[None, :, :]

    with ``M[q, j] = sum_i W[i, q] * qwl[i, j] / evid[j]**2``.

    Callers:

    * ``StandardDeviationMeasure._jacobian_fused`` passes
      ``(qoi, qoi**2)`` to get ``(first_mom_part, second_mom_part)``.
    * ``EntropicDeviationMeasure._jacobian_fused`` passes
      ``(exp(alpha*qoi), qoi)`` to get ``(d_expectation, mean_jac_part)``.

    Shapes are transposed to (i, k) / (j, k) / (q, i) for contiguous
    memory access. Each prange-j thread forms a local (ninner, nobs)
    buffer ``W[i, k] = qwl_ratio[i, j] * Jlog[i, j, k]`` and performs two
    per-thread dgemms to contract with ``weights_a`` and ``weights_b``.

    Parameters
    ----------
    shapes_ik : np.ndarray
        Model outputs, transposed layout. Shape: (ninner, nobs).
    obs_jk : np.ndarray
        Observations, transposed. Shape: (nouter, nobs).
    latent_jk : np.ndarray
        Latent samples, transposed. Shape: (nouter, nobs). Ignored if
        ``has_latent=False``.
    base_variances : np.ndarray
        Base noise variances. Shape: (nobs,).
    design_weights : np.ndarray
        Design weights. Shape: (nobs, 1).
    qwl_ratio : np.ndarray
        Quadrature-weighted likelihood divided by evidence:
        ``qwl[i, j] / evid[j]``. Shape: (ninner, nouter).
    weights_a_qi : np.ndarray
        First per-inner weight matrix, transposed. Shape: (npred, ninner).
    weights_b_qi : np.ndarray
        Second per-inner weight matrix, transposed. Shape: (npred, ninner).
    has_latent : bool
        Whether to include the reparameterization term.

    Returns
    -------
    part_a : np.ndarray
        Shape: (npred, nouter, nobs).
    part_b : np.ndarray
        Shape: (npred, nouter, nobs).
    """
    nobs = base_variances.shape[0]
    ninner = shapes_ik.shape[0]
    nouter = obs_jk.shape[0]
    npred = weights_a_qi.shape[0]

    inv_var = 1.0 / base_variances
    det_term_arr = 0.5 / design_weights[:, 0]
    sqrt_var_w_arr = np.sqrt(base_variances * design_weights[:, 0])

    part_a = np.zeros((npred, nouter, nobs))
    part_b = np.zeros((npred, nouter, nobs))

    for j in prange(nouter):
        W = np.empty((ninner, nobs))
        for i in range(ninner):
            wij = qwl_ratio[i, j]
            for k in range(nobs):
                r = obs_jk[j, k] - shapes_ik[i, k]
                jlog = -0.5 * r * r * inv_var[k] + det_term_arr[k]
                if has_latent:
                    jlog += 0.5 * r * latent_jk[j, k] / sqrt_var_w_arr[k]
                W[i, k] = wij * jlog
        part_a[:, j, :] = weights_a_qi @ W
        part_b[:, j, :] = weights_b_qi @ W
    return part_a, part_b
