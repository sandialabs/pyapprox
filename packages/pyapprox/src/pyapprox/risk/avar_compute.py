"""
Vectorized batch kernels for smoothed AVaR computation.

Replaces the Python-level loop over nqoi with vectorized operations
across all QoIs simultaneously. The binary search is unrolled into a
fixed number of iterations (ceil(log2(2*nsamples))), enabling parallel
state updates across QoIs.

All float operations use backend methods to preserve the PyTorch autograd
computation graph. Only integer index bookkeeping uses numpy arrays.
"""

import math

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend


def _residual_batch(
    x: Array,
    dvalues: Array,
    weights: Array,
    lbnd: Array,
    ubnd: Array,
    bkd: Backend[Array],
) -> Array:
    """Compute residual for each QoI at given Lagrange multipliers.

    Parameters
    ----------
    x : Array
        Lagrange multipliers. Shape: (nqoi,)
    dvalues : Array
        values / weights. Shape: (nqoi, nsamples)
    weights : Array
        Probability weights. Shape: (1, nsamples)
    lbnd : Array
        Lower bound scalar.
    ubnd : Array
        Upper bound 1/(1-alpha).
    bkd : Backend[Array]

    Returns
    -------
    Array
        Residual for each QoI. Shape: (nqoi,)
    """
    clamped = bkd.maximum(lbnd, bkd.minimum(ubnd, dvalues + x[:, None]))
    return 1.0 - bkd.sum(weights * clamped, axis=1)


def project_batch(
    values: Array,
    weights: Array,
    alpha: Array,
    bkd: Backend[Array],
) -> Array:
    """Project scaled values onto the CVaR risk envelope for all QoIs.

    Vectorized binary search across all QoIs simultaneously. Integer
    index state uses numpy; all float tensor operations use backend
    methods to preserve autograd.

    Parameters
    ----------
    values : Array
        Scaled values. Shape: (nqoi, nsamples)
    weights : Array
        Probability weights. Shape: (1, nsamples)
    alpha : Array
        Risk level. Shape: (1,)
    bkd : Backend[Array]

    Returns
    -------
    Array
        Projection. Shape: (nqoi, nsamples)
    """
    nqoi = values.shape[0]
    nsamples = values.shape[1]

    lbnd = bkd.asarray(0.0)
    ubnd = 1.0 / (1.0 - alpha)
    dvalues = values / weights  # (nqoi, nsamples)

    # Compute kinks per QoI, sorted descending. Shape: (nqoi, 2*nsamples)
    kinks_lo = lbnd - dvalues  # (nqoi, nsamples)
    kinks_hi = ubnd - dvalues  # (nqoi, nsamples)
    K_asc = bkd.sort(bkd.hstack([kinks_lo, kinks_hi]), axis=-1)
    K = bkd.flip(K_asc, axis=(-1,))  # descending

    # Binary search state: integer indices in numpy, float values in backend
    nkinks = 2 * nsamples
    niter = math.ceil(math.log2(nkinks)) if nkinks > 1 else 1

    ibeg = np.zeros(nqoi, dtype=np.int64)
    imid = np.full(nqoi, nsamples, dtype=np.int64)
    iend = np.full(nqoi, nkinks, dtype=np.int64)
    row_idx = np.arange(nqoi)

    # Gather from K using advanced indexing — preserves autograd
    x1 = K[row_idx, ibeg]  # (nqoi,)
    y1 = _residual_batch(x1, dvalues, weights, lbnd, ubnd, bkd)

    # Handle alpha=0 edge case: residual at ibeg=0 is ~0
    # Extract sign info to numpy for branching decisions only
    alpha_val = float(bkd.to_numpy(bkd.atleast_1d(alpha))[0])
    if alpha_val < 1e-15:
        y1_np = bkd.to_numpy(y1)
        near_zero = np.abs(y1_np) < 3e-16
        imid = np.where(near_zero, 1, imid)
        # Replace y1 with small negative value where near zero
        # Use bkd.where to keep on the computation graph
        mask = bkd.asarray(near_zero.astype(np.float64))
        y1 = y1 * (1.0 - mask) + mask * bkd.asarray(-3.0e-16)

    x2 = K[row_idx, imid]  # (nqoi,)
    y2 = _residual_batch(x2, dvalues, weights, lbnd, ubnd, bkd)

    for _ in range(niter):
        # Extract signs to numpy for index decisions only
        y1_signs = np.sign(bkd.to_numpy(y1))
        y2_signs = np.sign(bkd.to_numpy(y2))
        sign_match = y1_signs == y2_signs  # numpy bool array

        # Update integer indices (numpy only)
        iend = np.where(~sign_match, imid, iend)
        ibeg = np.where(sign_match, imid, ibeg)

        # Update float state using bkd.where — preserves autograd
        sign_match_arr = bkd.asarray(sign_match.astype(np.float64))
        # where sign_match: x1 <- x2, y1 <- y2; else keep
        x1 = sign_match_arr * x2 + (1.0 - sign_match_arr) * x1
        y1 = sign_match_arr * y2 + (1.0 - sign_match_arr) * y1

        # Update imid
        gap = iend - ibeg
        imid = np.where(
            gap == 1, iend, ibeg + np.round(gap / 2).astype(np.int64)
        )

        # Gather new x2 from K — preserves autograd
        x2 = K[row_idx, imid]  # (nqoi,)
        y2 = _residual_batch(x2, dvalues, weights, lbnd, ubnd, bkd)

    # Linear interpolation — all backend ops, preserves autograd
    denom = y2 - y1
    # Guard against zero denominator: use x1 where denom is tiny
    denom_safe = bkd.where(
        bkd.abs(denom) < bkd.asarray(1e-30),
        bkd.ones(denom.shape),
        denom,
    )
    lam_interp = (y2 * x1 - y1 * x2) / denom_safe
    # Where denom was tiny, fall back to x1
    lam_interp = bkd.where(
        bkd.abs(denom) < bkd.asarray(1e-30), x1, lam_interp
    )

    return weights * bkd.maximum(
        lbnd, bkd.minimum(ubnd, dvalues + lam_interp[:, None])
    )


def avar_values_batch(
    values: Array,
    weights: Array,
    alpha: Array,
    delta: float,
    lam: float,
    bkd: Backend[Array],
) -> Array:
    """Compute smoothed AVaR for all QoIs in a single batch.

    Parameters
    ----------
    values : Array
        Sample values. Shape: (nqoi, nsamples)
    weights : Array
        Quadrature weights. Shape: (1, nsamples)
    alpha : Array
        Risk level. Shape: (1,)
    delta : float
        Smoothing parameter.
    lam : float
        Regularization parameter.
    bkd : Backend[Array]

    Returns
    -------
    Array
        AVaR values. Shape: (nqoi, 1)
    """
    scaled = weights * values * delta + lam  # (nqoi, nsamples)
    proj = project_batch(scaled, weights, alpha, bkd)  # (nqoi, nsamples)

    term1 = bkd.sum(proj * values, axis=1)  # (nqoi,)
    diff = (proj - lam) / weights  # (nqoi, nsamples)
    term2 = 1.0 / (2.0 * delta) * bkd.sum(diff * (proj - lam), axis=1)
    return (term1 - term2)[:, None]  # (nqoi, 1)


def avar_jacobian_batch(
    values: Array,
    jac_values: Array,
    weights: Array,
    alpha: Array,
    delta: float,
    lam: float,
    bkd: Backend[Array],
) -> Array:
    """Compute AVaR Jacobian for all QoIs in a single batch.

    Parameters
    ----------
    values : Array
        Sample values. Shape: (nqoi, nsamples)
    jac_values : Array
        Jacobians. Shape: (nqoi, nsamples, nvars)
    weights : Array
        Quadrature weights. Shape: (1, nsamples)
    alpha : Array
        Risk level. Shape: (1,)
    delta : float
        Smoothing parameter.
    lam : float
        Regularization parameter.
    bkd : Backend[Array]

    Returns
    -------
    Array
        Jacobian. Shape: (nqoi, nvars)
    """
    scaled = weights * values * delta + lam
    proj = project_batch(scaled, weights, alpha, bkd)  # (nqoi, nsamples)
    return bkd.einsum("ij,ijk->ik", proj, jac_values)
