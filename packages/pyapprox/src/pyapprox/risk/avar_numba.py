"""
Numba JIT-compiled parallel kernels for smoothed AVaR computation.

Parallelizes the binary search across QoIs using prange. Each thread
runs the original scalar binary search for one QoI, avoiding the overhead
of Python-level loops.

All functions operate on raw NumPy arrays. The dispatch layer in
avar_dispatch.py handles backend conversion.

Note: This module requires numba. If numba is not available, importing
this module will raise ImportError, which avar_dispatch.py handles
gracefully.
"""

import math

import numpy as np
from numba import njit, prange


@njit(cache=True)
def _project_single(
    values: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    lam: float,
) -> np.ndarray:
    """Project scaled values onto CVaR risk envelope for a single QoI.

    Parameters
    ----------
    values : np.ndarray
        Scaled values. Shape: (nsamples,)
    weights : np.ndarray
        Probability weights. Shape: (nsamples,)
    alpha : float
        Risk level.
    lam : float
        Regularization parameter (unused in projection, kept for API).

    Returns
    -------
    np.ndarray
        Projection. Shape: (nsamples,)
    """
    nsamp = len(values)
    lbnd = 0.0
    ubnd = 1.0 / (1.0 - alpha) if alpha < 1.0 else 1e30

    dvalues = np.empty(nsamp)
    for i in range(nsamp):
        dvalues[i] = values[i] / weights[i]

    # Compute kinks and sort descending
    kinks = np.empty(2 * nsamp)
    for i in range(nsamp):
        kinks[i] = lbnd - dvalues[i]
        kinks[i + nsamp] = ubnd - dvalues[i]
    kinks.sort()
    # Reverse to descending
    for i in range(nsamp):
        kinks[i], kinks[2 * nsamp - 1 - i] = (
            kinks[2 * nsamp - 1 - i],
            kinks[i],
        )

    def residual(x: float) -> float:
        s = 0.0
        for i in range(nsamp):
            val = dvalues[i] + x
            if val < lbnd:
                val = lbnd
            elif val > ubnd:
                val = ubnd
            s += weights[i] * val
        return 1.0 - s

    ibeg = 0
    imid = nsamp
    iend = 2 * nsamp
    x1 = kinks[ibeg]
    y1 = residual(x1)

    # Handle alpha=0 edge case
    if abs(y1) < 3e-16:
        if alpha > 1e-15:
            return np.empty(0)  # error signal
        imid = 1
        y1 = -3.0e-16

    x2 = kinks[imid]
    y2 = residual(x2)

    while True:
        if (y1 >= 0.0) != (y2 >= 0.0):
            # Different signs
            iend = imid
        else:
            ibeg = imid
            x1 = x2
            y1 = y2

        if iend - ibeg == 1:
            imid = iend
        else:
            imid = ibeg + int(round((iend - ibeg) / 2))

        x2 = kinks[imid]
        y2 = residual(x2)

        if iend - ibeg == 1:
            break

    # Linear interpolation
    denom = y2 - y1
    if abs(denom) < 1e-30:
        lam_val = x1
    else:
        lam_val = (y2 * x1 - y1 * x2) / denom

    result = np.empty(nsamp)
    for i in range(nsamp):
        val = dvalues[i] + lam_val
        if val < lbnd:
            val = lbnd
        elif val > ubnd:
            val = ubnd
        result[i] = weights[i] * val
    return result


@njit(cache=True, parallel=True)
def avar_values_numba(
    values: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    delta: float,
    lam: float,
) -> np.ndarray:
    """Compute smoothed AVaR for all QoIs in parallel.

    Parameters
    ----------
    values : np.ndarray
        Sample values. Shape: (nqoi, nsamples)
    weights : np.ndarray
        Probability weights. Shape: (nsamples,)
    alpha : float
        Risk level.
    delta : float
        Smoothing parameter.
    lam : float
        Regularization parameter.

    Returns
    -------
    np.ndarray
        AVaR values. Shape: (nqoi,)
    """
    nqoi = values.shape[0]
    nsamples = values.shape[1]
    result = np.empty(nqoi)

    for q in prange(nqoi):
        # Scale
        scaled = np.empty(nsamples)
        for i in range(nsamples):
            scaled[i] = weights[i] * values[q, i] * delta + lam

        # Project
        proj = _project_single(scaled, weights, alpha, lam)

        # Compute AVaR
        term1 = 0.0
        term2 = 0.0
        for i in range(nsamples):
            term1 += proj[i] * values[q, i]
            diff = (proj[i] - lam) / weights[i]
            term2 += diff * (proj[i] - lam)
        term2 *= 1.0 / (2.0 * delta)
        result[q] = term1 - term2

    return result


@njit(cache=True, parallel=True)
def avar_jacobian_numba(
    values: np.ndarray,
    jac_values: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    delta: float,
    lam: float,
) -> np.ndarray:
    """Compute AVaR Jacobian for all QoIs in parallel.

    Parameters
    ----------
    values : np.ndarray
        Sample values. Shape: (nqoi, nsamples)
    jac_values : np.ndarray
        Jacobians. Shape: (nqoi, nsamples, nvars)
    weights : np.ndarray
        Probability weights. Shape: (nsamples,)
    alpha : float
        Risk level.
    delta : float
        Smoothing parameter.
    lam : float
        Regularization parameter.

    Returns
    -------
    np.ndarray
        Jacobian. Shape: (nqoi, nvars)
    """
    nqoi = values.shape[0]
    nsamples = values.shape[1]
    nvars = jac_values.shape[2]
    result = np.zeros((nqoi, nvars))

    for q in prange(nqoi):
        # Scale
        scaled = np.empty(nsamples)
        for i in range(nsamples):
            scaled[i] = weights[i] * values[q, i] * delta + lam

        # Project
        proj = _project_single(scaled, weights, alpha, lam)

        # einsum "i,ij->j" for this QoI
        for i in range(nsamples):
            for j in range(nvars):
                result[q, j] += proj[i] * jac_values[q, i, j]

    return result
