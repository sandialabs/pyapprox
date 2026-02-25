"""Numba JIT-compiled tensor product evaluation kernel.

Implements general N-D contraction using flat workspace arrays with prange
parallelism. Operates on raw NumPy arrays: a padded (nvars, npoints, max_n1d)
basis array and a (nvars,) int array of actual nterms per dimension.

Note: This module requires numba. If numba is not available, importing
this module will raise ImportError, which dispatch.py handles gracefully.
"""

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def tp_eval_numba(
    values_flat: np.ndarray,
    basis_vals_pad: np.ndarray,
    nterms_1d_arr: np.ndarray,
    nvars: int,
    nqoi: int,
    npoints: int,
) -> np.ndarray:
    """Evaluate tensor product interpolant via dimension-by-dimension contraction.

    Contracts dimensions from last (d=nvars-1) to first (d=0). At each step
    the intermediate tensor is stored in a flat 1D workspace to avoid Numba
    limitations with dynamic-shape arrays.

    Parameters
    ----------
    values_flat : np.ndarray
        Coefficient values with shape (nqoi, total_terms) where
        total_terms = prod(nterms_1d_arr).
    basis_vals_pad : np.ndarray
        Padded 1D basis evaluations with shape (nvars, npoints, max_n1d).
        Dimension d uses columns 0..nterms_1d_arr[d]-1.
    nterms_1d_arr : np.ndarray
        Number of terms per dimension, shape (nvars,), dtype int64.
    nvars : int
        Number of dimensions.
    nqoi : int
        Number of quantities of interest.
    npoints : int
        Number of evaluation points.

    Returns
    -------
    np.ndarray
        Result with shape (nqoi, npoints).
    """
    # Compute total terms
    total_terms = 1
    for d in range(nvars):
        total_terms *= nterms_1d_arr[d]

    # After contracting dimension d, the intermediate shape is conceptually:
    #   (nqoi, n0, n1, ..., n_{d-1}, npoints)
    # with "remaining" = n0 * n1 * ... * n_{d-1} and npoints at the end.
    # We store it flat: inter[q * remaining * npoints + r * npoints + p]

    # Start: current = values_flat reshaped conceptually as
    #   (nqoi, n0*n1*...*n_{D-1})
    # We'll contract the last dim first.

    # Workspace: double-buffer to avoid extra allocation
    # Max workspace size: nqoi * max(total_terms, npoints * total_terms / min_n)
    # Conservative: nqoi * total_terms is always enough for source
    # After first contraction: nqoi * (total_terms / n_{D-1}) * npoints
    # We need max of all intermediate sizes
    max_inter = 0
    remaining = total_terms
    for d in range(nvars - 1, -1, -1):
        n_d = nterms_1d_arr[d]
        remaining_before = remaining // n_d
        out_size = nqoi * remaining_before * npoints
        if out_size > max_inter:
            max_inter = out_size
        remaining = remaining_before

    buf_a = np.empty(max(nqoi * total_terms, max_inter))
    buf_b = np.empty(max_inter)

    # Copy values_flat into buf_a
    for q in range(nqoi):
        for i in range(total_terms):
            buf_a[q * total_terms + i] = values_flat[q, i]

    remaining = total_terms
    src = buf_a
    dst = buf_b

    for d in range(nvars - 1, -1, -1):
        n_d = nterms_1d_arr[d]
        remaining_before = remaining // n_d

        if d == nvars - 1:
            # First contraction:
            # src layout: (nqoi, remaining_before, n_d) flat
            # basis: (npoints, n_d) in basis_vals_pad[d, :, :n_d]
            # dst layout: (nqoi, remaining_before, npoints) flat
            for q in range(nqoi):
                for r in prange(remaining_before):
                    src_off = q * remaining + r * n_d
                    dst_off = q * remaining_before * npoints + r * npoints
                    for p in range(npoints):
                        acc = 0.0
                        for k in range(n_d):
                            acc += src[src_off + k] * basis_vals_pad[d, p, k]
                        dst[dst_off + p] = acc
        else:
            # Subsequent contractions:
            # src layout: (nqoi, remaining_before, n_d, npoints) flat
            # basis: (npoints, n_d) in basis_vals_pad[d, :, :n_d]
            # Contract over n_d (matching p in both), result:
            #   (nqoi, remaining_before, npoints) flat
            for q in range(nqoi):
                for r in prange(remaining_before):
                    src_off = q * remaining * npoints + r * n_d * npoints
                    dst_off = q * remaining_before * npoints + r * npoints
                    for p in range(npoints):
                        acc = 0.0
                        for k in range(n_d):
                            acc += (
                                src[src_off + k * npoints + p] * basis_vals_pad[d, p, k]
                            )
                        dst[dst_off + p] = acc

        remaining = remaining_before
        # Swap buffers
        src, dst = dst, src

    # Result is in src, layout: (nqoi, npoints) flat
    result = np.empty((nqoi, npoints))
    for q in range(nqoi):
        for p in prange(npoints):
            result[q, p] = src[q * npoints + p]

    return result
