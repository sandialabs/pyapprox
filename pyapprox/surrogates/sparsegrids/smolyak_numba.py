"""Numba JIT-compiled Smolyak coefficient computation.

Fuses the double loop over shifts and subspaces into a tight kernel with
binary search for membership checks. All Python overhead is eliminated.
"""

from typing import Any

import numpy as np
import numpy.typing as npt
from numba import njit, prange


@njit(cache=True)
def _encode_indices(indices: npt.NDArray[np.int64], nvars: int, n: int, base: int) -> npt.NDArray[np.int64]:
    """Encode multi-indices as single int64 via mixed-radix packing.

    Parameters
    ----------
    indices : np.ndarray
        Shape (nvars, n). Each column is a multi-index.
    nvars : int
        Number of variables.
    n : int
        Number of indices.
    base : int
        Radix base (must exceed max index value + max shift).

    Returns
    -------
    np.ndarray
        Shape (n,) of int64 encoded values.
    """
    encoded = np.zeros(n, dtype=np.int64)
    for j in range(n):
        val = np.int64(0)
        for d in range(nvars):
            val = val * base + indices[d, j]
        encoded[j] = val
    return encoded


@njit(cache=True)
def _binary_search(sorted_arr: npt.NDArray[np.int64], target: np.int64, n: int) -> bool:
    """Return True if target is in sorted_arr using binary search."""
    lo = 0
    hi = n - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if sorted_arr[mid] == target:
            return True
        elif sorted_arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return False


@njit(cache=True, parallel=True)
def smolyak_coefficients_numba(
    np_indices: npt.NDArray[np.int64],
    np_shifts: npt.NDArray[np.int64],
    np_signs: npt.NDArray[np.floating[Any]],
    nvars: int,
    nsubspaces: int,
    nshifts: int,
) -> npt.NDArray[np.floating[Any]]:
    """Compute Smolyak combination coefficients using numba.

    Parameters
    ----------
    np_indices : np.ndarray
        Multi-indices, shape (nvars, nsubspaces), dtype int64.
    np_shifts : np.ndarray
        All 2^nvars shift vectors, shape (nvars, nshifts), dtype int64.
    np_signs : np.ndarray
        Signs (-1)^|e| for each shift, shape (nshifts,), dtype float64.
    nvars : int
        Number of variables.
    nsubspaces : int
        Number of subspaces.
    nshifts : int
        Number of shifts (2^nvars).

    Returns
    -------
    np.ndarray
        Combination coefficients, shape (nsubspaces,), dtype float64.
    """
    # Determine mixed-radix base: must exceed max possible index value + 1
    max_val = np.int64(0)
    for d in range(nvars):
        for j in range(nsubspaces):
            if np_indices[d, j] > max_val:
                max_val = np_indices[d, j]
    # Shifted indices can be at most max_val + 1
    base = max_val + 2

    # Encode and sort the index set for binary search
    index_encoded = _encode_indices(np_indices, nvars, nsubspaces, base)
    index_sorted = np.sort(index_encoded)

    # Precompute shift encodings as deltas
    # For shift s, the encoded shifted index j is:
    #   encode(indices[:,j] + shifts[:,s])
    # We can precompute the delta for each shift
    shift_deltas = np.zeros(nshifts, dtype=np.int64)
    for s in range(nshifts):
        delta = np.int64(0)
        for d in range(nvars):
            delta = delta * base + np_shifts[d, s]
        shift_deltas[s] = delta

    # Main computation: for each subspace, sum signs over matching shifts
    np_coefs = np.zeros(nsubspaces, dtype=np.float64)

    for j in prange(nsubspaces):
        encoded_j = index_encoded[j]
        coef = 0.0
        for s in range(nshifts):
            target = encoded_j + shift_deltas[s]
            if _binary_search(index_sorted, target, nsubspaces):
                coef += np_signs[s]
        np_coefs[j] = coef

    return np_coefs
