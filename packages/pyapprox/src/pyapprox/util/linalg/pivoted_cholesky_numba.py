"""Numba JIT-compiled pivoted Cholesky factorization kernels.

Two variants:

1. ``pivoted_cholesky_econ_numba`` — dense matrix input.
2. ``pivoted_cholesky_fused_numba`` — kernel evaluation fused into the
   pivot loop (matrix-free).  Avoids materializing the dense kernel matrix.
   Accepts any ``@njit`` scalar kernel function as a parameter.

Note: This module requires numba. If numba is not available, importing
this module will raise ImportError, which the dispatch in
pivoted_cholesky.py handles gracefully.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numba import njit


@njit(cache=True)
def _find_init_pivot(perm, n, ii, init_pivot_val):
    """Find position of init_pivot_val in perm[ii:]."""
    for j in range(ii, n):
        if perm[j] == init_pivot_val:
            return j
    return ii


@njit(cache=True)
def pivoted_cholesky_econ_numba(
    K: np.ndarray,
    npivots: int,
    tol: float,
    weights: np.ndarray,
    init_pivots: np.ndarray,
    n_init_pivots: int,
    use_weights: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Pivoted Cholesky with diagonal pivot selection.

    Parameters
    ----------
    K : (n, n) float64 array
        Symmetric positive semi-definite matrix.
    npivots : int
        Number of pivots to compute.
    tol : float
        Relative residual tolerance for early termination.
    weights : (n,) float64 array
        Pivot selection weights. Ignored if ``use_weights`` is False.
    init_pivots : (n_init_pivots,) int64 array
        Forced initial pivots. Pass empty array if none.
    n_init_pivots : int
        Number of forced initial pivots.
    use_weights : bool
        Whether to use weighted pivot selection.

    Returns
    -------
    L : (n, npivots) float64 array
        Lower-triangular Cholesky factor.
    perm : (n,) int64 array
        Permutation indices (first ``ncompleted`` are the pivots).
    diag : (n,) float64 array
        Residual diagonal after factorization.
    ncompleted : int
        Number of pivots actually computed.
    """
    n = K.shape[0]
    diag = np.empty(n, dtype=np.float64)
    init_error = 0.0
    for i in range(n):
        diag[i] = K[i, i]
        init_error += abs(diag[i])

    L = np.zeros((n, npivots), dtype=np.float64)
    perm = np.arange(n, dtype=np.int64)
    ncompleted = 0

    for ii in range(npivots):
        if ii < n_init_pivots:
            best = _find_init_pivot(perm, n, ii, init_pivots[ii])
        elif use_weights:
            best = ii
            best_val = weights[perm[ii]] * diag[perm[ii]]
            for j in range(ii + 1, n):
                v = weights[perm[j]] * diag[perm[j]]
                if v > best_val:
                    best_val = v
                    best = j
        else:
            best = ii
            best_val = diag[perm[ii]]
            for j in range(ii + 1, n):
                v = diag[perm[j]]
                if v > best_val:
                    best_val = v
                    best = j

        perm[ii], perm[best] = perm[best], perm[ii]
        p_ii = perm[ii]

        if diag[p_ii] <= 1e-14:
            break

        sqrt_d = np.sqrt(diag[p_ii])
        L[p_ii, ii] = sqrt_d
        inv_sqrt = 1.0 / sqrt_d

        for j in range(ii + 1, n):
            p_j = perm[j]
            s = K[p_j, p_ii]
            for k in range(ii):
                s -= L[p_j, k] * L[p_ii, k]
            L[p_j, ii] = s * inv_sqrt
            diag[p_j] -= L[p_j, ii] * L[p_j, ii]

        ncompleted = ii + 1

        rel_error = 0.0
        for j in range(ii + 1, n):
            rel_error += diag[perm[j]]
        rel_error /= init_error
        if rel_error < tol:
            break

    return L, perm, diag, ncompleted


# -------------------------------------------------------------------
# Fused kernel + pchol: kernel function passed as parameter
# -------------------------------------------------------------------


@njit(cache=True)
def pivoted_cholesky_fused_numba(
    X: np.ndarray,
    kernel_func: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    kernel_params: np.ndarray,
    npivots: int,
    tol: float,
    weights: np.ndarray,
    init_pivots: np.ndarray,
    n_init_pivots: int,
    use_weights: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    """Pivoted Cholesky with fused kernel evaluation.

    Evaluates kernel_func(X[:, i], X[:, j], params) on the fly instead
    of reading from a precomputed dense kernel matrix.

    Parameters
    ----------
    X : (nvars, n) float64 array
        Data points (columns are samples).
    kernel_func : numba jitted callable
        Scalar kernel ``f(xi, xj, params) -> float64`` where
        ``xi``, ``xj`` are ``(nvars,)`` arrays.
    kernel_params : (nparams,) float64 array
        Kernel parameters (e.g. exponentiated length scales).
    npivots : int
        Number of pivots to compute.
    tol : float
        Relative residual tolerance for early termination.
    weights : (n,) float64 array
        Pivot selection weights. Ignored if ``use_weights`` is False.
    init_pivots : (n_init_pivots,) int64 array
        Forced initial pivots.
    n_init_pivots : int
        Number of forced initial pivots.
    use_weights : bool
        Whether to use weighted pivot selection.

    Returns
    -------
    L : (n, npivots) float64 array
    perm : (n,) int64 array
    diag : (n,) float64 array
    ncompleted : int
    init_error : float
    """
    n = X.shape[1]
    diag = np.empty(n, dtype=np.float64)
    init_error = 0.0
    for i in range(n):
        diag[i] = kernel_func(X[:, i], X[:, i], kernel_params)
        init_error += abs(diag[i])

    L = np.zeros((n, npivots), dtype=np.float64)
    perm = np.arange(n, dtype=np.int64)
    ncompleted = 0

    for ii in range(npivots):
        if ii < n_init_pivots:
            best = _find_init_pivot(perm, n, ii, init_pivots[ii])
        elif use_weights:
            best = ii
            best_val = weights[perm[ii]] * diag[perm[ii]]
            for j in range(ii + 1, n):
                v = weights[perm[j]] * diag[perm[j]]
                if v > best_val:
                    best_val = v
                    best = j
        else:
            best = ii
            best_val = diag[perm[ii]]
            for j in range(ii + 1, n):
                v = diag[perm[j]]
                if v > best_val:
                    best_val = v
                    best = j

        perm[ii], perm[best] = perm[best], perm[ii]
        p_ii = perm[ii]

        if diag[p_ii] <= 1e-14:
            break

        sqrt_d = np.sqrt(diag[p_ii])
        L[p_ii, ii] = sqrt_d
        inv_sqrt = 1.0 / sqrt_d

        x_pivot = X[:, p_ii]
        for j in range(ii + 1, n):
            p_j = perm[j]
            k_val = kernel_func(X[:, p_j], x_pivot, kernel_params)
            s = k_val
            for k in range(ii):
                s -= L[p_j, k] * L[p_ii, k]
            L[p_j, ii] = s * inv_sqrt
            diag[p_j] -= L[p_j, ii] * L[p_j, ii]

        ncompleted = ii + 1

        rel_error = 0.0
        for j in range(ii + 1, n):
            rel_error += diag[perm[j]]
        rel_error /= init_error
        if rel_error < tol:
            break

    return L, perm, diag, ncompleted, init_error
