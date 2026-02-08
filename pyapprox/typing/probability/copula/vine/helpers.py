"""
Helper functions for D-vine copula construction.

Provides utilities for:
- Determining precision matrix bandwidth
- Computing D-vine partial correlations from a correlation matrix
- Reconstructing a correlation matrix from partial correlations
"""

from typing import Dict, List

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.linalg.cholesky_factor import CholeskyFactor


def precision_bandwidth(
    precision: Array, bkd: Backend[Array], tol: float = 1e-12
) -> int:
    """
    Compute the bandwidth of a precision matrix.

    The bandwidth k is the smallest integer such that
    |Omega_{ij}| <= tol for all |i - j| > k.

    Parameters
    ----------
    precision : Array
        Precision matrix. Shape: (n, n)
    bkd : Backend[Array]
        Computational backend.
    tol : float
        Tolerance for zero entries.

    Returns
    -------
    int
        Bandwidth (0 = diagonal, 1 = tridiagonal, etc.)
    """
    n = precision.shape[0]
    bandwidth = 0
    for i in range(n):
        for j in range(i + 1, n):
            val = bkd.abs(precision[i, j])
            if bkd.to_numpy(val) > tol:
                bandwidth = max(bandwidth, j - i)
    return bandwidth


def compute_dvine_partial_correlations(
    correlation: Array,
    truncation_level: int,
    bkd: Backend[Array],
) -> Dict[int, List[float]]:
    """
    Compute partial correlations for a D-vine from a correlation matrix.

    For each tree level t = 1, ..., truncation_level, compute the partial
    correlation for each edge (e, e+t | {e+1,...,e+t-1}) using submatrix
    inversion of the correlation matrix.

    Parameters
    ----------
    correlation : Array
        Correlation matrix. Shape: (n, n)
    truncation_level : int
        Maximum tree level.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Dict[int, List[float]]
        Maps tree level (1-indexed) to list of partial correlations.
        Tree t has n-t entries.
    """
    n = correlation.shape[0]
    result: Dict[int, List[float]] = {}

    for t in range(1, truncation_level + 1):
        result[t] = []
        for e in range(n - t):
            if t == 1:
                rho = correlation[e, e + 1]
            else:
                sub = correlation[e : e + t + 1, e : e + t + 1]
                L = bkd.cholesky(sub)
                chol = CholeskyFactor(L, bkd)
                P = chol.matrix_inverse()
                rho = -P[0, t] / bkd.sqrt(P[0, 0] * P[t, t])
            result[t].append(float(bkd.to_numpy(rho)))

    return result


def correlation_from_partial_correlations(
    partial_correlations: Dict[int, List[float]],
    nvars: int,
    bkd: Backend[Array],
) -> Array:
    """
    Reconstruct a correlation matrix from D-vine partial correlations.

    Uses the reverse Schur complement formula. Processes entries from
    small separation t to large, so all needed intermediate entries
    are available.

    Parameters
    ----------
    partial_correlations : Dict[int, List[float]]
        Maps tree level (1-indexed) to list of partial correlations.
    nvars : int
        Number of variables.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        Correlation matrix. Shape: (nvars, nvars)
    """
    R = bkd.eye(nvars)

    for t in range(1, nvars):
        for e in range(nvars - t):
            if t in partial_correlations and e < len(
                partial_correlations[t]
            ):
                rho_partial = partial_correlations[t][e]
            else:
                rho_partial = 0.0

            if t == 1:
                R[e, e + 1] = rho_partial
                R[e + 1, e] = rho_partial
            else:
                S = list(range(e + 1, e + t))
                R_inner = R[S][:, S]
                L_inner = bkd.cholesky(R_inner)
                chol_inner = CholeskyFactor(L_inner, bkd)
                R_inner_inv = chol_inner.matrix_inverse()

                r_0S = R[e, S]
                r_tS = R[e + t, S]

                a = r_0S @ R_inner_inv @ r_0S
                b = r_tS @ R_inner_inv @ r_tS
                c = r_0S @ R_inner_inv @ r_tS

                r = c + rho_partial * bkd.sqrt(
                    (1.0 - a) * (1.0 - b)
                )
                R[e, e + t] = r
                R[e + t, e] = r

    return R
