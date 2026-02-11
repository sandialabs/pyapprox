"""Numba JIT-compiled barycentric Lagrange polynomial evaluation and derivatives.

Evaluates all Lagrange basis functions at all sample points using the
identity L_j(x) = w_j * P(x) / (x - x_j), where w_j are barycentric
weights and P(x) = prod_i(x - x_i).

Operates on raw NumPy arrays. The dispatch layer handles backend conversion.
"""

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def lagrange_eval_numba(
    abscissa: np.ndarray,
    samples: np.ndarray,
    bary_weights: np.ndarray,
) -> np.ndarray:
    """Evaluate Lagrange basis polynomials via barycentric formula.

    Parameters
    ----------
    abscissa : np.ndarray
        Interpolation nodes, shape (nabscissa,).
    samples : np.ndarray
        Evaluation points, shape (nsamples,).
    bary_weights : np.ndarray
        Precomputed barycentric weights, shape (nabscissa,).

    Returns
    -------
    np.ndarray
        Basis values, shape (nsamples, nabscissa).
    """
    nsamples = samples.shape[0]
    nabscissa = abscissa.shape[0]
    result = np.empty((nsamples, nabscissa))

    for i in prange(nsamples):
        x = samples[i]

        # Check if x coincides with any node
        hit_idx = -1
        for j in range(nabscissa):
            if x == abscissa[j]:
                hit_idx = j
                break

        if hit_idx >= 0:
            # Delta function: 1 at matching node, 0 elsewhere
            for j in range(nabscissa):
                result[i, j] = 0.0
            result[i, hit_idx] = 1.0
        else:
            # Barycentric formula: L_j(x) = w_j * P(x) / (x - x_j)
            # Compute P(x) = prod_k(x - x_k)
            full_prod = 1.0
            for j in range(nabscissa):
                full_prod *= x - abscissa[j]

            for j in range(nabscissa):
                result[i, j] = bary_weights[j] * full_prod / (x - abscissa[j])

    return result


@njit(cache=True)
def _node_first_derivs(
    abscissa: np.ndarray,
    bary_weights: np.ndarray,
) -> np.ndarray:
    """Precompute first derivative matrix at nodes.

    D1[m, j] = L'_j(x_m).

    Returns
    -------
    np.ndarray
        Shape (nabscissa, nabscissa).
    """
    n = abscissa.shape[0]
    D1 = np.empty((n, n))
    for m in range(n):
        row_sum = 0.0
        for j in range(n):
            if j != m:
                val = bary_weights[j] / (
                    bary_weights[m] * (abscissa[m] - abscissa[j])
                )
                D1[m, j] = val
                row_sum += val
            else:
                D1[m, j] = 0.0  # placeholder
        D1[m, m] = -row_sum
    return D1


@njit(cache=True, parallel=True)
def lagrange_jacobian_numba(
    abscissa: np.ndarray,
    samples: np.ndarray,
    bary_weights: np.ndarray,
) -> np.ndarray:
    """Evaluate first derivatives of Lagrange basis polynomials.

    Uses L'_j(x) = L_j(x) * S_j(x) where S_j = sum_{k!=j} 1/(x-x_k).
    At exact nodes, uses precomputed closed-form limits.

    Parameters
    ----------
    abscissa : np.ndarray
        Interpolation nodes, shape (nabscissa,).
    samples : np.ndarray
        Evaluation points, shape (nsamples,).
    bary_weights : np.ndarray
        Precomputed barycentric weights, shape (nabscissa,).

    Returns
    -------
    np.ndarray
        First derivatives, shape (nsamples, nabscissa).
    """
    nsamples = samples.shape[0]
    nabscissa = abscissa.shape[0]
    result = np.empty((nsamples, nabscissa))

    if nabscissa == 1:
        for i in range(nsamples):
            result[i, 0] = 0.0
        return result

    # Precompute node derivative matrix
    D1 = _node_first_derivs(abscissa, bary_weights)

    # Tolerance for near-node detection: derivatives use 1/(x-x_k) which
    # suffers catastrophic cancellation when x ≈ x_k
    tol = 1e-12 * (1.0 + np.max(np.abs(abscissa)))

    for i in prange(nsamples):
        x = samples[i]

        # Check if x is near any node
        hit_idx = -1
        min_dist = np.inf
        for j in range(nabscissa):
            d = abs(x - abscissa[j])
            if d < min_dist:
                min_dist = d
                hit_idx = j

        if min_dist < tol:
            # Use precomputed node derivatives
            for j in range(nabscissa):
                result[i, j] = D1[hit_idx, j]
        else:
            # Compute L_j(x) via barycentric formula
            full_prod = 1.0
            for j in range(nabscissa):
                full_prod *= x - abscissa[j]

            # Compute L_j(x) and S_j(x) simultaneously
            # S_j(x) = sum_{k!=j} 1/(x - x_k) = total_sum - 1/(x - x_j)
            total_inv_sum = 0.0
            for k in range(nabscissa):
                total_inv_sum += 1.0 / (x - abscissa[k])

            for j in range(nabscissa):
                diff_j = x - abscissa[j]
                Lj = bary_weights[j] * full_prod / diff_j
                Sj = total_inv_sum - 1.0 / diff_j
                result[i, j] = Lj * Sj

    return result


@njit(cache=True, parallel=True)
def lagrange_hessian_numba(
    abscissa: np.ndarray,
    samples: np.ndarray,
    bary_weights: np.ndarray,
) -> np.ndarray:
    """Evaluate second derivatives of Lagrange basis polynomials.

    Uses L''_j(x) = L_j(x) * (S_j(x)^2 - T_j(x)) where
    S_j = sum_{k!=j} 1/(x-x_k), T_j = sum_{k!=j} 1/(x-x_k)^2.
    At exact nodes, uses precomputed closed-form limits.

    Parameters
    ----------
    abscissa : np.ndarray
        Interpolation nodes, shape (nabscissa,).
    samples : np.ndarray
        Evaluation points, shape (nsamples,).
    bary_weights : np.ndarray
        Precomputed barycentric weights, shape (nabscissa,).

    Returns
    -------
    np.ndarray
        Second derivatives, shape (nsamples, nabscissa).
    """
    nsamples = samples.shape[0]
    nabscissa = abscissa.shape[0]
    result = np.empty((nsamples, nabscissa))

    if nabscissa == 1:
        for i in range(nsamples):
            result[i, 0] = 0.0
        return result

    # Precompute node derivative matrices
    D1 = _node_first_derivs(abscissa, bary_weights)

    # D2[m, j] = L''_j(x_m)
    D2 = np.empty((nabscissa, nabscissa))
    for m in range(nabscissa):
        row_sum = 0.0
        for j in range(nabscissa):
            if j != m:
                inv_diff = 1.0 / (abscissa[m] - abscissa[j])
                val = 2.0 * D1[m, j] * (D1[m, m] - inv_diff)
                D2[m, j] = val
                row_sum += val
            else:
                D2[m, j] = 0.0  # placeholder
        D2[m, m] = -row_sum

    # Tolerance for near-node detection
    tol = 1e-12 * (1.0 + np.max(np.abs(abscissa)))

    for i in prange(nsamples):
        x = samples[i]

        # Check if x is near any node
        hit_idx = -1
        min_dist = np.inf
        for j in range(nabscissa):
            d = abs(x - abscissa[j])
            if d < min_dist:
                min_dist = d
                hit_idx = j

        if min_dist < tol:
            for j in range(nabscissa):
                result[i, j] = D2[hit_idx, j]
        else:
            # Compute L_j(x) via barycentric formula
            full_prod = 1.0
            for j in range(nabscissa):
                full_prod *= x - abscissa[j]

            # Compute totals for S and T
            total_inv = 0.0
            total_inv_sq = 0.0
            for k in range(nabscissa):
                inv_k = 1.0 / (x - abscissa[k])
                total_inv += inv_k
                total_inv_sq += inv_k * inv_k

            for j in range(nabscissa):
                diff_j = x - abscissa[j]
                inv_j = 1.0 / diff_j
                Lj = bary_weights[j] * full_prod / diff_j
                Sj = total_inv - inv_j
                Tj = total_inv_sq - inv_j * inv_j
                result[i, j] = Lj * (Sj * Sj - Tj)

    return result
