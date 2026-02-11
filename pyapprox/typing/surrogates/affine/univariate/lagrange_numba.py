"""Numba JIT-compiled barycentric Lagrange polynomial evaluation.

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
