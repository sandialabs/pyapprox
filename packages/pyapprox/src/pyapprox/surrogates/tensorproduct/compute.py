"""Backend-agnostic vectorized tensor product evaluation.

Implements dimension-by-dimension contraction using bkd.einsum, avoiding
the materialization of the full (npoints, nterms) interpolation matrix.

This is the fallback strategy used when neither Numba nor torch.compile
dispatch is active.
"""

from typing import List

from pyapprox.util.backends.protocols import Array, Backend


def tp_eval_vectorized(
    basis_vals_1d: List[Array],
    values: Array,
    nterms_1d: List[int],
    bkd: Backend[Array],
) -> Array:
    """Evaluate tensor product interpolant via dimension-by-dimension contraction.

    Reshapes the coefficient tensor to (nqoi, n0, n1, ..., n_{D-1}), then
    contracts each dimension with its 1D basis matrix using einsum. This
    avoids materializing the full (npoints, prod(nterms_1d)) interpolation
    matrix, reducing both memory and compute.

    Parameters
    ----------
    basis_vals_1d : List[Array]
        1D basis evaluations, each with shape (npoints, nterms_1d[d]).
    values : Array
        Coefficient values with shape (nqoi, prod(nterms_1d)).
    nterms_1d : List[int]
        Number of terms in each dimension.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        Evaluation result with shape (nqoi, npoints).
    """
    nvars = len(nterms_1d)
    nqoi = values.shape[0]

    # Reshape values to (nqoi, n0, n1, ..., n_{D-1})
    coeff = bkd.reshape(values, [nqoi] + list(nterms_1d))

    # Contract dimensions from last to first.
    # After contracting dim d, the axis that was nterms_1d[d] becomes npoints.
    # The contracted npoints axis is always the last axis.
    for d in range(nvars - 1, -1, -1):
        if d == nvars - 1:
            # First contraction: coeff has shape (..., n_d)
            # basis has shape (npoints, n_d)
            # Result: (..., npoints)
            coeff = bkd.einsum("...k,pk->...p", coeff, basis_vals_1d[d])
        else:
            # Subsequent contractions: coeff has shape (..., n_d, npoints)
            # basis has shape (npoints, n_d)
            # Contract n_d, keep npoints aligned
            coeff = bkd.einsum("...kp,pk->...p", coeff, basis_vals_1d[d])

    # coeff now has shape (nqoi, npoints)
    return coeff
