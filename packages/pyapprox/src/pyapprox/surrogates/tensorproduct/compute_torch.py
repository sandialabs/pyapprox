"""torch.compile-ready tensor product evaluation.

Uses torch.einsum directly (bypassing bkd.*) to avoid graph breaks
during torch.compile tracing. Same algorithm as the vectorized fallback
but with torch-native calls.
"""

from typing import List

import torch


def tp_eval_torch(
    basis_vals_1d: List[torch.Tensor],
    values: torch.Tensor,
    nterms_1d: List[int],
) -> torch.Tensor:
    """Evaluate tensor product interpolant via dimension-by-dimension contraction.

    Parameters
    ----------
    basis_vals_1d : List[torch.Tensor]
        1D basis evaluations, each with shape (npoints, nterms_1d[d]).
    values : torch.Tensor
        Coefficient values with shape (nqoi, prod(nterms_1d)).
    nterms_1d : List[int]
        Number of terms in each dimension.

    Returns
    -------
    torch.Tensor
        Evaluation result with shape (nqoi, npoints).
    """
    nvars = len(nterms_1d)

    # Reshape values to (nqoi, n0, n1, ..., n_{D-1})
    coeff = values.reshape([values.shape[0]] + list(nterms_1d))

    for d in range(nvars - 1, -1, -1):
        if d == nvars - 1:
            coeff = torch.einsum("...k,pk->...p", coeff, basis_vals_1d[d])
        else:
            coeff = torch.einsum("...kp,pk->...p", coeff, basis_vals_1d[d])

    return coeff
