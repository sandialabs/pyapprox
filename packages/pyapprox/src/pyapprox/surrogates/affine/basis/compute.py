"""
Vectorized backend-agnostic tensor product assembly for MultiIndexBasis.

Uses Backend[Array] protocol only. Compatible with NumPy, PyTorch, and any
future backend. Preserves autograd graph for PyTorch.
"""

from typing import List, Tuple

from pyapprox.util.backends.protocols import Array, Backend


def basis_eval_vectorized(
    vals_1d: List[Array],
    indices: Array,
    nvars: int,
    bkd: Backend[Array],
) -> Array:
    """Assemble tensor product basis matrix from 1D values.

    Gathers selected 1D values into a 3D array and reduces with prod(),
    replacing the Python multiply-loop.

    Parameters
    ----------
    vals_1d : List[Array]
        Univariate basis values. vals_1d[d] has shape (nsamples, nterms_1d_d).
    indices : Array
        Multi-indices. Shape: (nvars, nterms). Integer dtype.
    nvars : int
        Number of variables.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        Basis matrix. Shape: (nsamples, nterms).
    """
    gathered = [vals_1d[dd][:, indices[dd, :]] for dd in range(nvars)]
    stacked = bkd.stack(gathered, axis=0)  # (nvars, nsamples, nterms)
    return bkd.prod(stacked, axis=0)  # (nsamples, nterms)


def _gather_and_prefix_suffix(
    vals_1d: List[Array],
    indices: Array,
    nvars: int,
    bkd: Backend[Array],
) -> Tuple[List[Array], List[Array], List[Array]]:
    """Gather selected 1D values and compute prefix/suffix products.

    Returns
    -------
    gathered_vals : List[Array]
        Each element has shape (nsamples, nterms).
    prefix : List[Array]
        prefix[d] = prod(gathered_vals[0], ..., gathered_vals[d-1]).
    suffix : List[Array]
        suffix[d] = prod(gathered_vals[d+1], ..., gathered_vals[nvars-1]).
    """
    gathered_vals = [vals_1d[dd][:, indices[dd, :]] for dd in range(nvars)]

    nsamples = vals_1d[0].shape[0]
    nterms = indices.shape[1]
    ones = bkd.ones((nsamples, nterms))

    prefix: List[Array] = [None] * nvars  # type: ignore[list-item]
    prefix[0] = ones
    for dd in range(1, nvars):
        prefix[dd] = prefix[dd - 1] * gathered_vals[dd - 1]

    suffix: List[Array] = [None] * nvars  # type: ignore[list-item]
    suffix[nvars - 1] = ones
    for dd in range(nvars - 2, -1, -1):
        suffix[dd] = suffix[dd + 1] * gathered_vals[dd + 1]

    return gathered_vals, prefix, suffix


def basis_jacobian_vectorized(
    vals_1d: List[Array],
    derivs_1d: List[Array],
    indices: Array,
    nvars: int,
    bkd: Backend[Array],
) -> Array:
    """Assemble Jacobian of tensor product basis using prefix/suffix products.

    Uses the leave-one-out product trick: for each dimension d,
    jac[:,:,d] = derivs[d] * prefix[d] * suffix[d]
    where prefix[d] = prod(vals[0..d-1]) and suffix[d] = prod(vals[d+1..end]).

    This is O(nvars) multiplies instead of O(nvars^2).

    Parameters
    ----------
    vals_1d : List[Array]
        Univariate basis values. vals_1d[d] shape: (nsamples, nterms_1d_d).
    derivs_1d : List[Array]
        Univariate first derivatives. derivs_1d[d] shape: (nsamples, nterms_1d_d).
    indices : Array
        Multi-indices. Shape: (nvars, nterms). Integer dtype.
    nvars : int
        Number of variables.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        Jacobian matrix. Shape: (nsamples, nterms, nvars).
    """
    gathered_vals, prefix, suffix = _gather_and_prefix_suffix(
        vals_1d, indices, nvars, bkd
    )
    gathered_derivs = [derivs_1d[dd][:, indices[dd, :]] for dd in range(nvars)]

    jac_list = []
    for dd in range(nvars):
        jac_dd = gathered_derivs[dd] * prefix[dd] * suffix[dd]
        jac_list.append(jac_dd)

    return bkd.moveaxis(bkd.stack(jac_list, axis=0), 0, -1)


def basis_hessian_vectorized(
    vals_1d: List[Array],
    derivs_1d: List[Array],
    hess_1d: List[Array],
    indices: Array,
    nvars: int,
    bkd: Backend[Array],
) -> Array:
    """Assemble Hessian of tensor product basis.

    Diagonal (d==k): hess_1d[d] * leave_one_out[d]
    Off-diagonal (d!=k): derivs[d] * derivs[k] * leave_two_out[d,k]

    Reuses prefix/suffix from the Jacobian computation for leave-one-out.

    Parameters
    ----------
    vals_1d : List[Array]
        Univariate basis values. vals_1d[d] shape: (nsamples, nterms_1d_d).
    derivs_1d : List[Array]
        Univariate first derivatives. derivs_1d[d] shape: (nsamples, nterms_1d_d).
    hess_1d : List[Array]
        Univariate second derivatives. hess_1d[d] shape: (nsamples, nterms_1d_d).
    indices : Array
        Multi-indices. Shape: (nvars, nterms). Integer dtype.
    nvars : int
        Number of variables.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        Hessian. Shape: (nsamples, nterms, nvars, nvars).
    """
    gathered_vals, prefix, suffix = _gather_and_prefix_suffix(
        vals_1d, indices, nvars, bkd
    )
    gathered_derivs = [derivs_1d[dd][:, indices[dd, :]] for dd in range(nvars)]
    gathered_hess = [hess_1d[dd][:, indices[dd, :]] for dd in range(nvars)]

    nsamples = vals_1d[0].shape[0]
    nterms = indices.shape[1]
    result = bkd.zeros((nsamples, nterms, nvars, nvars))

    for dd in range(nvars):
        # leave_one_out[dd] = prefix[dd] * suffix[dd]
        leave_one_out_dd = prefix[dd] * suffix[dd]

        # Diagonal: hess_1d[dd] * leave_one_out[dd]
        result[:, :, dd, dd] = gathered_hess[dd] * leave_one_out_dd

        for kk in range(dd + 1, nvars):
            # Off-diagonal: derivs[dd] * derivs[kk] * leave_two_out[dd,kk]
            # leave_two_out[dd,kk] = prefix[dd] * (product of vals[dd+1..kk-1])
            #                        * suffix[kk]
            # But it's simpler to compute as:
            # leave_one_out[dd] / vals[kk] — but that risks division by zero.
            # Instead, directly compute the product excluding dd and kk:
            hess_dk = gathered_derivs[dd] * gathered_derivs[kk]
            for ll in range(nvars):
                if ll != dd and ll != kk:
                    hess_dk = hess_dk * gathered_vals[ll]
            result[:, :, dd, kk] = hess_dk
            result[:, :, kk, dd] = hess_dk

    return result
