"""
Numba JIT-compiled fused kernels for MultiIndexBasis tensor product assembly.

These kernels avoid materializing intermediate arrays by fusing the gather
and product operations into tight loops. Each kernel operates on a pre-stacked
3D array (nvars, nsamples, max_nterms_1d) produced by the dispatch layer.

All functions operate on raw NumPy arrays (not backend-wrapped).
The dispatch layer in dispatch.py handles the conversion.
"""

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def basis_eval_numba(
    vals_1d: np.ndarray,
    indices: np.ndarray,
    nvars: int,
    nsamples: int,
    nterms: int,
) -> np.ndarray:
    """Evaluate tensor product basis via fused gather-multiply kernel.

    Parameters
    ----------
    vals_1d : np.ndarray
        Stacked univariate basis values. Shape: (nvars, nsamples, max_nterms_1d).
    indices : np.ndarray
        Multi-indices. Shape: (nvars, nterms). Integer dtype.
    nvars : int
        Number of variables.
    nsamples : int
        Number of sample points.
    nterms : int
        Number of basis terms.

    Returns
    -------
    np.ndarray
        Basis matrix. Shape: (nsamples, nterms).
    """
    result = np.empty((nsamples, nterms))
    for ss in prange(nsamples):
        for tt in range(nterms):
            val = 1.0
            for dd in range(nvars):
                val *= vals_1d[dd, ss, indices[dd, tt]]
            result[ss, tt] = val
    return result


@njit(cache=True, parallel=True)
def basis_jacobian_numba(
    vals_1d: np.ndarray,
    derivs_1d: np.ndarray,
    indices: np.ndarray,
    nvars: int,
    nsamples: int,
    nterms: int,
) -> np.ndarray:
    """Evaluate Jacobian of tensor product basis via fused kernel.

    For each dimension d, computes:
        jac[s, t, d] = derivs_1d[d, s, idx_d] * prod_{k != d} vals_1d[k, s, idx_k]

    Parameters
    ----------
    vals_1d : np.ndarray
        Stacked univariate basis values. Shape: (nvars, nsamples, max_nterms_1d).
    derivs_1d : np.ndarray
        Stacked univariate first derivatives. Shape: (nvars, nsamples, max_nterms_1d).
    indices : np.ndarray
        Multi-indices. Shape: (nvars, nterms). Integer dtype.
    nvars : int
        Number of variables.
    nsamples : int
        Number of sample points.
    nterms : int
        Number of basis terms.

    Returns
    -------
    np.ndarray
        Jacobian. Shape: (nsamples, nterms, nvars).
    """
    result = np.empty((nsamples, nterms, nvars))
    for ss in prange(nsamples):
        for tt in range(nterms):
            for dd in range(nvars):
                val = derivs_1d[dd, ss, indices[dd, tt]]
                for kk in range(nvars):
                    if kk != dd:
                        val *= vals_1d[kk, ss, indices[kk, tt]]
                result[ss, tt, dd] = val
    return result


@njit(cache=True, parallel=True)
def basis_hessian_numba(
    vals_1d: np.ndarray,
    derivs_1d: np.ndarray,
    hess_1d: np.ndarray,
    indices: np.ndarray,
    nvars: int,
    nsamples: int,
    nterms: int,
) -> np.ndarray:
    """Evaluate Hessian of tensor product basis via fused kernel.

    Diagonal (d == k):
        H[s, t, d, d] = hess_1d[d, s, idx_d] * prod_{l != d} vals_1d[l, s, idx_l]

    Off-diagonal (d != k):
        H[s, t, d, k] = derivs_1d[d, s, idx_d] * derivs_1d[k, s, idx_k]
                         * prod_{l != d, l != k} vals_1d[l, s, idx_l]

    Parameters
    ----------
    vals_1d : np.ndarray
        Stacked univariate basis values. Shape: (nvars, nsamples, max_nterms_1d).
    derivs_1d : np.ndarray
        Stacked univariate first derivatives. Shape: (nvars, nsamples, max_nterms_1d).
    hess_1d : np.ndarray
        Stacked univariate second derivatives. Shape: (nvars, nsamples, max_nterms_1d).
    indices : np.ndarray
        Multi-indices. Shape: (nvars, nterms). Integer dtype.
    nvars : int
        Number of variables.
    nsamples : int
        Number of sample points.
    nterms : int
        Number of basis terms.

    Returns
    -------
    np.ndarray
        Hessian. Shape: (nsamples, nterms, nvars, nvars).
    """
    result = np.empty((nsamples, nterms, nvars, nvars))
    for ss in prange(nsamples):
        for tt in range(nterms):
            for dd in range(nvars):
                # Diagonal: second derivative in dimension dd
                val = hess_1d[dd, ss, indices[dd, tt]]
                for ll in range(nvars):
                    if ll != dd:
                        val *= vals_1d[ll, ss, indices[ll, tt]]
                result[ss, tt, dd, dd] = val

                # Off-diagonal: d < k only, then use symmetry
                for kk in range(dd + 1, nvars):
                    val = (
                        derivs_1d[dd, ss, indices[dd, tt]]
                        * derivs_1d[kk, ss, indices[kk, tt]]
                    )
                    for ll in range(nvars):
                        if ll != dd and ll != kk:
                            val *= vals_1d[ll, ss, indices[ll, tt]]
                    result[ss, tt, dd, kk] = val
                    result[ss, tt, kk, dd] = val
    return result
