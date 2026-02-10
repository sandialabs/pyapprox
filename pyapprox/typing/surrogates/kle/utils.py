"""Utility functions for Karhunen-Loève Expansion computations."""

from typing import Generic, Tuple

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend


def adjust_sign_eig(U: Array, bkd: Backend[Array]) -> Array:
    """Ensure uniqueness of eigenvalue decomposition.

    Adjusts signs so the largest-magnitude entry in the first row of U
    is positive. This ensures consistent sign convention across platforms.

    Parameters
    ----------
    U : Array, shape (M, K)
        Eigenvectors as columns.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array, shape (M, K)
        Sign-adjusted eigenvectors.
    """
    idx = bkd.argmax(bkd.abs(U[0, :]))
    s = bkd.sign(U[idx, :])
    II = bkd.where(s == 0)[0]
    s[II] = 1.0
    U *= s
    return U


def sort_eigenpairs(
    eig_vals: Array,
    eig_vecs: Array,
    nterms: int,
    bkd: Backend[Array],
) -> Tuple[Array, Array]:
    """Sort eigenpairs by descending eigenvalue with tie-breaking.

    For eigenvalues that are equal up to 12 decimal places, breaks ties
    using the magnitude of the first entry in the eigenvector. This ensures
    cross-platform consistency.

    Parameters
    ----------
    eig_vals : Array, shape (nterms,)
        Eigenvalues.
    eig_vecs : Array, shape (ncoords, nterms)
        Eigenvectors as columns.
    nterms : int
        Number of terms to keep.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    sorted_eig_vals : Array, shape (nterms,)
        Sorted eigenvalues (descending).
    sorted_eig_vecs : Array, shape (ncoords, nterms)
        Correspondingly sorted eigenvectors.
    """
    # Sort by eigenvalue descending, then by magnitude of first
    # eigenvector entry (for tie-breaking across platforms)
    rounded_vals = bkd.asarray(
        np.round(bkd.to_numpy(eig_vals), decimals=12)
    )
    tuples = zip(
        bkd.arange(nterms, dtype=int),
        rounded_vals,
        -bkd.abs(eig_vecs[0, :]),
    )
    tuples = sorted(tuples, key=lambda tup: (tup[1], tup[2]), reverse=True)
    II = bkd.hstack([tup[0] for tup in tuples])
    return eig_vals[II], eig_vecs[:, II]


def eigendecomposition_unweighted(
    K: Array,
    nterms: int,
    bkd: Backend[Array],
) -> Tuple[Array, Array]:
    """Compute eigendecomposition of a symmetric kernel matrix.

    Parameters
    ----------
    K : Array, shape (N, N)
        Symmetric positive semi-definite kernel matrix.
    nterms : int
        Number of eigenpairs to keep (largest eigenvalues).
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    eig_vals : Array, shape (nterms,)
        Largest eigenvalues in descending order.
    eig_vecs : Array, shape (N, nterms)
        Corresponding eigenvectors.
    """
    eig_vals, eig_vecs = bkd.eigh(K)
    # eigh returns eigenvalues in ascending order; take last nterms
    eig_vals = eig_vals[-nterms:]
    eig_vecs = eig_vecs[:, -nterms:]
    # Sort with tie-breaking and adjust signs
    eig_vals, eig_vecs = sort_eigenpairs(eig_vals, eig_vecs, nterms, bkd)
    eig_vecs = adjust_sign_eig(eig_vecs, bkd)
    return eig_vals, eig_vecs


def eigendecomposition_weighted(
    K: Array,
    quad_weights: Array,
    nterms: int,
    bkd: Backend[Array],
) -> Tuple[Array, Array]:
    """Compute weighted eigendecomposition of a kernel matrix.

    Uses symmetrization: W^{1/2} K W^{1/2} = V D V^T, then
    eigenvectors = W^{-1/2} V.

    See https://etheses.lse.ac.uk/2950/1/U615901.pdf, page 42.

    Parameters
    ----------
    K : Array, shape (N, N)
        Symmetric positive semi-definite kernel matrix.
    quad_weights : Array, shape (N,)
        Quadrature weights for orthogonalization.
    nterms : int
        Number of eigenpairs to keep (largest eigenvalues).
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    eig_vals : Array, shape (nterms,)
        Largest eigenvalues in descending order.
    eig_vecs : Array, shape (N, nterms)
        Corresponding eigenvectors (unweighted, orthonormal under
        the quadrature weight inner product).
    """
    sqrt_weights = bkd.sqrt(quad_weights)
    # Symmetrize: K_sym = sqrt_w * K * sqrt_w
    K_sym = sqrt_weights[:, None] * K * sqrt_weights
    sym_eig_vals, sym_eig_vecs = bkd.eigh(K_sym)
    # Take largest nterms
    sym_eig_vals = sym_eig_vals[-nterms:]
    sym_eig_vecs = sym_eig_vecs[:, -nterms:]
    # Undo symmetrization
    eig_vecs = (1.0 / sqrt_weights[:, None]) * sym_eig_vecs
    # Sort with tie-breaking and adjust signs
    eig_vals, eig_vecs = sort_eigenpairs(
        sym_eig_vals, eig_vecs, nterms, bkd
    )
    eig_vecs = adjust_sign_eig(eig_vecs, bkd)
    return eig_vals, eig_vecs
