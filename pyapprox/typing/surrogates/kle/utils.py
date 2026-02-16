"""Utility functions for Karhunen-Loève Expansion computations."""

from typing import Tuple

import numpy as np
from scipy.linalg import eigh as scipy_eigh
from scipy.sparse.linalg import eigsh

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


def _partial_eigsh(
    K_np: np.ndarray,
    nterms: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the nterms largest eigenpairs using iterative Lanczos.

    Uses scipy.sparse.linalg.eigsh which is O(N*k) instead of O(N^3)
    for the full eigendecomposition.

    .. warning::
        This always operates on NumPy arrays. When called from a Torch
        backend, the input is converted to NumPy via ``bkd.to_numpy()``
        and results are converted back via ``bkd.asarray()``. This
        breaks the PyTorch autograd computation graph. KLE basis
        construction is typically a one-time setup cost and does not
        need to be differentiated through.
    """
    eig_vals, eig_vecs = eigsh(K_np, k=nterms, which="LM")
    return eig_vals, eig_vecs


def eigendecomposition_unweighted(
    K: Array,
    nterms: int,
    bkd: Backend[Array],
) -> Tuple[Array, Array]:
    """Compute eigendecomposition of a symmetric kernel matrix.

    Uses partial eigensolve (scipy eigsh) when nterms < N for O(N*k)
    cost instead of O(N^3) full decomposition.

    .. warning::
        When nterms < N, the eigensolve is performed in NumPy via
        scipy.sparse.linalg.eigsh regardless of backend. For Torch
        backend this breaks the autograd computation graph for the
        eigendecomposition step. This is acceptable because KLE basis
        construction is a one-time setup cost.

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
    N = K.shape[0]
    if nterms < N:
        K_np = bkd.to_numpy(K)
        eig_vals_np, eig_vecs_np = _partial_eigsh(K_np, nterms)
        eig_vals = bkd.asarray(eig_vals_np)
        eig_vecs = bkd.asarray(eig_vecs_np)
    else:
        eig_vals, eig_vecs = bkd.eigh(K)
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

    .. warning::
        When nterms < N, the eigensolve is performed in NumPy via
        scipy.sparse.linalg.eigsh regardless of backend. For Torch
        backend this breaks the autograd computation graph for the
        eigendecomposition step. This is acceptable because KLE basis
        construction is a one-time setup cost.

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
    N = K_sym.shape[0]
    if nterms < N:
        K_sym_np = bkd.to_numpy(K_sym)
        sym_eig_vals_np, sym_eig_vecs_np = _partial_eigsh(K_sym_np, nterms)
        sym_eig_vals = bkd.asarray(sym_eig_vals_np)
        sym_eig_vecs = bkd.asarray(sym_eig_vecs_np)
    else:
        sym_eig_vals, sym_eig_vecs = bkd.eigh(K_sym)
    # Undo symmetrization
    eig_vecs = (1.0 / sqrt_weights[:, None]) * sym_eig_vecs
    # Sort with tie-breaking and adjust signs
    eig_vals, eig_vecs = sort_eigenpairs(
        sym_eig_vals, eig_vecs, nterms, bkd
    )
    eig_vecs = adjust_sign_eig(eig_vecs, bkd)
    return eig_vals, eig_vecs


def eigendecomposition_generalized(
    A: Array,
    M: Array,
    nterms: int,
    bkd: Backend[Array],
) -> Tuple[Array, Array]:
    """Solve generalized eigenproblem A v = lambda M v for largest eigenvalues.

    Uses scipy.linalg.eigh (full) or scipy.sparse.linalg.eigsh (partial)
    depending on whether nterms < N.

    .. warning::
        The eigensolve is always performed in NumPy regardless of backend.
        For Torch backend this breaks the autograd computation graph.
        KLE basis construction is typically a one-time setup cost and does
        not need to be differentiated through.

    Parameters
    ----------
    A : Array, shape (N, N)
        Symmetric matrix (e.g. covariance matrix C_h).
    M : Array, shape (N, N)
        Symmetric positive definite matrix (e.g. mass matrix).
    nterms : int
        Number of eigenpairs to keep (largest eigenvalues).
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    eig_vals : Array, shape (nterms,)
        Largest eigenvalues in descending order.
    eig_vecs : Array, shape (N, nterms)
        Corresponding eigenvectors, M-orthonormal.
    """
    from scipy.sparse import issparse

    A_np = bkd.to_numpy(A) if not isinstance(A, np.ndarray) else A
    M_np = bkd.to_numpy(M) if not isinstance(M, np.ndarray) else M
    if issparse(A_np):
        A_np = A_np.toarray()
    if issparse(M_np):
        M_np = M_np.toarray()
    N = A_np.shape[0]
    if nterms < N:
        eig_vals_np, eig_vecs_np = eigsh(
            A_np, k=nterms, M=M_np, which="LM"
        )
    else:
        eig_vals_np, eig_vecs_np = scipy_eigh(A_np, M_np)
    eig_vals = bkd.asarray(eig_vals_np)
    eig_vecs = bkd.asarray(eig_vecs_np)
    eig_vals, eig_vecs = sort_eigenpairs(eig_vals, eig_vecs, nterms, bkd)
    eig_vecs = adjust_sign_eig(eig_vecs, bkd)
    return eig_vals, eig_vecs
