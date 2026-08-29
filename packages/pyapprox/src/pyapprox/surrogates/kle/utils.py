"""Utility functions for Karhunen-Loève Expansion computations."""

from typing import Optional, Tuple

import numpy as np
from scipy.linalg import eigh as scipy_eigh
from scipy.sparse.linalg import eigsh

from pyapprox.util.backends.protocols import Array, Backend


def adjust_sign_eig(U: Array, bkd: Backend[Array]) -> Array:
    """Fix the sign each eigenvector is otherwise free to choose.

    An eigenvector is determined only up to scale: if ``A v = lambda v``
    then ``A (-v) = lambda (-v)``, so a unit-norm eigenvector is one of
    ``+v`` or ``-v`` and LAPACK's choice between them is not portable.
    Every quantity a KLE reports is invariant under the flip -- the
    eigenvalues, the orthonormality, the spectral reconstruction and
    hence the covariance of the field -- so fixing a convention costs
    nothing and makes bases comparable across platforms, entry points
    and stored archives.

    **Each column is decided by itself**, by making its
    largest-magnitude entry positive. Both properties matter:

    *Per column*, because the sign of one eigenvector says nothing about
    another. A rule that lets column ``j`` depend on the other columns
    is not stable under truncation, so canonicalizing ten modes and
    keeping three would disagree with canonicalizing three -- and then
    two callers requesting different term counts from identical data get
    different-signed bases, which is the inconsistency this exists to
    remove.

    *Largest magnitude*, because that entry's sign is the one furthest
    from rounding error. Deciding on the first entry instead fails
    whenever a mode has a node at the start of the domain, which
    Dirichlet conditions make routine: the value is near zero and its
    sign is noise. Deciding on the sum fails harder, being exactly zero
    for every antisymmetric mode.

    Ties are common rather than pathological -- KLE eigenfunctions are
    symmetric or antisymmetric about the domain centre, so their extreme
    values match at both ends -- and are broken by taking the lowest
    index, which keeps the result deterministic and backend-independent.
    A zero column has no sign to fix and is left alone.

    Parameters
    ----------
    U : Array, shape (M, K)
        Eigenvectors as columns. Modified in place and returned.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array, shape (M, K)
        The same array, with each column's sign canonicalized.
    """
    if U.ndim != 2:
        raise ValueError(f"U must be 2D (M, K), got ndim={U.ndim}")
    # argmax over each column, ties resolved to the lowest row index by
    # both backends, so the pivot entry is a property of that column
    # alone and survives truncation of any other.
    pivots = bkd.argmax(bkd.abs(U), axis=0)
    pivot_vals = bkd.get_diagonal(U[pivots, :])
    signs = bkd.sign(pivot_vals)
    # sign() is 0 for a zero column, which would erase it rather than
    # flip it; such a column has no orientation to canonicalize.
    signs = bkd.where(bkd.equal(signs, 0.0), bkd.full(signs.shape, 1.0), signs)
    U *= signs
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
    rounded_vals = bkd.asarray(np.round(bkd.to_numpy(eig_vals), decimals=12))
    sorted_tuples = sorted(
        zip(
            bkd.arange(nterms, dtype=int),
            rounded_vals,
            -bkd.abs(eig_vecs[0, :]),
        ),
        key=lambda tup: (tup[1], tup[2]),
        reverse=True,
    )
    II = bkd.hstack([tup[0] for tup in sorted_tuples])
    return eig_vals[II], eig_vecs[:, II]


def _partial_eigsh(
    K_np: np.ndarray,
    nterms: int,
    seed: Optional[int] = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the nterms largest eigenpairs using iterative Lanczos.

    Uses scipy.sparse.linalg.eigsh which is O(N*k) instead of O(N^3)
    for the full eigendecomposition.

    Parameters
    ----------
    K_np : np.ndarray
        Symmetric matrix, shape (N, N).
    nterms : int
        Number of leading eigenpairs.
    seed : int or None
        Seeds the Lanczos start vector, so two solves of the same
        problem return the same basis. Pass None for ARPACK's own
        random start, which is *not* reproducible: it draws from a
        stream numpy's global seed does not reach, so seeding before
        the call does not help. Measured on three full-rank matrices,
        successive unseeded solves agreed on eigenvalues to 5e-15 while
        individual eigenvectors differed by 4.4e-01 to 8.3e-01. That is
        not degenerate-subspace rotation -- one was a random SPD matrix
        with well-separated eigenvalues -- and ``adjust_sign_eig``
        cannot repair it, since the vectors it canonicalizes already
        differ by O(1).

        A seeded random draw rather than a constant vector: a constant
        is orthogonal to any antisymmetric leading eigenvector and
        would stall there. The local stream never perturbs the global
        RNG, matching the convention in ``util.linalg.randomized``.

    .. warning::
        This always operates on NumPy arrays. When called from a Torch
        backend, the input is converted to NumPy via ``bkd.to_numpy()``
        and results are converted back via ``bkd.asarray()``. This
        breaks the PyTorch autograd computation graph. KLE basis
        construction is typically a one-time setup cost and does not
        need to be differentiated through.
    """
    start = (
        None
        if seed is None
        else np.random.RandomState(seed).normal(size=K_np.shape[0])
    )
    eig_vals, eig_vecs = eigsh(K_np, k=nterms, which="LM", v0=start)
    return eig_vals, eig_vecs


def eigendecomposition_unweighted(
    K: Array,
    nterms: int,
    bkd: Backend[Array],
    seed: Optional[int] = 0,
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
        eig_vals_np, eig_vecs_np = _partial_eigsh(K_np, nterms, seed)
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
    seed: Optional[int] = 0,
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
        sym_eig_vals_np, sym_eig_vecs_np = _partial_eigsh(
            K_sym_np, nterms, seed
        )
        sym_eig_vals = bkd.asarray(sym_eig_vals_np)
        sym_eig_vecs = bkd.asarray(sym_eig_vecs_np)
    else:
        sym_eig_vals, sym_eig_vecs = bkd.eigh(K_sym)
    # Undo symmetrization
    eig_vecs = (1.0 / sqrt_weights[:, None]) * sym_eig_vecs
    # Sort with tie-breaking and adjust signs
    eig_vals, eig_vecs = sort_eigenpairs(sym_eig_vals, eig_vecs, nterms, bkd)
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
        eig_vals_np, eig_vecs_np = eigsh(A_np, k=nterms, M=M_np, which="LM")
    else:
        eig_vals_np, eig_vecs_np = scipy_eigh(A_np, M_np)
    eig_vals = bkd.asarray(eig_vals_np)
    eig_vecs = bkd.asarray(eig_vecs_np)
    eig_vals, eig_vecs = sort_eigenpairs(eig_vals, eig_vecs, nterms, bkd)
    eig_vecs = adjust_sign_eig(eig_vecs, bkd)
    return eig_vals, eig_vecs
