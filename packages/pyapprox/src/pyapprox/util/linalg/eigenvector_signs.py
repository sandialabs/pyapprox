"""A sign convention for eigenvectors.

An eigenvector is determined only up to sign, so a convention has to be
imposed for bases to be comparable across platforms, callers and stored
archives.
"""

from typing import Tuple

from pyapprox.util.backends.protocols import Array, Backend


def adjust_sign_eig(U: Array, bkd: Backend[Array]) -> Array:
    """Fix the sign each eigenvector is otherwise free to choose.

    An eigenvector is determined only up to scale: if ``A v = lambda v``
    then ``A (-v) = lambda (-v)``, so a unit-norm eigenvector is one of
    ``+v`` or ``-v`` and LAPACK's choice between them is not portable.
    Every quantity derived from the decomposition is invariant under the
    flip -- the eigenvalues, the orthonormality and the spectral
    reconstruction -- so fixing a convention costs nothing and makes bases
    comparable across platforms, entry points and stored archives.

    **Each column is decided by itself**, by making its
    largest-magnitude entry positive. Both properties matter:

    *Per column*, because the sign of one eigenvector says nothing about
    another. A rule that lets column ``j`` depend on the other columns
    is not stable under truncation, so canonicalizing ten columns and
    keeping three would disagree with canonicalizing three -- and then
    two callers requesting different numbers of eigenvectors from
    identical data get different-signed bases, which is the inconsistency
    this exists to remove.

    *Largest magnitude*, because that entry's sign is the one furthest
    from rounding error. Deciding on the first entry instead fails
    whenever a vector's first entry is near zero, which is routine (a
    vector vanishing at a constrained index, for example): its sign is
    then noise. Deciding on the sum fails harder, being exactly zero for
    every antisymmetric vector.

    Ties are common rather than pathological -- eigenvectors of a matrix
    invariant under a reversal of indices are symmetric or antisymmetric,
    so their extreme entries match at both ends -- and are broken by
    taking the lowest index, which keeps the result deterministic and
    backend-independent. A zero column has no sign to fix and is left
    alone.

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
    return eigenvector_signs(U, bkd)[0]


def eigenvector_signs(
    U: Array, bkd: Backend[Array]
) -> Tuple[Array, Array]:
    """Canonicalize column signs, and report the signs applied.

    The same operation as :func:`adjust_sign_eig`, which delegates here,
    but returning the sign vector as well as the signed array.

    A caller holding a second factor of the same decomposition needs
    this. Flipping the sign of an eigenvector is only free when nothing
    else is paired with it: for a factorization ``A = U S V^T`` the
    flip must be applied to the matching column of ``V`` too, or the
    factors no longer reconstruct ``A``. Returning the signs lets that
    caller apply them rather than re-deriving a rule that must agree
    with this one exactly.

    Parameters
    ----------
    U : Array, shape (M, K)
        Eigenvectors as columns.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    U : Array, shape (M, K)
        The same array, with each column's sign canonicalized.
    signs : Array, shape (K,)
        The sign applied to each column, ``+1`` or ``-1``.
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
    return U, signs
