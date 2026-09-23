r"""Orthonormalizing a tall array, as an injected choice.

A randomized decomposition orthonormalizes its sketch repeatedly: once
per power iteration, and once more before projecting. The sketch is
:math:`(n_{\mathrm{rows}}, k)` with :math:`k` small, so this is the only
step in the algorithm whose cost and memory are set by the ambient
dimension rather than by the rank.

Which factorization does it is a genuine choice rather than a detail:

- A dense Householder QR is the accurate one, and needs the whole array
  resident.
- A blocked method -- Cholesky-QR, TSQR -- reads the array in row blocks
  and never holds it, at some cost in conditioning or in complexity.

So it is a protocol, injected into the decomposition, rather than a call
to :func:`numpy.linalg.qr` written six times. A caller whose sketch fits
keeps the dense one and pays nothing; a caller whose sketch does not
supplies a blocked implementation without the decomposition changing.

**Only :math:`Q` is returned.** Every call site here discards :math:`R`,
and a blocked implementation that had to produce it would do strictly
more work -- Cholesky-QR gets :math:`R` on the way, but a method that
streams :math:`Q` to storage would have to keep it only to satisfy a
signature nobody reads.
"""

from typing import Generic, Optional, Protocol, runtime_checkable

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class OrthonormalizerProtocol(Protocol, Generic[Array]):
    r"""Turns a tall array into one with orthonormal columns."""

    def __call__(self, array: Array) -> Array:
        r"""Return :math:`Q` with orthonormal columns spanning ``array``.

        Parameters
        ----------
        array : Array, shape (nrows, k)
            Tall and thin: ``nrows`` is the ambient dimension and ``k``
            the sketch width. Columns need not be independent, though a
            rank-deficient input leaves the spanning columns arbitrary.

        Returns
        -------
        Array, shape (nrows, k)
            Orthonormal columns with the same span. Reduced rather than
            full: the ``nrows - k`` columns completing the basis are
            never wanted here and at a large ambient dimension cannot be
            formed.
        """
        ...


class HouseholderQR(Generic[Array]):
    """Dense Householder QR, via LAPACK.

    The accurate implementation and the default. Backward stable, so the
    computed :math:`Q` is orthonormal to roughly machine precision
    regardless of the input's conditioning -- the property a blocked
    alternative gives up.

    Holds the whole array, which at a large ambient dimension is what
    makes an alternative necessary rather than merely interesting.

    Runs in NumPy whatever the backend, because LAPACK is what performs
    the factorization; the conversion is at the boundary of the call and
    is why this is a class rather than a free function reaching for the
    backend's own decomposition.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def __call__(self, array: Array) -> Array:
        """Return the reduced QR factor of ``array``."""
        validate_tall_array(array)
        factor, _ = np.linalg.qr(
            self._bkd.to_numpy(array), mode="reduced"
        )
        return self._bkd.asarray(factor)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class CholeskyQR(Generic[Array]):
    r"""Orthonormalize through the Gram, in row blocks.

    .. math:: G = Y^T Y, \quad G = R^T R, \quad Q = Y R^{-1}

    Only the last step touches the ambient dimension, and it does so a
    block at a time. The Gram is :math:`(k, k)` for a sketch of width
    :math:`k` and accumulates over row blocks, so :math:`d` never
    enters it: at ``nterms=200`` with 20 oversampling it is 378 KB and
    its Cholesky takes under a millisecond. The same asymmetry the
    method of snapshots rests on.

    That is what makes this the implementation for a sketch too large
    to hold. :class:`HouseholderQR` hands the whole array to LAPACK,
    which copies it -- measured at 153 MB for a 76 MB sketch, against
    77 MB here.

    **Repeated by default, and the repetition is not optional at
    scale.** Forming :math:`G` squares the condition number, so one
    pass loses half the digits: on a sketch with
    :math:`\mathrm{cond} = 10^5` a single pass left an orthonormality
    error of 1.6e-07 against Householder's 6.7e-16, and by
    :math:`10^8` it was 8.6e-02 -- no orthogonality at all. A second
    pass over the improved factor restores 4.4e-16 at every
    conditioning the first pass survives.

    **Past that it raises rather than returning a wrong factor.** The
    Cholesky fails once :math:`\mathrm{cond}(Y)^2` approaches
    :math:`1/\epsilon`, near :math:`\mathrm{cond}(Y) = 10^8`, and it
    fails as an exception rather than as a plausible factor. A sketch
    that ill-conditioned is not what a randomized decomposition
    produces -- data whose spectrum decays to :math:`10^{-10}` gave a
    sketch with :math:`\mathrm{cond} \approx 1.6 \times 10^4`, because
    sketching mixes the spectrum rather than inheriting it.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    max_bytes : int, optional
        Byte budget for one row block. None uses a 1 GB default.
    npasses : int
        How many times to repeat. Two is the default and the
        recommendation; one is available to a caller who has measured
        their own conditioning and wants the pass back.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        max_bytes: Optional[int] = None,
        npasses: int = 2,
    ) -> None:
        if npasses < 1:
            raise ValueError(
                f"npasses must be at least 1, got {npasses}"
            )
        self._bkd = bkd
        self._max_bytes = max_bytes
        self._npasses = int(npasses)

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def npasses(self) -> int:
        """Return how many times the factorization is repeated."""
        return self._npasses

    def __call__(self, array: Array) -> Array:
        """Return an orthonormal factor spanning ``array``'s columns."""
        validate_tall_array(array)
        factor = array
        for _ in range(self._npasses):
            factor = self._one_pass(factor)
        return factor

    def _one_pass(self, array: Array) -> Array:
        """One Gram, Cholesky and solve sweep over row blocks."""
        bkd = self._bkd
        nrows, ncols = int(array.shape[0]), int(array.shape[1])
        height = _rows_per_block(ncols, array.dtype, self._max_bytes)
        gram = bkd.zeros((ncols, ncols))
        for start in range(0, nrows, height):
            block = array[start : min(start + height, nrows), :]
            gram = gram + bkd.dot(block.T, block)
        # Symmetrize before factorizing: the two triangles differ at
        # rounding level and cholesky reads only one, silently
        # discarding the discrepancy rather than averaging it.
        upper = self._cholesky_factor((gram + gram.T) / 2.0)
        out = bkd.zeros((nrows, ncols))
        for start in range(0, nrows, height):
            rows = slice(start, min(start + height, nrows))
            out[rows, :] = bkd.solve(upper.T, array[rows, :].T).T
        return out

    def _cholesky_factor(self, gram: Array) -> Array:
        """Return ``R`` with ``R.T @ R == gram``, naming why it failed.

        NumPy reports a failed Cholesky as "matrix is not positive
        definite", which is true and unhelpful here: the matrix is a
        Gram and is positive definite in exact arithmetic. What
        actually happened is that squaring the condition number
        exhausted the available precision.
        """
        try:
            lower = np.linalg.cholesky(self._bkd.to_numpy(gram))
        except np.linalg.LinAlgError as error:
            raise np.linalg.LinAlgError(
                "Cholesky of the Gram failed: forming Y^T Y squares "
                "the condition number, so this happens once cond(Y) "
                "reaches about 1e8. Use HouseholderQR, which is "
                "unaffected by conditioning, or reduce the "
                "oversampling that made the sketch rank deficient."
            ) from error
        return self._bkd.asarray(lower.T)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(npasses={self._npasses})"


def _rows_per_block(
    ncols: int, dtype: object, max_bytes: Optional[int] = None
) -> int:
    """How many rows of a ``(nrows, ncols)`` array fit in the budget.

    Local rather than shared with the snapshot sources, which divide a
    budget by a *snapshot* count: coupling the two would make a change
    to how snapshots are read silently change how a sketch is
    factorized.
    """
    budget = (1 << 30) if max_bytes is None else max_bytes
    if budget < 1:
        raise ValueError(f"max_bytes must be positive, got {max_bytes}")
    itemsize = getattr(dtype, "itemsize", 8)
    return max(1, int(budget // (max(ncols, 1) * itemsize)))


def validate_tall_array(array: Array) -> None:
    """Check ``array`` is 2D and has at least as many rows as columns.

    A wide array is not an error for ``qr`` itself, which happily
    returns a reduced factor with fewer columns than it was given. It is
    an error *here*: the sketch is meant to be tall, and a wide one
    means the rank requested exceeds the ambient dimension -- which
    surfaces later as a basis with missing columns rather than as the
    argument that was wrong.
    """
    if array.ndim != 2:
        raise ValueError(
            f"array must be 2D (nrows, k), got ndim={array.ndim}"
        )
    nrows, ncols = int(array.shape[0]), int(array.shape[1])
    if nrows < ncols:
        raise ValueError(
            f"array must be tall, got {nrows} rows and {ncols} "
            "columns; the requested rank exceeds the ambient dimension"
        )
