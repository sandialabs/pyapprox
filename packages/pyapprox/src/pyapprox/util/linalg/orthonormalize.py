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

from typing import Generic, Protocol, runtime_checkable

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
