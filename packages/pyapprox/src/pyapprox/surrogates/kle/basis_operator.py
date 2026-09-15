r"""A reduced basis as operations rather than as an array.

Every consumer of a ``(nstates, nterms)`` basis in this package does one
of four things with it: select columns, scale columns, contract over
*rows*, or ask its dimensions. Catalogued from the call sites --
``eig_vecs[:, :nterms]`` and ``eigenvectors[:, selected]`` are selection,
``eig_vecs * sqrt_eig_vals`` is scaling, ``dot(basis().T, fields)`` and
``dot(basis(), coefs)`` and ``dot(basis()**2, variances)`` are
contractions, and ``.shape[0]`` is a dimension.

None of those needs the basis to be an array, and the array form is the
one thing that cannot survive a basis too large to hold: at :math:`10^7`
states and 50 terms it is 4 GB, and the datasets this exists for are
larger. So the seam is drawn at the operations.

:class:`ArrayBasis` wraps an array and is what every current caller
gets. A streaming implementation satisfies the same protocol by reading
row blocks, and a consumer written against the protocol cannot tell
which it holds.

**Why selection and scaling return a basis.** A consumer that slices
before contracting -- and both encoders do -- would otherwise force a
streaming implementation to materialize at the slice. Returning a basis
lets the restriction be recorded and applied per block, so
``select(...).scale(...).apply_transpose(...)`` never forms anything
ambient. That composition is the property worth testing: deferring two
transformations to block time is where an implementation goes wrong, and
a resident one gets it right by accident.

**The metric is deliberately absent.** ``KLEEncoder.encode`` applies
:math:`M` to the *fields* before contracting, so the basis itself is
metric-free. Folding the metric in would make
:meth:`BasisOperatorProtocol.apply_transpose` mean different things for
a weighted and an unweighted basis, and would duplicate a
responsibility ``InnerProductProtocol`` already owns.

The ``apply``/``apply_transpose``/``nrows``/``ncols`` names are
:class:`~pyapprox.util.linalg.randomized.MatVecOperator`'s, so a basis
can feed the randomized machinery there without an adapter. ``nstates``
and ``nterms`` are the same two integers under the names this package
uses for them.
"""

from typing import (
    Generic,
    Protocol,
    Sequence,
    runtime_checkable,
)

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class BasisOperatorProtocol(Protocol, Generic[Array]):
    r"""A :math:`(n_{\mathrm{states}}, n_{\mathrm{terms}})` basis, as operations.

    Implementations must be immutable: :meth:`select` and :meth:`scale`
    return a new basis rather than mutating. A consumer holding one can
    then pass it on without copying defensively, which matters when the
    copy would be gigabytes.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def nstates(self) -> int:
        """Return the ambient dimension, the number of rows."""
        ...

    def nterms(self) -> int:
        """Return the number of basis vectors, the number of columns."""
        ...

    def select(
        self, columns: Sequence[int]
    ) -> "BasisOperatorProtocol[Array]":
        """Return the basis restricted to ``columns``, in that order.

        Reordering as well as restricting, since a greedy selection
        produces indices in selection order rather than ascending order
        and the basis has to follow. Indices address the basis as it
        stands, so selecting twice narrows against the current columns
        rather than the original -- otherwise a restricted basis would
        leak what it had been restricted from.
        """
        ...

    def scale(self, factors: Array) -> "BasisOperatorProtocol[Array]":
        """Return the basis with column ``j`` multiplied by ``factors[j]``.

        ``factors`` has shape ``(nterms,)``. Separate from
        :meth:`select` because the eigenvalue weighting a KLE applies is
        a different operation from the truncation it applies, and
        composing them should not require forming either result.
        """
        ...

    def square(self) -> "BasisOperatorProtocol[Array]":
        """Return the elementwise square of the basis.

        Needed by variance propagation, which contracts :math:`V^2`
        against a vector of variances. Elementwise, so it commutes with
        row blocking like everything else here.
        """
        ...

    def apply(self, coefs: Array) -> Array:
        r"""Return :math:`V c`, shape ``(nstates, ncols)``.

        The one operation whose result is ambient-sized, and so the one
        a caller working out of core must think about before calling.
        """
        ...

    def apply_transpose(self, fields: Array) -> Array:
        r"""Return :math:`V^T f`, shape ``(nterms, ncols)``.

        Contracts over rows, so the result is small whatever the ambient
        dimension. Apply the metric to ``fields`` beforehand if the
        inner product is weighted.
        """
        ...

    def to_array(self) -> Array:
        """Return the basis as a dense ``(nstates, nterms)`` array.

        For code that genuinely needs one -- serialization, plotting, an
        external library. An implementation backed by something larger
        than memory raises rather than thrashing, so a caller reaching
        for this at scale is told immediately and pointed at the
        operations above.
        """
        ...


class ArrayBasis(Generic[Array]):
    """A basis held as a dense array.

    What every current caller gets, and the reference a streaming
    implementation is checked against. Each method is the obvious array
    expression; the class exists so consumers can be written against the
    protocol without the resident case paying for the abstraction.

    Parameters
    ----------
    basis : Array
        Shape ``(nstates, nterms)``.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, basis: Array, bkd: Backend[Array]) -> None:
        if basis.ndim != 2:
            raise ValueError(
                f"basis must be 2D (nstates, nterms), got "
                f"ndim={basis.ndim}"
            )
        self._basis = basis
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nstates(self) -> int:
        """Return the ambient dimension."""
        return int(self._basis.shape[0])

    def nterms(self) -> int:
        """Return the number of basis vectors."""
        return int(self._basis.shape[1])

    def nrows(self) -> int:
        """Alias for :meth:`nstates`, for operator consumers."""
        return self.nstates()

    def ncols(self) -> int:
        """Alias for :meth:`nterms`, for operator consumers."""
        return self.nterms()

    def select(self, columns: Sequence[int]) -> "ArrayBasis[Array]":
        """Return the basis restricted to ``columns``, in that order."""
        index = self._bkd.asarray(list(columns), dtype=int)
        return ArrayBasis(self._basis[:, index], self._bkd)

    def scale(self, factors: Array) -> "ArrayBasis[Array]":
        """Return the basis with column ``j`` scaled by ``factors[j]``."""
        if factors.ndim != 1 or int(factors.shape[0]) != self.nterms():
            raise ValueError(
                f"factors must be 1D with one entry per term "
                f"({self.nterms()}), got shape {tuple(factors.shape)}"
            )
        return ArrayBasis(self._basis * factors, self._bkd)

    def square(self) -> "ArrayBasis[Array]":
        """Return the elementwise square of the basis."""
        return ArrayBasis(self._basis**2, self._bkd)

    def apply(self, coefs: Array) -> Array:
        r"""Return :math:`V c`, shape ``(nstates, ncols)``."""
        return self._bkd.dot(self._basis, coefs)

    def apply_transpose(self, fields: Array) -> Array:
        r"""Return :math:`V^T f`, shape ``(nterms, ncols)``."""
        return self._bkd.dot(self._basis.T, fields)

    def to_array(self) -> Array:
        """Return the underlying array."""
        return self._basis

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(nstates={self.nstates()}, "
            f"nterms={self.nterms()})"
        )


def as_basis_operator(
    basis: "Array | BasisOperatorProtocol[Array]",
    bkd: Backend[Array],
) -> BasisOperatorProtocol[Array]:
    """Return ``basis`` as an operator, wrapping a bare array.

    Lets a consumer accept either without branching, so widening a
    signature costs its callers nothing.
    """
    if isinstance(basis, BasisOperatorProtocol):
        return basis
    return ArrayBasis(basis, bkd)
