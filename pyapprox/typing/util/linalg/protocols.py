"""Protocols for linear algebra utilities."""

from typing import Generic, Protocol, Tuple, runtime_checkable

from pyapprox.typing.util.backends.protocols import Array, Backend


@runtime_checkable
class PivotedFactorizerProtocol(Protocol, Generic[Array]):
    """Protocol for pivoted matrix factorizers."""

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def factorize(self, npivots: int) -> Tuple[Array, Array]:
        """Factorize matrix with given number of pivots.

        Returns
        -------
        Tuple[Array, Array]
            The L and U factors (for LU) or Q and R factors (for QR).
        """
        ...

    def npivots(self) -> int:
        """Return number of completed pivots."""
        ...

    def pivots(self) -> Array:
        """Return pivot indices."""
        ...

    def success(self) -> bool:
        """Return True if factorization was successful."""
        ...


@runtime_checkable
class IncrementalFactorizerProtocol(Protocol, Generic[Array]):
    """Protocol for incremental matrix factorization.

    Used for sequential Leja point selection where we add columns/rows
    incrementally.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def add_column(self, column: Array) -> int:
        """Add a column and return the pivot row index."""
        ...

    def add_row(self, row: Array) -> None:
        """Add a row to the matrix."""
        ...

    def get_pivot_indices(self) -> Array:
        """Return all pivot indices so far."""
        ...

    def reset(self) -> None:
        """Reset the factorization state."""
        ...
