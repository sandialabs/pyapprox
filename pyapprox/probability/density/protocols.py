"""Protocols for density estimation basis functions."""

from typing import Generic, Protocol, Tuple, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class DensityBasisProtocol(Protocol, Generic[Array]):
    """Protocol for basis functions used in density estimation.

    A density basis provides evaluation of basis functions and the
    mass matrix M_ij = int phi_i(y) phi_j(y) dy needed for L2 projection.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def nbasis(self) -> int:
        """Return the number of basis functions."""
        ...

    def domain(self) -> Tuple[float, float]:
        """Return the domain (y_min, y_max) of the basis functions."""
        ...

    def evaluate(self, y_values: Array) -> Array:
        """Evaluate all basis functions at given points.

        Parameters
        ----------
        y_values : Array
            Query points. Shape: (1, npts).

        Returns
        -------
        Array
            Basis values. Shape: (nbasis, npts).
        """
        ...

    def mass_matrix(self) -> Array:
        """Return the mass matrix M_ij = int phi_i(y) phi_j(y) dy.

        Returns
        -------
        Array
            Mass matrix. Shape: (nbasis, nbasis).
        """
        ...


__all__ = ["DensityBasisProtocol"]
