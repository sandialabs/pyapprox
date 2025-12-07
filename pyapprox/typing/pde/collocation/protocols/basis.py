"""Basis protocols for spectral collocation methods.

Defines interfaces for polynomial bases used in collocation discretization.
"""

from typing import Protocol, Generic, runtime_checkable

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.pde.collocation.protocols.mesh import MeshProtocol


@runtime_checkable
class BasisProtocol(Protocol, Generic[Array]):
    """Protocol for collocation basis.

    A basis provides derivative matrices for computing spatial derivatives
    on a collocation mesh.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def mesh(self) -> MeshProtocol[Array]:
        """Return the associated mesh."""
        ...

    def derivative_matrix(self, order: int, dim: int) -> Array:
        """Get derivative matrix for specified order and dimension.

        Parameters
        ----------
        order : int
            Derivative order (1 for first derivative, 2 for second, etc.)
        dim : int
            Spatial dimension (0 for x, 1 for y, 2 for z)

        Returns
        -------
        Array
            Derivative matrix of shape (npts, npts).
            Multiplying by solution values gives derivative values.
        """
        ...

    def interpolate(self, values: Array, new_points: Array) -> Array:
        """Interpolate values to new points.

        Parameters
        ----------
        values : Array
            Values at mesh points. Shape: (npts,) or (ncomponents, npts)
        new_points : Array
            Points to interpolate to. Shape: (ndim, new_npts)

        Returns
        -------
        Array
            Interpolated values. Shape matches input.
        """
        ...


@runtime_checkable
class BasisWithQuadratureProtocol(Protocol, Generic[Array]):
    """Protocol for basis with quadrature rule.

    Extends BasisProtocol with quadrature weights for integration.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def mesh(self) -> MeshProtocol[Array]:
        """Return the associated mesh."""
        ...

    def derivative_matrix(self, order: int, dim: int) -> Array:
        """Get derivative matrix for specified order and dimension."""
        ...

    def interpolate(self, values: Array, new_points: Array) -> Array:
        """Interpolate values to new points."""
        ...

    def quadrature_weights(self) -> Array:
        """Return quadrature weights at mesh points.

        Returns
        -------
        Array
            Quadrature weights. Shape: (npts,)
            Includes Jacobian determinant scaling for non-affine transforms.
        """
        ...

    def integrate(self, values: Array) -> Array:
        """Integrate values over the domain.

        Parameters
        ----------
        values : Array
            Values at mesh points. Shape: (npts,) or (ncomponents, npts)

        Returns
        -------
        Array
            Integral value. Shape: () for scalar, (ncomponents,) for vector.
        """
        ...
