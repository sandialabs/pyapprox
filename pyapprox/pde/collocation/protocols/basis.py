"""Basis protocols for spectral collocation methods.

Defines interfaces for polynomial bases used in collocation discretization.
"""

from typing import Generic, Protocol, Tuple, runtime_checkable

from pyapprox.pde.collocation.protocols.mesh import MeshProtocol
from pyapprox.util.backends.protocols import Array, Backend

# =============================================================================
# Extensibility Protocols for 1D Components
# =============================================================================


@runtime_checkable
class NodesGenerator1DProtocol(Protocol, Generic[Array]):
    """Protocol for generating 1D collocation nodes.

    Different polynomial families (Chebyshev, Legendre, etc.) use different
    node distributions. This protocol allows swapping node generators.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def generate(self, npts: int) -> Array:
        """Generate npts nodes on reference interval [-1, 1].

        Parameters
        ----------
        npts : int
            Number of nodes to generate.

        Returns
        -------
        Array
            1D array of nodes. Shape: (npts,)
        """
        ...


@runtime_checkable
class DerivativeMatrix1DProtocol(Protocol, Generic[Array]):
    """Protocol for computing 1D derivative matrix given nodes.

    Different polynomial families use different derivative matrix formulas.
    This protocol decouples matrix computation from node generation.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def compute(self, nodes: Array) -> Array:
        """Compute derivative matrix for given nodes.

        Parameters
        ----------
        nodes : Array
            1D collocation nodes. Shape: (npts,)

        Returns
        -------
        Array
            Derivative matrix. Shape: (npts, npts)
            D[i,j] approximates d(L_j)/dx evaluated at nodes[i],
            where L_j is the j-th Lagrange basis polynomial.
        """
        ...


# =============================================================================
# Tensor Product Basis Protocol
# =============================================================================


@runtime_checkable
class TensorProductBasisProtocol(Protocol, Generic[Array]):
    """Protocol for tensor-product bases with Kronecker structure.

    Tensor product bases construct multi-D derivative matrices from
    1D components using Kronecker products. This is one approach to
    multi-D bases; future implementations may use simplex or unstructured
    approaches that don't follow this protocol.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def ndim(self) -> int:
        """Return the number of spatial dimensions."""
        ...

    def npts_per_dim(self) -> Tuple[int, ...]:
        """Return number of points in each dimension.

        Returns
        -------
        Tuple[int, ...]
            Tuple of length ndim with point counts per dimension.
        """
        ...

    def npts(self) -> int:
        """Return total number of points (product of npts_per_dim)."""
        ...

    def nodes_1d(self, dim: int) -> Array:
        """Return 1D nodes for specified dimension.

        Parameters
        ----------
        dim : int
            Dimension index (0 for x, 1 for y, 2 for z).

        Returns
        -------
        Array
            1D nodes. Shape: (npts_dim,)
        """
        ...

    def derivative_matrix_1d(self, dim: int) -> Array:
        """Return 1D derivative matrix for specified dimension.

        Parameters
        ----------
        dim : int
            Dimension index.

        Returns
        -------
        Array
            1D derivative matrix. Shape: (npts_dim, npts_dim)
        """
        ...

    def derivative_matrix(self, order: int, dim: int) -> Array:
        """Return full derivative matrix via Kronecker product.

        Parameters
        ----------
        order : int
            Derivative order (1 for first derivative, 2 for second, etc.)
        dim : int
            Spatial dimension (0 for x, 1 for y, 2 for z).

        Returns
        -------
        Array
            Full derivative matrix. Shape: (npts, npts)
        """
        ...


# =============================================================================
# High-Level Basis Protocols
# =============================================================================


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
