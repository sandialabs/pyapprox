"""2D Chebyshev collocation basis.

Convenience wrapper combining Chebyshev nodes and derivative matrices
with the tensor product basis infrastructure for 2D domains.
"""

from typing import Generic, Tuple

from pyapprox.pde.collocation.basis.chebyshev.derivative import (
    ChebyshevDerivativeMatrix1D,
)
from pyapprox.pde.collocation.basis.tensor_product import (
    TensorProductBasis,
)
from pyapprox.pde.collocation.protocols.mesh import (
    MeshWithTransformProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class ChebyshevBasis2D(Generic[Array]):
    """Chebyshev collocation basis for 2D domains.

    Uses Chebyshev-Gauss-Lobatto nodes and barycentric derivative matrices
    in each direction, assembled via Kronecker products.

    Parameters
    ----------
    mesh : MeshWithTransformProtocol
        Mesh providing reference nodes, physical nodes, and gradient factors.
        Use TransformedMesh2D with an optional transform.
    bkd : Backend
        Computational backend.

    Examples
    --------
    Reference domain [-1, 1]^2:

    >>> mesh = TransformedMesh2D(10, 10, bkd)
    >>> basis = ChebyshevBasis2D(mesh, bkd)

    Physical domain [0, 1] x [0, 2] with affine transform:

    >>> transform = AffineTransform2D((0.0, 1.0, 0.0, 2.0), bkd)
    >>> mesh = TransformedMesh2D(10, 10, bkd, transform)
    >>> basis = ChebyshevBasis2D(mesh, bkd)
    """

    def __init__(
        self,
        mesh: MeshWithTransformProtocol[Array],
        bkd: Backend[Array],
    ):
        self._bkd = bkd
        self._mesh = mesh

        # Create derivative matrix computer
        deriv_computer = ChebyshevDerivativeMatrix1D(bkd)

        # Build tensor product basis with mesh
        self._tensor_basis = TensorProductBasis(mesh, deriv_computer, bkd)

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def ndim(self) -> int:
        """Return the number of spatial dimensions."""
        return 2

    def npts_per_dim(self) -> Tuple[int, ...]:
        """Return number of points per dimension."""
        return self._mesh.npts_per_dim()

    def npts(self) -> int:
        """Return total number of points."""
        return self._mesh.npts()

    def nodes_x(self) -> Array:
        """Return 1D reference collocation nodes in x direction.

        Returns
        -------
        Array
            Chebyshev-Gauss-Lobatto nodes. Shape: (npts_x,)
        """
        return self._tensor_basis.nodes_1d(0)

    def nodes_y(self) -> Array:
        """Return 1D reference collocation nodes in y direction.

        Returns
        -------
        Array
            Chebyshev-Gauss-Lobatto nodes. Shape: (npts_y,)
        """
        return self._tensor_basis.nodes_1d(1)

    def mesh(self) -> MeshWithTransformProtocol[Array]:
        """Return the underlying mesh."""
        return self._mesh

    def derivative_matrix(self, order: int, dim: int) -> Array:
        """Return physical derivative matrix for specified order and dimension.

        Parameters
        ----------
        order : int
            Derivative order (1 for first derivative, 2 for second, etc.)
        dim : int
            Spatial dimension (0 for x, 1 for y).

        Returns
        -------
        Array
            Derivative matrix in physical coordinates. Shape: (npts, npts)
        """
        return self._tensor_basis.derivative_matrix(order, dim)

    def derivative_matrix_x(self, order: int = 1) -> Array:
        """Return physical derivative matrix in x direction.

        Parameters
        ----------
        order : int, optional
            Derivative order. Default is 1.

        Returns
        -------
        Array
            Derivative matrix in physical coordinates. Shape: (npts, npts)
        """
        return self._tensor_basis.derivative_matrix(order, dim=0)

    def derivative_matrix_y(self, order: int = 1) -> Array:
        """Return physical derivative matrix in y direction.

        Parameters
        ----------
        order : int, optional
            Derivative order. Default is 1.

        Returns
        -------
        Array
            Derivative matrix in physical coordinates. Shape: (npts, npts)
        """
        return self._tensor_basis.derivative_matrix(order, dim=1)

    def reference_derivative_matrix(self, order: int, dim: int) -> Array:
        """Return reference derivative matrix for specified order and dimension.

        Parameters
        ----------
        order : int
            Derivative order (1 for first derivative, 2 for second, etc.)
        dim : int
            Spatial dimension (0 for x, 1 for y).

        Returns
        -------
        Array
            Derivative matrix in reference coordinates. Shape: (npts, npts)
        """
        return self._tensor_basis.reference_derivative_matrix(order, dim)
