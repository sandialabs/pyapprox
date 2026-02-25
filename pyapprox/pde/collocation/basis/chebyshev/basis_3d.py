"""3D Chebyshev collocation basis.

Convenience wrapper combining Chebyshev nodes and derivative matrices
with the tensor product basis infrastructure for 3D domains.
"""

from typing import Generic, Tuple

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.pde.collocation.basis.tensor_product import (
    TensorProductBasis,
)
from pyapprox.pde.collocation.basis.chebyshev.derivative import (
    ChebyshevDerivativeMatrix1D,
)
from pyapprox.pde.collocation.protocols.mesh import (
    MeshWithTransformProtocol,
)


class ChebyshevBasis3D(Generic[Array]):
    """Chebyshev collocation basis for 3D domains.

    Uses Chebyshev-Gauss-Lobatto nodes and barycentric derivative matrices
    in each direction, assembled via Kronecker products.

    Parameters
    ----------
    mesh : MeshWithTransformProtocol
        Mesh providing reference nodes, physical nodes, and gradient factors.
        Use TransformedMesh3D with an optional transform.
    bkd : Backend
        Computational backend.

    Examples
    --------
    Reference domain [-1, 1]^3:

    >>> mesh = TransformedMesh3D(10, 10, 10, bkd)
    >>> basis = ChebyshevBasis3D(mesh, bkd)

    Physical domain [0, 1]^3 with affine transform:

    >>> transform = AffineTransform3D((0.0, 1.0, 0.0, 1.0, 0.0, 1.0), bkd)
    >>> mesh = TransformedMesh3D(10, 10, 10, bkd, transform)
    >>> basis = ChebyshevBasis3D(mesh, bkd)
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
        return 3

    def npts_per_dim(self) -> Tuple[int, ...]:
        """Return number of points per dimension."""
        return self._mesh.npts_per_dim()

    def npts(self) -> int:
        """Return total number of points."""
        return self._mesh.npts()

    def nodes_x(self) -> Array:
        """Return 1D reference collocation nodes in x direction."""
        return self._tensor_basis.nodes_1d(0)

    def nodes_y(self) -> Array:
        """Return 1D reference collocation nodes in y direction."""
        return self._tensor_basis.nodes_1d(1)

    def nodes_z(self) -> Array:
        """Return 1D reference collocation nodes in z direction."""
        return self._tensor_basis.nodes_1d(2)

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
            Spatial dimension (0 for x, 1 for y, 2 for z).

        Returns
        -------
        Array
            Derivative matrix in physical coordinates. Shape: (npts, npts)
        """
        return self._tensor_basis.derivative_matrix(order, dim)

    def derivative_matrix_x(self, order: int = 1) -> Array:
        """Return physical derivative matrix in x direction."""
        return self._tensor_basis.derivative_matrix(order, dim=0)

    def derivative_matrix_y(self, order: int = 1) -> Array:
        """Return physical derivative matrix in y direction."""
        return self._tensor_basis.derivative_matrix(order, dim=1)

    def derivative_matrix_z(self, order: int = 1) -> Array:
        """Return physical derivative matrix in z direction."""
        return self._tensor_basis.derivative_matrix(order, dim=2)

    def reference_derivative_matrix(self, order: int, dim: int) -> Array:
        """Return reference derivative matrix for specified order and dimension.

        Parameters
        ----------
        order : int
            Derivative order (1 for first derivative, 2 for second, etc.)
        dim : int
            Spatial dimension (0 for x, 1 for y, 2 for z).

        Returns
        -------
        Array
            Derivative matrix in reference coordinates. Shape: (npts, npts)
        """
        return self._tensor_basis.reference_derivative_matrix(order, dim)
