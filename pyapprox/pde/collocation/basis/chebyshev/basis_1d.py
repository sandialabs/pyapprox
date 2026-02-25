"""1D Chebyshev collocation basis.

Convenience wrapper combining Chebyshev nodes and derivative matrices
with the tensor product basis infrastructure.
"""

from typing import Generic, Optional, Tuple

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
from pyapprox.pde.collocation.mesh.transformed import TransformedMesh1D


class ChebyshevBasis1D(Generic[Array]):
    """Chebyshev collocation basis for 1D domains.

    Uses Chebyshev-Gauss-Lobatto nodes and barycentric derivative matrices.

    Parameters
    ----------
    mesh : MeshWithTransformProtocol
        Mesh providing reference nodes, physical nodes, and gradient factors.
        Use TransformedMesh1D with an optional transform.
    bkd : Backend
        Computational backend.

    Examples
    --------
    Reference domain [-1, 1]:

    >>> mesh = TransformedMesh1D(10, bkd)
    >>> basis = ChebyshevBasis1D(mesh, bkd)

    Physical domain [0, 1] with affine transform:

    >>> transform = AffineTransform1D((0.0, 1.0), bkd)
    >>> mesh = TransformedMesh1D(10, bkd, transform)
    >>> basis = ChebyshevBasis1D(mesh, bkd)
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
        return 1

    def npts_per_dim(self) -> Tuple[int, ...]:
        """Return number of points per dimension."""
        return self._mesh.npts_per_dim()

    def npts(self) -> int:
        """Return total number of points."""
        return self._mesh.npts()

    def nodes(self) -> Array:
        """Return 1D reference collocation nodes.

        Returns
        -------
        Array
            Chebyshev-Gauss-Lobatto nodes. Shape: (npts,)
        """
        return self._tensor_basis.nodes_1d(0)

    def mesh(self) -> MeshWithTransformProtocol[Array]:
        """Return the underlying mesh."""
        return self._mesh

    def derivative_matrix(self, order: int = 1, dim: int = 0) -> Array:
        """Return physical derivative matrix.

        Parameters
        ----------
        order : int, optional
            Derivative order. Default is 1.
        dim : int, optional
            Spatial dimension. Must be 0 for 1D. Default is 0.

        Returns
        -------
        Array
            Derivative matrix in physical coordinates. Shape: (npts, npts)
        """
        if dim != 0:
            raise ValueError(f"dim must be 0 for 1D basis, got {dim}")
        return self._tensor_basis.derivative_matrix(order, dim=0)

    def reference_derivative_matrix(self, order: int = 1, dim: int = 0) -> Array:
        """Return reference derivative matrix.

        Parameters
        ----------
        order : int, optional
            Derivative order. Default is 1.
        dim : int, optional
            Spatial dimension. Must be 0 for 1D. Default is 0.

        Returns
        -------
        Array
            Derivative matrix in reference coordinates. Shape: (npts, npts)
        """
        if dim != 0:
            raise ValueError(f"dim must be 0 for 1D basis, got {dim}")
        return self._tensor_basis.reference_derivative_matrix(order, dim=0)
