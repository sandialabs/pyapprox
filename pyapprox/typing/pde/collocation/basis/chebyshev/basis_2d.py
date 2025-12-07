"""2D Chebyshev collocation basis.

Convenience wrapper combining Chebyshev nodes and derivative matrices
with the tensor product basis infrastructure for 2D domains.
"""

from typing import Generic, Tuple

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.pde.collocation.basis.tensor_product import (
    TensorProductBasis,
)
from pyapprox.typing.pde.collocation.basis.chebyshev.nodes import (
    ChebyshevGaussLobattoNodes1D,
)
from pyapprox.typing.pde.collocation.basis.chebyshev.derivative import (
    ChebyshevDerivativeMatrix1D,
)


class ChebyshevBasis2D(Generic[Array]):
    """Chebyshev collocation basis for 2D domains.

    Uses Chebyshev-Gauss-Lobatto nodes and barycentric derivative matrices
    in each direction, assembled via Kronecker products.

    Parameters
    ----------
    npts_x : int
        Number of collocation points in x direction.
    npts_y : int
        Number of collocation points in y direction.
    bkd : Backend
        Computational backend.
    """

    def __init__(self, npts_x: int, npts_y: int, bkd: Backend[Array]):
        self._bkd = bkd
        self._npts_per_dim = (npts_x, npts_y)

        # Create 1D components for each dimension
        nodes_gen_x = ChebyshevGaussLobattoNodes1D(bkd)
        nodes_gen_y = ChebyshevGaussLobattoNodes1D(bkd)
        deriv_computer_x = ChebyshevDerivativeMatrix1D(bkd)
        deriv_computer_y = ChebyshevDerivativeMatrix1D(bkd)

        # Build tensor product basis
        self._tensor_basis = TensorProductBasis(
            [nodes_gen_x, nodes_gen_y],
            [deriv_computer_x, deriv_computer_y],
            (npts_x, npts_y),
            bkd,
        )

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def ndim(self) -> int:
        """Return the number of spatial dimensions."""
        return 2

    def npts_per_dim(self) -> Tuple[int, int]:
        """Return number of points per dimension."""
        return self._npts_per_dim

    def npts(self) -> int:
        """Return total number of points."""
        return self._npts_per_dim[0] * self._npts_per_dim[1]

    def nodes_x(self) -> Array:
        """Return 1D collocation nodes in x direction.

        Returns
        -------
        Array
            Chebyshev-Gauss-Lobatto nodes. Shape: (npts_x,)
        """
        return self._tensor_basis.nodes_1d(0)

    def nodes_y(self) -> Array:
        """Return 1D collocation nodes in y direction.

        Returns
        -------
        Array
            Chebyshev-Gauss-Lobatto nodes. Shape: (npts_y,)
        """
        return self._tensor_basis.nodes_1d(1)

    def derivative_matrix(self, order: int, dim: int) -> Array:
        """Return derivative matrix for specified order and dimension.

        Parameters
        ----------
        order : int
            Derivative order (1 for first derivative, 2 for second, etc.)
        dim : int
            Spatial dimension (0 for x, 1 for y).

        Returns
        -------
        Array
            Full derivative matrix. Shape: (npts, npts)
        """
        return self._tensor_basis.derivative_matrix(order, dim)

    def derivative_matrix_x(self, order: int = 1) -> Array:
        """Return derivative matrix in x direction.

        Parameters
        ----------
        order : int, optional
            Derivative order. Default is 1.

        Returns
        -------
        Array
            Derivative matrix. Shape: (npts, npts)
        """
        return self._tensor_basis.derivative_matrix(order, dim=0)

    def derivative_matrix_y(self, order: int = 1) -> Array:
        """Return derivative matrix in y direction.

        Parameters
        ----------
        order : int, optional
            Derivative order. Default is 1.

        Returns
        -------
        Array
            Derivative matrix. Shape: (npts, npts)
        """
        return self._tensor_basis.derivative_matrix(order, dim=1)
