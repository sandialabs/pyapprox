"""1D Chebyshev collocation basis.

Convenience wrapper combining Chebyshev nodes and derivative matrices
with the tensor product basis infrastructure.
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


class ChebyshevBasis1D(Generic[Array]):
    """Chebyshev collocation basis for 1D domains.

    Uses Chebyshev-Gauss-Lobatto nodes and barycentric derivative matrices.

    Parameters
    ----------
    npts : int
        Number of collocation points.
    bkd : Backend
        Computational backend.
    """

    def __init__(self, npts: int, bkd: Backend[Array]):
        self._bkd = bkd
        self._npts = npts

        # Create 1D components
        nodes_gen = ChebyshevGaussLobattoNodes1D(bkd)
        deriv_computer = ChebyshevDerivativeMatrix1D(bkd)

        # Build tensor product basis (1D is just a special case)
        self._tensor_basis = TensorProductBasis(
            [nodes_gen], [deriv_computer], (npts,), bkd
        )

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def ndim(self) -> int:
        """Return the number of spatial dimensions."""
        return 1

    def npts_per_dim(self) -> Tuple[int]:
        """Return number of points per dimension."""
        return (self._npts,)

    def npts(self) -> int:
        """Return total number of points."""
        return self._npts

    def nodes(self) -> Array:
        """Return collocation nodes.

        Returns
        -------
        Array
            Chebyshev-Gauss-Lobatto nodes. Shape: (npts,)
        """
        return self._tensor_basis.nodes_1d(0)

    def derivative_matrix(self, order: int = 1) -> Array:
        """Return derivative matrix.

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
