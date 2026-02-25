"""2D quadrature weights at tensor-product collocation nodes.

Computes integration weights on a 2D domain by taking the tensor product
of 1D reference quadrature weights and multiplying by the full 2D Jacobian
determinant. This handles curvilinear coordinate transforms (e.g., polar)
correctly.
"""

from typing import Generic, Optional, Tuple

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.pde.collocation.basis.chebyshev.basis_1d import (
    ChebyshevBasis1D,
)
from pyapprox.pde.collocation.basis.chebyshev.basis_2d import (
    ChebyshevBasis2D,
)
from pyapprox.pde.collocation.mesh import TransformedMesh1D
from pyapprox.pde.collocation.quadrature.collocation_quadrature import (
    CollocationQuadrature1D,
)


class CollocationQuadrature2D(Generic[Array]):
    """Quadrature weights at 2D tensor-product collocation nodes.

    Computes weights w such that ``w @ f_values`` approximates the
    integral of f over the 2D physical domain. Works with curvilinear
    coordinate transforms (e.g., polar).

    The algorithm:
    1. Build 1D reference quadrature weights on [-1, 1] for each
       dimension using ``CollocationQuadrature1D``.
    2. Tensor-product the 1D weights (first dimension varies fastest).
    3. Multiply by the full 2D Jacobian determinant at each node.

    Parameters
    ----------
    basis : ChebyshevBasis2D
        The 2D Chebyshev collocation basis.
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self, basis: ChebyshevBasis2D[Array], bkd: Backend[Array]
    ) -> None:
        self._bkd = bkd
        self._basis = basis
        npts_x, npts_y = basis.npts_per_dim()

        # Create temporary 1D bases on reference [-1, 1] (no transform)
        mesh_x = TransformedMesh1D(npts_x, bkd)
        mesh_y = TransformedMesh1D(npts_y, bkd)
        basis_x = ChebyshevBasis1D(mesh_x, bkd)
        basis_y = ChebyshevBasis1D(mesh_y, bkd)

        self._quad_x = CollocationQuadrature1D(basis_x, bkd)
        self._quad_y = CollocationQuadrature1D(basis_y, bkd)

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def full_domain_weights(self) -> Array:
        """Compute quadrature weights for the full 2D physical domain.

        Returns
        -------
        Array, shape (npts,)
            Weights such that ``w @ f_values`` approximates the
            integral of f over the entire domain.
        """
        w_x = self._quad_x.full_domain_weights()  # (npts_x,)
        w_y = self._quad_y.full_domain_weights()  # (npts_y,)
        return self._tensor_product_with_jacobian(w_x, w_y)

    def weights(
        self,
        x_bounds: Optional[Tuple[float, float]] = None,
        y_bounds: Optional[Tuple[float, float]] = None,
    ) -> Array:
        """Compute quadrature weights for a subdomain.

        Bounds are in **reference** coordinates (each in [-1, 1]).
        If a bound is None, the full range [-1, 1] is used for that
        dimension.

        Parameters
        ----------
        x_bounds : tuple of float, optional
            (xi_a, xi_b) in reference first-dimension coordinates.
        y_bounds : tuple of float, optional
            (eta_a, eta_b) in reference second-dimension coordinates.

        Returns
        -------
        Array, shape (npts,)
            Subdomain quadrature weights.
        """
        if x_bounds is not None:
            w_x = self._quad_x.weights(x_bounds[0], x_bounds[1])
        else:
            w_x = self._quad_x.full_domain_weights()

        if y_bounds is not None:
            w_y = self._quad_y.weights(y_bounds[0], y_bounds[1])
        else:
            w_y = self._quad_y.full_domain_weights()

        return self._tensor_product_with_jacobian(w_x, w_y)

    def _tensor_product_with_jacobian(
        self, w_x: Array, w_y: Array
    ) -> Array:
        """Tensor-product 1D weights and multiply by 2D Jacobian determinant.

        Uses first-dim-fastest ordering matching ``compute_cartesian_product``.
        """
        bkd = self._bkd
        # kron(w_y, w_x) gives w_2d[i + j*nx] = w_x[i] * w_y[j]
        w_ref_2d = bkd.kron(w_y, w_x)  # (npts,)

        # Multiply by full 2D Jacobian determinant
        mesh = self._basis.mesh()
        jac_det = mesh.jacobian_determinant()  # (npts,)
        return w_ref_2d * jac_det
