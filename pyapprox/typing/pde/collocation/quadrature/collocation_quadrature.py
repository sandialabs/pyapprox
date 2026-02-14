"""Quadrature weights at collocation nodes for full-domain and subdomain integration.

Computes integration weights at Chebyshev-Gauss-Lobatto collocation nodes
by projecting Gauss-Legendre quadrature weights through the Lagrange
interpolation matrix. This follows the approach used in the legacy
ChebyshevCollocationBasis._set_quadrature_weights_at_mesh_pts().
"""

from typing import Generic, cast

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.pde.collocation.basis.chebyshev.basis_1d import (
    ChebyshevBasis1D,
)
from pyapprox.typing.pde.decomposition.interface.interpolation import (
    lagrange_interpolation_matrix,
)
from pyapprox.typing.surrogates.affine.univariate.globalpoly.jacobi import (
    LegendrePolynomial1D,
)
from pyapprox.typing.surrogates.affine.univariate.globalpoly.quadrature import (
    GaussQuadratureRule,
)


class CollocationQuadrature1D(Generic[Array]):
    """Quadrature weights at 1D collocation nodes for integration.

    Given function values at CGL collocation nodes, computes weights w
    such that ``w @ f_values`` approximates the integral of f over a
    specified interval. Weights are exact for polynomials of degree
    at most n-1 where n is the number of collocation nodes.

    The weights are computed by projecting Gauss-Legendre quadrature
    weights through the Lagrange interpolation matrix defined on the
    collocation nodes.

    Parameters
    ----------
    basis : ChebyshevBasis1D
        The 1D Chebyshev collocation basis.
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self, basis: ChebyshevBasis1D[Array], bkd: Backend[Array]
    ) -> None:
        self._bkd = bkd
        self._basis = basis
        self._ref_nodes = basis.nodes()  # shape (npts,)
        npts = self._ref_nodes.shape[0]

        # Pre-compute GL quadrature on [-1, 1] with n+1 points
        # (Lebesgue measure: weights sum to 2)
        n_gl = npts + 1
        poly = LegendrePolynomial1D(bkd)
        poly.set_nterms(n_gl)
        gl_rule = GaussQuadratureRule(poly)
        gl_pts, gl_wts = gl_rule(n_gl)
        self._gl_pts = gl_pts[0, :]       # shape (n_gl,)
        self._gl_wts = gl_wts[:, 0] * 2.0  # probability -> Lebesgue

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def weights(self, a_sub: float, b_sub: float) -> Array:
        """Compute quadrature weights for interval [a_sub, b_sub].

        Parameters
        ----------
        a_sub : float
            Left endpoint in physical coordinates.
        b_sub : float
            Right endpoint in physical coordinates.

        Returns
        -------
        Array, shape (npts,)
            Weights such that ``w @ f_values`` approximates
            the integral of f over [a_sub, b_sub].
        """
        bkd = self._bkd
        transform = self._basis.mesh().transform()

        # Map physical bounds to reference coordinates
        if transform is not None:
            xi_a_arr = transform.map_to_reference(bkd.asarray([[a_sub]]))
            xi_b_arr = transform.map_to_reference(bkd.asarray([[b_sub]]))
            xi_a = xi_a_arr[0, 0]
            xi_b = xi_b_arr[0, 0]
            # Jacobian determinant (constant for affine)
            jac_det = transform.jacobian_determinant(
                bkd.asarray([[0.0]])
            )[0]
        else:
            xi_a = a_sub
            xi_b = b_sub
            jac_det = 1.0

        # Map GL points from [-1, 1] to reference subdomain [xi_a, xi_b]
        scale = (xi_b - xi_a) / 2.0
        shift = (xi_a + xi_b) / 2.0
        gl_mapped = scale * self._gl_pts + shift
        gl_wts_scaled = scale * self._gl_wts

        # Build Lagrange interpolation matrix: L[k, j] = L_j(gl_mapped_k)
        L = lagrange_interpolation_matrix(
            self._ref_nodes, gl_mapped, bkd
        )  # shape (n_gl, npts)

        # Project GL weights to collocation nodes
        weights_ref: Array = L.T @ gl_wts_scaled  # shape (npts,)

        if transform is not None:
            return cast(Array, jac_det * weights_ref)
        return weights_ref

    def full_domain_weights(self) -> Array:
        """Compute quadrature weights for the entire physical domain.

        Returns
        -------
        Array, shape (npts,)
            Weights for full-domain integration.
        """
        bkd = self._bkd
        phys_pts = self._basis.mesh().points()  # shape (1, npts)
        a = float(bkd.to_numpy(phys_pts[0, 0]))
        b = float(bkd.to_numpy(phys_pts[0, -1]))
        return self.weights(a, b)
