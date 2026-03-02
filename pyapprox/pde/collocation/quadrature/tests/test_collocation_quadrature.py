"""Tests for collocation quadrature weights."""


from pyapprox.pde.collocation.basis.chebyshev.basis_1d import (
    ChebyshevBasis1D,
)
from pyapprox.pde.collocation.mesh.transformed import TransformedMesh1D
from pyapprox.pde.collocation.mesh.transforms.affine import (
    AffineTransform1D,
)
from pyapprox.pde.collocation.quadrature.collocation_quadrature import (
    CollocationQuadrature1D,
)
from pyapprox.surrogates.affine.univariate.globalpoly.quadrature import (
    ClenshawCurtisQuadratureRule,
)
from pyapprox.util.backends.protocols import Array


class TestCollocationQuadrature:
    """Base test class for CollocationQuadrature1D."""

    def _make_reference_basis(self, bkd, npts: int) -> ChebyshevBasis1D[Array]:
        mesh = TransformedMesh1D(npts, bkd)
        return ChebyshevBasis1D(mesh, bkd)

    def _make_physical_basis(
        self, bkd, npts: int, a: float, b: float
    ) -> ChebyshevBasis1D[Array]:
        transform = AffineTransform1D((a, b), bkd)
        mesh = TransformedMesh1D(npts, bkd, transform)
        return ChebyshevBasis1D(mesh, bkd)

    def test_full_domain_weights_shape(self, bkd) -> None:
        """Full-domain weights have correct shape and are all positive."""
        npts = 9
        basis = self._make_reference_basis(bkd, npts)
        quad = CollocationQuadrature1D(basis, bkd)
        w = quad.full_domain_weights()
        assert w.shape == (npts,)
        # All weights should be positive for full-domain on CGL nodes
        min_w = bkd.min(w)
        assert float(bkd.to_numpy(bkd.asarray([min_w]))[0]) > 0.0

    def test_full_domain_matches_cc(self, bkd) -> None:
        """Full-domain weights match Clenshaw-Curtis Lebesgue weights."""
        npts = 9  # 2^3 + 1
        basis = self._make_reference_basis(bkd, npts)
        quad = CollocationQuadrature1D(basis, bkd)
        w = quad.full_domain_weights()

        # _unordered_pts_wts returns probability weights (sum=1);
        # multiply by 2 for Lebesgue measure on [-1, 1]
        cc = ClenshawCurtisQuadratureRule(bkd)
        _, cc_wts = cc._unordered_pts_wts(3)  # level 3 = 9 points
        cc_wts_lebesgue = cc_wts * 2.0

        bkd.assert_allclose(w, cc_wts_lebesgue, atol=1e-12)

    def test_full_domain_polynomial_exactness(self, bkd) -> None:
        """Full-domain weights integrate polynomials exactly up to deg n-1."""
        npts = 9
        basis = self._make_reference_basis(bkd, npts)
        quad = CollocationQuadrature1D(basis, bkd)
        w = quad.full_domain_weights()
        nodes = basis.nodes()

        for degree in range(npts):
            f = nodes**degree
            computed = bkd.sum(w * f)
            # integral of x^d from -1 to 1
            if degree % 2 == 1:
                exact = 0.0  # odd function on symmetric interval
            else:
                exact = 2.0 / (degree + 1)
            bkd.assert_allclose(
                bkd.asarray([computed]),
                bkd.asarray([exact]),
                atol=1e-12,
            )

    def test_subdomain_polynomial_exactness(self, bkd) -> None:
        """Subdomain weights integrate polynomials exactly up to deg n-1."""
        npts = 9
        basis = self._make_reference_basis(bkd, npts)
        quad = CollocationQuadrature1D(basis, bkd)
        nodes = basis.nodes()
        a_sub, b_sub = -0.5, 0.7

        for degree in range(npts):
            f = nodes**degree
            w = quad.weights(a_sub, b_sub)
            computed = bkd.sum(w * f)
            # integral of x^d from a to b = (b^{d+1} - a^{d+1}) / (d+1)
            exact = (b_sub ** (degree + 1) - a_sub ** (degree + 1)) / (degree + 1)
            bkd.assert_allclose(
                bkd.asarray([computed]),
                bkd.asarray([exact]),
                atol=1e-10,
            )

    def test_subinterval_additivity(self, bkd) -> None:
        """Weights for [a,c] equal sum of weights for [a,b] and [b,c]."""
        npts = 9
        basis = self._make_reference_basis(bkd, npts)
        quad = CollocationQuadrature1D(basis, bkd)

        a, b, c = -0.6, 0.1, 0.8
        w_ac = quad.weights(a, c)
        w_ab = quad.weights(a, b)
        w_bc = quad.weights(b, c)
        bkd.assert_allclose(w_ac, w_ab + w_bc, atol=1e-12)

    def test_physical_domain(self, bkd) -> None:
        """Integration on physical domain [0, L] works correctly."""
        npts = 9
        L = 2.0
        basis = self._make_physical_basis(bkd, npts, 0.0, L)
        quad = CollocationQuadrature1D(basis, bkd)
        phys_nodes = basis.mesh().points()[0, :]

        # integral_0^L x^2 dx = L^3 / 3
        f = phys_nodes**2
        w = quad.full_domain_weights()
        computed = bkd.sum(w * f)
        exact = L**3 / 3.0
        bkd.assert_allclose(bkd.asarray([computed]), bkd.asarray([exact]), atol=1e-10)

    def test_zero_width_interval(self, bkd) -> None:
        """Weights for zero-width interval are all zero."""
        npts = 9
        basis = self._make_reference_basis(bkd, npts)
        quad = CollocationQuadrature1D(basis, bkd)
        w = quad.weights(0.3, 0.3)
        bkd.assert_allclose(w, bkd.zeros((npts,)), atol=1e-14)

    def test_subdomain_physical(self, bkd) -> None:
        """Subdomain integration on physical domain [0, 2]."""
        npts = 11
        basis = self._make_physical_basis(bkd, npts, 0.0, 2.0)
        quad = CollocationQuadrature1D(basis, bkd)
        phys_nodes = basis.mesh().points()[0, :]

        # integral_{0.5}^{1.5} x^2 dx = (1.5^3 - 0.5^3) / 3
        f = phys_nodes**2
        w = quad.weights(0.5, 1.5)
        computed = bkd.sum(w * f)
        exact = (1.5**3 - 0.5**3) / 3.0
        bkd.assert_allclose(bkd.asarray([computed]), bkd.asarray([exact]), atol=1e-10)
