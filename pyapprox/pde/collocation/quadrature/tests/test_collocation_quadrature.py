"""Tests for collocation quadrature weights."""

import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.test_utils import load_tests  # noqa: F401
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


class TestCollocationQuadrature(Generic[Array], unittest.TestCase):
    """Base test class for CollocationQuadrature1D."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def _make_reference_basis(self, npts: int) -> ChebyshevBasis1D[Array]:
        bkd = self.bkd()
        mesh = TransformedMesh1D(npts, bkd)
        return ChebyshevBasis1D(mesh, bkd)

    def _make_physical_basis(
        self, npts: int, a: float, b: float
    ) -> ChebyshevBasis1D[Array]:
        bkd = self.bkd()
        transform = AffineTransform1D((a, b), bkd)
        mesh = TransformedMesh1D(npts, bkd, transform)
        return ChebyshevBasis1D(mesh, bkd)

    def test_full_domain_weights_shape(self) -> None:
        """Full-domain weights have correct shape and are all positive."""
        bkd = self.bkd()
        npts = 9
        basis = self._make_reference_basis(npts)
        quad = CollocationQuadrature1D(basis, bkd)
        w = quad.full_domain_weights()
        self.assertEqual(w.shape, (npts,))
        # All weights should be positive for full-domain on CGL nodes
        min_w = bkd.min(w)
        self.assertGreater(float(bkd.to_numpy(bkd.asarray([min_w]))[0]), 0.0)

    def test_full_domain_matches_cc(self) -> None:
        """Full-domain weights match Clenshaw-Curtis Lebesgue weights."""
        bkd = self.bkd()
        npts = 9  # 2^3 + 1
        basis = self._make_reference_basis(npts)
        quad = CollocationQuadrature1D(basis, bkd)
        w = quad.full_domain_weights()

        # _unordered_pts_wts returns probability weights (sum=1);
        # multiply by 2 for Lebesgue measure on [-1, 1]
        cc = ClenshawCurtisQuadratureRule(bkd)
        _, cc_wts = cc._unordered_pts_wts(3)  # level 3 = 9 points
        cc_wts_lebesgue = cc_wts * 2.0

        bkd.assert_allclose(w, cc_wts_lebesgue, atol=1e-12)

    def test_full_domain_polynomial_exactness(self) -> None:
        """Full-domain weights integrate polynomials exactly up to deg n-1."""
        bkd = self.bkd()
        npts = 9
        basis = self._make_reference_basis(npts)
        quad = CollocationQuadrature1D(basis, bkd)
        w = quad.full_domain_weights()
        nodes = basis.nodes()

        for degree in range(npts):
            f = nodes ** degree
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

    def test_subdomain_polynomial_exactness(self) -> None:
        """Subdomain weights integrate polynomials exactly up to deg n-1."""
        bkd = self.bkd()
        npts = 9
        basis = self._make_reference_basis(npts)
        quad = CollocationQuadrature1D(basis, bkd)
        nodes = basis.nodes()
        a_sub, b_sub = -0.5, 0.7

        for degree in range(npts):
            f = nodes ** degree
            w = quad.weights(a_sub, b_sub)
            computed = bkd.sum(w * f)
            # integral of x^d from a to b = (b^{d+1} - a^{d+1}) / (d+1)
            exact = (
                b_sub ** (degree + 1) - a_sub ** (degree + 1)
            ) / (degree + 1)
            bkd.assert_allclose(
                bkd.asarray([computed]),
                bkd.asarray([exact]),
                atol=1e-10,
            )

    def test_subinterval_additivity(self) -> None:
        """Weights for [a,c] equal sum of weights for [a,b] and [b,c]."""
        bkd = self.bkd()
        npts = 9
        basis = self._make_reference_basis(npts)
        quad = CollocationQuadrature1D(basis, bkd)

        a, b, c = -0.6, 0.1, 0.8
        w_ac = quad.weights(a, c)
        w_ab = quad.weights(a, b)
        w_bc = quad.weights(b, c)
        bkd.assert_allclose(w_ac, w_ab + w_bc, atol=1e-12)

    def test_physical_domain(self) -> None:
        """Integration on physical domain [0, L] works correctly."""
        bkd = self.bkd()
        npts = 9
        L = 2.0
        basis = self._make_physical_basis(npts, 0.0, L)
        quad = CollocationQuadrature1D(basis, bkd)
        phys_nodes = basis.mesh().points()[0, :]

        # integral_0^L x^2 dx = L^3 / 3
        f = phys_nodes ** 2
        w = quad.full_domain_weights()
        computed = bkd.sum(w * f)
        exact = L ** 3 / 3.0
        bkd.assert_allclose(
            bkd.asarray([computed]), bkd.asarray([exact]), atol=1e-10
        )

    def test_zero_width_interval(self) -> None:
        """Weights for zero-width interval are all zero."""
        bkd = self.bkd()
        npts = 9
        basis = self._make_reference_basis(npts)
        quad = CollocationQuadrature1D(basis, bkd)
        w = quad.weights(0.3, 0.3)
        bkd.assert_allclose(w, bkd.zeros((npts,)), atol=1e-14)

    def test_subdomain_physical(self) -> None:
        """Subdomain integration on physical domain [0, 2]."""
        bkd = self.bkd()
        npts = 11
        basis = self._make_physical_basis(npts, 0.0, 2.0)
        quad = CollocationQuadrature1D(basis, bkd)
        phys_nodes = basis.mesh().points()[0, :]

        # integral_{0.5}^{1.5} x^2 dx = (1.5^3 - 0.5^3) / 3
        f = phys_nodes ** 2
        w = quad.weights(0.5, 1.5)
        computed = bkd.sum(w * f)
        exact = (1.5 ** 3 - 0.5 ** 3) / 3.0
        bkd.assert_allclose(
            bkd.asarray([computed]), bkd.asarray([exact]), atol=1e-10
        )


class TestCollocationQuadratureNumpy(
    TestCollocationQuadrature[NDArray[Any]]
):
    """NumPy backend tests."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestCollocationQuadratureTorch(
    TestCollocationQuadrature[torch.Tensor]
):
    """PyTorch backend tests."""

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)


if __name__ == "__main__":
    unittest.main()
