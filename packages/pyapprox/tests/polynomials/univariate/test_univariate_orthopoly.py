"""Tests for orthonormal polynomial univariate bases."""

from abc import ABC, abstractmethod

import numpy as np

from pyapprox.surrogates.affine.univariate import (
    Chebyshev1stKindPolynomial1D,
    Chebyshev2ndKindPolynomial1D,
    GaussLobattoQuadratureRule,
    GaussQuadratureRule,
    HermitePolynomial1D,
    JacobiPolynomial1D,
    LegendrePolynomial1D,
)
from pyapprox.util.backends.numpy import NumpyBkd


class _OrthonormalPolynomialTestBase(ABC):
    """Base class for orthonormal polynomial tests."""

    @abstractmethod
    def _create_polynomial(self):
        """Create the polynomial instance for testing."""
        raise NotImplementedError

    @abstractmethod
    def _get_integration_bounds(self):
        """Return the integration bounds for orthonormality testing."""
        raise NotImplementedError

    @abstractmethod
    def _get_weight_function(self):
        """Return the weight function for orthonormality testing."""
        raise NotImplementedError

    def test_set_nterms(self):
        """Test that nterms can be set and retrieved."""
        poly = self._create_polynomial()
        assert poly.nterms() == 0

        poly.set_nterms(5)
        assert poly.nterms() == 5

        poly.set_nterms(10)
        assert poly.nterms() == 10

        # Can also reduce
        poly.set_nterms(3)
        assert poly.nterms() == 3

    def test_evaluation_shape(self):
        """Test that evaluation returns correct shape."""
        bkd = NumpyBkd
        poly = self._create_polynomial()
        poly.set_nterms(5)

        samples = bkd.linspace(-0.9, 0.9, 10)[None, :]
        vals = poly(samples)

        assert vals.shape == (10, 5)

    def test_jacobian_batch_shape(self):
        """Test that Jacobians have correct shape."""
        bkd = NumpyBkd
        poly = self._create_polynomial()
        poly.set_nterms(5)

        samples = bkd.linspace(-0.9, 0.9, 10)[None, :]
        jacs = poly.jacobian_batch(samples)

        assert jacs.shape == (10, 5)

    def test_hessian_batch_shape(self):
        """Test that Hessians have correct shape."""
        bkd = NumpyBkd
        poly = self._create_polynomial()
        poly.set_nterms(5)

        samples = bkd.linspace(-0.9, 0.9, 10)[None, :]
        hess = poly.hessian_batch(samples)

        assert hess.shape == (10, 5)

    def test_jacobian_batch_finite_difference(self):
        """Test Jacobians against finite differences."""
        bkd = NumpyBkd
        poly = self._create_polynomial()
        poly.set_nterms(5)

        samples = bkd.linspace(-0.8, 0.8, 5)[None, :]
        jacs = poly.jacobian_batch(samples)

        # Finite difference check
        eps = 1e-7
        for ii in range(samples.shape[1]):
            x = samples[:, ii : ii + 1]
            xp = x + eps
            xm = x - eps
            fd_jac = (poly(xp) - poly(xm)) / (2 * eps)
            bkd.assert_allclose(jacs[ii, :], fd_jac[0, :], rtol=1e-5, atol=1e-5)

    def test_quadrature_rule_shape(self):
        """Test that quadrature rule returns correct shapes."""
        poly = self._create_polynomial()
        poly.set_nterms(10)

        points, weights = poly.gauss_quadrature_rule(5)

        assert points.shape == (1, 5)
        assert weights.shape == (5, 1)

    def test_orthonormality(self):
        """Test that polynomials are orthonormal under quadrature."""
        bkd = NumpyBkd
        poly = self._create_polynomial()
        nterms = 8
        nquad = nterms + 2
        poly.set_nterms(nquad)  # Need enough terms for quadrature

        # Use enough quadrature points for exact integration
        points, weights = poly.gauss_quadrature_rule(nquad)

        # Evaluate polynomials at quadrature points
        vals = poly(points)[:, :nterms]  # (nquad, nterms)

        # Compute mass matrix: M_ij = sum_k w_k * phi_i(x_k) * phi_j(x_k)
        mass = vals.T @ (vals * weights)

        # Should be identity
        bkd.assert_allclose(mass, bkd.eye(nterms), rtol=1e-10, atol=1e-10)


class TestLegendrePolynomial(_OrthonormalPolynomialTestBase):
    """Tests for Legendre polynomials."""

    def _create_polynomial(self):
        return LegendrePolynomial1D(NumpyBkd)

    def _get_integration_bounds(self):
        return (-1, 1)

    def _get_weight_function(self):
        return lambda x: np.ones_like(x)

    def test_legendre_specific_values(self):
        """Test specific Legendre polynomial values."""
        bkd = NumpyBkd
        poly = LegendrePolynomial1D(bkd)
        poly.set_nterms(4)

        # P_0(x) = 1/sqrt(2) for orthonormal Legendre on [-1,1] with prob meas
        # For probability measure, P_0(x) = 1
        x = bkd.array([[0.0]])
        vals = poly(x)

        # P_0(0) = 1 (normalized)
        bkd.assert_allclose(vals[0, 0], 1.0, rtol=1e-10)

        # P_1(x) = sqrt(3)*x for probability measure
        # P_1(0) = 0
        bkd.assert_allclose(vals[0, 1], 0.0, atol=1e-10)


class TestJacobiPolynomial(_OrthonormalPolynomialTestBase):
    """Tests for Jacobi polynomials."""

    def _create_polynomial(self):
        return JacobiPolynomial1D(1.0, 2.0, NumpyBkd)

    def _get_integration_bounds(self):
        return (-1, 1)

    def _get_weight_function(self):
        return lambda x: (1 - x) ** 1.0 * (1 + x) ** 2.0


class TestChebyshev1stKindPolynomial(_OrthonormalPolynomialTestBase):
    """Tests for Chebyshev polynomials of the first kind."""

    def _create_polynomial(self):
        return Chebyshev1stKindPolynomial1D(NumpyBkd)

    def _get_integration_bounds(self):
        return (-1, 1)

    def _get_weight_function(self):
        return lambda x: 1 / np.sqrt(1 - x**2)

    def test_orthonormality(self):
        """Test Chebyshev 1st kind orthonormality.

        For Chebyshev 1st kind with our normalization:
        - T_0 is not scaled (integral = pi)
        - T_n for n>=1 are scaled by 1/sqrt(2) (integral = pi/2 * 1/2 = pi/4)
        Actually after scaling, they all integrate to pi/2 except T_0.
        """
        bkd = NumpyBkd
        poly = self._create_polynomial()
        nterms = 8
        nquad = nterms + 2
        poly.set_nterms(nquad)

        points, weights = poly.gauss_quadrature_rule(nquad)
        vals = poly(points)[:, :nterms]
        mass = vals.T @ (vals * weights)

        # Check orthogonality (off-diagonal should be zero)
        off_diag_mask = ~bkd.to_numpy(bkd.eye(nterms)).astype(bool)
        off_diag = mass[off_diag_mask]
        bkd.assert_allclose(off_diag, bkd.zeros(off_diag.shape), rtol=1e-10, atol=1e-10)

        # Check normalization: T_0 integrates to pi, rest to pi/2
        expected_diag = np.array([np.pi] + [np.pi / 2] * (nterms - 1))
        actual_diag = bkd.get_diagonal(mass)
        bkd.assert_allclose(actual_diag, expected_diag, rtol=1e-10, atol=1e-10)


class TestChebyshev2ndKindPolynomial(_OrthonormalPolynomialTestBase):
    """Tests for Chebyshev polynomials of the second kind."""

    def _create_polynomial(self):
        return Chebyshev2ndKindPolynomial1D(NumpyBkd)

    def _get_integration_bounds(self):
        return (-1, 1)

    def _get_weight_function(self):
        return lambda x: np.sqrt(1 - x**2)

    def test_orthonormality(self):
        """Test Chebyshev 2nd kind orthonormality.

        Note: Quadrature weights are scaled by pi/2, so mass matrix diagonal = pi/2.
        """
        bkd = NumpyBkd
        poly = self._create_polynomial()
        nterms = 8
        nquad = nterms + 2
        poly.set_nterms(nquad)

        points, weights = poly.gauss_quadrature_rule(nquad)
        vals = poly(points)[:, :nterms]
        mass = vals.T @ (vals * weights)

        # For Chebyshev 2nd kind with pi/2-scaled weights, diagonal is pi/2
        expected = (np.pi / 2) * bkd.eye(nterms)
        bkd.assert_allclose(mass, expected, rtol=1e-10, atol=1e-10)


class TestHermitePolynomial(_OrthonormalPolynomialTestBase):
    """Tests for Hermite polynomials."""

    def _create_polynomial(self):
        return HermitePolynomial1D(NumpyBkd)

    def _get_integration_bounds(self):
        return (-np.inf, np.inf)

    def _get_weight_function(self):
        return lambda x: np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi)

    def test_hermite_orthonormality(self):
        """Test Hermite orthonormality with standard normal weight."""
        bkd = NumpyBkd
        poly = HermitePolynomial1D(bkd)
        nterms = 8
        nquad = nterms + 2
        poly.set_nterms(nquad)  # Need enough terms for quadrature

        # Use Gauss-Hermite quadrature
        points, weights = poly.gauss_quadrature_rule(nquad)

        vals = poly(points)[:, :nterms]
        mass = vals.T @ (vals * weights)

        bkd.assert_allclose(mass, bkd.eye(nterms), rtol=1e-10, atol=1e-10)


class TestGaussQuadratureRule:
    """Tests for Gaussian quadrature rule wrapper."""

    def test_quadrature_rule_accuracy(self):
        """Test that quadrature exactly integrates polynomials."""
        bkd = NumpyBkd
        poly = LegendrePolynomial1D(bkd)
        poly.set_nterms(10)
        quad = GaussQuadratureRule(poly)

        # N-point Gauss quadrature exactly integrates degree 2N-1 polynomials
        # For uniform measure on [-1,1], integral of 1 is 1 (prob measure)
        points, weights = quad(5)
        integral = bkd.sum(weights)
        bkd.assert_allclose(integral, 1.0, rtol=1e-14)

        # Integral of x^2 on [-1,1] with uniform prob measure = 1/3
        integral_x2 = bkd.sum(points[0, :] ** 2 * weights[:, 0])
        bkd.assert_allclose(integral_x2, 1.0 / 3.0, rtol=1e-14)

    def test_caching(self):
        """Test that caching works correctly."""
        bkd = NumpyBkd
        poly = LegendrePolynomial1D(bkd)
        poly.set_nterms(10)
        quad = GaussQuadratureRule(poly, store=True)

        points1, weights1 = quad(5)
        points2, weights2 = quad(5)

        # Should be the same objects if cached
        assert points1 is points2
        assert weights1 is weights2


class TestGaussLobattoQuadratureRule:
    """Tests for Gauss-Lobatto quadrature rule."""

    def test_endpoints_included(self):
        """Test that Lobatto rule includes endpoints."""
        bkd = NumpyBkd
        poly = LegendrePolynomial1D(bkd)
        poly.set_nterms(10)
        quad = GaussLobattoQuadratureRule(poly)

        points, weights = quad(5)

        # First and last points should be -1 and 1
        bkd.assert_allclose(points[0, 0], -1.0, rtol=1e-14)
        bkd.assert_allclose(points[0, -1], 1.0, rtol=1e-14)
