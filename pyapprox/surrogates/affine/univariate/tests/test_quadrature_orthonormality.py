"""Tests for Gauss quadrature exactness and orthonormality.

These tests verify:
1. Gauss quadrature exactly integrates polynomials up to degree 2n-1
2. The Gramian matrix of orthonormal polynomials is the identity

Theory:
- n-point Gaussian quadrature is exact for polynomials of degree <= 2n-1
- Orthonormal polynomials satisfy <p_i, p_j> = delta_ij (Kronecker delta)
- The Gramian G_ij = sum_k w_k * p_i(x_k) * p_j(x_k) should equal I
"""

import math
import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.surrogates.affine.univariate.globalpoly import (
    CharlierPolynomial1D,
    Chebyshev1stKindPolynomial1D,
    Chebyshev2ndKindPolynomial1D,
    DiscreteChebyshevPolynomial1D,
    DiscreteNumericOrthonormalPolynomial1D,
    HahnPolynomial1D,
    HermitePolynomial1D,
    JacobiPolynomial1D,
    KrawtchoukPolynomial1D,
    LaguerrePolynomial1D,
    LegendrePolynomial1D,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd


class TestQuadratureOrthonormalityBase(Generic[Array], unittest.TestCase):
    """Base class for quadrature and orthonormality tests."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    # =========================================================================
    # Gauss Quadrature Exactness Tests
    # =========================================================================

    def test_legendre_quadrature_exact_for_polynomials(self) -> None:
        """n-point Legendre-Gauss quadrature integrates x^k exactly for k<2n."""
        # For Legendre: integral of x^k on [-1,1] with prob measure is:
        #   0 if k odd
        #   1/(k+1) if k even (after normalizing by 1/2)
        for npoints in [3, 5, 7]:
            poly = LegendrePolynomial1D(self._bkd)
            poly.set_nterms(npoints)
            x, w = poly.gauss_quadrature_rule(npoints)

            # Test monomials x^k for k = 0, 1, ..., 2*npoints-1
            for degree in range(2 * npoints):
                monomial_vals = x[0] ** degree
                integral = self._bkd.sum(w[:, 0] * monomial_vals)

                # Exact integral of x^k on [-1,1] normalized by 1/2
                if degree % 2 == 0:
                    exact = 1.0 / (degree + 1)
                else:
                    exact = 0.0

                self._bkd.assert_allclose(
                    self._bkd.asarray([integral]),
                    self._bkd.asarray([exact]),
                    rtol=1e-12,
                    atol=1e-14,
                )

    def test_jacobi_quadrature_exact_for_polynomials(self) -> None:
        """Jacobi-Gauss quadrature integrates x^k exactly up to degree 2n-1."""
        alpha, beta = 1.0, 2.0

        for npoints in [3, 5, 7]:
            poly = JacobiPolynomial1D(alpha, beta, self._bkd)
            poly.set_nterms(npoints)
            x, w = poly.gauss_quadrature_rule(npoints)

            # Test constant (x^0) - should equal integral of weight
            # For probability measure, this should be 1.0
            ones = self._bkd.ones(x.shape[1])
            integral_one = self._bkd.sum(w[:, 0] * ones)
            self._bkd.assert_allclose(
                self._bkd.asarray([integral_one]),
                self._bkd.asarray([1.0]),
                rtol=1e-12,
                atol=1e-14,
            )

            # Test x^1
            integral_x = self._bkd.sum(w[:, 0] * x[0])
            # For Jacobi with alpha, beta the mean is (beta-alpha)/(alpha+beta+2)
            expected_mean = (beta - alpha) / (alpha + beta + 2)
            self._bkd.assert_allclose(
                self._bkd.asarray([integral_x]),
                self._bkd.asarray([expected_mean]),
                rtol=1e-10,
                atol=1e-12,
            )

    def test_hermite_quadrature_exact_for_polynomials(self) -> None:
        """Hermite-Gauss quadrature integrates x^k exactly up to degree 2n-1."""
        for npoints in [3, 5, 7]:
            poly = HermitePolynomial1D(self._bkd, rho=0.0, prob_meas=True)
            poly.set_nterms(npoints)
            x, w = poly.gauss_quadrature_rule(npoints)

            # For standard normal (prob_meas=True):
            # E[x^k] = 0 for odd k
            # E[x^k] = (k-1)!! for even k (double factorial)
            for degree in range(2 * npoints):
                monomial_vals = x[0] ** degree
                integral = self._bkd.sum(w[:, 0] * monomial_vals)

                if degree % 2 == 1:
                    expected = 0.0
                else:
                    # (k-1)!! for even k
                    expected = 1.0
                    for i in range(1, degree, 2):
                        expected *= i

                # Use atol=1e-9 to handle zero comparisons with numerical noise
                # (higher-degree monomials on Hermite have more numerical error)
                self._bkd.assert_allclose(
                    self._bkd.asarray([integral]),
                    self._bkd.asarray([expected]),
                    rtol=1e-8,
                    atol=1e-9,
                )

    def test_laguerre_quadrature_exact_for_polynomials(self) -> None:
        """Laguerre-Gauss quadrature integrates x^k exactly up to degree 2n-1."""
        for npoints in [3, 5, 7]:
            poly = LaguerrePolynomial1D(self._bkd, rho=0.0)
            poly.set_nterms(npoints)
            x, w = poly.gauss_quadrature_rule(npoints)

            # For rho=0, weight is exp(-x), integral over [0,inf)
            # With probability normalization, E[x^k] = k! (factorial)
            for degree in range(2 * npoints):
                monomial_vals = x[0] ** degree
                integral = self._bkd.sum(w[:, 0] * monomial_vals)

                # k! for Gamma(1) distribution (which is exponential)
                expected = float(math.factorial(degree))

                self._bkd.assert_allclose(
                    self._bkd.asarray([integral]),
                    self._bkd.asarray([expected]),
                    rtol=1e-8,
                    atol=1e-10,
                )

    # =========================================================================
    # Gramian Matrix / Orthonormality Tests
    # =========================================================================

    def test_legendre_gramian_is_identity(self) -> None:
        """Gramian of Legendre polynomials should be identity."""
        for nterms in [3, 5, 8]:
            poly = LegendrePolynomial1D(self._bkd)
            poly.set_nterms(nterms)

            # Use enough quadrature points to exactly integrate degree 2*(n-1)
            nquad = nterms
            x, w = poly.gauss_quadrature_rule(nquad)

            # Evaluate all polynomials at quadrature points
            # Shape: (nquad, nterms)
            vals = poly(x)

            # Gramian: G_ij = sum_k w_k * p_i(x_k) * p_j(x_k)
            # = vals.T @ diag(w) @ vals
            gramian = vals.T @ (w * vals)

            expected = self._bkd.eye(nterms)
            self._bkd.assert_allclose(gramian, expected, rtol=1e-12, atol=1e-14)

    def test_jacobi_gramian_is_identity(self) -> None:
        """Gramian of Jacobi polynomials should be identity."""
        params = [(0.0, 0.0), (1.0, 2.0), (-0.5, -0.5), (0.5, 0.5)]

        for alpha, beta in params:
            for nterms in [3, 5, 7]:
                poly = JacobiPolynomial1D(alpha, beta, self._bkd)
                poly.set_nterms(nterms)

                nquad = nterms
                x, w = poly.gauss_quadrature_rule(nquad)
                vals = poly(x)

                gramian = vals.T @ (w * vals)
                expected = self._bkd.eye(nterms)

                self._bkd.assert_allclose(
                    gramian,
                    expected,
                    rtol=1e-11,
                    atol=1e-13,
                )

    def test_hermite_gramian_is_identity(self) -> None:
        """Gramian of Hermite polynomials should be identity."""
        for nterms in [3, 5, 8]:
            poly = HermitePolynomial1D(self._bkd, rho=0.0, prob_meas=True)
            poly.set_nterms(nterms)

            nquad = nterms
            x, w = poly.gauss_quadrature_rule(nquad)
            vals = poly(x)

            gramian = vals.T @ (w * vals)
            expected = self._bkd.eye(nterms)

            self._bkd.assert_allclose(gramian, expected, rtol=1e-12, atol=1e-14)

    def test_laguerre_gramian_is_identity(self) -> None:
        """Gramian of Laguerre polynomials should be identity."""
        for nterms in [3, 5, 8]:
            poly = LaguerrePolynomial1D(self._bkd, rho=0.0)
            poly.set_nterms(nterms)

            nquad = nterms
            x, w = poly.gauss_quadrature_rule(nquad)
            vals = poly(x)

            gramian = vals.T @ (w * vals)
            expected = self._bkd.eye(nterms)

            self._bkd.assert_allclose(gramian, expected, rtol=1e-11, atol=1e-13)

    def test_laguerre_general_rho_gramian_is_identity(self) -> None:
        """Gramian of Laguerre polynomials with general rho should be identity."""
        for rho in [0.5, 1.0, 2.0]:
            for nterms in [3, 5, 7]:
                poly = LaguerrePolynomial1D(self._bkd, rho=rho)
                poly.set_nterms(nterms)

                nquad = nterms
                x, w = poly.gauss_quadrature_rule(nquad)
                vals = poly(x)

                gramian = vals.T @ (w * vals)
                expected = self._bkd.eye(nterms)

                self._bkd.assert_allclose(
                    gramian,
                    expected,
                    rtol=1e-10,
                    atol=1e-12,
                )

    # =========================================================================
    # Chebyshev Polynomial Tests
    # =========================================================================

    def test_chebyshev_1st_kind_gramian_diagonal(self) -> None:
        """Chebyshev 1st kind uses classical normalization.

        The __call__ method applies scaling vals[:, 1:] /= 2**0.5 and
        quadrature returns weights * pi. Expected Gramian:
        - G[0,0] = pi (first polynomial is 1)
        - G[i,i] = pi/2 for i > 0 (due to 2^0.5 scaling)
        """
        for nterms in [3, 5, 7]:
            poly = Chebyshev1stKindPolynomial1D(self._bkd)
            poly.set_nterms(nterms)

            nquad = nterms
            x, w = poly.gauss_quadrature_rule(nquad)
            vals = poly(x)

            gramian = vals.T @ (w * vals)

            # Check diagonals match expected classical normalization
            # G[0,0] = pi, G[i,i] = pi/2 for i > 0
            expected_diag = self._bkd.asarray([math.pi] + [math.pi / 2] * (nterms - 1))
            actual_diag = self._bkd.asarray([gramian[i, i] for i in range(nterms)])
            self._bkd.assert_allclose(
                actual_diag, expected_diag, rtol=1e-10, atol=1e-12
            )

            # Off-diagonals should be ~0
            for i in range(nterms):
                for j in range(i + 1, nterms):
                    self._bkd.assert_allclose(
                        self._bkd.asarray([gramian[i, j]]),
                        self._bkd.asarray([0.0]),
                        atol=1e-10,
                    )

    def test_chebyshev_2nd_kind_gramian_diagonal(self) -> None:
        """Chebyshev 2nd kind uses classical normalization.

        quadrature returns weights * pi/2. Expected Gramian:
        - G[i,i] = pi/2 for all i
        """
        for nterms in [3, 5, 7]:
            poly = Chebyshev2ndKindPolynomial1D(self._bkd)
            poly.set_nterms(nterms)

            nquad = nterms
            x, w = poly.gauss_quadrature_rule(nquad)
            vals = poly(x)

            gramian = vals.T @ (w * vals)

            # All diagonals should be pi/2
            expected_diag = self._bkd.asarray([math.pi / 2] * nterms)
            actual_diag = self._bkd.asarray([gramian[i, i] for i in range(nterms)])
            self._bkd.assert_allclose(
                actual_diag, expected_diag, rtol=1e-10, atol=1e-12
            )

            # Off-diagonals should be ~0
            for i in range(nterms):
                for j in range(i + 1, nterms):
                    self._bkd.assert_allclose(
                        self._bkd.asarray([gramian[i, j]]),
                        self._bkd.asarray([0.0]),
                        atol=1e-10,
                    )

    # =========================================================================
    # Discrete Polynomial Tests (Krawtchouk, Charlier, Hahn, DiscreteChebyshev)
    # =========================================================================

    def _binomial_pmf(self, k: int, n: int, p: float) -> float:
        """Compute binomial probability mass function."""
        from scipy.special import comb

        return comb(n, k, exact=True) * (p**k) * ((1 - p) ** (n - k))

    def _poisson_pmf(self, k: int, mu: float) -> float:
        """Compute Poisson probability mass function."""
        return (mu**k) * np.exp(-mu) / math.factorial(k)

    def test_krawtchouk_gramian_is_identity(self) -> None:
        """Krawtchouk polynomials orthonormal w.r.t. binomial distribution."""
        n_trials = 10
        p = 0.4

        for nterms in [3, 5, 7]:
            poly = KrawtchoukPolynomial1D(self._bkd, n_trials, p)
            poly.set_nterms(nterms)

            # Discrete quadrature: sample at k = 0, 1, ..., n_trials
            k_vals = self._bkd.asarray([float(k) for k in range(n_trials + 1)])
            weights = self._bkd.asarray(
                [self._binomial_pmf(k, n_trials, p) for k in range(n_trials + 1)]
            )

            # Evaluate polynomials at discrete points
            samples = self._bkd.reshape(k_vals, (1, -1))
            vals = poly(samples)

            # Compute Gramian: G_ij = sum_k w_k * p_i(k) * p_j(k)
            gramian = vals.T @ (self._bkd.reshape(weights, (-1, 1)) * vals)
            expected = self._bkd.eye(nterms)

            self._bkd.assert_allclose(gramian, expected, rtol=1e-10, atol=1e-12)

    def test_charlier_gramian_is_identity(self) -> None:
        """Charlier polynomials orthonormal w.r.t. Poisson distribution."""
        mu = 3.0

        for nterms in [3, 5, 7]:
            poly = CharlierPolynomial1D(self._bkd, mu)
            poly.set_nterms(nterms)

            # Discrete quadrature: sample at k = 0, 1, ..., K (truncated)
            # Need enough terms to capture the distribution
            K = max(30, int(mu + 6 * np.sqrt(mu)))
            k_vals = self._bkd.asarray([float(k) for k in range(K + 1)])
            weights = self._bkd.asarray(
                [self._poisson_pmf(k, mu) for k in range(K + 1)]
            )

            # Evaluate polynomials at discrete points
            samples = self._bkd.reshape(k_vals, (1, -1))
            vals = poly(samples)

            # Compute Gramian
            gramian = vals.T @ (self._bkd.reshape(weights, (-1, 1)) * vals)
            expected = self._bkd.eye(nterms)

            # Higher nterms has more numerical error
            self._bkd.assert_allclose(gramian, expected, rtol=1e-8, atol=1e-9)

    def test_discrete_chebyshev_gramian_is_identity(self) -> None:
        """DiscreteChebyshev polynomials orthonormal w.r.t. uniform discrete."""
        N = 15

        for nterms in [3, 5, 7]:
            poly = DiscreteChebyshevPolynomial1D(self._bkd, N)
            poly.set_nterms(nterms)

            # Discrete quadrature: uniform on {0, 1, ..., N-1}
            k_vals = self._bkd.asarray([float(k) for k in range(N)])
            weights = self._bkd.ones((N,)) / N

            # Evaluate polynomials at discrete points
            samples = self._bkd.reshape(k_vals, (1, -1))
            vals = poly(samples)

            # Compute Gramian
            gramian = vals.T @ (self._bkd.reshape(weights, (-1, 1)) * vals)
            expected = self._bkd.eye(nterms)

            self._bkd.assert_allclose(gramian, expected, rtol=1e-10, atol=1e-12)

    def test_hahn_gramian_is_identity(self) -> None:
        """Hahn polynomials orthonormal w.r.t. hypergeometric distribution.

        Hahn polynomials use the weight function:
            w(x) = C(n, x) * C(M-n, N-x) / C(M, N)
        where alpha = -n+1 and beta = -M-1+n.

        For this test, we use the Gauss quadrature from the polynomial class
        which should give correct weights for the orthogonality measure.
        """
        N = 10
        alpha = 1.0
        beta = 2.0

        for nterms in [3, 5]:
            poly = HahnPolynomial1D(self._bkd, N, alpha, beta)
            poly.set_nterms(nterms)

            # Use Gauss quadrature from the polynomial class
            x, w = poly.gauss_quadrature_rule(nterms)
            vals = poly(x)

            gramian = vals.T @ (w * vals)
            expected = self._bkd.eye(nterms)

            self._bkd.assert_allclose(gramian, expected, rtol=1e-10, atol=1e-12)

    # =========================================================================
    # Numeric/Lanczos Polynomial Tests
    # =========================================================================

    def test_lanczos_uniform_gramian_is_identity(self) -> None:
        """Lanczos polynomials with uniform weights are orthonormal."""
        # Test with uniform discrete measure on specific points
        samples = self._bkd.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        weights = self._bkd.ones((8,)) / 8

        for nterms in [3, 5, 7]:
            poly = DiscreteNumericOrthonormalPolynomial1D(self._bkd, samples, weights)
            poly.set_nterms(nterms)

            # Evaluate at the sample points
            samples_2d = self._bkd.reshape(samples, (1, -1))
            vals = poly(samples_2d)

            # Compute Gramian with discrete weights
            gramian = vals.T @ (self._bkd.reshape(weights, (-1, 1)) * vals)
            expected = self._bkd.eye(nterms)

            self._bkd.assert_allclose(gramian, expected, rtol=1e-10, atol=1e-12)

    def test_lanczos_nonuniform_gramian_is_identity(self) -> None:
        """Lanczos polynomials with non-uniform weights are orthonormal."""
        # Test with non-uniform weights
        samples = self._bkd.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        weights_raw = self._bkd.asarray([0.1, 0.2, 0.3, 0.2, 0.15, 0.05])
        weights = weights_raw / self._bkd.sum(weights_raw)

        for nterms in [3, 4, 5]:
            poly = DiscreteNumericOrthonormalPolynomial1D(self._bkd, samples, weights)
            poly.set_nterms(nterms)

            # Evaluate at the sample points
            samples_2d = self._bkd.reshape(samples, (1, -1))
            vals = poly(samples_2d)

            # Compute Gramian with discrete weights
            gramian = vals.T @ (self._bkd.reshape(weights, (-1, 1)) * vals)
            expected = self._bkd.eye(nterms)

            self._bkd.assert_allclose(gramian, expected, rtol=1e-10, atol=1e-12)

    def test_lanczos_matches_discrete_chebyshev(self) -> None:
        """Lanczos on uniform discrete should match DiscreteChebyshev."""
        N = 8
        samples = self._bkd.asarray([float(k) for k in range(N)])
        weights = self._bkd.ones((N,)) / N

        nterms = 5

        # Lanczos polynomial
        lanczos_poly = DiscreteNumericOrthonormalPolynomial1D(
            self._bkd, samples, weights
        )
        lanczos_poly.set_nterms(nterms)

        # DiscreteChebyshev polynomial
        dc_poly = DiscreteChebyshevPolynomial1D(self._bkd, N)
        dc_poly.set_nterms(nterms)

        # Evaluate both at sample points
        samples_2d = self._bkd.reshape(samples, (1, -1))
        lanczos_vals = lanczos_poly(samples_2d)
        dc_vals = dc_poly(samples_2d)

        # Both should give orthonormal polynomials (may differ by sign)
        # Check that both Gramians are identity
        gramian_lanczos = lanczos_vals.T @ (
            self._bkd.reshape(weights, (-1, 1)) * lanczos_vals
        )
        gramian_dc = dc_vals.T @ (self._bkd.reshape(weights, (-1, 1)) * dc_vals)

        expected = self._bkd.eye(nterms)
        self._bkd.assert_allclose(gramian_lanczos, expected, rtol=1e-10, atol=1e-12)
        self._bkd.assert_allclose(gramian_dc, expected, rtol=1e-10, atol=1e-12)

    # =========================================================================
    # Combined Tests for Different Probability Settings
    # =========================================================================

    def test_legendre_orthonormality_both_prob_settings(self) -> None:
        """Legendre orthonormality should hold for both probability settings."""
        nterms = 5

        for prob_meas in [True, False]:
            poly = LegendrePolynomial1D(self._bkd)
            poly._prob_meas = prob_meas
            poly.set_nterms(nterms)

            x, w = poly.gauss_quadrature_rule(nterms)
            vals = poly(x)
            gramian = vals.T @ (w * vals)

            expected = self._bkd.eye(nterms)
            self._bkd.assert_allclose(gramian, expected, rtol=1e-11, atol=1e-13)

    def test_hermite_orthonormality_both_prob_settings(self) -> None:
        """Hermite orthonormality should hold for both probability settings."""
        nterms = 5

        for prob_meas in [True, False]:
            poly = HermitePolynomial1D(self._bkd, rho=0.0, prob_meas=prob_meas)
            poly.set_nterms(nterms)

            x, w = poly.gauss_quadrature_rule(nterms)
            vals = poly(x)
            gramian = vals.T @ (w * vals)

            expected = self._bkd.eye(nterms)
            self._bkd.assert_allclose(gramian, expected, rtol=1e-11, atol=1e-13)

    # =========================================================================
    # 1-Point Gauss Quadrature Tests
    # =========================================================================
    # These tests verify that recursion coefficients for n=1 are correct by
    # checking that 1-point Gauss quadrature produces the correct mean.
    # 1-point Gauss quadrature: x = a[0], w = b[0]^2 (or b[0] for prob measure)
    # It should exactly compute E[1] = 1 and E[x] = mean of distribution.

    def test_legendre_one_point_quadrature(self) -> None:
        """1-point Legendre quadrature: x=0, w=1 (mean of uniform[-1,1])."""
        poly = LegendrePolynomial1D(self._bkd)
        poly.set_nterms(1)
        x, w = poly.gauss_quadrature_rule(1)

        # For uniform on [-1, 1]: mean = 0
        self._bkd.assert_allclose(x, self._bkd.asarray([[0.0]]), atol=1e-14)
        self._bkd.assert_allclose(w, self._bkd.asarray([[1.0]]), atol=1e-14)

        # E[1] = 1
        self._bkd.assert_allclose(
            self._bkd.sum(w) * self._bkd.asarray([1.0]),
            self._bkd.asarray([1.0]),
            atol=1e-14,
        )

    def test_hermite_one_point_quadrature(self) -> None:
        """1-point Hermite quadrature: x=0, w=1 (mean of standard normal)."""
        poly = HermitePolynomial1D(self._bkd, rho=0.0, prob_meas=True)
        poly.set_nterms(1)
        x, w = poly.gauss_quadrature_rule(1)

        # For standard normal: mean = 0
        self._bkd.assert_allclose(x, self._bkd.asarray([[0.0]]), atol=1e-14)
        self._bkd.assert_allclose(w, self._bkd.asarray([[1.0]]), atol=1e-14)

    def test_laguerre_one_point_quadrature(self) -> None:
        """1-point Laguerre quadrature: x=1, w=1 (mean of Gamma(1,1))."""
        poly = LaguerrePolynomial1D(self._bkd, rho=0.0)
        poly.set_nterms(1)
        x, w = poly.gauss_quadrature_rule(1)

        # For exponential (Gamma(1,1)): mean = 1
        self._bkd.assert_allclose(x, self._bkd.asarray([[1.0]]), atol=1e-14)
        self._bkd.assert_allclose(w, self._bkd.asarray([[1.0]]), atol=1e-14)

        # E[x] = 1
        integral_x = self._bkd.sum(w[:, 0] * x[0, :])
        self._bkd.assert_allclose(
            self._bkd.asarray([integral_x]),
            self._bkd.asarray([1.0]),
            atol=1e-14,
        )

    def test_krawtchouk_one_point_quadrature(self) -> None:
        """1-point Krawtchouk quadrature: x=n*p (mean of Binomial(n,p))."""
        n_trials, p = 10, 0.5
        poly = KrawtchoukPolynomial1D(self._bkd, n_trials=n_trials, p=p)
        poly.set_nterms(1)
        x, w = poly.gauss_quadrature_rule(1)

        # For Binomial(10, 0.5): mean = 5
        self._bkd.assert_allclose(x, self._bkd.asarray([[n_trials * p]]), atol=1e-12)
        self._bkd.assert_allclose(w, self._bkd.asarray([[1.0]]), atol=1e-14)

    def test_charlier_one_point_quadrature(self) -> None:
        """1-point Charlier quadrature: x=mu (mean of Poisson(mu))."""
        mu = 3.0
        poly = CharlierPolynomial1D(self._bkd, mu=mu)
        poly.set_nterms(1)
        x, w = poly.gauss_quadrature_rule(1)

        # For Poisson(mu): mean = mu
        self._bkd.assert_allclose(x, self._bkd.asarray([[mu]]), atol=1e-14)
        self._bkd.assert_allclose(w, self._bkd.asarray([[1.0]]), atol=1e-14)

    def test_discrete_chebyshev_one_point_quadrature(self) -> None:
        """1-point discrete Chebyshev: x=(N-1)/2 (mean of uniform{0,...,N-1})."""
        N = 5
        poly = DiscreteChebyshevPolynomial1D(self._bkd, N=N)
        poly.set_nterms(1)
        x, w = poly.gauss_quadrature_rule(1)

        # For uniform on {0, 1, ..., N-1}: mean = (N-1)/2
        expected_mean = (N - 1) / 2.0
        self._bkd.assert_allclose(x, self._bkd.asarray([[expected_mean]]), atol=1e-12)
        self._bkd.assert_allclose(w, self._bkd.asarray([[1.0]]), atol=1e-14)

    def test_hahn_one_point_quadrature(self) -> None:
        """1-point Hahn quadrature gives mean of hypergeometric distribution."""
        N, alpha, beta = 10, 1.0, 1.0
        poly = HahnPolynomial1D(self._bkd, N=N, alpha=alpha, beta=beta)
        poly.set_nterms(1)
        x, w = poly.gauss_quadrature_rule(1)

        # Weight should be 1 for probability measure
        self._bkd.assert_allclose(w, self._bkd.asarray([[1.0]]), atol=1e-14)

        # E[1] should be 1
        integral_1 = self._bkd.sum(w[:, 0] * 1.0)
        self._bkd.assert_allclose(
            self._bkd.asarray([integral_1]),
            self._bkd.asarray([1.0]),
            atol=1e-14,
        )

    def test_jacobi_general_one_point_quadrature(self) -> None:
        """1-point Jacobi quadrature: x = (beta-alpha)/(alpha+beta+2)."""
        alpha, beta = 1.0, 2.0
        poly = JacobiPolynomial1D(alpha, beta, self._bkd)
        poly.set_nterms(1)
        x, w = poly.gauss_quadrature_rule(1)

        # a[0] = (beta - alpha) / (alpha + beta + 2)
        expected_x = (beta - alpha) / (alpha + beta + 2.0)
        self._bkd.assert_allclose(x, self._bkd.asarray([[expected_x]]), atol=1e-14)
        self._bkd.assert_allclose(w, self._bkd.asarray([[1.0]]), atol=1e-14)


class TestQuadratureOrthonormalityNumpy(TestQuadratureOrthonormalityBase[NDArray[Any]]):
    """NumPy backend tests for quadrature and orthonormality."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestQuadratureOrthonormalityTorch(TestQuadratureOrthonormalityBase[torch.Tensor]):
    """PyTorch backend tests for quadrature and orthonormality."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
