"""Tests for extended orthonormal polynomials (Laguerre, discrete, numeric)."""

import unittest

import numpy as np

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.surrogates.affine.univariate.globalpoly import (
    LaguerrePolynomial1D,
    laguerre_recurrence,
    KrawtchoukPolynomial1D,
    krawtchouk_recurrence,
    HahnPolynomial1D,
    hahn_recurrence,
    CharlierPolynomial1D,
    charlier_recurrence,
    DiscreteChebyshevPolynomial1D,
    discrete_chebyshev_recurrence,
    DiscreteNumericOrthonormalPolynomial1D,
    WeightedSamplePolynomial1D,
    lanczos_recursion,
    LegendrePolynomial1D,
)


class OrthopolyTestBase:
    """Base class for orthonormal polynomial tests."""

    __test__ = False

    def setUp(self):
        np.random.seed(42)
        self.bkd = NumpyBkd()


class TestLaguerrePolynomial(OrthopolyTestBase, unittest.TestCase):
    """Tests for Laguerre polynomials."""

    __test__ = True

    def test_evaluation_shape(self):
        """Test that polynomial evaluation has correct shape."""
        poly = LaguerrePolynomial1D(self.bkd, rho=0.0)
        poly.set_nterms(5)
        samples = self.bkd.reshape(self.bkd.linspace(0.0, 10.0, 20), (1, -1))
        values = poly(samples)
        self.assertEqual(values.shape, (20, 5))

    def test_recursion_coefficients_shape(self):
        """Test recursion coefficients shape."""
        ab = laguerre_recurrence(10, rho=0.5, bkd=self.bkd)
        self.assertEqual(ab.shape, (10, 2))

    def test_orthonormality(self):
        """Test orthonormality with respect to gamma weight."""
        rho = 0.5
        nterms = 5
        poly = LaguerrePolynomial1D(self.bkd, rho=rho)
        poly.set_nterms(20)  # Need more terms for quadrature
        x, w = poly.gauss_quadrature_rule(20)

        # Reset to 5 terms for test
        poly.set_nterms(nterms)

        # Evaluate at quadrature points
        samples = self.bkd.reshape(x, (1, -1))
        values = poly(samples)  # (npoints, nterms)

        # Compute inner product matrix
        gram = self.bkd.dot(values.T * w.T, values)

        # Should be identity
        eye = self.bkd.eye(nterms)
        self.bkd.assert_allclose(gram, eye, atol=1e-10)

    def test_invalid_rho(self):
        """Test that invalid rho raises error."""
        with self.assertRaises(ValueError):
            LaguerrePolynomial1D(self.bkd, rho=-1.5)


class TestKrawtchoukPolynomial(OrthopolyTestBase, unittest.TestCase):
    """Tests for Krawtchouk polynomials (Binomial)."""

    __test__ = True

    def test_evaluation_shape(self):
        """Test polynomial evaluation shape."""
        poly = KrawtchoukPolynomial1D(self.bkd, n_trials=10, p=0.3)
        poly.set_nterms(5)
        samples = self.bkd.reshape(self.bkd.asarray([0, 2, 5, 7, 10]), (1, -1))
        values = poly(samples)
        self.assertEqual(values.shape, (5, 5))

    def test_orthonormality_binomial(self):
        """Test orthonormality with binomial weights."""
        from scipy.stats import binom

        n_trials = 10
        p = 0.4
        poly = KrawtchoukPolynomial1D(self.bkd, n_trials=n_trials, p=p)
        poly.set_nterms(5)

        # Discrete samples and binomial weights
        samples = self.bkd.asarray(np.arange(n_trials + 1), dtype=float)
        weights = self.bkd.asarray(binom.pmf(np.arange(n_trials + 1), n_trials, p))

        # Evaluate
        samples_2d = self.bkd.reshape(samples, (1, -1))
        values = poly(samples_2d)

        # Compute inner product
        gram = self.bkd.dot(values.T * weights, values)
        eye = self.bkd.eye(5)
        self.bkd.assert_allclose(gram, eye, atol=1e-10)

    def test_invalid_parameters(self):
        """Test invalid parameter validation."""
        with self.assertRaises(ValueError):
            KrawtchoukPolynomial1D(self.bkd, n_trials=10, p=1.5)
        with self.assertRaises(ValueError):
            KrawtchoukPolynomial1D(self.bkd, n_trials=0, p=0.5)


class TestHahnPolynomial(OrthopolyTestBase, unittest.TestCase):
    """Tests for Hahn polynomials (Hypergeometric)."""

    __test__ = True

    def test_evaluation_shape(self):
        """Test polynomial evaluation shape."""
        poly = HahnPolynomial1D(self.bkd, N=10, alpha=1.0, beta=2.0)
        poly.set_nterms(5)
        samples = self.bkd.reshape(self.bkd.asarray([0, 2, 5, 7, 10]), (1, -1))
        values = poly(samples)
        self.assertEqual(values.shape, (5, 5))

    def test_recursion_shape(self):
        """Test recursion coefficients shape."""
        ab = hahn_recurrence(5, N=10, alpha=1.0, beta=2.0, bkd=self.bkd)
        self.assertEqual(ab.shape, (5, 2))


class TestCharlierPolynomial(OrthopolyTestBase, unittest.TestCase):
    """Tests for Charlier polynomials (Poisson)."""

    __test__ = True

    def test_evaluation_shape(self):
        """Test polynomial evaluation shape."""
        poly = CharlierPolynomial1D(self.bkd, mu=3.0)
        poly.set_nterms(5)
        samples = self.bkd.reshape(self.bkd.asarray([0, 1, 2, 3, 4, 5]), (1, -1))
        values = poly(samples)
        self.assertEqual(values.shape, (6, 5))

    def test_orthonormality_poisson(self):
        """Test orthonormality with Poisson weights."""
        from scipy.stats import poisson

        mu = 3.0
        poly = CharlierPolynomial1D(self.bkd, mu=mu)
        poly.set_nterms(5)

        # Discrete samples and Poisson weights (truncated)
        # Using more samples for better convergence with infinite support
        max_k = 30
        samples = self.bkd.asarray(np.arange(max_k + 1), dtype=float)
        weights = self.bkd.asarray(poisson.pmf(np.arange(max_k + 1), mu))

        # Evaluate
        samples_2d = self.bkd.reshape(samples, (1, -1))
        values = poly(samples_2d)

        # Compute inner product
        gram = self.bkd.dot(values.T * weights, values)
        eye = self.bkd.eye(5)
        # Tolerance is looser due to truncation of infinite support
        self.bkd.assert_allclose(gram, eye, atol=1e-6)

    def test_invalid_mu(self):
        """Test invalid mu raises error."""
        with self.assertRaises(ValueError):
            CharlierPolynomial1D(self.bkd, mu=-1.0)


class TestDiscreteChebyshevPolynomial(OrthopolyTestBase, unittest.TestCase):
    """Tests for Discrete Chebyshev polynomials."""

    __test__ = True

    def test_evaluation_shape(self):
        """Test polynomial evaluation shape."""
        N = 10
        poly = DiscreteChebyshevPolynomial1D(self.bkd, N=N)
        poly.set_nterms(5)
        samples = self.bkd.reshape(self.bkd.asarray([0, 2, 5, 7, 9]), (1, -1))
        values = poly(samples)
        self.assertEqual(values.shape, (5, 5))

    def test_orthonormality_uniform_discrete(self):
        """Test orthonormality with uniform discrete weights."""
        N = 10
        poly = DiscreteChebyshevPolynomial1D(self.bkd, N=N)
        poly.set_nterms(5)

        # Discrete samples and uniform weights
        samples = self.bkd.asarray(np.arange(N), dtype=float)
        weights = self.bkd.ones((N,)) / N

        # Evaluate
        samples_2d = self.bkd.reshape(samples, (1, -1))
        values = poly(samples_2d)

        # Compute inner product
        gram = self.bkd.dot(values.T * weights, values)
        eye = self.bkd.eye(5)
        self.bkd.assert_allclose(gram, eye, atol=1e-10)

    def test_invalid_N(self):
        """Test invalid N raises error."""
        with self.assertRaises(ValueError):
            DiscreteChebyshevPolynomial1D(self.bkd, N=0)


class TestDiscreteNumericPolynomial(OrthopolyTestBase, unittest.TestCase):
    """Tests for DiscreteNumericOrthonormalPolynomial1D."""

    __test__ = True

    def test_uniform_samples(self):
        """Test with uniform weighted samples."""
        samples = self.bkd.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
        weights = self.bkd.ones((5,)) / 5
        poly = DiscreteNumericOrthonormalPolynomial1D(self.bkd, samples, weights)
        poly.set_nterms(5)

        # Check orthonormality
        self.assertTrue(poly.check_orthonormality(tol=1e-10))

    def test_matches_discrete_chebyshev(self):
        """Test that numeric matches DiscreteChebyshev for uniform."""
        N = 10
        samples = self.bkd.asarray(np.arange(N), dtype=float)
        weights = self.bkd.ones((N,)) / N

        numeric_poly = DiscreteNumericOrthonormalPolynomial1D(
            self.bkd, samples, weights
        )
        numeric_poly.set_nterms(5)
        chebyshev_poly = DiscreteChebyshevPolynomial1D(self.bkd, N=N)
        chebyshev_poly.set_nterms(5)

        # Evaluate both at sample points
        samples_2d = self.bkd.reshape(samples, (1, -1))
        numeric_vals = numeric_poly(samples_2d)
        chebyshev_vals = chebyshev_poly(samples_2d)

        # Values should match (up to sign)
        for j in range(5):
            # Check if columns match or are negatives
            diff1 = self.bkd.norm(numeric_vals[:, j] - chebyshev_vals[:, j])
            diff2 = self.bkd.norm(numeric_vals[:, j] + chebyshev_vals[:, j])
            min_diff = min(float(diff1), float(diff2))
            self.assertLess(min_diff, 1e-10)

    def test_weighted_sample_polynomial(self):
        """Test WeightedSamplePolynomial1D convenience class."""
        samples = self.bkd.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
        poly = WeightedSamplePolynomial1D(self.bkd, samples)
        poly.set_nterms(5)
        self.assertTrue(poly.check_orthonormality(tol=1e-10))


class TestLanczosRecursion(OrthopolyTestBase, unittest.TestCase):
    """Tests for Lanczos recursion algorithm."""

    __test__ = True

    def test_matches_legendre(self):
        """Test that Lanczos matches Legendre for uniform continuous."""
        # Use many uniform samples to approximate continuous uniform
        n = 100
        samples = self.bkd.linspace(-1, 1, n)
        weights = self.bkd.ones((n,)) / n

        # Get Lanczos recursion coefficients
        ab_lanczos = lanczos_recursion(samples, weights, 5, self.bkd)

        # Get Legendre recursion coefficients
        legendre = LegendrePolynomial1D(self.bkd)
        legendre.set_nterms(5)
        ab_legendre = legendre._get_recursion_coefficients(5)

        # a coefficients should be close to 0 (symmetric)
        self.bkd.assert_allclose(ab_lanczos[:, 0], ab_legendre[:, 0], atol=0.1)

    def test_too_many_terms_raises(self):
        """Test that requesting too many terms raises error."""
        samples = self.bkd.asarray([0.0, 1.0, 2.0])
        weights = self.bkd.ones((3,)) / 3

        with self.assertRaises(ValueError):
            lanczos_recursion(samples, weights, 10, self.bkd)


if __name__ == "__main__":
    unittest.main()
