"""Tests for extended orthonormal polynomials (Laguerre, discrete, numeric)."""

import pytest

import numpy as np

from pyapprox.surrogates.affine.univariate.globalpoly import (
    CharlierPolynomial1D,
    DiscreteChebyshevPolynomial1D,
    DiscreteNumericOrthonormalPolynomial1D,
    HahnPolynomial1D,
    KrawtchoukPolynomial1D,
    LaguerrePolynomial1D,
    LegendrePolynomial1D,
    WeightedSamplePolynomial1D,
    hahn_recurrence,
    laguerre_recurrence,
    lanczos_recursion,
)
from pyapprox.util.backends.numpy import NumpyBkd


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(42)


class TestLaguerrePolynomial:
    """Tests for Laguerre polynomials."""

    def test_evaluation_shape(self, numpy_bkd):
        """Test that polynomial evaluation has correct shape."""
        bkd = numpy_bkd
        poly = LaguerrePolynomial1D(bkd, rho=0.0)
        poly.set_nterms(5)
        samples = bkd.reshape(bkd.linspace(0.0, 10.0, 20), (1, -1))
        values = poly(samples)
        assert values.shape == (20, 5)

    def test_recursion_coefficients_shape(self, numpy_bkd):
        """Test recursion coefficients shape."""
        bkd = numpy_bkd
        ab = laguerre_recurrence(10, rho=0.5, bkd=bkd)
        assert ab.shape == (10, 2)

    def test_orthonormality(self, numpy_bkd):
        """Test orthonormality with respect to gamma weight."""
        bkd = numpy_bkd
        rho = 0.5
        nterms = 5
        poly = LaguerrePolynomial1D(bkd, rho=rho)
        poly.set_nterms(20)  # Need more terms for quadrature
        x, w = poly.gauss_quadrature_rule(20)

        # Reset to 5 terms for test
        poly.set_nterms(nterms)

        # Evaluate at quadrature points
        samples = bkd.reshape(x, (1, -1))
        values = poly(samples)  # (npoints, nterms)

        # Compute inner product matrix
        gram = bkd.dot(values.T * w.T, values)

        # Should be identity
        eye = bkd.eye(nterms)
        bkd.assert_allclose(gram, eye, atol=1e-10)

    def test_invalid_rho(self, numpy_bkd):
        """Test that invalid rho raises error."""
        bkd = numpy_bkd
        with pytest.raises(ValueError):
            LaguerrePolynomial1D(bkd, rho=-1.5)


class TestKrawtchoukPolynomial:
    """Tests for Krawtchouk polynomials (Binomial)."""

    def test_evaluation_shape(self, numpy_bkd):
        """Test polynomial evaluation shape."""
        bkd = numpy_bkd
        poly = KrawtchoukPolynomial1D(bkd, n_trials=10, p=0.3)
        poly.set_nterms(5)
        samples = bkd.reshape(bkd.asarray([0, 2, 5, 7, 10]), (1, -1))
        values = poly(samples)
        assert values.shape == (5, 5)

    def test_orthonormality_binomial(self, numpy_bkd):
        """Test orthonormality with binomial weights."""
        from scipy.stats import binom

        bkd = numpy_bkd
        n_trials = 10
        p = 0.4
        poly = KrawtchoukPolynomial1D(bkd, n_trials=n_trials, p=p)
        poly.set_nterms(5)

        # Discrete samples and binomial weights
        samples = bkd.asarray(np.arange(n_trials + 1), dtype=float)
        weights = bkd.asarray(binom.pmf(np.arange(n_trials + 1), n_trials, p))

        # Evaluate
        samples_2d = bkd.reshape(samples, (1, -1))
        values = poly(samples_2d)

        # Compute inner product
        gram = bkd.dot(values.T * weights, values)
        eye = bkd.eye(5)
        bkd.assert_allclose(gram, eye, atol=1e-10)

    def test_invalid_parameters(self, numpy_bkd):
        """Test invalid parameter validation."""
        bkd = numpy_bkd
        with pytest.raises(ValueError):
            KrawtchoukPolynomial1D(bkd, n_trials=10, p=1.5)
        with pytest.raises(ValueError):
            KrawtchoukPolynomial1D(bkd, n_trials=0, p=0.5)


class TestHahnPolynomial:
    """Tests for Hahn polynomials (Hypergeometric)."""

    def test_evaluation_shape(self, numpy_bkd):
        """Test polynomial evaluation shape."""
        bkd = numpy_bkd
        poly = HahnPolynomial1D(bkd, N=10, alpha=1.0, beta=2.0)
        poly.set_nterms(5)
        samples = bkd.reshape(bkd.asarray([0, 2, 5, 7, 10]), (1, -1))
        values = poly(samples)
        assert values.shape == (5, 5)

    def test_recursion_shape(self, numpy_bkd):
        """Test recursion coefficients shape."""
        bkd = numpy_bkd
        ab = hahn_recurrence(5, N=10, alpha=1.0, beta=2.0, bkd=bkd)
        assert ab.shape == (5, 2)


class TestCharlierPolynomial:
    """Tests for Charlier polynomials (Poisson)."""

    def test_evaluation_shape(self, numpy_bkd):
        """Test polynomial evaluation shape."""
        bkd = numpy_bkd
        poly = CharlierPolynomial1D(bkd, mu=3.0)
        poly.set_nterms(5)
        samples = bkd.reshape(bkd.asarray([0, 1, 2, 3, 4, 5]), (1, -1))
        values = poly(samples)
        assert values.shape == (6, 5)

    def test_orthonormality_poisson(self, numpy_bkd):
        """Test orthonormality with Poisson weights."""
        from scipy.stats import poisson

        bkd = numpy_bkd
        mu = 3.0
        poly = CharlierPolynomial1D(bkd, mu=mu)
        poly.set_nterms(5)

        # Discrete samples and Poisson weights (truncated)
        # Using more samples for better convergence with infinite support
        max_k = 30
        samples = bkd.asarray(np.arange(max_k + 1), dtype=float)
        weights = bkd.asarray(poisson.pmf(np.arange(max_k + 1), mu))

        # Evaluate
        samples_2d = bkd.reshape(samples, (1, -1))
        values = poly(samples_2d)

        # Compute inner product
        gram = bkd.dot(values.T * weights, values)
        eye = bkd.eye(5)
        # Tolerance is looser due to truncation of infinite support
        bkd.assert_allclose(gram, eye, atol=1e-6)

    def test_invalid_mu(self, numpy_bkd):
        """Test invalid mu raises error."""
        bkd = numpy_bkd
        with pytest.raises(ValueError):
            CharlierPolynomial1D(bkd, mu=-1.0)


class TestDiscreteChebyshevPolynomial:
    """Tests for Discrete Chebyshev polynomials."""

    def test_evaluation_shape(self, numpy_bkd):
        """Test polynomial evaluation shape."""
        bkd = numpy_bkd
        N = 10
        poly = DiscreteChebyshevPolynomial1D(bkd, N=N)
        poly.set_nterms(5)
        samples = bkd.reshape(bkd.asarray([0, 2, 5, 7, 9]), (1, -1))
        values = poly(samples)
        assert values.shape == (5, 5)

    def test_orthonormality_uniform_discrete(self, numpy_bkd):
        """Test orthonormality with uniform discrete weights."""
        bkd = numpy_bkd
        N = 10
        poly = DiscreteChebyshevPolynomial1D(bkd, N=N)
        poly.set_nterms(5)

        # Discrete samples and uniform weights
        samples = bkd.asarray(np.arange(N), dtype=float)
        weights = bkd.ones((N,)) / N

        # Evaluate
        samples_2d = bkd.reshape(samples, (1, -1))
        values = poly(samples_2d)

        # Compute inner product
        gram = bkd.dot(values.T * weights, values)
        eye = bkd.eye(5)
        bkd.assert_allclose(gram, eye, atol=1e-10)

    def test_invalid_N(self, numpy_bkd):
        """Test invalid N raises error."""
        bkd = numpy_bkd
        with pytest.raises(ValueError):
            DiscreteChebyshevPolynomial1D(bkd, N=0)


class TestDiscreteNumericPolynomial:
    """Tests for DiscreteNumericOrthonormalPolynomial1D."""

    def test_uniform_samples(self, numpy_bkd):
        """Test with uniform weighted samples."""
        bkd = numpy_bkd
        samples = bkd.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
        weights = bkd.ones((5,)) / 5
        poly = DiscreteNumericOrthonormalPolynomial1D(bkd, samples, weights)
        poly.set_nterms(5)

        # Check orthonormality
        assert poly.check_orthonormality(tol=1e-10)

    def test_matches_discrete_chebyshev(self, numpy_bkd):
        """Test that numeric matches DiscreteChebyshev for uniform."""
        bkd = numpy_bkd
        N = 10
        samples = bkd.asarray(np.arange(N), dtype=float)
        weights = bkd.ones((N,)) / N

        numeric_poly = DiscreteNumericOrthonormalPolynomial1D(
            bkd, samples, weights
        )
        numeric_poly.set_nterms(5)
        chebyshev_poly = DiscreteChebyshevPolynomial1D(bkd, N=N)
        chebyshev_poly.set_nterms(5)

        # Evaluate both at sample points
        samples_2d = bkd.reshape(samples, (1, -1))
        numeric_vals = numeric_poly(samples_2d)
        chebyshev_vals = chebyshev_poly(samples_2d)

        # Values should match (up to sign)
        for j in range(5):
            # Check if columns match or are negatives
            diff1 = bkd.norm(numeric_vals[:, j] - chebyshev_vals[:, j])
            diff2 = bkd.norm(numeric_vals[:, j] + chebyshev_vals[:, j])
            min_diff = min(float(diff1), float(diff2))
            assert min_diff < 1e-10

    def test_weighted_sample_polynomial(self, numpy_bkd):
        """Test WeightedSamplePolynomial1D convenience class."""
        bkd = numpy_bkd
        samples = bkd.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
        poly = WeightedSamplePolynomial1D(bkd, samples)
        poly.set_nterms(5)
        assert poly.check_orthonormality(tol=1e-10)


class TestLanczosRecursion:
    """Tests for Lanczos recursion algorithm."""

    def test_matches_legendre(self, numpy_bkd):
        """Test that Lanczos matches Legendre for uniform continuous."""
        bkd = numpy_bkd
        # Use many uniform samples to approximate continuous uniform
        n = 100
        samples = bkd.linspace(-1, 1, n)
        weights = bkd.ones((n,)) / n

        # Get Lanczos recursion coefficients
        ab_lanczos = lanczos_recursion(samples, weights, 5, bkd)

        # Get Legendre recursion coefficients
        legendre = LegendrePolynomial1D(bkd)
        legendre.set_nterms(5)
        ab_legendre = legendre._get_recursion_coefficients(5)

        # a coefficients should be close to 0 (symmetric)
        bkd.assert_allclose(ab_lanczos[:, 0], ab_legendre[:, 0], atol=0.1)

    def test_too_many_terms_raises(self, numpy_bkd):
        """Test that requesting too many terms raises error."""
        bkd = numpy_bkd
        samples = bkd.asarray([0.0, 1.0, 2.0])
        weights = bkd.ones((3,)) / 3

        with pytest.raises(ValueError):
            lanczos_recursion(samples, weights, 10, bkd)
