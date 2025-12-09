"""
Tests for univariate distributions.
"""

import unittest
import numpy as np
from scipy import stats

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.probability.univariate import (
    GaussianMarginal,
    ScipyContinuousMarginal,
    ScipyDiscreteMarginal,
)


class TestGaussianMarginal(unittest.TestCase):
    """Tests for GaussianMarginal."""

    def setUp(self):
        self.bkd = NumpyBkd()
        self.mean = 2.0
        self.stdev = 0.5
        self.dist = GaussianMarginal(self.mean, self.stdev, self.bkd)

    def test_nvars(self):
        """Test nvars returns 1."""
        self.assertEqual(self.dist.nvars(), 1)

    def test_pdf_at_mean(self):
        """Test PDF is maximized at mean."""
        samples = np.array([self.mean - 1, self.mean, self.mean + 1])
        pdf_vals = self.dist(samples)
        self.assertEqual(np.argmax(pdf_vals), 1)

    def test_cdf_at_mean(self):
        """Test CDF at mean is 0.5."""
        cdf_val = self.dist.cdf(np.array([self.mean]))[0]
        self.assertAlmostEqual(cdf_val, 0.5)

    def test_cdf_monotonic(self):
        """Test CDF is monotonically increasing."""
        samples = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        cdf_vals = self.dist.cdf(samples)
        self.assertTrue(np.all(np.diff(cdf_vals) >= 0))

    def test_invcdf_cdf_inverse(self):
        """Test invcdf(cdf(x)) = x."""
        samples = np.array([1.0, 2.0, 3.0])
        recovered = self.dist.invcdf(self.dist.cdf(samples))
        np.testing.assert_array_almost_equal(recovered, samples)

    def test_cdf_invcdf_inverse(self):
        """Test cdf(invcdf(p)) = p."""
        probs = np.array([0.1, 0.5, 0.9])
        recovered = self.dist.cdf(self.dist.invcdf(probs))
        np.testing.assert_array_almost_equal(recovered, probs)

    def test_rvs_shape(self):
        """Test rvs returns correct shape."""
        samples = self.dist.rvs(100)
        self.assertEqual(samples.shape, (1, 100))

    def test_logpdf_exp_equals_pdf(self):
        """Test exp(logpdf) = pdf."""
        samples = np.array([1.0, 2.0, 3.0])
        pdf_vals = self.dist(samples)
        logpdf_vals = self.dist.logpdf(samples)
        np.testing.assert_array_almost_equal(np.exp(logpdf_vals), pdf_vals)

    def test_mean_variance(self):
        """Test mean and variance."""
        self.assertAlmostEqual(self.dist.mean_value(), self.mean)
        self.assertAlmostEqual(self.dist.variance(), self.stdev**2)
        self.assertAlmostEqual(self.dist.std(), self.stdev)

    def test_is_bounded(self):
        """Test Gaussian is unbounded."""
        self.assertFalse(self.dist.is_bounded())

    def test_interval(self):
        """Test interval contains correct probability."""
        alpha = 0.95
        interval = self.dist.interval(alpha)
        # For standard normal, 95% interval is approximately [-1.96, 1.96]
        # For N(mean, stdev), it's [mean - 1.96*stdev, mean + 1.96*stdev]
        self.assertEqual(len(interval), 2)
        self.assertLess(interval[0], self.mean)
        self.assertGreater(interval[1], self.mean)

    def test_invcdf_jacobian(self):
        """Test invcdf Jacobian = 1/pdf at quantile."""
        probs = np.array([0.25, 0.5, 0.75])
        jacobian = self.dist.invcdf_jacobian(probs)
        quantiles = self.dist.invcdf(probs)
        pdf_at_quantiles = self.dist(quantiles)
        np.testing.assert_array_almost_equal(jacobian, 1.0 / pdf_at_quantiles)

    def test_equality(self):
        """Test equality comparison."""
        dist2 = GaussianMarginal(self.mean, self.stdev, self.bkd)
        dist3 = GaussianMarginal(0.0, 1.0, self.bkd)
        self.assertEqual(self.dist, dist2)
        self.assertNotEqual(self.dist, dist3)


class TestScipyContinuousMarginal(unittest.TestCase):
    """Tests for ScipyContinuousMarginal."""

    def setUp(self):
        self.bkd = NumpyBkd()
        # Beta distribution on [0, 1]
        self.scipy_rv = stats.beta(a=2, b=5)
        self.dist = ScipyContinuousMarginal(self.scipy_rv, self.bkd)

    def test_nvars(self):
        """Test nvars returns 1."""
        self.assertEqual(self.dist.nvars(), 1)

    def test_pdf(self):
        """Test PDF matches scipy."""
        samples = np.array([0.1, 0.3, 0.5, 0.7])
        pdf_vals = self.dist(samples)
        expected = self.scipy_rv.pdf(samples)
        np.testing.assert_array_almost_equal(pdf_vals, expected)

    def test_cdf(self):
        """Test CDF matches scipy."""
        samples = np.array([0.1, 0.3, 0.5, 0.7])
        cdf_vals = self.dist.cdf(samples)
        expected = self.scipy_rv.cdf(samples)
        np.testing.assert_array_almost_equal(cdf_vals, expected)

    def test_invcdf(self):
        """Test invcdf matches scipy ppf."""
        probs = np.array([0.1, 0.5, 0.9])
        invcdf_vals = self.dist.invcdf(probs)
        expected = self.scipy_rv.ppf(probs)
        np.testing.assert_array_almost_equal(invcdf_vals, expected)

    def test_rvs_shape(self):
        """Test rvs returns correct shape."""
        samples = self.dist.rvs(100)
        self.assertEqual(samples.shape, (1, 100))

    def test_is_bounded(self):
        """Test beta is bounded."""
        self.assertTrue(self.dist.is_bounded())

    def test_unbounded_distribution(self):
        """Test unbounded distribution detection."""
        normal_rv = stats.norm(loc=0, scale=1)
        normal_dist = ScipyContinuousMarginal(normal_rv, self.bkd)
        self.assertFalse(normal_dist.is_bounded())

    def test_invalid_distribution(self):
        """Test that discrete distributions raise error."""
        with self.assertRaises(ValueError):
            discrete_rv = stats.binom(n=10, p=0.5)
            ScipyContinuousMarginal(discrete_rv, self.bkd)

    def test_distribution_info(self):
        """Test distribution info extraction."""
        self.assertEqual(self.dist.name, "beta")
        self.assertIn("a", self.dist.shapes)
        self.assertIn("b", self.dist.shapes)


class TestScipyDiscreteMarginal(unittest.TestCase):
    """Tests for ScipyDiscreteMarginal."""

    def setUp(self):
        self.bkd = NumpyBkd()
        # Binomial distribution
        self.scipy_rv = stats.binom(n=10, p=0.3)
        self.dist = ScipyDiscreteMarginal(self.scipy_rv, self.bkd)

    def test_nvars(self):
        """Test nvars returns 1."""
        self.assertEqual(self.dist.nvars(), 1)

    def test_pmf(self):
        """Test PMF matches scipy."""
        samples = np.array([0, 3, 5, 10])
        pmf_vals = self.dist(samples)
        expected = self.scipy_rv.pmf(samples)
        np.testing.assert_array_almost_equal(pmf_vals, expected)

    def test_cdf(self):
        """Test CDF matches scipy."""
        samples = np.array([0, 3, 5, 10])
        cdf_vals = self.dist.cdf(samples)
        expected = self.scipy_rv.cdf(samples)
        np.testing.assert_array_almost_equal(cdf_vals, expected)

    def test_invcdf(self):
        """Test invcdf matches scipy ppf."""
        probs = np.array([0.1, 0.5, 0.9])
        invcdf_vals = self.dist.invcdf(probs)
        expected = self.scipy_rv.ppf(probs)
        np.testing.assert_array_almost_equal(invcdf_vals, expected)

    def test_rvs_shape(self):
        """Test rvs returns correct shape."""
        samples = self.dist.rvs(100)
        self.assertEqual(samples.shape, (1, 100))

    def test_is_bounded(self):
        """Test binomial is bounded."""
        self.assertTrue(self.dist.is_bounded())

    def test_unbounded_distribution(self):
        """Test unbounded distribution detection."""
        poisson_rv = stats.poisson(mu=5)
        poisson_dist = ScipyDiscreteMarginal(poisson_rv, self.bkd)
        self.assertFalse(poisson_dist.is_bounded())

    def test_invalid_distribution(self):
        """Test that continuous distributions raise error."""
        with self.assertRaises(ValueError):
            continuous_rv = stats.norm(loc=0, scale=1)
            ScipyDiscreteMarginal(continuous_rv, self.bkd)


class TestProtocolCompliance(unittest.TestCase):
    """Test that distributions satisfy protocol interfaces."""

    def test_gaussian_marginal_protocol(self):
        """Test GaussianMarginal has required methods."""
        bkd = NumpyBkd()
        dist = GaussianMarginal(0, 1, bkd)

        # MarginalProtocol methods
        self.assertTrue(callable(getattr(dist, "bkd", None)))
        self.assertTrue(callable(getattr(dist, "nvars", None)))
        self.assertTrue(callable(getattr(dist, "rvs", None)))
        self.assertTrue(callable(getattr(dist, "logpdf", None)))
        self.assertTrue(callable(getattr(dist, "cdf", None)))
        self.assertTrue(callable(getattr(dist, "invcdf", None)))

        # MarginalWithJacobianProtocol methods
        self.assertTrue(callable(getattr(dist, "invcdf_jacobian", None)))

    def test_scipy_continuous_protocol(self):
        """Test ScipyContinuousMarginal has required methods."""
        bkd = NumpyBkd()
        dist = ScipyContinuousMarginal(stats.norm(0, 1), bkd)

        self.assertTrue(callable(getattr(dist, "bkd", None)))
        self.assertTrue(callable(getattr(dist, "nvars", None)))
        self.assertTrue(callable(getattr(dist, "rvs", None)))
        self.assertTrue(callable(getattr(dist, "logpdf", None)))
        self.assertTrue(callable(getattr(dist, "cdf", None)))
        self.assertTrue(callable(getattr(dist, "invcdf", None)))

    def test_scipy_discrete_protocol(self):
        """Test ScipyDiscreteMarginal has required methods."""
        bkd = NumpyBkd()
        dist = ScipyDiscreteMarginal(stats.binom(10, 0.5), bkd)

        self.assertTrue(callable(getattr(dist, "bkd", None)))
        self.assertTrue(callable(getattr(dist, "nvars", None)))
        self.assertTrue(callable(getattr(dist, "rvs", None)))
        self.assertTrue(callable(getattr(dist, "logpdf", None)))
        self.assertTrue(callable(getattr(dist, "cdf", None)))
        self.assertTrue(callable(getattr(dist, "invcdf", None)))


if __name__ == "__main__":
    unittest.main()
