"""
Tests for joint probability distributions.
"""

import unittest
import numpy as np
from scipy import stats

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.probability.univariate import (
    ScipyContinuousMarginal,
    GaussianMarginal,
)
from pyapprox.typing.probability.joint import IndependentJoint


class TestIndependentJoint(unittest.TestCase):
    """Tests for IndependentJoint."""

    def setUp(self):
        self.bkd = NumpyBkd()
        # Create marginals: standard normal, beta, uniform
        self.marginals = [
            ScipyContinuousMarginal(stats.norm(0, 1), self.bkd),
            ScipyContinuousMarginal(stats.beta(2, 5), self.bkd),
            ScipyContinuousMarginal(stats.uniform(0, 1), self.bkd),
        ]
        self.joint = IndependentJoint(self.marginals, self.bkd)

    def test_nvars(self):
        """Test nvars returns correct dimension."""
        self.assertEqual(self.joint.nvars(), 3)

    def test_marginals(self):
        """Test marginals returns list of marginals."""
        marginals = self.joint.marginals()
        self.assertEqual(len(marginals), 3)

    def test_marginal(self):
        """Test accessing individual marginal."""
        marginal = self.joint.marginal(1)
        self.assertEqual(marginal.name, "beta")

    def test_rvs_shape(self):
        """Test rvs returns correct shape."""
        samples = self.joint.rvs(100)
        self.assertEqual(samples.shape, (3, 100))

    def test_logpdf_sum_of_marginals(self):
        """Test logpdf equals sum of marginal logpdfs."""
        samples = np.array([[0.0, 0.5, -1.0], [0.3, 0.5, 0.2], [0.5, 0.2, 0.8]])

        logpdf_joint = self.joint.logpdf(samples)

        # Compute manually as sum
        logpdf_expected = np.zeros(3)
        for i, marginal in enumerate(self.marginals):
            logpdf_expected += marginal.logpdf(samples[i])

        np.testing.assert_array_almost_equal(logpdf_joint, logpdf_expected)

    def test_pdf_exp_logpdf(self):
        """Test pdf = exp(logpdf)."""
        samples = np.array([[0.0, 0.5], [0.3, 0.5], [0.5, 0.2]])
        pdf_vals = self.joint.pdf(samples)
        logpdf_vals = self.joint.logpdf(samples)
        np.testing.assert_array_almost_equal(pdf_vals, np.exp(logpdf_vals))

    def test_cdf_product_of_marginals(self):
        """Test cdf equals product of marginal cdfs."""
        samples = np.array([[0.0, 0.5], [0.3, 0.5], [0.5, 0.2]])

        cdf_joint = self.joint.cdf(samples)

        # Compute manually as product
        cdf_expected = np.ones(2)
        for i, marginal in enumerate(self.marginals):
            cdf_expected *= marginal.cdf(samples[i])

        np.testing.assert_array_almost_equal(cdf_joint, cdf_expected)

    def test_invcdf_component_wise(self):
        """Test invcdf applies to each component."""
        probs = np.array([[0.5, 0.25], [0.5, 0.75], [0.5, 0.1]])

        quantiles = self.joint.invcdf(probs)

        # Verify each component
        for i, marginal in enumerate(self.marginals):
            expected = marginal.invcdf(probs[i])
            np.testing.assert_array_almost_equal(quantiles[i], expected)

    def test_correlation_matrix_identity(self):
        """Test correlation matrix is identity for independent marginals."""
        corr = self.joint.correlation_matrix()
        expected = np.eye(3)
        np.testing.assert_array_almost_equal(corr, expected)

    def test_covariance_diagonal(self):
        """Test covariance is diagonal for independent marginals."""
        cov = self.joint.covariance()
        # Check diagonal
        diag = np.diag(cov)
        self.assertTrue(np.all(diag > 0))
        # Check off-diagonal is zero
        off_diag = cov - np.diag(diag)
        np.testing.assert_array_almost_equal(off_diag, np.zeros((3, 3)))

    def test_mean(self):
        """Test mean computation."""
        mean = self.joint.mean()
        self.assertEqual(mean.shape, (3,))
        # Check first marginal (standard normal)
        self.assertAlmostEqual(float(mean[0]), 0.0, places=1)

    def test_variance(self):
        """Test variance computation."""
        var = self.joint.variance()
        self.assertEqual(var.shape, (3,))
        # Check first marginal (standard normal, var=1)
        self.assertAlmostEqual(float(var[0]), 1.0, places=1)

    def test_bounds(self):
        """Test bounds computation."""
        bounds = self.joint.bounds()
        self.assertEqual(bounds.shape, (2, 3))
        # Uniform has bounds [0, 1]
        self.assertAlmostEqual(float(bounds[0, 2]), 0.0)
        self.assertAlmostEqual(float(bounds[1, 2]), 1.0)

    def test_empty_marginals_raises(self):
        """Test empty marginals raises error."""
        with self.assertRaises(ValueError):
            IndependentJoint([], self.bkd)


class TestIndependentJointGaussian(unittest.TestCase):
    """Tests for IndependentJoint with Gaussian marginals."""

    def setUp(self):
        self.bkd = NumpyBkd()
        self.means = [0.0, 1.0, 2.0]
        self.stds = [1.0, 2.0, 0.5]
        self.marginals = [
            GaussianMarginal(m, s, self.bkd)
            for m, s in zip(self.means, self.stds)
        ]
        self.joint = IndependentJoint(self.marginals, self.bkd)

    def test_nvars(self):
        """Test nvars returns correct dimension."""
        self.assertEqual(self.joint.nvars(), 3)

    def test_logpdf_vs_multivariate_normal(self):
        """Test logpdf matches multivariate normal with diagonal covariance."""
        from scipy.stats import multivariate_normal

        cov = np.diag([s**2 for s in self.stds])
        scipy_dist = multivariate_normal(self.means, cov)

        samples = np.array([[0.0, 0.5, -1.0], [1.0, 0.5, 2.0], [2.0, 2.5, 1.5]])

        logpdf_ours = self.joint.logpdf(samples)
        logpdf_scipy = scipy_dist.logpdf(samples.T)

        np.testing.assert_array_almost_equal(logpdf_ours, logpdf_scipy)

    def test_mean_matches_marginal_means(self):
        """Test mean matches marginal means."""
        mean = self.joint.mean()
        np.testing.assert_array_almost_equal(mean, self.means)

    def test_variance_matches_marginal_variances(self):
        """Test variance matches marginal variances."""
        var = self.joint.variance()
        expected = np.array([s**2 for s in self.stds])
        np.testing.assert_array_almost_equal(var, expected)


class TestIndependentJointSingleVariable(unittest.TestCase):
    """Tests for IndependentJoint with single variable."""

    def setUp(self):
        self.bkd = NumpyBkd()
        self.marginal = ScipyContinuousMarginal(stats.norm(0, 1), self.bkd)
        self.joint = IndependentJoint([self.marginal], self.bkd)

    def test_nvars(self):
        """Test nvars returns 1."""
        self.assertEqual(self.joint.nvars(), 1)

    def test_rvs_shape(self):
        """Test rvs returns correct shape."""
        samples = self.joint.rvs(50)
        self.assertEqual(samples.shape, (1, 50))

    def test_logpdf_matches_marginal(self):
        """Test logpdf matches marginal logpdf."""
        samples = np.array([[0.0, 1.0, -1.0]])

        logpdf_joint = self.joint.logpdf(samples)
        logpdf_marginal = self.marginal.logpdf(samples[0])

        np.testing.assert_array_almost_equal(logpdf_joint, logpdf_marginal)


class TestIndependentJointProtocol(unittest.TestCase):
    """Tests for JointDistributionProtocol compliance."""

    def setUp(self):
        self.bkd = NumpyBkd()
        self.marginals = [
            ScipyContinuousMarginal(stats.norm(0, 1), self.bkd),
            ScipyContinuousMarginal(stats.uniform(0, 1), self.bkd),
        ]
        self.joint = IndependentJoint(self.marginals, self.bkd)

    def test_has_bkd(self):
        """Test has bkd method."""
        self.assertIsNotNone(self.joint.bkd())

    def test_has_nvars(self):
        """Test has nvars method."""
        self.assertEqual(self.joint.nvars(), 2)

    def test_has_rvs(self):
        """Test has rvs method."""
        samples = self.joint.rvs(10)
        self.assertEqual(samples.shape, (2, 10))

    def test_has_logpdf(self):
        """Test has logpdf method."""
        samples = np.array([[0.0, 0.5], [0.5, 0.8]])
        logpdf = self.joint.logpdf(samples)
        self.assertEqual(logpdf.shape, (2,))

    def test_has_marginals(self):
        """Test has marginals method."""
        marginals = self.joint.marginals()
        self.assertEqual(len(marginals), 2)

    def test_has_correlation_matrix(self):
        """Test has correlation_matrix method."""
        corr = self.joint.correlation_matrix()
        self.assertEqual(corr.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
