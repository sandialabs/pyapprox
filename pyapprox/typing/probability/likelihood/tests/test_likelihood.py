"""
Tests for likelihood functions.
"""

import unittest
import numpy as np
from scipy import stats

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.probability.covariance import (
    DenseCholeskyCovarianceOperator,
    DiagonalCovarianceOperator,
)
from pyapprox.typing.probability.likelihood import (
    GaussianLogLikelihood,
    DiagonalGaussianLogLikelihood,
)


class TestGaussianLogLikelihood(unittest.TestCase):
    """Tests for GaussianLogLikelihood."""

    def setUp(self):
        self.bkd = NumpyBkd()
        self.nobs = 3
        self.noise_cov = 0.01 * np.eye(self.nobs)
        self.noise_op = DenseCholeskyCovarianceOperator(self.noise_cov, self.bkd)
        self.likelihood = GaussianLogLikelihood(self.noise_op, self.bkd)

    def test_nobs(self):
        """Test nobs returns correct dimension."""
        self.assertEqual(self.likelihood.nobs(), 3)

    def test_noise_covariance_operator(self):
        """Test noise covariance operator accessor."""
        self.assertIsNotNone(self.likelihood.noise_covariance_operator())

    def test_set_observations(self):
        """Test setting observations."""
        obs = np.array([[1.0], [2.0], [3.0]])
        self.likelihood.set_observations(obs)
        # Should not raise

    def test_logpdf_at_observations(self):
        """Test logpdf at exact observations is maximum."""
        obs = np.array([[1.0], [2.0], [3.0]])
        self.likelihood.set_observations(obs)

        # At observations, logpdf should be maximum
        logpdf_at_obs = self.likelihood.logpdf(obs)

        # Perturbed
        model_perturbed = obs + 0.1
        logpdf_perturbed = self.likelihood.logpdf(model_perturbed)

        self.assertGreater(float(logpdf_at_obs), float(logpdf_perturbed))

    def test_logpdf_vs_scipy(self):
        """Test logpdf matches scipy multivariate normal."""
        obs = np.array([1.0, 2.0, 3.0])
        self.likelihood.set_observations(obs)

        model = np.array([1.01, 1.99, 3.02])

        # Our likelihood
        logpdf_ours = self.likelihood.logpdf(model)

        # Scipy (note: scipy uses obs as the random variable, model as mean)
        scipy_dist = stats.multivariate_normal(model, self.noise_cov)
        logpdf_scipy = scipy_dist.logpdf(obs)

        np.testing.assert_array_almost_equal(logpdf_ours, [logpdf_scipy])

    def test_logpdf_batch(self):
        """Test logpdf with multiple model samples."""
        obs = np.array([[1.0], [2.0], [3.0]])
        self.likelihood.set_observations(obs)

        model = np.array([[1.0, 1.1, 0.9], [2.0, 2.1, 1.9], [3.0, 3.1, 2.9]])
        logpdf = self.likelihood.logpdf(model)

        self.assertEqual(logpdf.shape, (3,))

    def test_rvs_shape(self):
        """Test rvs returns correct shape."""
        model = np.array([[1.0], [2.0], [3.0]])
        samples = self.likelihood.rvs(model, nsamples=10)
        self.assertEqual(samples.shape, (3, 10))

    def test_rvs_mean(self):
        """Test rvs has correct mean."""
        np.random.seed(42)
        model = np.array([[1.0], [2.0], [3.0]])
        samples = self.likelihood.rvs(model, nsamples=10000)

        mean = np.mean(samples, axis=1)
        np.testing.assert_array_almost_equal(mean, model.flatten(), decimal=1)

    def test_gradient_shape(self):
        """Test gradient has correct shape."""
        obs = np.array([[1.0], [2.0], [3.0]])
        self.likelihood.set_observations(obs)

        model = np.array([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        grad = self.likelihood.gradient(model)

        self.assertEqual(grad.shape, (3, 2))

    def test_gradient_at_observations_zero(self):
        """Test gradient at observations is zero."""
        obs = np.array([[1.0], [2.0], [3.0]])
        self.likelihood.set_observations(obs)

        grad = self.likelihood.gradient(obs)
        np.testing.assert_array_almost_equal(grad, np.zeros((3, 1)))

    def test_observations_not_set_raises(self):
        """Test logpdf without observations raises error."""
        model = np.array([[1.0], [2.0], [3.0]])
        with self.assertRaises(ValueError):
            self.likelihood.logpdf(model)


class TestGaussianLogLikelihoodCorrelated(unittest.TestCase):
    """Tests for GaussianLogLikelihood with correlated noise."""

    def setUp(self):
        self.bkd = NumpyBkd()
        self.nobs = 2
        self.noise_cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        self.noise_op = DenseCholeskyCovarianceOperator(self.noise_cov, self.bkd)
        self.likelihood = GaussianLogLikelihood(self.noise_op, self.bkd)

    def test_logpdf_vs_scipy(self):
        """Test logpdf matches scipy for correlated noise."""
        obs = np.array([1.0, 2.0])
        self.likelihood.set_observations(obs)

        model = np.array([1.1, 1.9])

        logpdf_ours = self.likelihood.logpdf(model)

        scipy_dist = stats.multivariate_normal(model, self.noise_cov)
        logpdf_scipy = scipy_dist.logpdf(obs)

        np.testing.assert_array_almost_equal(logpdf_ours, [logpdf_scipy])


class TestDiagonalGaussianLogLikelihood(unittest.TestCase):
    """Tests for DiagonalGaussianLogLikelihood."""

    def setUp(self):
        self.bkd = NumpyBkd()
        self.noise_var = np.array([0.01, 0.02, 0.01])
        self.likelihood = DiagonalGaussianLogLikelihood(self.noise_var, self.bkd)

    def test_nobs(self):
        """Test nobs returns correct dimension."""
        self.assertEqual(self.likelihood.nobs(), 3)

    def test_logpdf_at_observations(self):
        """Test logpdf at exact observations is maximum."""
        obs = np.array([[1.0], [2.0], [3.0]])
        self.likelihood.set_observations(obs)

        logpdf_at_obs = self.likelihood.logpdf(obs)
        model_perturbed = obs + 0.1
        logpdf_perturbed = self.likelihood.logpdf(model_perturbed)

        self.assertGreater(float(logpdf_at_obs), float(logpdf_perturbed))

    def test_logpdf_vs_full_gaussian(self):
        """Test diagonal matches full Gaussian with diagonal cov."""
        obs = np.array([[1.0], [2.0], [3.0]])
        self.likelihood.set_observations(obs)

        model = np.array([[1.01], [1.99], [3.02]])

        # Full Gaussian likelihood
        noise_cov = np.diag(self.noise_var)
        noise_op = DenseCholeskyCovarianceOperator(noise_cov, self.bkd)
        full_likelihood = GaussianLogLikelihood(noise_op, self.bkd)
        full_likelihood.set_observations(obs)

        logpdf_diag = self.likelihood.logpdf(model)
        logpdf_full = full_likelihood.logpdf(model)

        np.testing.assert_array_almost_equal(logpdf_diag, logpdf_full)

    def test_rvs_shape(self):
        """Test rvs returns correct shape."""
        model = np.array([[1.0], [2.0], [3.0]])
        samples = self.likelihood.rvs(model, nsamples=10)
        self.assertEqual(samples.shape, (3, 10))

    def test_gradient_shape(self):
        """Test gradient has correct shape."""
        obs = np.array([[1.0], [2.0], [3.0]])
        self.likelihood.set_observations(obs)

        model = np.array([[1.0], [2.0], [3.0]])
        grad = self.likelihood.gradient(model)

        self.assertEqual(grad.shape, (3, 1))


class TestDesignWeights(unittest.TestCase):
    """Tests for design weights functionality."""

    def setUp(self):
        self.bkd = NumpyBkd()
        self.noise_var = np.array([0.01, 0.01, 0.01])
        self.likelihood = DiagonalGaussianLogLikelihood(self.noise_var, self.bkd)

    def test_set_design_weights(self):
        """Test setting design weights."""
        weights = np.array([1.0, 0.5, 0.25])
        self.likelihood.set_design_weights(weights)
        # Should not raise

    def test_weights_affect_logpdf(self):
        """Test design weights affect logpdf."""
        obs = np.array([[1.0], [2.0], [3.0]])
        model = np.array([[1.1], [2.1], [3.1]])  # Same error per obs

        self.likelihood.set_observations(obs)

        logpdf_no_weights = self.likelihood.logpdf(model)

        # Zero out third observation
        weights = np.array([1.0, 1.0, 0.0])
        self.likelihood.set_design_weights(weights)
        logpdf_with_weights = self.likelihood.logpdf(model)

        # With zero weight on third obs, logpdf should be higher
        self.assertGreater(float(logpdf_with_weights), float(logpdf_no_weights))


class TestVectorizedLogLikelihood(unittest.TestCase):
    """Tests for vectorized log-likelihood evaluation."""

    def setUp(self):
        self.bkd = NumpyBkd()
        self.noise_cov = 0.01 * np.eye(2)
        self.noise_op = DenseCholeskyCovarianceOperator(self.noise_cov, self.bkd)
        self.likelihood = GaussianLogLikelihood(self.noise_op, self.bkd)

    def test_vectorized_shape(self):
        """Test vectorized logpdf returns correct shape."""
        model = np.array([[1.0, 1.1, 1.2], [2.0, 2.1, 2.2]])
        obs = np.array([[1.0, 1.5], [2.0, 2.5]])

        logpdf = self.likelihood.logpdf_vectorized(model, obs)
        self.assertEqual(logpdf.shape, (3, 2))

    def test_vectorized_matches_loop(self):
        """Test vectorized matches sequential evaluation."""
        model = np.array([[1.0, 1.1], [2.0, 2.1]])
        obs = np.array([[1.0, 1.2], [2.0, 2.2]])

        logpdf_vec = self.likelihood.logpdf_vectorized(model, obs)

        # Sequential
        for j in range(obs.shape[1]):
            self.likelihood.set_observations(obs[:, j:j+1])
            logpdf_seq = self.likelihood.logpdf(model)
            np.testing.assert_array_almost_equal(
                logpdf_vec[:, j], logpdf_seq
            )


if __name__ == "__main__":
    unittest.main()
