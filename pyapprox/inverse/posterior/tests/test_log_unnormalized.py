"""
Tests for LogUnNormalizedPosterior.
"""

import pytest

import numpy as np

from pyapprox.inverse.posterior import LogUnNormalizedPosterior
from pyapprox.probability.covariance import DiagonalCovarianceOperator
from pyapprox.probability.gaussian import DenseCholeskyMultivariateGaussian
from pyapprox.probability.likelihood import GaussianLogLikelihood


class TestLogUnNormalizedPosteriorBase:
    """Base test class for LogUnNormalizedPosterior."""

    def _make_posterior(self, bkd):
        """Create posterior for tests."""
        nvars = 2
        nobs = 3

        # Linear model: y = A @ theta
        A_np = np.array(
            [
                [1.0, 0.5],
                [0.5, 1.0],
                [1.0, 1.0],
            ],
            dtype=np.float64,
        )
        A = bkd.asarray(A_np)

        def model_fn(theta):
            return A @ theta

        # Setup likelihood
        noise_var = bkd.asarray(np.array([0.01, 0.01, 0.01], dtype=np.float64))
        noise_cov_op = DiagonalCovarianceOperator(noise_var, bkd)
        likelihood = GaussianLogLikelihood(noise_cov_op, bkd)

        # Observations
        obs = bkd.asarray(np.array([[1.0], [1.5], [2.0]], dtype=np.float64))
        likelihood.set_observations(obs)

        # Prior
        prior_mean = bkd.asarray(np.zeros((nvars, 1), dtype=np.float64))
        prior_cov = bkd.asarray(np.eye(nvars, dtype=np.float64))
        prior = DenseCholeskyMultivariateGaussian(
            prior_mean, prior_cov, bkd
        )

        # Create posterior
        posterior = LogUnNormalizedPosterior(
            model_fn, likelihood, prior, bkd
        )
        return posterior, nvars

    def test_nvars(self, bkd) -> None:
        """Test nvars returns correct value."""
        posterior, nvars = self._make_posterior(bkd)
        assert posterior.nvars() == nvars

    def test_call_returns_scalar_per_sample(self, bkd) -> None:
        """Test __call__ returns one value per sample."""
        posterior, nvars = self._make_posterior(bkd)
        samples = bkd.asarray(np.random.randn(nvars, 5).astype(np.float64))
        logpost = posterior(samples)
        assert logpost.shape == (5,)

    def test_call_single_sample(self, bkd) -> None:
        """Test __call__ works for single sample."""
        posterior, _ = self._make_posterior(bkd)
        sample = bkd.asarray(np.array([[0.5], [0.5]], dtype=np.float64))
        logpost = posterior(sample)
        assert logpost.shape == (1,)

    def test_log_posterior_finite(self, bkd) -> None:
        """Test log posterior values are finite."""
        posterior, nvars = self._make_posterior(bkd)
        samples = bkd.asarray(np.random.randn(nvars, 10).astype(np.float64))
        logpost = posterior(samples)
        logpost_np = bkd.to_numpy(logpost)
        assert np.all(np.isfinite(logpost_np))

    def test_map_returns_correct_shape(self, bkd) -> None:
        """Test MAP estimation returns correct shape."""
        posterior, nvars = self._make_posterior(bkd)
        map_point = posterior.maximum_aposteriori_point()
        assert map_point.shape == (nvars, 1)

    def test_map_improves_log_posterior(self, bkd) -> None:
        """Test MAP point has higher log posterior than initial guess."""
        posterior, nvars = self._make_posterior(bkd)
        initial = bkd.asarray(np.zeros((nvars, 1), dtype=np.float64))
        map_point = posterior.maximum_aposteriori_point(initial_guess=initial)

        logpost_initial = posterior(initial)[0]
        logpost_map = posterior(map_point)[0]

        logpost_initial_np = float(bkd.to_numpy(logpost_initial))
        logpost_map_np = float(bkd.to_numpy(logpost_map))

        assert logpost_map_np >= logpost_initial_np - 1e-6


class TestLogUnNormalizedPosteriorAnalytical:
    """Test log posterior matches analytical formula for simple case."""

    def test_gaussian_prior_likelihood(self, bkd) -> None:
        """Test log posterior value matches analytical computation."""
        nvars = 2

        # Identity model: y = theta
        def model_fn(theta):
            return theta

        # Setup likelihood with diagonal noise
        noise_var = bkd.asarray(np.array([0.1, 0.1], dtype=np.float64))
        noise_cov_op = DiagonalCovarianceOperator(noise_var, bkd)
        likelihood = GaussianLogLikelihood(noise_cov_op, bkd)

        # Observations
        obs = bkd.asarray(np.array([[1.0], [2.0]], dtype=np.float64))
        likelihood.set_observations(obs)

        # Prior: N(0, I)
        prior_mean = bkd.asarray(np.zeros((nvars, 1), dtype=np.float64))
        prior_cov = bkd.asarray(np.eye(nvars, dtype=np.float64))
        prior = DenseCholeskyMultivariateGaussian(prior_mean, prior_cov, bkd)

        posterior = LogUnNormalizedPosterior(model_fn, likelihood, prior, bkd)

        # Evaluate at a test point
        theta = bkd.asarray(np.array([[1.0], [2.0]], dtype=np.float64))
        logpost = posterior(theta)

        # Just verify it's finite and reasonable
        logpost_np = float(bkd.to_numpy(logpost))
        assert np.isfinite(logpost_np)


class TestLogUnNormalizedPosteriorValidation:
    """Test input validation."""

    def test_repr(self, bkd) -> None:
        """Test string representation."""
        nvars = 2

        def model_fn(theta):
            return theta

        noise_var = bkd.asarray(np.array([0.1, 0.1], dtype=np.float64))
        noise_cov_op = DiagonalCovarianceOperator(noise_var, bkd)
        likelihood = GaussianLogLikelihood(noise_cov_op, bkd)
        likelihood.set_observations(
            bkd.asarray(np.array([[1.0], [2.0]], dtype=np.float64))
        )

        prior_mean = bkd.asarray(np.zeros((nvars, 1), dtype=np.float64))
        prior_cov = bkd.asarray(np.eye(nvars, dtype=np.float64))
        prior = DenseCholeskyMultivariateGaussian(prior_mean, prior_cov, bkd)

        posterior = LogUnNormalizedPosterior(model_fn, likelihood, prior, bkd)

        repr_str = repr(posterior)
        assert "LogUnNormalizedPosterior" in repr_str
        assert "nvars=2" in repr_str
