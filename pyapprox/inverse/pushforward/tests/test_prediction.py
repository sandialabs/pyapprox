"""
Tests for Gaussian prediction (posterior pushforward).
"""

import pytest

import numpy as np

from pyapprox.inverse.conjugate.gaussian import DenseGaussianConjugatePosterior
from pyapprox.inverse.pushforward.gaussian import GaussianPushforward
from pyapprox.inverse.pushforward.prediction import DenseGaussianPrediction


class TestDenseGaussianPredictionBase:
    """Base test class for DenseGaussianPrediction."""

    def _make_predictor(self, bkd):
        """Create predictor for tests."""
        nvars = 3
        nobs = 2
        npred = 2

        # Observation model matrix
        obs_matrix = bkd.asarray([[1.0, 0.5, 0.0], [0.0, 0.5, 1.0]])

        # Prediction model matrix
        pred_matrix = bkd.asarray([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])

        # Prior
        prior_mean = bkd.zeros((nvars, 1))
        prior_cov = bkd.eye(nvars)

        # Noise
        noise_var = 0.1
        noise_cov = noise_var * bkd.eye(nobs)

        # Create predictor
        predictor = DenseGaussianPrediction(
            obs_matrix,
            pred_matrix,
            prior_mean,
            prior_cov,
            noise_cov,
            bkd,
        )

        # Observations
        obs = bkd.asarray([[1.0], [1.5]])

        return predictor, obs, nvars, nobs, npred

    def test_nvars(self, bkd) -> None:
        """Test nvars returns correct value."""
        predictor, _, nvars, _, _ = self._make_predictor(bkd)
        assert predictor.nvars() == nvars

    def test_nobs(self, bkd) -> None:
        """Test nobs returns correct value."""
        predictor, _, _, nobs, _ = self._make_predictor(bkd)
        assert predictor.nobs() == nobs

    def test_npred(self, bkd) -> None:
        """Test npred returns correct value."""
        predictor, _, _, _, npred = self._make_predictor(bkd)
        assert predictor.npred() == npred

    def test_mean_shape(self, bkd) -> None:
        """Test mean has correct shape after compute."""
        predictor, obs, _, _, npred = self._make_predictor(bkd)
        predictor.compute(obs)
        mean = predictor.mean()
        assert mean.shape == (npred, 1)

    def test_covariance_shape(self, bkd) -> None:
        """Test covariance has correct shape after compute."""
        predictor, obs, _, _, npred = self._make_predictor(bkd)
        predictor.compute(obs)
        cov = predictor.covariance()
        assert cov.shape == (npred, npred)

    def test_covariance_symmetric(self, bkd) -> None:
        """Test covariance is symmetric."""
        predictor, obs, _, _, _ = self._make_predictor(bkd)
        predictor.compute(obs)
        cov = predictor.covariance()
        cov_np = bkd.to_numpy(cov)
        np.testing.assert_array_almost_equal(cov_np, cov_np.T, decimal=5)

    def test_compute_not_called_raises(self, bkd) -> None:
        """Test accessing results before compute raises error."""
        predictor, _, _, _, _ = self._make_predictor(bkd)
        with pytest.raises(RuntimeError):
            predictor.mean()

    def test_wrong_obs_shape_raises(self, bkd) -> None:
        """Test wrong observation shape raises error."""
        predictor, _, _, _, _ = self._make_predictor(bkd)
        bad_obs = bkd.zeros((5, 1))  # Wrong nobs
        with pytest.raises(ValueError):
            predictor.compute(bad_obs)

    def test_pushforward_variable_returns_gaussian(self, bkd) -> None:
        """Test pushforward_variable returns a Gaussian distribution."""
        predictor, obs, _, _, _ = self._make_predictor(bkd)
        predictor.compute(obs)
        pf_var = predictor.pushforward_variable()
        assert hasattr(pf_var, "logpdf")
        assert hasattr(pf_var, "rvs")


class TestDenseGaussianPredictionVsNaive:
    """Test prediction against naive two-step computation."""

    def test_matches_two_step_computation(self, bkd) -> None:
        """
        Test that DenseGaussianPrediction matches:
        1. Compute posterior using DenseGaussianConjugatePosterior
        2. Push posterior through prediction model using GaussianPushforward
        """
        nvars = 2
        nobs = 2

        # Observation model
        obs_matrix = bkd.asarray([[1.0, 0.5], [0.5, 1.0]])

        # Prediction model
        pred_matrix = bkd.asarray([[1.0, 1.0]])

        # Prior
        prior_mean = bkd.zeros((nvars, 1))
        prior_cov = bkd.eye(nvars)

        # Noise
        noise_cov = 0.1 * bkd.eye(nobs)

        # Observations
        obs = bkd.asarray([[1.0], [1.5]])

        # Method 1: DenseGaussianPrediction (efficient)
        predictor = DenseGaussianPrediction(
            obs_matrix, pred_matrix, prior_mean, prior_cov, noise_cov, bkd
        )
        predictor.compute(obs)
        pred_mean_efficient = bkd.to_numpy(predictor.mean())
        pred_cov_efficient = bkd.to_numpy(predictor.covariance())

        # Method 2: Two-step (naive)
        # Step 1: Compute posterior
        posterior_solver = DenseGaussianConjugatePosterior(
            obs_matrix, prior_mean, prior_cov, noise_cov, bkd
        )
        posterior_solver.compute(obs)
        post_mean = posterior_solver.posterior_mean()
        post_cov = posterior_solver.posterior_covariance()

        # Step 2: Push posterior through prediction model
        pushforward = GaussianPushforward(pred_matrix, post_mean, post_cov, bkd)
        pred_mean_naive = bkd.to_numpy(pushforward.mean())
        pred_cov_naive = bkd.to_numpy(pushforward.covariance())

        # Compare results
        np.testing.assert_array_almost_equal(
            pred_mean_efficient, pred_mean_naive, decimal=4
        )
        np.testing.assert_array_almost_equal(
            pred_cov_efficient, pred_cov_naive, decimal=4
        )


class TestDenseGaussianPredictionValidation:
    """Test input validation."""

    def test_mismatched_nvars_raises(self, bkd) -> None:
        """Test mismatched nvars between obs and pred matrices raises error."""
        obs_matrix = bkd.eye(2)  # nvars=2
        pred_matrix = bkd.asarray([[1.0, 1.0, 1.0]])  # nvars=3
        prior_mean = bkd.zeros((2, 1))
        prior_cov = bkd.eye(2)
        noise_cov = bkd.eye(2)

        with pytest.raises(ValueError):
            DenseGaussianPrediction(
                obs_matrix, pred_matrix, prior_mean, prior_cov, noise_cov, bkd
            )

    def test_wrong_prior_mean_shape_raises(self, bkd) -> None:
        """Test wrong prior mean shape raises error."""
        obs_matrix = bkd.eye(2)
        pred_matrix = bkd.eye(2)
        prior_mean = bkd.zeros((3, 1))  # Wrong shape
        prior_cov = bkd.eye(2)
        noise_cov = bkd.eye(2)

        with pytest.raises(ValueError):
            DenseGaussianPrediction(
                obs_matrix, pred_matrix, prior_mean, prior_cov, noise_cov, bkd
            )

    def test_wrong_prior_cov_shape_raises(self, bkd) -> None:
        """Test wrong prior covariance shape raises error."""
        obs_matrix = bkd.eye(2)
        pred_matrix = bkd.eye(2)
        prior_mean = bkd.zeros((2, 1))
        prior_cov = bkd.eye(3)  # Wrong shape
        noise_cov = bkd.eye(2)

        with pytest.raises(ValueError):
            DenseGaussianPrediction(
                obs_matrix, pred_matrix, prior_mean, prior_cov, noise_cov, bkd
            )

    def test_wrong_noise_cov_shape_raises(self, bkd) -> None:
        """Test wrong noise covariance shape raises error."""
        obs_matrix = bkd.eye(2)
        pred_matrix = bkd.eye(2)
        prior_mean = bkd.zeros((2, 1))
        prior_cov = bkd.eye(2)
        noise_cov = bkd.eye(3)  # Wrong shape

        with pytest.raises(ValueError):
            DenseGaussianPrediction(
                obs_matrix, pred_matrix, prior_mean, prior_cov, noise_cov, bkd
            )
