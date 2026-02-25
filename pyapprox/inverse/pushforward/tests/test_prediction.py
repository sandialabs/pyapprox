"""
Tests for Gaussian prediction (posterior pushforward).
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.inverse.conjugate.gaussian import DenseGaussianConjugatePosterior
from pyapprox.inverse.pushforward.gaussian import GaussianPushforward
from pyapprox.inverse.pushforward.prediction import DenseGaussianPrediction
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd


class TestDenseGaussianPredictionBase(Generic[Array], unittest.TestCase):
    """Base test class for DenseGaussianPrediction."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        """Override in derived classes."""
        raise NotImplementedError

    def setUp(self) -> None:
        # Problem dimensions
        self.nvars = 3
        self.nobs = 2
        self.npred = 2

        # Observation model matrix
        self.obs_matrix = self.bkd().asarray([[1.0, 0.5, 0.0], [0.0, 0.5, 1.0]])

        # Prediction model matrix
        self.pred_matrix = self.bkd().asarray([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])

        # Prior
        self.prior_mean = self.bkd().zeros((self.nvars, 1))
        self.prior_cov = self.bkd().eye(self.nvars)

        # Noise
        self.noise_var = 0.1
        self.noise_cov = self.noise_var * self.bkd().eye(self.nobs)

        # Create predictor
        self.predictor = DenseGaussianPrediction(
            self.obs_matrix,
            self.pred_matrix,
            self.prior_mean,
            self.prior_cov,
            self.noise_cov,
            self.bkd(),
        )

        # Observations
        self.obs = self.bkd().asarray([[1.0], [1.5]])

    def test_nvars(self) -> None:
        """Test nvars returns correct value."""
        self.assertEqual(self.predictor.nvars(), self.nvars)

    def test_nobs(self) -> None:
        """Test nobs returns correct value."""
        self.assertEqual(self.predictor.nobs(), self.nobs)

    def test_npred(self) -> None:
        """Test npred returns correct value."""
        self.assertEqual(self.predictor.npred(), self.npred)

    def test_mean_shape(self) -> None:
        """Test mean has correct shape after compute."""
        self.predictor.compute(self.obs)
        mean = self.predictor.mean()
        self.assertEqual(mean.shape, (self.npred, 1))

    def test_covariance_shape(self) -> None:
        """Test covariance has correct shape after compute."""
        self.predictor.compute(self.obs)
        cov = self.predictor.covariance()
        self.assertEqual(cov.shape, (self.npred, self.npred))

    def test_covariance_symmetric(self) -> None:
        """Test covariance is symmetric."""
        self.predictor.compute(self.obs)
        cov = self.predictor.covariance()
        cov_np = self.bkd().to_numpy(cov)
        np.testing.assert_array_almost_equal(cov_np, cov_np.T, decimal=5)

    def test_compute_not_called_raises(self) -> None:
        """Test accessing results before compute raises error."""
        with self.assertRaises(RuntimeError):
            self.predictor.mean()

    def test_wrong_obs_shape_raises(self) -> None:
        """Test wrong observation shape raises error."""
        bad_obs = self.bkd().zeros((5, 1))  # Wrong nobs
        with self.assertRaises(ValueError):
            self.predictor.compute(bad_obs)

    def test_pushforward_variable_returns_gaussian(self) -> None:
        """Test pushforward_variable returns a Gaussian distribution."""
        self.predictor.compute(self.obs)
        pf_var = self.predictor.pushforward_variable()
        self.assertTrue(hasattr(pf_var, "logpdf"))
        self.assertTrue(hasattr(pf_var, "rvs"))


class TestDenseGaussianPredictionVsNaive(Generic[Array], unittest.TestCase):
    """Test prediction against naive two-step computation."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_matches_two_step_computation(self) -> None:
        """
        Test that DenseGaussianPrediction matches:
        1. Compute posterior using DenseGaussianConjugatePosterior
        2. Push posterior through prediction model using GaussianPushforward
        """
        nvars = 2
        nobs = 2

        # Observation model
        obs_matrix = self.bkd().asarray([[1.0, 0.5], [0.5, 1.0]])

        # Prediction model
        pred_matrix = self.bkd().asarray([[1.0, 1.0]])

        # Prior
        prior_mean = self.bkd().zeros((nvars, 1))
        prior_cov = self.bkd().eye(nvars)

        # Noise
        noise_cov = 0.1 * self.bkd().eye(nobs)

        # Observations
        obs = self.bkd().asarray([[1.0], [1.5]])

        # Method 1: DenseGaussianPrediction (efficient)
        predictor = DenseGaussianPrediction(
            obs_matrix, pred_matrix, prior_mean, prior_cov, noise_cov, self.bkd()
        )
        predictor.compute(obs)
        pred_mean_efficient = self.bkd().to_numpy(predictor.mean())
        pred_cov_efficient = self.bkd().to_numpy(predictor.covariance())

        # Method 2: Two-step (naive)
        # Step 1: Compute posterior
        posterior_solver = DenseGaussianConjugatePosterior(
            obs_matrix, prior_mean, prior_cov, noise_cov, self.bkd()
        )
        posterior_solver.compute(obs)
        post_mean = posterior_solver.posterior_mean()
        post_cov = posterior_solver.posterior_covariance()

        # Step 2: Push posterior through prediction model
        pushforward = GaussianPushforward(pred_matrix, post_mean, post_cov, self.bkd())
        pred_mean_naive = self.bkd().to_numpy(pushforward.mean())
        pred_cov_naive = self.bkd().to_numpy(pushforward.covariance())

        # Compare results
        np.testing.assert_array_almost_equal(
            pred_mean_efficient, pred_mean_naive, decimal=4
        )
        np.testing.assert_array_almost_equal(
            pred_cov_efficient, pred_cov_naive, decimal=4
        )


class TestDenseGaussianPredictionValidation(Generic[Array], unittest.TestCase):
    """Test input validation."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_mismatched_nvars_raises(self) -> None:
        """Test mismatched nvars between obs and pred matrices raises error."""
        obs_matrix = self.bkd().eye(2)  # nvars=2
        pred_matrix = self.bkd().asarray([[1.0, 1.0, 1.0]])  # nvars=3
        prior_mean = self.bkd().zeros((2, 1))
        prior_cov = self.bkd().eye(2)
        noise_cov = self.bkd().eye(2)

        with self.assertRaises(ValueError):
            DenseGaussianPrediction(
                obs_matrix, pred_matrix, prior_mean, prior_cov, noise_cov, self.bkd()
            )

    def test_wrong_prior_mean_shape_raises(self) -> None:
        """Test wrong prior mean shape raises error."""
        obs_matrix = self.bkd().eye(2)
        pred_matrix = self.bkd().eye(2)
        prior_mean = self.bkd().zeros((3, 1))  # Wrong shape
        prior_cov = self.bkd().eye(2)
        noise_cov = self.bkd().eye(2)

        with self.assertRaises(ValueError):
            DenseGaussianPrediction(
                obs_matrix, pred_matrix, prior_mean, prior_cov, noise_cov, self.bkd()
            )

    def test_wrong_prior_cov_shape_raises(self) -> None:
        """Test wrong prior covariance shape raises error."""
        obs_matrix = self.bkd().eye(2)
        pred_matrix = self.bkd().eye(2)
        prior_mean = self.bkd().zeros((2, 1))
        prior_cov = self.bkd().eye(3)  # Wrong shape
        noise_cov = self.bkd().eye(2)

        with self.assertRaises(ValueError):
            DenseGaussianPrediction(
                obs_matrix, pred_matrix, prior_mean, prior_cov, noise_cov, self.bkd()
            )

    def test_wrong_noise_cov_shape_raises(self) -> None:
        """Test wrong noise covariance shape raises error."""
        obs_matrix = self.bkd().eye(2)
        pred_matrix = self.bkd().eye(2)
        prior_mean = self.bkd().zeros((2, 1))
        prior_cov = self.bkd().eye(2)
        noise_cov = self.bkd().eye(3)  # Wrong shape

        with self.assertRaises(ValueError):
            DenseGaussianPrediction(
                obs_matrix, pred_matrix, prior_mean, prior_cov, noise_cov, self.bkd()
            )


# NumPy backend tests
class TestDenseGaussianPredictionNumpy(TestDenseGaussianPredictionBase[NDArray[Any]]):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestDenseGaussianPredictionVsNaiveNumpy(
    TestDenseGaussianPredictionVsNaive[NDArray[Any]]
):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestDenseGaussianPredictionValidationNumpy(
    TestDenseGaussianPredictionValidation[NDArray[Any]]
):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# PyTorch backend tests
class TestDenseGaussianPredictionTorch(TestDenseGaussianPredictionBase[torch.Tensor]):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


class TestDenseGaussianPredictionVsNaiveTorch(
    TestDenseGaussianPredictionVsNaive[torch.Tensor]
):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = TorchBkd()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


class TestDenseGaussianPredictionValidationTorch(
    TestDenseGaussianPredictionValidation[torch.Tensor]
):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = TorchBkd()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


if __name__ == "__main__":
    unittest.main()
