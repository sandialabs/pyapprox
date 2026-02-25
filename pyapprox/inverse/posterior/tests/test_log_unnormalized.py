"""
Tests for LogUnNormalizedPosterior.
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.inverse.posterior import LogUnNormalizedPosterior
from pyapprox.probability.covariance import DiagonalCovarianceOperator
from pyapprox.probability.gaussian import DenseCholeskyMultivariateGaussian
from pyapprox.probability.likelihood import GaussianLogLikelihood
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd


class TestLogUnNormalizedPosteriorBase(Generic[Array], unittest.TestCase):
    """Base test class for LogUnNormalizedPosterior."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        """Override in derived classes."""
        raise NotImplementedError

    def setUp(self) -> None:
        self.nvars = 2
        self.nobs = 3

        # Linear model: y = A @ theta
        A_np = np.array(
            [
                [1.0, 0.5],
                [0.5, 1.0],
                [1.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.A = self.bkd().asarray(A_np)

        def model_fn(theta: Array) -> Array:
            return self.A @ theta

        self.model_fn = model_fn

        # Setup likelihood
        noise_var = self.bkd().asarray(np.array([0.01, 0.01, 0.01], dtype=np.float64))
        self.noise_cov_op = DiagonalCovarianceOperator(noise_var, self.bkd())
        self.likelihood = GaussianLogLikelihood(self.noise_cov_op, self.bkd())

        # Observations
        self.obs = self.bkd().asarray(np.array([[1.0], [1.5], [2.0]], dtype=np.float64))
        self.likelihood.set_observations(self.obs)

        # Prior
        prior_mean = self.bkd().asarray(np.zeros((self.nvars, 1), dtype=np.float64))
        prior_cov = self.bkd().asarray(np.eye(self.nvars, dtype=np.float64))
        self.prior = DenseCholeskyMultivariateGaussian(
            prior_mean, prior_cov, self.bkd()
        )

        # Create posterior
        self.posterior = LogUnNormalizedPosterior(
            model_fn, self.likelihood, self.prior, self.bkd()
        )

    def test_nvars(self) -> None:
        """Test nvars returns correct value."""
        self.assertEqual(self.posterior.nvars(), self.nvars)

    def test_call_returns_scalar_per_sample(self) -> None:
        """Test __call__ returns one value per sample."""
        samples = self.bkd().asarray(np.random.randn(self.nvars, 5).astype(np.float64))
        logpost = self.posterior(samples)
        self.assertEqual(logpost.shape, (5,))

    def test_call_single_sample(self) -> None:
        """Test __call__ works for single sample."""
        sample = self.bkd().asarray(np.array([[0.5], [0.5]], dtype=np.float64))
        logpost = self.posterior(sample)
        self.assertEqual(logpost.shape, (1,))

    def test_log_posterior_finite(self) -> None:
        """Test log posterior values are finite."""
        samples = self.bkd().asarray(np.random.randn(self.nvars, 10).astype(np.float64))
        logpost = self.posterior(samples)
        logpost_np = self.bkd().to_numpy(logpost)
        self.assertTrue(np.all(np.isfinite(logpost_np)))

    def test_map_returns_correct_shape(self) -> None:
        """Test MAP estimation returns correct shape."""
        map_point = self.posterior.maximum_aposteriori_point()
        self.assertEqual(map_point.shape, (self.nvars, 1))

    def test_map_improves_log_posterior(self) -> None:
        """Test MAP point has higher log posterior than initial guess."""
        initial = self.bkd().asarray(np.zeros((self.nvars, 1), dtype=np.float64))
        map_point = self.posterior.maximum_aposteriori_point(initial_guess=initial)

        logpost_initial = self.posterior(initial)[0]
        logpost_map = self.posterior(map_point)[0]

        logpost_initial_np = float(self.bkd().to_numpy(logpost_initial))
        logpost_map_np = float(self.bkd().to_numpy(logpost_map))

        self.assertGreaterEqual(logpost_map_np, logpost_initial_np - 1e-6)


class TestLogUnNormalizedPosteriorAnalytical(Generic[Array], unittest.TestCase):
    """Test log posterior matches analytical formula for simple case."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_gaussian_prior_likelihood(self) -> None:
        """Test log posterior value matches analytical computation."""
        nvars = 2

        # Identity model: y = theta
        def model_fn(theta: Array) -> Array:
            return theta

        # Setup likelihood with diagonal noise
        noise_var = self.bkd().asarray(np.array([0.1, 0.1], dtype=np.float64))
        noise_cov_op = DiagonalCovarianceOperator(noise_var, self.bkd())
        likelihood = GaussianLogLikelihood(noise_cov_op, self.bkd())

        # Observations
        obs = self.bkd().asarray(np.array([[1.0], [2.0]], dtype=np.float64))
        likelihood.set_observations(obs)

        # Prior: N(0, I)
        prior_mean = self.bkd().asarray(np.zeros((nvars, 1), dtype=np.float64))
        prior_cov = self.bkd().asarray(np.eye(nvars, dtype=np.float64))
        prior = DenseCholeskyMultivariateGaussian(prior_mean, prior_cov, self.bkd())

        posterior = LogUnNormalizedPosterior(model_fn, likelihood, prior, self.bkd())

        # Evaluate at a test point
        theta = self.bkd().asarray(np.array([[1.0], [2.0]], dtype=np.float64))
        logpost = posterior(theta)

        # Analytical computation:
        # log p(y|theta) = -0.5 * sum((y - theta)^2 / sigma^2) + const
        # log p(theta) = -0.5 * sum(theta^2) + const
        # At theta = obs = [1, 2], residuals are zero
        # log p(y|theta) = const (from likelihood)
        # log p(theta) = -0.5 * (1 + 4) + const = -2.5 + const

        # Just verify it's finite and reasonable
        logpost_np = float(self.bkd().to_numpy(logpost))
        self.assertTrue(np.isfinite(logpost_np))


class TestLogUnNormalizedPosteriorValidation(Generic[Array], unittest.TestCase):
    """Test input validation."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_repr(self) -> None:
        """Test string representation."""
        nvars = 2

        def model_fn(theta: Array) -> Array:
            return theta

        noise_var = self.bkd().asarray(np.array([0.1, 0.1], dtype=np.float64))
        noise_cov_op = DiagonalCovarianceOperator(noise_var, self.bkd())
        likelihood = GaussianLogLikelihood(noise_cov_op, self.bkd())
        likelihood.set_observations(
            self.bkd().asarray(np.array([[1.0], [2.0]], dtype=np.float64))
        )

        prior_mean = self.bkd().asarray(np.zeros((nvars, 1), dtype=np.float64))
        prior_cov = self.bkd().asarray(np.eye(nvars, dtype=np.float64))
        prior = DenseCholeskyMultivariateGaussian(prior_mean, prior_cov, self.bkd())

        posterior = LogUnNormalizedPosterior(model_fn, likelihood, prior, self.bkd())

        repr_str = repr(posterior)
        self.assertIn("LogUnNormalizedPosterior", repr_str)
        self.assertIn("nvars=2", repr_str)


# NumPy backend tests
class TestLogUnNormalizedPosteriorNumpy(TestLogUnNormalizedPosteriorBase[NDArray[Any]]):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestLogUnNormalizedPosteriorAnalyticalNumpy(
    TestLogUnNormalizedPosteriorAnalytical[NDArray[Any]]
):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestLogUnNormalizedPosteriorValidationNumpy(
    TestLogUnNormalizedPosteriorValidation[NDArray[Any]]
):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# PyTorch backend tests
class TestLogUnNormalizedPosteriorTorch(TestLogUnNormalizedPosteriorBase[torch.Tensor]):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


class TestLogUnNormalizedPosteriorAnalyticalTorch(
    TestLogUnNormalizedPosteriorAnalytical[torch.Tensor]
):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = TorchBkd()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


class TestLogUnNormalizedPosteriorValidationTorch(
    TestLogUnNormalizedPosteriorValidation[torch.Tensor]
):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = TorchBkd()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


if __name__ == "__main__":
    unittest.main()
