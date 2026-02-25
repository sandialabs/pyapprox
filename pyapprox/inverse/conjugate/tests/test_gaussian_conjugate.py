"""
Tests for Gaussian conjugate posterior.
"""

import unittest
from typing import Generic, Any

import numpy as np
from numpy.typing import NDArray
import torch
from scipy import stats

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.inverse.conjugate.gaussian import DenseGaussianConjugatePosterior


class TestDenseGaussianConjugateBase(Generic[Array], unittest.TestCase):
    """Base test class for DenseGaussianConjugatePosterior."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        """Override in derived classes."""
        raise NotImplementedError

    def setUp(self) -> None:
        # Simple 2D problem
        self.nvars = 2
        self.nobs = 3

        # Linear model: y = A @ x + offset
        self.A = self.bkd().asarray(
            [[1.0, 0.5], [0.5, 1.0], [1.0, 1.0]]
        )
        self.offset = self.bkd().zeros((self.nobs, 1))

        # Prior
        self.prior_mean = self.bkd().zeros((self.nvars, 1))
        self.prior_cov = self.bkd().eye(self.nvars)

        # Noise
        self.noise_var = 0.1
        self.noise_cov = self.noise_var * self.bkd().eye(self.nobs)

        # Create solver
        self.solver = DenseGaussianConjugatePosterior(
            self.A,
            self.prior_mean,
            self.prior_cov,
            self.noise_cov,
            self.bkd(),
            self.offset,
        )

        # Observations
        self.obs = self.bkd().asarray([[1.0], [1.5], [2.0]])

    def test_nvars(self) -> None:
        """Test nvars returns correct value."""
        self.assertEqual(self.solver.nvars(), self.nvars)

    def test_nobs(self) -> None:
        """Test nobs returns correct value."""
        self.assertEqual(self.solver.nobs(), self.nobs)

    def test_posterior_mean_shape(self) -> None:
        """Test posterior mean has correct shape."""
        self.solver.compute(self.obs)
        mean = self.solver.posterior_mean()
        self.assertEqual(mean.shape, (self.nvars, 1))

    def test_posterior_covariance_shape(self) -> None:
        """Test posterior covariance has correct shape."""
        self.solver.compute(self.obs)
        cov = self.solver.posterior_covariance()
        self.assertEqual(cov.shape, (self.nvars, self.nvars))

    def test_posterior_covariance_symmetric(self) -> None:
        """Test posterior covariance is symmetric."""
        self.solver.compute(self.obs)
        cov = self.solver.posterior_covariance()
        cov_np = self.bkd().to_numpy(cov)
        np.testing.assert_array_almost_equal(cov_np, cov_np.T)

    def test_posterior_covariance_positive_definite(self) -> None:
        """Test posterior covariance is positive definite."""
        self.solver.compute(self.obs)
        cov = self.solver.posterior_covariance()
        cov_np = self.bkd().to_numpy(cov)
        eigenvalues = np.linalg.eigvalsh(cov_np)
        self.assertTrue(all(eigenvalues > 0))

    def test_evidence_positive(self) -> None:
        """Test evidence is positive."""
        self.solver.compute(self.obs)
        evidence = self.solver.evidence()
        self.assertGreater(evidence, 0)

    def test_expected_kl_divergence_nonnegative(self) -> None:
        """Test expected KL divergence is non-negative."""
        self.solver.compute(self.obs)
        kl = self.solver.expected_kl_divergence()
        self.assertGreaterEqual(kl, 0)

    def test_posterior_variable_returns_gaussian(self) -> None:
        """Test posterior_variable returns a Gaussian distribution."""
        self.solver.compute(self.obs)
        post = self.solver.posterior_variable()
        # Check it has the expected methods
        self.assertTrue(hasattr(post, 'logpdf'))
        self.assertTrue(hasattr(post, 'rvs'))

    def test_compute_not_called_raises(self) -> None:
        """Test accessing results before compute raises error."""
        with self.assertRaises(RuntimeError):
            self.solver.posterior_mean()

    def test_wrong_obs_shape_raises(self) -> None:
        """Test wrong observation shape raises error."""
        bad_obs = self.bkd().zeros((5, 1))  # Wrong nobs
        with self.assertRaises(ValueError):
            self.solver.compute(bad_obs)

    def test_posterior_differs_from_prior(self) -> None:
        """Test posterior mean differs from prior mean after data."""
        self.solver.compute(self.obs)
        post_mean = self.bkd().to_numpy(self.solver.posterior_mean())
        prior_mean = self.bkd().to_numpy(self.prior_mean)

        # Posterior should move toward data direction
        diff = np.linalg.norm(post_mean - prior_mean)
        self.assertGreater(diff, 0.01)  # Not identical to prior


class TestDenseGaussianConjugateAnalytical(Generic[Array], unittest.TestCase):
    """Test against analytical formulas."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        # 1D problem for easy analytical verification
        self.A = self.bkd().asarray([[1.0]])
        self.prior_mean = self.bkd().asarray([[0.0]])
        self.prior_var = 1.0
        self.prior_cov = self.prior_var * self.bkd().eye(1)
        self.noise_var = 0.5
        self.noise_cov = self.noise_var * self.bkd().eye(1)

        self.solver = DenseGaussianConjugatePosterior(
            self.A, self.prior_mean, self.prior_cov, self.noise_cov, self.bkd()
        )

    def test_posterior_mean_analytical(self) -> None:
        """Test posterior mean matches analytical formula."""
        obs = self.bkd().asarray([[2.0]])
        self.solver.compute(obs)

        # Analytical: post_mean = (1/prior_var + 1/noise_var)^{-1} * (obs/noise_var)
        # = noise_var * prior_var / (noise_var + prior_var) * obs / noise_var
        # = prior_var / (noise_var + prior_var) * obs
        expected = self.prior_var / (self.noise_var + self.prior_var) * 2.0

        post_mean = self.bkd().to_numpy(self.solver.posterior_mean())
        self.assertAlmostEqual(post_mean[0, 0], expected, places=5)

    def test_posterior_variance_analytical(self) -> None:
        """Test posterior variance matches analytical formula."""
        obs = self.bkd().asarray([[2.0]])
        self.solver.compute(obs)

        # Analytical: post_var = (1/prior_var + 1/noise_var)^{-1}
        expected = 1.0 / (1.0 / self.prior_var + 1.0 / self.noise_var)

        post_cov = self.bkd().to_numpy(self.solver.posterior_covariance())
        self.assertAlmostEqual(post_cov[0, 0], expected, places=5)


# NumPy backend tests
class TestDenseGaussianConjugateNumpy(TestDenseGaussianConjugateBase[NDArray[Any]]):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestDenseGaussianConjugateAnalyticalNumpy(
    TestDenseGaussianConjugateAnalytical[NDArray[Any]]
):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# PyTorch backend tests
class TestDenseGaussianConjugateTorch(TestDenseGaussianConjugateBase[torch.Tensor]):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


class TestDenseGaussianConjugateAnalyticalTorch(
    TestDenseGaussianConjugateAnalytical[torch.Tensor]
):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


from pyapprox.util.test_utils import load_tests


if __name__ == "__main__":
    unittest.main()
