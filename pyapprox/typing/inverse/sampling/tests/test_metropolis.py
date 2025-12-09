"""
Tests for Metropolis-Hastings MCMC samplers.
"""

import unittest
from typing import Generic, Any

import numpy as np
from numpy.typing import NDArray
import torch

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.inverse.sampling import (
    MetropolisHastingsSampler,
    AdaptiveMetropolisSampler,
)


class TestMetropolisHastingsSamplerBase(Generic[Array], unittest.TestCase):
    """Base test class for MetropolisHastingsSampler."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        """Override in derived classes."""
        raise NotImplementedError

    def setUp(self) -> None:
        np.random.seed(42)
        self.nvars = 2

        # Simple Gaussian target: log p(x) = -0.5 * x^T @ x
        def log_posterior(samples: Array) -> Array:
            samples_np = self.bkd().to_numpy(samples)
            return self.bkd().asarray(-0.5 * np.sum(samples_np**2, axis=0))

        self.log_posterior = log_posterior
        self.sampler = MetropolisHastingsSampler(
            log_posterior, self.nvars, self.bkd()
        )

    def test_nvars(self) -> None:
        """Test nvars returns correct value."""
        self.assertEqual(self.sampler.nvars(), self.nvars)

    def test_sample_returns_correct_shape(self) -> None:
        """Test sample returns correct shape."""
        nsamples = 100
        result = self.sampler.sample(nsamples)
        self.assertEqual(result.samples.shape, (self.nvars, nsamples))

    def test_sample_with_burn(self) -> None:
        """Test sampling with burn-in."""
        nsamples = 100
        burn = 50
        result = self.sampler.sample(nsamples, burn=burn)
        self.assertEqual(result.samples.shape, (self.nvars, nsamples))

    def test_sample_with_initial_state(self) -> None:
        """Test sampling with initial state."""
        initial = self.bkd().asarray(np.array([[1.0], [1.0]], dtype=np.float64))
        result = self.sampler.sample(50, initial_state=initial)
        self.assertEqual(result.samples.shape, (self.nvars, 50))

    def test_acceptance_rate_in_range(self) -> None:
        """Test acceptance rate is in [0, 1]."""
        result = self.sampler.sample(200)
        self.assertGreaterEqual(result.acceptance_rate, 0.0)
        self.assertLessEqual(result.acceptance_rate, 1.0)

    def test_log_posteriors_shape(self) -> None:
        """Test log posteriors have correct shape."""
        nsamples = 100
        result = self.sampler.sample(nsamples)
        self.assertEqual(result.log_posteriors.shape, (nsamples,))

    def test_samples_near_target_mean(self) -> None:
        """Test that samples are roughly centered at zero for standard normal."""
        np.random.seed(123)
        result = self.sampler.sample(500, burn=100)
        samples_np = self.bkd().to_numpy(result.samples)
        sample_mean = np.mean(samples_np, axis=1)
        # Should be close to zero (target mean)
        np.testing.assert_array_less(np.abs(sample_mean), 1.0)

    def test_set_proposal_covariance(self) -> None:
        """Test setting proposal covariance."""
        new_cov = self.bkd().asarray(
            np.array([[0.5, 0.0], [0.0, 0.5]], dtype=np.float64)
        )
        self.sampler.set_proposal_covariance(new_cov)
        cov = self.sampler.proposal_covariance()
        cov_np = self.bkd().to_numpy(cov)
        np.testing.assert_array_almost_equal(
            cov_np, np.array([[0.5, 0.0], [0.0, 0.5]])
        )

    def test_wrong_proposal_cov_shape_raises(self) -> None:
        """Test wrong proposal covariance shape raises error."""
        wrong_cov = self.bkd().asarray(np.eye(3))
        with self.assertRaises(ValueError):
            self.sampler.set_proposal_covariance(wrong_cov)

    def test_wrong_initial_state_shape_raises(self) -> None:
        """Test wrong initial state shape raises error."""
        wrong_initial = self.bkd().asarray(np.zeros((3, 1)))
        with self.assertRaises(ValueError):
            self.sampler.sample(100, initial_state=wrong_initial)


class TestAdaptiveMetropolisSamplerBase(Generic[Array], unittest.TestCase):
    """Base test class for AdaptiveMetropolisSampler."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        """Override in derived classes."""
        raise NotImplementedError

    def setUp(self) -> None:
        np.random.seed(42)
        self.nvars = 2

        # Simple Gaussian target
        def log_posterior(samples: Array) -> Array:
            samples_np = self.bkd().to_numpy(samples)
            return self.bkd().asarray(-0.5 * np.sum(samples_np**2, axis=0))

        self.log_posterior = log_posterior
        self.sampler = AdaptiveMetropolisSampler(
            log_posterior,
            self.nvars,
            self.bkd(),
            adaptation_start=50,
            adaptation_interval=25,
        )

    def test_sample_returns_correct_shape(self) -> None:
        """Test adaptive sampler returns correct shape."""
        nsamples = 200
        result = self.sampler.sample(nsamples)
        self.assertEqual(result.samples.shape, (self.nvars, nsamples))

    def test_acceptance_rate_reasonable(self) -> None:
        """Test adaptive sampler achieves reasonable acceptance rate."""
        np.random.seed(456)
        result = self.sampler.sample(500, burn=100)
        # Adaptive should achieve reasonable acceptance (typically 0.1-0.8)
        self.assertGreater(result.acceptance_rate, 0.05)

    def test_repr(self) -> None:
        """Test string representation."""
        repr_str = repr(self.sampler)
        self.assertIn("AdaptiveMetropolisSampler", repr_str)
        self.assertIn("nvars=2", repr_str)


class TestMetropolisWithBounds(Generic[Array], unittest.TestCase):
    """Test sampling with parameter bounds."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_samples_within_bounds(self) -> None:
        """Test samples respect parameter bounds."""
        np.random.seed(789)
        nvars = 2

        def log_posterior(samples: Array) -> Array:
            samples_np = self.bkd().to_numpy(samples)
            return self.bkd().asarray(-0.5 * np.sum(samples_np**2, axis=0))

        sampler = MetropolisHastingsSampler(log_posterior, nvars, self.bkd())

        # Bounds: [-2, 2] for both variables
        bounds = self.bkd().asarray(
            np.array([[-2.0, 2.0], [-2.0, 2.0]], dtype=np.float64)
        )
        result = sampler.sample(200, bounds=bounds)

        samples_np = self.bkd().to_numpy(result.samples)
        self.assertTrue(np.all(samples_np >= -2.0))
        self.assertTrue(np.all(samples_np <= 2.0))


# NumPy backend tests
class TestMetropolisHastingsSamplerNumpy(
    TestMetropolisHastingsSamplerBase[NDArray[Any]]
):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestAdaptiveMetropolisSamplerNumpy(
    TestAdaptiveMetropolisSamplerBase[NDArray[Any]]
):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestMetropolisWithBoundsNumpy(TestMetropolisWithBounds[NDArray[Any]]):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# PyTorch backend tests
class TestMetropolisHastingsSamplerTorch(
    TestMetropolisHastingsSamplerBase[torch.Tensor]
):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


class TestAdaptiveMetropolisSamplerTorch(
    TestAdaptiveMetropolisSamplerBase[torch.Tensor]
):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


class TestMetropolisWithBoundsTorch(TestMetropolisWithBounds[torch.Tensor]):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = TorchBkd()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


from pyapprox.typing.util.test_utils import load_tests


if __name__ == "__main__":
    unittest.main()
