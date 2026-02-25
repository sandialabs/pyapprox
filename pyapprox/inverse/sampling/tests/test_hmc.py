"""
Tests for Hamiltonian Monte Carlo sampler.
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.inverse.sampling.hmc import HamiltonianMonteCarlo
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd


class TestHMCBase(Generic[Array], unittest.TestCase):
    """Base test class for HamiltonianMonteCarlo."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        np.random.seed(42)
        self.nvars = 2

        # Standard Gaussian target
        def log_posterior(samples: Array) -> Array:
            samples_np = self.bkd().to_numpy(samples)
            return self.bkd().asarray(-0.5 * np.sum(samples_np**2, axis=0))

        def gradient(sample: Array) -> Array:
            return -sample

        self.log_posterior = log_posterior
        self.gradient = gradient
        self.sampler = HamiltonianMonteCarlo(
            log_posterior,
            gradient,
            self.nvars,
            self.bkd(),
            step_size=0.1,
            num_leapfrog_steps=10,
        )

    def test_nvars(self) -> None:
        """Test nvars returns correct value."""
        self.assertEqual(self.sampler.nvars(), self.nvars)

    def test_sample_returns_correct_shape(self) -> None:
        """Test sample returns correct shape."""
        nsamples = 50
        result = self.sampler.sample(nsamples)
        self.assertEqual(result.samples.shape, (self.nvars, nsamples))

    def test_sample_with_burn(self) -> None:
        """Test sampling with burn-in."""
        nsamples = 50
        burn = 20
        result = self.sampler.sample(nsamples, burn=burn)
        self.assertEqual(result.samples.shape, (self.nvars, nsamples))

    def test_sample_with_initial_state(self) -> None:
        """Test sampling with initial state."""
        initial = self.bkd().asarray(np.array([[1.0], [1.0]], dtype=np.float64))
        result = self.sampler.sample(30, initial_state=initial)
        self.assertEqual(result.samples.shape, (self.nvars, 30))

    def test_acceptance_rate_in_range(self) -> None:
        """Test acceptance rate is in [0, 1]."""
        result = self.sampler.sample(100)
        self.assertGreaterEqual(result.acceptance_rate, 0.0)
        self.assertLessEqual(result.acceptance_rate, 1.0)

    def test_log_posteriors_shape(self) -> None:
        """Test log posteriors have correct shape."""
        nsamples = 50
        result = self.sampler.sample(nsamples)
        self.assertEqual(result.log_posteriors.shape, (nsamples,))

    def test_wrong_initial_state_shape_raises(self) -> None:
        """Test wrong initial state shape raises error."""
        wrong_initial = self.bkd().asarray(np.zeros((3, 1)))
        with self.assertRaises(ValueError):
            self.sampler.sample(100, initial_state=wrong_initial)

    def test_repr(self) -> None:
        """Test string representation."""
        repr_str = repr(self.sampler)
        self.assertIn("HamiltonianMonteCarlo", repr_str)
        self.assertIn("nvars=2", repr_str)


class TestHMCConvergence(Generic[Array], unittest.TestCase):
    """Test HMC convergence on simple targets."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_gaussian_target_mean(self) -> None:
        """Test HMC samples have correct mean for Gaussian target."""
        np.random.seed(123)

        def log_posterior(samples: Array) -> Array:
            samples_np = self.bkd().to_numpy(samples)
            return self.bkd().asarray(-0.5 * np.sum(samples_np**2, axis=0))

        def gradient(sample: Array) -> Array:
            return -sample

        sampler = HamiltonianMonteCarlo(
            log_posterior,
            gradient,
            nvars=2,
            bkd=self.bkd(),
            step_size=0.15,
            num_leapfrog_steps=15,
        )

        result = sampler.sample(nsamples=500, burn=100)
        samples_np = self.bkd().to_numpy(result.samples)
        sample_mean = np.mean(samples_np, axis=1)

        # Mean should be close to 0
        np.testing.assert_array_less(np.abs(sample_mean), 0.3)


# NumPy backend tests
class TestHMCNumpy(TestHMCBase[NDArray[Any]]):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestHMCConvergenceNumpy(TestHMCConvergence[NDArray[Any]]):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# PyTorch backend tests
class TestHMCTorch(TestHMCBase[torch.Tensor]):
    __test__ = True

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> TorchBkd:
        return self._bkd


class TestHMCConvergenceTorch(TestHMCConvergence[torch.Tensor]):
    __test__ = True

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()

    def bkd(self) -> TorchBkd:
        return self._bkd


if __name__ == "__main__":
    unittest.main()
