"""
Tests for Delayed Rejection Adaptive Metropolis (DRAM) sampler.
"""

import unittest
from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray
import torch

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.inverse.sampling.dram import (
    DelayedRejectionAdaptiveMetropolis,
)


class TestDRAMBase(Generic[Array], unittest.TestCase):
    """Base test class for DelayedRejectionAdaptiveMetropolis."""

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

        self.log_posterior = log_posterior
        self.sampler = DelayedRejectionAdaptiveMetropolis(
            log_posterior,
            self.nvars,
            self.bkd(),
            adaptation_start=50,
            adaptation_interval=25,
            dr_scale=0.1,
        )

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

    def test_acceptance_rate_reasonable(self) -> None:
        """Test DRAM achieves reasonable acceptance rate."""
        np.random.seed(456)
        result = self.sampler.sample(300, burn=100)
        # DRAM should achieve reasonable acceptance (typically 0.1-0.9)
        self.assertGreater(result.acceptance_rate, 0.05)

    def test_log_posteriors_shape(self) -> None:
        """Test log posteriors have correct shape."""
        nsamples = 100
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
        self.assertIn("DelayedRejectionAdaptiveMetropolis", repr_str)
        self.assertIn("dr_scale", repr_str)


class TestDRAMWithBounds(Generic[Array], unittest.TestCase):
    """Test DRAM with parameter bounds."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_samples_within_bounds(self) -> None:
        """Test samples respect parameter bounds."""
        np.random.seed(789)

        def log_posterior(samples: Array) -> Array:
            samples_np = self.bkd().to_numpy(samples)
            return self.bkd().asarray(-0.5 * np.sum(samples_np**2, axis=0))

        sampler = DelayedRejectionAdaptiveMetropolis(
            log_posterior, nvars=2, bkd=self.bkd()
        )

        bounds = self.bkd().asarray(
            np.array([[-2.0, 2.0], [-2.0, 2.0]], dtype=np.float64)
        )
        result = sampler.sample(200, bounds=bounds)

        samples_np = self.bkd().to_numpy(result.samples)
        self.assertTrue(np.all(samples_np >= -2.0))
        self.assertTrue(np.all(samples_np <= 2.0))


class TestDRAMConvergence(Generic[Array], unittest.TestCase):
    """Test DRAM convergence on simple targets."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_gaussian_target_mean(self) -> None:
        """Test DRAM samples have correct mean for Gaussian target."""
        np.random.seed(123)

        def log_posterior(samples: Array) -> Array:
            samples_np = self.bkd().to_numpy(samples)
            return self.bkd().asarray(-0.5 * np.sum(samples_np**2, axis=0))

        sampler = DelayedRejectionAdaptiveMetropolis(
            log_posterior,
            nvars=2,
            bkd=self.bkd(),
            adaptation_start=100,
            adaptation_interval=50,
        )

        result = sampler.sample(nsamples=500, burn=200)
        samples_np = self.bkd().to_numpy(result.samples)
        sample_mean = np.mean(samples_np, axis=1)

        # Mean should be close to 0
        np.testing.assert_array_less(np.abs(sample_mean), 0.3)


# NumPy backend tests
class TestDRAMNumpy(TestDRAMBase[NDArray[Any]]):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestDRAMWithBoundsNumpy(TestDRAMWithBounds[NDArray[Any]]):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestDRAMConvergenceNumpy(TestDRAMConvergence[NDArray[Any]]):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# PyTorch backend tests
class TestDRAMTorch(TestDRAMBase[torch.Tensor]):
    __test__ = True

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> TorchBkd:
        return self._bkd


class TestDRAMWithBoundsTorch(TestDRAMWithBounds[torch.Tensor]):
    __test__ = True

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()

    def bkd(self) -> TorchBkd:
        return self._bkd


class TestDRAMConvergenceTorch(TestDRAMConvergence[torch.Tensor]):
    __test__ = True

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()

    def bkd(self) -> TorchBkd:
        return self._bkd


from pyapprox.util.test_utils import load_tests


if __name__ == "__main__":
    unittest.main()
