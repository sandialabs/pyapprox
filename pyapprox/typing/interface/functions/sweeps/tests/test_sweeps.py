"""Tests for parameter sweeps module.

Tests bounded and Gaussian parameter sweepers with dual backend support.
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array
from pyapprox.typing.util.test_utils import load_tests

from pyapprox.typing.interface.functions.sweeps import (
    BoundedParameterSweeper,
    GaussianParameterSweeper,
)


class TestBoundedParameterSweeper(Generic[Array], unittest.TestCase):
    """Base test class for BoundedParameterSweeper."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        np.random.seed(42)

    def test_basic_sweep_generation(self):
        """Test basic sweep generation."""
        bounds = self._bkd.asarray([[0.0, 1.0], [0.0, 2.0], [-1.0, 1.0]])
        sweeper = BoundedParameterSweeper(
            bounds, nsamples_per_sweep=50, bkd=self._bkd
        )

        self.assertEqual(sweeper.nvars(), 3)
        self.assertEqual(sweeper.nsamples_per_sweep(), 50)

        samples = sweeper.rvs(nsweeps=5)
        self.assertEqual(samples.shape, (3, 250))

    def test_samples_within_bounds(self):
        """Test that all samples are within bounds."""
        bounds = self._bkd.asarray([[0.0, 1.0], [0.0, 2.0], [-1.0, 1.0]])
        sweeper = BoundedParameterSweeper(
            bounds, nsamples_per_sweep=100, bkd=self._bkd
        )

        samples = sweeper.rvs(nsweeps=10)
        samples_np = self._bkd.to_numpy(samples)
        bounds_np = self._bkd.to_numpy(bounds)

        for ii in range(3):
            self.assertTrue(np.all(samples_np[ii, :] >= bounds_np[ii, 0] - 1e-10))
            self.assertTrue(np.all(samples_np[ii, :] <= bounds_np[ii, 1] + 1e-10))

    def test_canonical_active_samples_shape(self):
        """Test that canonical samples have correct shape."""
        bounds = self._bkd.asarray([[0.0, 1.0], [0.0, 1.0]])
        sweeper = BoundedParameterSweeper(
            bounds, nsamples_per_sweep=30, bkd=self._bkd
        )

        samples = sweeper.rvs(nsweeps=3)
        canonical = sweeper.canonical_active_samples()

        self.assertEqual(canonical.shape, (3, 30))

    def test_set_rotation_matrices(self):
        """Test setting custom rotation matrices."""
        bounds = self._bkd.asarray([[0.0, 1.0], [0.0, 1.0]])
        sweeper = BoundedParameterSweeper(
            bounds, nsamples_per_sweep=20, bkd=self._bkd
        )

        # Set custom rotation matrix (orthogonal vectors)
        rotation = self._bkd.asarray([[1.0, 0.0], [0.0, 1.0]])
        sweeper.set_sweep_rotation_matrices(rotation)

        samples = sweeper.rvs(nsweeps=2)
        self.assertEqual(samples.shape, (2, 40))

    def test_invalid_bounds_shape(self):
        """Test that invalid bounds raise errors."""
        with self.assertRaises(ValueError):
            BoundedParameterSweeper(
                self._bkd.asarray([0.0, 1.0]),  # 1D instead of 2D
                nsamples_per_sweep=10,
                bkd=self._bkd,
            )

    def test_sweep_bounds_computation(self):
        """Test sweep bounds computation."""
        bounds = self._bkd.asarray([[0.0, 1.0], [0.0, 1.0]])
        sweeper = BoundedParameterSweeper(
            bounds, nsamples_per_sweep=20, bkd=self._bkd
        )

        # Test along a coordinate axis
        rotation_vec = self._bkd.asarray([[1.0], [0.0]])
        lb, ub = sweeper.sweep_bounds(rotation_vec)

        # For unit hypercube [-1,1]^2, sweep along (1,0) should give [-1, 1]
        self.assertTrue(lb >= -1.5)
        self.assertTrue(ub <= 1.5)
        self.assertTrue(lb < ub)


class TestGaussianParameterSweeper(Generic[Array], unittest.TestCase):
    """Base test class for GaussianParameterSweeper."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        np.random.seed(42)

    def test_basic_sweep_generation(self):
        """Test basic Gaussian sweep generation."""
        mean = self._bkd.asarray([0.0, 1.0, 2.0])
        # Identity covariance sqrt
        cov_sqrt_op = lambda x: x

        sweeper = GaussianParameterSweeper(
            mean,
            cov_sqrt_op,
            sweep_radius=3.0,
            nsamples_per_sweep=50,
            bkd=self._bkd,
        )

        self.assertEqual(sweeper.nvars(), 3)
        self.assertEqual(sweeper.nsamples_per_sweep(), 50)

        samples = sweeper.rvs(nsweeps=5)
        self.assertEqual(samples.shape, (3, 250))

    def test_sweep_centered_at_mean(self):
        """Test that sweep center is at the mean."""
        mean = self._bkd.asarray([1.0, 2.0])
        cov_sqrt_op = lambda x: x

        sweeper = GaussianParameterSweeper(
            mean,
            cov_sqrt_op,
            sweep_radius=2.0,
            nsamples_per_sweep=51,  # Odd so middle sample exists
            bkd=self._bkd,
        )

        samples = sweeper.rvs(nsweeps=1)
        # Middle sample (index 25) should be at mean
        middle_sample = samples[:, 25]
        self._bkd.assert_allclose(middle_sample, mean, atol=0.1)

    def test_sweep_bounds(self):
        """Test sweep bounds are +-sweep_radius."""
        mean = self._bkd.asarray([0.0, 0.0])
        cov_sqrt_op = lambda x: x

        sweeper = GaussianParameterSweeper(
            mean,
            cov_sqrt_op,
            sweep_radius=3.0,
            nsamples_per_sweep=10,
            bkd=self._bkd,
        )

        rotation_vec = self._bkd.asarray([[1.0], [0.0]])
        lb, ub = sweeper.sweep_bounds(rotation_vec)

        self.assertEqual(lb, -3.0)
        self.assertEqual(ub, 3.0)

    def test_correlated_sweep(self):
        """Test sweep with correlated Gaussian."""
        mean = self._bkd.asarray([0.0, 0.0])
        # Cholesky factor for covariance [[1, 0.5], [0.5, 1]]
        L = self._bkd.asarray([[1.0, 0.0], [0.5, 0.866]])
        cov_sqrt_op = lambda x: L @ x

        sweeper = GaussianParameterSweeper(
            mean,
            cov_sqrt_op,
            sweep_radius=2.0,
            nsamples_per_sweep=30,
            bkd=self._bkd,
        )

        samples = sweeper.rvs(nsweeps=3)
        self.assertEqual(samples.shape, (2, 90))

    def test_canonical_active_samples_shape(self):
        """Test canonical samples shape."""
        mean = self._bkd.asarray([0.0, 0.0])
        cov_sqrt_op = lambda x: x

        sweeper = GaussianParameterSweeper(
            mean,
            cov_sqrt_op,
            sweep_radius=2.0,
            nsamples_per_sweep=20,
            bkd=self._bkd,
        )

        sweeper.rvs(nsweeps=4)
        canonical = sweeper.canonical_active_samples()

        self.assertEqual(canonical.shape, (4, 20))


class TestBoundedParameterSweeperNumpy(TestBoundedParameterSweeper[NDArray[Any]]):
    """NumPy backend tests for BoundedParameterSweeper."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestBoundedParameterSweeperTorch(TestBoundedParameterSweeper[torch.Tensor]):
    """PyTorch backend tests for BoundedParameterSweeper."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestGaussianParameterSweeperNumpy(TestGaussianParameterSweeper[NDArray[Any]]):
    """NumPy backend tests for GaussianParameterSweeper."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGaussianParameterSweeperTorch(TestGaussianParameterSweeper[torch.Tensor]):
    """PyTorch backend tests for GaussianParameterSweeper."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
