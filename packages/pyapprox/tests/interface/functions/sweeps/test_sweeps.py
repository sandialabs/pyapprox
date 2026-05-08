"""Tests for parameter sweeps module.

Tests bounded and Gaussian parameter sweepers with dual backend support.
"""

import numpy as np
import pytest

from pyapprox.interface.functions.sweeps import (
    BoundedParameterSweeper,
    GaussianParameterSweeper,
)


class TestBoundedParameterSweeper:
    """Base test class for BoundedParameterSweeper."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def test_basic_sweep_generation(self, bkd):
        """Test basic sweep generation."""
        bounds = bkd.asarray([[0.0, 1.0], [0.0, 2.0], [-1.0, 1.0]])
        sweeper = BoundedParameterSweeper(bounds, nsamples_per_sweep=50, bkd=bkd)

        assert sweeper.nvars() == 3
        assert sweeper.nsamples_per_sweep() == 50

        samples = sweeper.rvs(nsweeps=5)
        assert samples.shape == (3, 250)

    def test_samples_within_bounds(self, bkd):
        """Test that all samples are within bounds."""
        bounds = bkd.asarray([[0.0, 1.0], [0.0, 2.0], [-1.0, 1.0]])
        sweeper = BoundedParameterSweeper(bounds, nsamples_per_sweep=100, bkd=bkd)

        samples = sweeper.rvs(nsweeps=10)
        samples_np = bkd.to_numpy(samples)
        bounds_np = bkd.to_numpy(bounds)

        for ii in range(3):
            assert np.all(samples_np[ii, :] >= bounds_np[ii, 0] - 1e-10)
            assert np.all(samples_np[ii, :] <= bounds_np[ii, 1] + 1e-10)

    def test_canonical_active_samples_shape(self, bkd):
        """Test that canonical samples have correct shape."""
        bounds = bkd.asarray([[0.0, 1.0], [0.0, 1.0]])
        sweeper = BoundedParameterSweeper(bounds, nsamples_per_sweep=30, bkd=bkd)

        sweeper.rvs(nsweeps=3)
        canonical = sweeper.canonical_active_samples()

        assert canonical.shape == (3, 30)

    def test_set_rotation_matrices(self, bkd):
        """Test setting custom rotation matrices."""
        bounds = bkd.asarray([[0.0, 1.0], [0.0, 1.0]])
        sweeper = BoundedParameterSweeper(bounds, nsamples_per_sweep=20, bkd=bkd)

        # Set custom rotation matrix (orthogonal vectors)
        rotation = bkd.asarray([[1.0, 0.0], [0.0, 1.0]])
        sweeper.set_sweep_rotation_matrices(rotation)

        samples = sweeper.rvs(nsweeps=2)
        assert samples.shape == (2, 40)

    def test_invalid_bounds_shape(self, bkd):
        """Test that invalid bounds raise errors."""
        with pytest.raises(ValueError):
            BoundedParameterSweeper(
                bkd.asarray([0.0, 1.0]),  # 1D instead of 2D
                nsamples_per_sweep=10,
                bkd=bkd,
            )

    def test_sweep_bounds_computation(self, bkd):
        """Test sweep bounds computation."""
        bounds = bkd.asarray([[0.0, 1.0], [0.0, 1.0]])
        sweeper = BoundedParameterSweeper(bounds, nsamples_per_sweep=20, bkd=bkd)

        # Test along a coordinate axis
        rotation_vec = bkd.asarray([[1.0], [0.0]])
        lb, ub = sweeper.sweep_bounds(rotation_vec)

        # For unit hypercube [-1,1]^2, sweep along (1,0) should give [-1, 1]
        assert lb >= -1.5
        assert ub <= 1.5
        assert lb < ub


class TestGaussianParameterSweeper:
    """Base test class for GaussianParameterSweeper."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def test_basic_sweep_generation(self, bkd):
        """Test basic Gaussian sweep generation."""
        mean = bkd.asarray([0.0, 1.0, 2.0])

        # Identity covariance sqrt
        def cov_sqrt_op(x):
            return x

        sweeper = GaussianParameterSweeper(
            mean,
            cov_sqrt_op,
            sweep_radius=3.0,
            nsamples_per_sweep=50,
            bkd=bkd,
        )

        assert sweeper.nvars() == 3
        assert sweeper.nsamples_per_sweep() == 50

        samples = sweeper.rvs(nsweeps=5)
        assert samples.shape == (3, 250)

    def test_sweep_centered_at_mean(self, bkd):
        """Test that sweep center is at the mean."""
        mean = bkd.asarray([1.0, 2.0])

        def cov_sqrt_op(x):
            return x

        sweeper = GaussianParameterSweeper(
            mean,
            cov_sqrt_op,
            sweep_radius=2.0,
            nsamples_per_sweep=51,  # Odd so middle sample exists
            bkd=bkd,
        )

        samples = sweeper.rvs(nsweeps=1)
        # Middle sample (index 25) should be at mean
        middle_sample = samples[:, 25]
        bkd.assert_allclose(middle_sample, mean, atol=0.1)

    def test_sweep_bounds(self, bkd):
        """Test sweep bounds are +-sweep_radius."""
        mean = bkd.asarray([0.0, 0.0])

        def cov_sqrt_op(x):
            return x

        sweeper = GaussianParameterSweeper(
            mean,
            cov_sqrt_op,
            sweep_radius=3.0,
            nsamples_per_sweep=10,
            bkd=bkd,
        )

        rotation_vec = bkd.asarray([[1.0], [0.0]])
        lb, ub = sweeper.sweep_bounds(rotation_vec)

        assert lb == -3.0
        assert ub == 3.0

    def test_correlated_sweep(self, bkd):
        """Test sweep with correlated Gaussian."""
        mean = bkd.asarray([0.0, 0.0])
        # Cholesky factor for covariance [[1, 0.5], [0.5, 1]]
        L = bkd.asarray([[1.0, 0.0], [0.5, 0.866]])

        def cov_sqrt_op(x):
            return L @ x

        sweeper = GaussianParameterSweeper(
            mean,
            cov_sqrt_op,
            sweep_radius=2.0,
            nsamples_per_sweep=30,
            bkd=bkd,
        )

        samples = sweeper.rvs(nsweeps=3)
        assert samples.shape == (2, 90)

    def test_canonical_active_samples_shape(self, bkd):
        """Test canonical samples shape."""
        mean = bkd.asarray([0.0, 0.0])

        def cov_sqrt_op(x):
            return x

        sweeper = GaussianParameterSweeper(
            mean,
            cov_sqrt_op,
            sweep_radius=2.0,
            nsamples_per_sweep=20,
            bkd=bkd,
        )

        sweeper.rvs(nsweeps=4)
        canonical = sweeper.canonical_active_samples()

        assert canonical.shape == (4, 20)
