"""
Tests for Delayed Rejection Adaptive Metropolis (DRAM) sampler.
"""

import pytest

import numpy as np

from pyapprox.inverse.sampling.dram import (
    DelayedRejectionAdaptiveMetropolis,
)


class TestDRAMBase:
    """Base test class for DelayedRejectionAdaptiveMetropolis."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _make_sampler(self, bkd):
        """Create sampler for tests."""
        nvars = 2

        # Standard Gaussian target
        def log_posterior(samples):
            samples_np = bkd.to_numpy(samples)
            return bkd.asarray(-0.5 * np.sum(samples_np**2, axis=0))

        sampler = DelayedRejectionAdaptiveMetropolis(
            log_posterior,
            nvars,
            bkd,
            adaptation_start=50,
            adaptation_interval=25,
            dr_scale=0.1,
        )
        return sampler, nvars

    def test_sample_returns_correct_shape(self, bkd) -> None:
        """Test sample returns correct shape."""
        sampler, nvars = self._make_sampler(bkd)
        nsamples = 100
        result = sampler.sample(nsamples)
        assert result.samples.shape == (nvars, nsamples)

    def test_sample_with_burn(self, bkd) -> None:
        """Test sampling with burn-in."""
        sampler, nvars = self._make_sampler(bkd)
        nsamples = 100
        burn = 50
        result = sampler.sample(nsamples, burn=burn)
        assert result.samples.shape == (nvars, nsamples)

    def test_sample_with_initial_state(self, bkd) -> None:
        """Test sampling with initial state."""
        sampler, nvars = self._make_sampler(bkd)
        initial = bkd.asarray(np.array([[1.0], [1.0]], dtype=np.float64))
        result = sampler.sample(50, initial_state=initial)
        assert result.samples.shape == (nvars, 50)

    def test_acceptance_rate_reasonable(self, bkd) -> None:
        """Test DRAM achieves reasonable acceptance rate."""
        sampler, _ = self._make_sampler(bkd)
        np.random.seed(456)
        result = sampler.sample(300, burn=100)
        # DRAM should achieve reasonable acceptance (typically 0.1-0.9)
        assert result.acceptance_rate > 0.05

    def test_log_posteriors_shape(self, bkd) -> None:
        """Test log posteriors have correct shape."""
        sampler, _ = self._make_sampler(bkd)
        nsamples = 100
        result = sampler.sample(nsamples)
        assert result.log_posteriors.shape == (nsamples,)

    def test_wrong_initial_state_shape_raises(self, bkd) -> None:
        """Test wrong initial state shape raises error."""
        sampler, _ = self._make_sampler(bkd)
        wrong_initial = bkd.asarray(np.zeros((3, 1)))
        with pytest.raises(ValueError):
            sampler.sample(100, initial_state=wrong_initial)

    def test_repr(self, bkd) -> None:
        """Test string representation."""
        sampler, _ = self._make_sampler(bkd)
        repr_str = repr(sampler)
        assert "DelayedRejectionAdaptiveMetropolis" in repr_str
        assert "dr_scale" in repr_str


class TestDRAMWithBounds:
    """Test DRAM with parameter bounds."""

    def test_samples_within_bounds(self, bkd) -> None:
        """Test samples respect parameter bounds."""
        np.random.seed(789)

        def log_posterior(samples):
            samples_np = bkd.to_numpy(samples)
            return bkd.asarray(-0.5 * np.sum(samples_np**2, axis=0))

        sampler = DelayedRejectionAdaptiveMetropolis(
            log_posterior, nvars=2, bkd=bkd
        )

        bounds = bkd.asarray(
            np.array([[-2.0, 2.0], [-2.0, 2.0]], dtype=np.float64)
        )
        result = sampler.sample(200, bounds=bounds)

        samples_np = bkd.to_numpy(result.samples)
        assert np.all(samples_np >= -2.0)
        assert np.all(samples_np <= 2.0)


class TestDRAMConvergence:
    """Test DRAM convergence on simple targets."""

    def test_gaussian_target_mean(self, bkd) -> None:
        """Test DRAM samples have correct mean for Gaussian target."""
        np.random.seed(123)

        def log_posterior(samples):
            samples_np = bkd.to_numpy(samples)
            return bkd.asarray(-0.5 * np.sum(samples_np**2, axis=0))

        sampler = DelayedRejectionAdaptiveMetropolis(
            log_posterior,
            nvars=2,
            bkd=bkd,
            adaptation_start=100,
            adaptation_interval=50,
        )

        result = sampler.sample(nsamples=500, burn=200)
        samples_np = bkd.to_numpy(result.samples)
        sample_mean = np.mean(samples_np, axis=1)

        # Mean should be close to 0
        np.testing.assert_array_less(np.abs(sample_mean), 0.3)
