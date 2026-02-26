"""
Tests for Hamiltonian Monte Carlo sampler.
"""

import pytest

import numpy as np

from pyapprox.inverse.sampling.hmc import HamiltonianMonteCarlo


class TestHMCBase:
    """Base test class for HamiltonianMonteCarlo."""

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

        def gradient(sample):
            return -sample

        sampler = HamiltonianMonteCarlo(
            log_posterior,
            gradient,
            nvars,
            bkd,
            step_size=0.1,
            num_leapfrog_steps=10,
        )
        return sampler, nvars

    def test_nvars(self, bkd) -> None:
        """Test nvars returns correct value."""
        sampler, nvars = self._make_sampler(bkd)
        assert sampler.nvars() == nvars

    def test_sample_returns_correct_shape(self, bkd) -> None:
        """Test sample returns correct shape."""
        sampler, nvars = self._make_sampler(bkd)
        nsamples = 50
        result = sampler.sample(nsamples)
        assert result.samples.shape == (nvars, nsamples)

    def test_sample_with_burn(self, bkd) -> None:
        """Test sampling with burn-in."""
        sampler, nvars = self._make_sampler(bkd)
        nsamples = 50
        burn = 20
        result = sampler.sample(nsamples, burn=burn)
        assert result.samples.shape == (nvars, nsamples)

    def test_sample_with_initial_state(self, bkd) -> None:
        """Test sampling with initial state."""
        sampler, nvars = self._make_sampler(bkd)
        initial = bkd.asarray(np.array([[1.0], [1.0]], dtype=np.float64))
        result = sampler.sample(30, initial_state=initial)
        assert result.samples.shape == (nvars, 30)

    def test_acceptance_rate_in_range(self, bkd) -> None:
        """Test acceptance rate is in [0, 1]."""
        sampler, _ = self._make_sampler(bkd)
        result = sampler.sample(100)
        assert result.acceptance_rate >= 0.0
        assert result.acceptance_rate <= 1.0

    def test_log_posteriors_shape(self, bkd) -> None:
        """Test log posteriors have correct shape."""
        sampler, _ = self._make_sampler(bkd)
        nsamples = 50
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
        assert "HamiltonianMonteCarlo" in repr_str
        assert "nvars=2" in repr_str


class TestHMCConvergence:
    """Test HMC convergence on simple targets."""

    def test_gaussian_target_mean(self, bkd) -> None:
        """Test HMC samples have correct mean for Gaussian target."""
        np.random.seed(123)

        def log_posterior(samples):
            samples_np = bkd.to_numpy(samples)
            return bkd.asarray(-0.5 * np.sum(samples_np**2, axis=0))

        def gradient(sample):
            return -sample

        sampler = HamiltonianMonteCarlo(
            log_posterior,
            gradient,
            nvars=2,
            bkd=bkd,
            step_size=0.15,
            num_leapfrog_steps=15,
        )

        result = sampler.sample(nsamples=500, burn=100)
        samples_np = bkd.to_numpy(result.samples)
        sample_mean = np.mean(samples_np, axis=1)

        # Mean should be close to 0
        np.testing.assert_array_less(np.abs(sample_mean), 0.3)
