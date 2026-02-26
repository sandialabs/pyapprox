"""
Tests for Metropolis-Hastings MCMC samplers.
"""

import pytest

import numpy as np

from pyapprox.inverse.sampling import (
    AdaptiveMetropolisSampler,
    MetropolisHastingsSampler,
)


class TestMetropolisHastingsSamplerBase:
    """Base test class for MetropolisHastingsSampler."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _make_sampler(self, bkd):
        """Create sampler for tests."""
        nvars = 2

        # Simple Gaussian target: log p(x) = -0.5 * x^T @ x
        def log_posterior(samples):
            samples_np = bkd.to_numpy(samples)
            return bkd.asarray(-0.5 * np.sum(samples_np**2, axis=0))

        sampler = MetropolisHastingsSampler(log_posterior, nvars, bkd)
        return sampler, nvars

    def test_nvars(self, bkd) -> None:
        """Test nvars returns correct value."""
        sampler, nvars = self._make_sampler(bkd)
        assert sampler.nvars() == nvars

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

    def test_acceptance_rate_in_range(self, bkd) -> None:
        """Test acceptance rate is in [0, 1]."""
        sampler, _ = self._make_sampler(bkd)
        result = sampler.sample(200)
        assert result.acceptance_rate >= 0.0
        assert result.acceptance_rate <= 1.0

    def test_log_posteriors_shape(self, bkd) -> None:
        """Test log posteriors have correct shape."""
        sampler, _ = self._make_sampler(bkd)
        nsamples = 100
        result = sampler.sample(nsamples)
        assert result.log_posteriors.shape == (nsamples,)

    def test_samples_near_target_mean(self, bkd) -> None:
        """Test that samples are roughly centered at zero for standard normal."""
        sampler, _ = self._make_sampler(bkd)
        np.random.seed(123)
        result = sampler.sample(500, burn=100)
        samples_np = bkd.to_numpy(result.samples)
        sample_mean = np.mean(samples_np, axis=1)
        # Should be close to zero (target mean)
        np.testing.assert_array_less(np.abs(sample_mean), 1.0)

    def test_set_proposal_covariance(self, bkd) -> None:
        """Test setting proposal covariance."""
        sampler, _ = self._make_sampler(bkd)
        new_cov = bkd.asarray(
            np.array([[0.5, 0.0], [0.0, 0.5]], dtype=np.float64)
        )
        sampler.set_proposal_covariance(new_cov)
        cov = sampler.proposal_covariance()
        cov_np = bkd.to_numpy(cov)
        np.testing.assert_array_almost_equal(cov_np, np.array([[0.5, 0.0], [0.0, 0.5]]))

    def test_wrong_proposal_cov_shape_raises(self, bkd) -> None:
        """Test wrong proposal covariance shape raises error."""
        sampler, _ = self._make_sampler(bkd)
        wrong_cov = bkd.asarray(np.eye(3))
        with pytest.raises(ValueError):
            sampler.set_proposal_covariance(wrong_cov)

    def test_wrong_initial_state_shape_raises(self, bkd) -> None:
        """Test wrong initial state shape raises error."""
        sampler, _ = self._make_sampler(bkd)
        wrong_initial = bkd.asarray(np.zeros((3, 1)))
        with pytest.raises(ValueError):
            sampler.sample(100, initial_state=wrong_initial)


class TestAdaptiveMetropolisSamplerBase:
    """Base test class for AdaptiveMetropolisSampler."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _make_sampler(self, bkd):
        """Create sampler for tests."""
        nvars = 2

        # Simple Gaussian target
        def log_posterior(samples):
            samples_np = bkd.to_numpy(samples)
            return bkd.asarray(-0.5 * np.sum(samples_np**2, axis=0))

        sampler = AdaptiveMetropolisSampler(
            log_posterior,
            nvars,
            bkd,
            adaptation_start=50,
            adaptation_interval=25,
        )
        return sampler, nvars

    def test_sample_returns_correct_shape(self, bkd) -> None:
        """Test adaptive sampler returns correct shape."""
        sampler, nvars = self._make_sampler(bkd)
        nsamples = 200
        result = sampler.sample(nsamples)
        assert result.samples.shape == (nvars, nsamples)

    def test_acceptance_rate_reasonable(self, bkd) -> None:
        """Test adaptive sampler achieves reasonable acceptance rate."""
        sampler, _ = self._make_sampler(bkd)
        np.random.seed(456)
        result = sampler.sample(500, burn=100)
        # Adaptive should achieve reasonable acceptance (typically 0.1-0.8)
        assert result.acceptance_rate > 0.05

    def test_repr(self, bkd) -> None:
        """Test string representation."""
        sampler, _ = self._make_sampler(bkd)
        repr_str = repr(sampler)
        assert "AdaptiveMetropolisSampler" in repr_str
        assert "nvars=2" in repr_str


class TestMetropolisWithBounds:
    """Test sampling with parameter bounds."""

    def test_samples_within_bounds(self, bkd) -> None:
        """Test samples respect parameter bounds."""
        np.random.seed(789)
        nvars = 2

        def log_posterior(samples):
            samples_np = bkd.to_numpy(samples)
            return bkd.asarray(-0.5 * np.sum(samples_np**2, axis=0))

        sampler = MetropolisHastingsSampler(log_posterior, nvars, bkd)

        # Bounds: [-2, 2] for both variables
        bounds = bkd.asarray(
            np.array([[-2.0, 2.0], [-2.0, 2.0]], dtype=np.float64)
        )
        result = sampler.sample(200, bounds=bounds)

        samples_np = bkd.to_numpy(result.samples)
        assert np.all(samples_np >= -2.0)
        assert np.all(samples_np <= 2.0)
