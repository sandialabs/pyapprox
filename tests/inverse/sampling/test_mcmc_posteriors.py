"""
Tests comparing MCMC samples to known analytical posterior statistics.

These tests validate that MCMC samplers correctly sample from distributions
with known analytical properties.
"""

import numpy as np
import pytest

from pyapprox.inverse.sampling import (
    AdaptiveMetropolisSampler,
    DelayedRejectionAdaptiveMetropolis,
    HamiltonianMonteCarlo,
    MetropolisHastingsSampler,
    effective_sample_size,
    rhat,
)
from tests._helpers.sampling_distributions import (
    BananaLogPosterior,
    CorrelatedGaussianLogPosterior,
    GaussianMixtureLogPosterior,
)
from pyapprox.util.test_utils import (
    slower_test,
)

# TODO: Should this be split into test specific to samplers, such files already exist
# TODO: we should be using bkd.assert_allclose and other bkd generic functions
# not functions from np.testing. This will avoid a lot of unneeded
# bkd.to_numpy conversions. Use bkd.all_bool instead of np.all

class TestMCMCPosteriorBase:
    """Base class for MCMC posterior validation tests."""

    def test_correlated_gaussian_metropolis(self, bkd) -> None:
        """Test Metropolis-Hastings on correlated Gaussian target."""
        np.random.seed(42)

        mean = bkd.asarray(np.array([1.0, -1.0]))
        cov = bkd.asarray(np.array([[1.0, 0.5], [0.5, 2.0]]))
        target = CorrelatedGaussianLogPosterior(bkd, mean, cov)

        sampler = MetropolisHastingsSampler(
            target,
            nvars=2,
            bkd=bkd,
            proposal_cov=bkd.asarray(np.array([[0.5, 0.0], [0.0, 0.5]])),
        )

        result = sampler.sample(nsamples=2000, burn=500)
        samples_np = bkd.to_numpy(result.samples)

        # Check mean
        sample_mean = np.mean(samples_np, axis=1)
        true_mean = target.true_mean()
        np.testing.assert_array_less(
            np.abs(sample_mean - true_mean),
            0.3,
            "Sample mean should be close to true mean",
        )

    def test_correlated_gaussian_adaptive(self, bkd) -> None:
        """Test Adaptive Metropolis on correlated Gaussian target."""
        np.random.seed(123)

        mean = bkd.asarray(np.array([0.5, 0.5]))
        cov = bkd.asarray(np.array([[1.0, -0.3], [-0.3, 1.0]]))
        target = CorrelatedGaussianLogPosterior(bkd, mean, cov)

        sampler = AdaptiveMetropolisSampler(
            target,
            nvars=2,
            bkd=bkd,
            adaptation_start=100,
            adaptation_interval=50,
        )

        result = sampler.sample(nsamples=3000, burn=1000)
        samples_np = bkd.to_numpy(result.samples)

        # Check mean
        sample_mean = np.mean(samples_np, axis=1)
        true_mean = target.true_mean()
        np.testing.assert_array_less(
            np.abs(sample_mean - true_mean),
            0.2,
            "Sample mean should be close to true mean",
        )

        # Check covariance (more relaxed tolerance)
        sample_cov = np.cov(samples_np)
        true_cov = target.true_covariance()
        np.testing.assert_array_less(
            np.abs(sample_cov - true_cov),
            0.4,
            "Sample covariance should be close to true covariance",
        )

    def test_correlated_gaussian_hmc(self, bkd) -> None:
        """Test HMC on correlated Gaussian target."""
        np.random.seed(456)

        mean = bkd.asarray(np.array([0.0, 0.0]))
        cov = bkd.asarray(np.array([[1.0, 0.8], [0.8, 1.0]]))
        target = CorrelatedGaussianLogPosterior(bkd, mean, cov)

        sampler = HamiltonianMonteCarlo(
            target,
            target.gradient,
            nvars=2,
            bkd=bkd,
            step_size=0.15,
            num_leapfrog_steps=15,
        )

        result = sampler.sample(nsamples=1000, burn=200)
        samples_np = bkd.to_numpy(result.samples)

        # HMC should achieve good acceptance rate
        assert result.acceptance_rate > 0.5

        # Check mean
        sample_mean = np.mean(samples_np, axis=1)
        true_mean = target.true_mean()
        np.testing.assert_array_less(
            np.abs(sample_mean - true_mean),
            0.2,
            "HMC sample mean should be close to true mean",
        )

    def test_correlated_gaussian_dram(self, bkd) -> None:
        """Test DRAM on correlated Gaussian target."""
        np.random.seed(789)

        mean = bkd.asarray(np.array([1.0, 1.0]))
        cov = bkd.asarray(np.array([[2.0, 0.5], [0.5, 1.0]]))
        target = CorrelatedGaussianLogPosterior(bkd, mean, cov)

        sampler = DelayedRejectionAdaptiveMetropolis(
            target,
            nvars=2,
            bkd=bkd,
            adaptation_start=100,
            adaptation_interval=50,
            dr_scale=0.1,
        )

        result = sampler.sample(nsamples=2000, burn=500)
        samples_np = bkd.to_numpy(result.samples)

        # Check mean
        sample_mean = np.mean(samples_np, axis=1)
        true_mean = target.true_mean()
        np.testing.assert_array_less(
            np.abs(sample_mean - true_mean),
            0.3,
            "DRAM sample mean should be close to true mean",
        )

    @slower_test
    def test_banana_distribution_hmc(self, bkd) -> None:
        """Test HMC on banana-shaped (non-Gaussian) distribution."""
        np.random.seed(111)

        target = BananaLogPosterior(bkd, mu0=0.0, mu1=0.0, s0=1.0, s1=1.0)

        sampler = HamiltonianMonteCarlo(
            target,
            target.gradient,
            nvars=2,
            bkd=bkd,
            step_size=0.1,
            num_leapfrog_steps=20,
        )

        result = sampler.sample(nsamples=2000, burn=500)
        samples_np = bkd.to_numpy(result.samples)

        # Check means against analytical values
        sample_mean = np.mean(samples_np, axis=1)
        true_mean = target.true_mean()

        # x0 marginal is Gaussian with mean mu0
        assert np.abs(sample_mean[0] - true_mean[0]) < 0.3, (
            "x0 mean should be close to mu0"
        )

        # x1 marginal has mean s0^2 + mu0^2 + mu1
        assert np.abs(sample_mean[1] - true_mean[1]) < 0.5, (
            "x1 mean should be close to s0^2 + mu0^2 + mu1"
        )

    def test_banana_distribution_dram(self, bkd) -> None:
        """Test DRAM on banana-shaped distribution."""
        np.random.seed(222)

        target = BananaLogPosterior(bkd, mu0=0.0, mu1=0.0, s0=1.0, s1=0.5)

        sampler = DelayedRejectionAdaptiveMetropolis(
            target,
            nvars=2,
            bkd=bkd,
            adaptation_start=200,
            adaptation_interval=100,
            dr_scale=0.1,
        )

        result = sampler.sample(nsamples=3000, burn=1000)
        samples_np = bkd.to_numpy(result.samples)

        sample_mean = np.mean(samples_np, axis=1)
        true_mean = target.true_mean()

        # Check x0 mean (should be mu0=0)
        assert np.abs(sample_mean[0] - true_mean[0]) < 0.3

    def test_gaussian_mixture_adaptive(self, bkd) -> None:
        """Test Adaptive Metropolis on bimodal Gaussian mixture."""
        np.random.seed(333)

        # Use modes that are not too far apart for easier mixing
        target = GaussianMixtureLogPosterior(bkd, mu1=-1.5, mu2=1.5, sigma=1.0)

        sampler = AdaptiveMetropolisSampler(
            target,
            nvars=1,
            bkd=bkd,
            adaptation_start=100,
            adaptation_interval=50,
        )

        # Need more samples for bimodal
        result = sampler.sample(nsamples=5000, burn=1000)
        samples_np = bkd.to_numpy(result.samples)

        sample_mean = np.mean(samples_np)
        true_mean = target.true_mean()

        # Mean should be close to 0 (average of -1.5 and 1.5)
        assert np.abs(sample_mean - true_mean) < 0.5, (
            "Sample mean should be close to true mean for mixture"
        )

    @pytest.mark.slow_on("TorchBkd")
    def test_diagnostics_on_converged_chain(self, bkd) -> None:
        """Test that diagnostics indicate convergence for well-tuned sampler."""
        np.random.seed(444)

        mean = bkd.asarray(np.array([0.0, 0.0]))
        cov = bkd.asarray(np.array([[1.0, 0.0], [0.0, 1.0]]))
        target = CorrelatedGaussianLogPosterior(bkd, mean, cov)

        sampler = HamiltonianMonteCarlo(
            target,
            target.gradient,
            nvars=2,
            bkd=bkd,
            step_size=0.2,
            num_leapfrog_steps=10,
        )

        result = sampler.sample(nsamples=1000, burn=200)

        # ESS should be reasonable for HMC
        ess = effective_sample_size(result.samples, bkd)
        ess_np = bkd.to_numpy(ess)

        # ESS should be at least 10% of nsamples for HMC
        assert np.all(ess_np > 100), f"ESS should be > 100, got {ess_np}"

    def test_multiple_chains_rhat(self, bkd) -> None:
        """Test R-hat diagnostic with multiple chains."""
        np.random.seed(555)

        mean = bkd.asarray(np.array([0.0]))
        cov = bkd.asarray(np.array([[1.0]]))
        target = CorrelatedGaussianLogPosterior(bkd, mean, cov)

        # Run multiple chains from different starting points
        chains = []
        for start in [-2.0, 0.0, 2.0]:
            sampler = MetropolisHastingsSampler(
                target,
                nvars=1,
                bkd=bkd,
                proposal_cov=bkd.asarray(np.array([[0.5]])),
            )
            initial = bkd.asarray(np.array([[start]]))
            result = sampler.sample(nsamples=1000, burn=200, initial_state=initial)
            chains.append(result.samples)

        # Stack chains
        chains_array = bkd.asarray(
            np.stack([bkd.to_numpy(c) for c in chains], axis=0)
        )

        # Compute R-hat
        r_hat = rhat(chains_array, bkd)
        r_hat_np = bkd.to_numpy(r_hat)

        # R-hat should be close to 1 for converged chains
        assert np.all(r_hat_np < 1.2), f"R-hat should be < 1.2, got {r_hat_np}"
