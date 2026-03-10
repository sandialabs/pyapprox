"""
Tests comparing MCMC samples to known conjugate posterior analytics.

These tests validate MCMC samplers by comparing sample statistics to
exact analytical values from conjugate posteriors.
"""

import numpy as np

from pyapprox.inverse.conjugate.beta import BetaConjugatePosterior
from pyapprox.inverse.conjugate.dirichlet import DirichletConjugatePosterior
from pyapprox.inverse.conjugate.gaussian import DenseGaussianConjugatePosterior
from pyapprox.inverse.sampling import (
    AdaptiveMetropolisSampler,
    HamiltonianMonteCarlo,
    MetropolisHastingsSampler,
)
from pyapprox.util.test_utils import slow_test

#TODO: Should this be in mcmc module?


class TestBetaConjugateMCMC:
    """Test MCMC against exact Beta posterior."""

    def test_beta_bernoulli_metropolis(self, bkd) -> None:
        """Test Metropolis-Hastings matches Beta posterior mean/variance."""
        np.random.seed(42)

        # Generate Bernoulli data
        obs = bkd.asarray(np.array([[1, 1, 0, 1, 0, 1, 1, 0]]))
        # 5 successes, 3 failures

        # Analytical posterior using existing conjugate class
        conjugate = BetaConjugatePosterior(alpha=1.0, beta=1.0, bkd=bkd)
        conjugate.compute(obs)
        true_mean = conjugate.posterior_mean()
        true_var = conjugate.posterior_variance()
        alpha_post = conjugate.posterior_alpha()  # 1 + 5 = 6
        beta_post = conjugate.posterior_beta()  # 1 + 3 = 4

        # Define log posterior for MCMC (Beta(6, 4))
        def log_posterior(samples):
            samples_np = bkd.to_numpy(samples)
            p = samples_np[0, :]
            # Clip to avoid log(0)
            p = np.clip(p, 1e-10, 1 - 1e-10)
            # Beta log pdf: (a-1)log(p) + (b-1)log(1-p) - log(B(a,b))
            log_pdf = (alpha_post - 1) * np.log(p) + (beta_post - 1) * np.log(1 - p)
            return bkd.asarray(log_pdf)

        sampler = MetropolisHastingsSampler(
            log_posterior,
            nvars=1,
            bkd=bkd,
            proposal_cov=bkd.asarray(np.array([[0.02]])),
        )

        # Start near mode
        initial = bkd.asarray(np.array([[0.6]]))
        result = sampler.sample(nsamples=5000, burn=1000, initial_state=initial)
        samples_np = bkd.to_numpy(result.samples)

        # Clip samples to valid range
        samples_np = np.clip(samples_np, 1e-10, 1 - 1e-10)

        sample_mean = np.mean(samples_np)
        sample_var = np.var(samples_np)

        # Compare to analytical values
        assert np.abs(sample_mean - true_mean) < 0.05, (
            f"Sample mean {sample_mean:.4f} should be close to {true_mean:.4f}"
        )
        assert np.abs(sample_var - true_var) < 0.01, (
            f"Sample var {sample_var:.4f} should be close to {true_var:.4f}"
        )

    def test_beta_posterior_with_more_data(self, bkd) -> None:
        """Test with more observations for tighter posterior."""
        np.random.seed(123)

        # More data: 15 successes, 5 failures
        obs = bkd.asarray(
            np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])
        )

        conjugate = BetaConjugatePosterior(
            alpha=2.0, beta=2.0, bkd=bkd
        )  # Informative prior
        conjugate.compute(obs)
        true_mean = conjugate.posterior_mean()
        alpha_post = conjugate.posterior_alpha()  # 2 + 15 = 17
        beta_post = conjugate.posterior_beta()  # 2 + 5 = 7

        def log_posterior(samples):
            samples_np = bkd.to_numpy(samples)
            p = np.clip(samples_np[0, :], 1e-10, 1 - 1e-10)
            log_pdf = (alpha_post - 1) * np.log(p) + (beta_post - 1) * np.log(1 - p)
            return bkd.asarray(log_pdf)

        sampler = AdaptiveMetropolisSampler(
            log_posterior,
            nvars=1,
            bkd=bkd,
            adaptation_start=100,
            adaptation_interval=50,
        )

        initial = bkd.asarray(np.array([[0.7]]))
        result = sampler.sample(nsamples=3000, burn=500, initial_state=initial)
        samples_np = np.clip(bkd.to_numpy(result.samples), 1e-10, 1 - 1e-10)

        sample_mean = np.mean(samples_np)
        assert np.abs(sample_mean - true_mean) < 0.03, (
            f"Sample mean {sample_mean:.4f} should be close to {true_mean:.4f}"
        )


class TestDirichletConjugateMCMC:
    """Test MCMC against exact Dirichlet posterior."""

    def test_dirichlet_multinomial_metropolis(self, bkd) -> None:
        """Test Metropolis-Hastings matches Dirichlet posterior mean."""
        np.random.seed(456)

        # Category observations: 4 in cat 0, 3 in cat 1, 3 in cat 2
        obs = bkd.asarray(np.array([[0, 0, 0, 0, 1, 1, 1, 2, 2, 2]]))

        # Analytical posterior
        conjugate = DirichletConjugatePosterior(
            alphas=bkd.asarray(np.array([1.0, 1.0, 1.0])),
            bkd=bkd,
        )
        conjugate.compute(obs)
        true_mean = bkd.to_numpy(conjugate.posterior_mean())
        alphas_post = bkd.to_numpy(conjugate.posterior_alphas())
        # Should be [1+4, 1+3, 1+3] = [5, 4, 4]

        # Log Dirichlet PDF (on simplex)
        def log_posterior(samples):
            """Log Dirichlet pdf for K-1 dimensional samples (last dim implicit)."""
            samples_np = bkd.to_numpy(samples)
            # samples has shape (K-1, nsamples), last component is implicit
            p1 = samples_np[0, :]
            p2 = samples_np[1, :]
            p3 = 1 - p1 - p2  # Last component

            # Check if in simplex
            valid = (p1 > 0) & (p2 > 0) & (p3 > 0) & (p1 + p2 < 1)

            log_pdf = np.full(len(p1), -1e10)
            mask = valid
            if np.any(mask):
                log_pdf[mask] = (
                    (alphas_post[0] - 1) * np.log(np.clip(p1[mask], 1e-10, 1))
                    + (alphas_post[1] - 1) * np.log(np.clip(p2[mask], 1e-10, 1))
                    + (alphas_post[2] - 1) * np.log(np.clip(p3[mask], 1e-10, 1))
                )
            return bkd.asarray(log_pdf)

        sampler = MetropolisHastingsSampler(
            log_posterior,
            nvars=2,  # K-1 dimensions
            bkd=bkd,
            proposal_cov=bkd.asarray(np.array([[0.01, 0], [0, 0.01]])),
        )

        # Start near mode
        initial = bkd.asarray(np.array([[0.38], [0.31]]))
        result = sampler.sample(nsamples=5000, burn=1000, initial_state=initial)
        samples_np = bkd.to_numpy(result.samples)

        # Compute sample means for p1, p2, p3
        p1_mean = np.mean(samples_np[0, :])
        p2_mean = np.mean(samples_np[1, :])
        p3_mean = np.mean(1 - samples_np[0, :] - samples_np[1, :])

        # Compare to true Dirichlet means
        assert np.abs(p1_mean - true_mean[0]) < 0.05, (
            f"p1 mean {p1_mean:.3f} should be close to {true_mean[0]:.3f}"
        )
        assert np.abs(p2_mean - true_mean[1]) < 0.05, (
            f"p2 mean {p2_mean:.3f} should be close to {true_mean[1]:.3f}"
        )
        assert np.abs(p3_mean - true_mean[2]) < 0.05, (
            f"p3 mean {p3_mean:.3f} should be close to {true_mean[2]:.3f}"
        )


class TestGaussianConjugateMCMC:
    """Test MCMC against exact Gaussian posterior."""

    @slow_test
    def test_gaussian_conjugate_metropolis(self, bkd) -> None:
        """Test Metropolis-Hastings matches Gaussian posterior mean/cov."""
        np.random.seed(789)

        # Linear model: y = A @ x + noise
        A = bkd.asarray(np.array([[1.0, 0.5], [0.5, 1.0], [1.0, 1.0]]))
        prior_mean = bkd.asarray(np.zeros((2, 1)))
        prior_cov = bkd.asarray(np.eye(2) * 2.0)
        noise_cov = bkd.asarray(np.eye(3) * 0.1)

        # Generate synthetic data from true parameters
        true_params = np.array([[1.0], [0.5]])
        A_np = bkd.to_numpy(A)
        noise = np.random.randn(3, 1) * 0.316  # sqrt(0.1)
        obs = bkd.asarray(A_np @ true_params + noise)

        # Analytical posterior
        conjugate = DenseGaussianConjugatePosterior(
            A, prior_mean, prior_cov, noise_cov, bkd
        )
        conjugate.compute(obs)
        true_post_mean = bkd.to_numpy(conjugate.posterior_mean())
        true_post_cov = bkd.to_numpy(conjugate.posterior_covariance())

        # Define log posterior for MCMC
        prior_prec = bkd.to_numpy(bkd.inv(prior_cov))
        noise_prec = bkd.to_numpy(bkd.inv(noise_cov))
        obs_np = bkd.to_numpy(obs)

        def log_posterior(samples):
            samples_np = bkd.to_numpy(samples)
            nsamples = samples_np.shape[1]
            log_pdf = np.zeros(nsamples)

            for i in range(nsamples):
                x = samples_np[:, i : i + 1]
                # Log prior
                diff = x - bkd.to_numpy(prior_mean)
                log_prior = -0.5 * float((diff.T @ prior_prec @ diff)[0, 0])
                # Log likelihood
                residual = obs_np - A_np @ x
                log_like = -0.5 * float((residual.T @ noise_prec @ residual)[0, 0])
                log_pdf[i] = log_prior + log_like

            return bkd.asarray(log_pdf)

        def gradient(sample):
            """Gradient for HMC."""
            sample_np = bkd.to_numpy(sample)
            x = sample_np
            # Grad log prior
            grad_prior = -prior_prec @ (x - bkd.to_numpy(prior_mean))
            # Grad log likelihood
            residual = obs_np - A_np @ x
            grad_like = A_np.T @ noise_prec @ residual
            return bkd.asarray((grad_prior + grad_like).astype(np.float64))

        # Use HMC for better mixing
        sampler = HamiltonianMonteCarlo(
            log_posterior,
            gradient,
            nvars=2,
            bkd=bkd,
            step_size=0.1,
            num_leapfrog_steps=10,
        )

        result = sampler.sample(nsamples=2000, burn=500)
        samples_np = bkd.to_numpy(result.samples)

        sample_mean = np.mean(samples_np, axis=1, keepdims=True)
        sample_cov = np.cov(samples_np)

        # Compare to analytical posterior
        np.testing.assert_array_less(
            np.abs(sample_mean - true_post_mean),
            0.3,
            "Sample mean should be close to analytical posterior mean",
        )
        np.testing.assert_array_less(
            np.abs(sample_cov - true_post_cov),
            0.3,
            "Sample covariance should be close to analytical posterior covariance",
        )

    def test_gaussian_conjugate_adaptive(self, bkd) -> None:
        """Test Adaptive Metropolis matches Gaussian posterior."""
        np.random.seed(321)

        # Simple 1D case
        A = bkd.asarray(np.array([[1.0]]))
        prior_mean = bkd.asarray(np.zeros((1, 1)))
        prior_cov = bkd.asarray(np.array([[4.0]]))  # Prior variance = 4
        noise_cov = bkd.asarray(np.array([[1.0]]))  # Noise variance = 1

        # Observe y = 2.0
        obs = bkd.asarray(np.array([[2.0]]))

        conjugate = DenseGaussianConjugatePosterior(
            A, prior_mean, prior_cov, noise_cov, bkd
        )
        conjugate.compute(obs)
        true_mean = float(bkd.to_numpy(conjugate.posterior_mean())[0, 0])
        true_var = float(bkd.to_numpy(conjugate.posterior_covariance())[0, 0])

        # For 1D: posterior precision = prior_prec + noise_prec = 0.25 + 1 = 1.25
        # posterior variance = 1/1.25 = 0.8
        # posterior mean = 0.8 * (1 * 2 / 1) = 1.6

        def log_posterior(samples):
            samples_np = bkd.to_numpy(samples)
            x = samples_np[0, :]
            # Prior: N(0, 4)
            log_prior = -0.5 * x**2 / 4.0
            # Likelihood: N(x, 1) at y=2
            log_like = -0.5 * (x - 2.0) ** 2 / 1.0
            return bkd.asarray(log_prior + log_like)

        sampler = AdaptiveMetropolisSampler(
            log_posterior,
            nvars=1,
            bkd=bkd,
            adaptation_start=100,
            adaptation_interval=50,
        )

        result = sampler.sample(nsamples=3000, burn=500)
        samples_np = bkd.to_numpy(result.samples)

        sample_mean = np.mean(samples_np)
        sample_var = np.var(samples_np)

        assert np.abs(sample_mean - true_mean) < 0.1, (
            f"Sample mean {sample_mean:.3f} should be close to {true_mean:.3f}"
        )
        assert np.abs(sample_var - true_var) < 0.1, (
            f"Sample var {sample_var:.3f} should be close to {true_var:.3f}"
        )
