"""
Tests comparing MCMC samples to known conjugate posterior analytics.

These tests validate MCMC samplers by comparing sample statistics to
exact analytical values from conjugate posteriors.
"""

import unittest
from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray
import torch
from scipy.special import gammaln

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.inverse.sampling import (
    MetropolisHastingsSampler,
    AdaptiveMetropolisSampler,
    HamiltonianMonteCarlo,
)
from pyapprox.inverse.conjugate.beta import BetaConjugatePosterior
from pyapprox.inverse.conjugate.dirichlet import DirichletConjugatePosterior
from pyapprox.inverse.conjugate.gaussian import DenseGaussianConjugatePosterior


class TestBetaConjugateMCMC(Generic[Array], unittest.TestCase):
    """Test MCMC against exact Beta posterior."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_beta_bernoulli_metropolis(self) -> None:
        """Test Metropolis-Hastings matches Beta posterior mean/variance."""
        np.random.seed(42)

        # Generate Bernoulli data
        obs = self.bkd().asarray(np.array([[1, 1, 0, 1, 0, 1, 1, 0]]))
        # 5 successes, 3 failures

        # Analytical posterior using existing conjugate class
        conjugate = BetaConjugatePosterior(
            alpha=1.0, beta=1.0, bkd=self.bkd()
        )
        conjugate.compute(obs)
        true_mean = conjugate.posterior_mean()
        true_var = conjugate.posterior_variance()
        alpha_post = conjugate.posterior_alpha()  # 1 + 5 = 6
        beta_post = conjugate.posterior_beta()    # 1 + 3 = 4

        # Define log posterior for MCMC (Beta(6, 4))
        def log_posterior(samples: Array) -> Array:
            samples_np = self.bkd().to_numpy(samples)
            p = samples_np[0, :]
            # Clip to avoid log(0)
            p = np.clip(p, 1e-10, 1 - 1e-10)
            # Beta log pdf: (a-1)log(p) + (b-1)log(1-p) - log(B(a,b))
            log_pdf = (
                (alpha_post - 1) * np.log(p)
                + (beta_post - 1) * np.log(1 - p)
            )
            return self.bkd().asarray(log_pdf)

        sampler = MetropolisHastingsSampler(
            log_posterior,
            nvars=1,
            bkd=self.bkd(),
            proposal_cov=self.bkd().asarray(np.array([[0.02]])),
        )

        # Start near mode
        initial = self.bkd().asarray(np.array([[0.6]]))
        result = sampler.sample(nsamples=5000, burn=1000, initial_state=initial)
        samples_np = self.bkd().to_numpy(result.samples)

        # Clip samples to valid range
        samples_np = np.clip(samples_np, 1e-10, 1 - 1e-10)

        sample_mean = np.mean(samples_np)
        sample_var = np.var(samples_np)

        # Compare to analytical values
        self.assertLess(
            np.abs(sample_mean - true_mean), 0.05,
            f"Sample mean {sample_mean:.4f} should be close to {true_mean:.4f}"
        )
        self.assertLess(
            np.abs(sample_var - true_var), 0.01,
            f"Sample var {sample_var:.4f} should be close to {true_var:.4f}"
        )

    def test_beta_posterior_with_more_data(self) -> None:
        """Test with more observations for tighter posterior."""
        np.random.seed(123)

        # More data: 15 successes, 5 failures
        obs = self.bkd().asarray(np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        ]))

        conjugate = BetaConjugatePosterior(
            alpha=2.0, beta=2.0, bkd=self.bkd()
        )  # Informative prior
        conjugate.compute(obs)
        true_mean = conjugate.posterior_mean()
        alpha_post = conjugate.posterior_alpha()  # 2 + 15 = 17
        beta_post = conjugate.posterior_beta()    # 2 + 5 = 7

        def log_posterior(samples: Array) -> Array:
            samples_np = self.bkd().to_numpy(samples)
            p = np.clip(samples_np[0, :], 1e-10, 1 - 1e-10)
            log_pdf = (
                (alpha_post - 1) * np.log(p)
                + (beta_post - 1) * np.log(1 - p)
            )
            return self.bkd().asarray(log_pdf)

        sampler = AdaptiveMetropolisSampler(
            log_posterior,
            nvars=1,
            bkd=self.bkd(),
            adaptation_start=100,
            adaptation_interval=50,
        )

        initial = self.bkd().asarray(np.array([[0.7]]))
        result = sampler.sample(nsamples=3000, burn=500, initial_state=initial)
        samples_np = np.clip(self.bkd().to_numpy(result.samples), 1e-10, 1 - 1e-10)

        sample_mean = np.mean(samples_np)
        self.assertLess(
            np.abs(sample_mean - true_mean), 0.03,
            f"Sample mean {sample_mean:.4f} should be close to {true_mean:.4f}"
        )


class TestDirichletConjugateMCMC(Generic[Array], unittest.TestCase):
    """Test MCMC against exact Dirichlet posterior."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_dirichlet_multinomial_metropolis(self) -> None:
        """Test Metropolis-Hastings matches Dirichlet posterior mean."""
        np.random.seed(456)

        # Category observations: 4 in cat 0, 3 in cat 1, 3 in cat 2
        obs = self.bkd().asarray(np.array([[0, 0, 0, 0, 1, 1, 1, 2, 2, 2]]))

        # Analytical posterior
        conjugate = DirichletConjugatePosterior(
            alphas=self.bkd().asarray(np.array([1.0, 1.0, 1.0])),
            bkd=self.bkd(),
        )
        conjugate.compute(obs)
        true_mean = self.bkd().to_numpy(conjugate.posterior_mean())
        alphas_post = self.bkd().to_numpy(conjugate.posterior_alphas())
        # Should be [1+4, 1+3, 1+3] = [5, 4, 4]

        # Log Dirichlet PDF (on simplex)
        def log_posterior(samples: Array) -> Array:
            """Log Dirichlet pdf for K-1 dimensional samples (last dim implicit)."""
            samples_np = self.bkd().to_numpy(samples)
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
            return self.bkd().asarray(log_pdf)

        sampler = MetropolisHastingsSampler(
            log_posterior,
            nvars=2,  # K-1 dimensions
            bkd=self.bkd(),
            proposal_cov=self.bkd().asarray(np.array([[0.01, 0], [0, 0.01]])),
        )

        # Start near mode
        initial = self.bkd().asarray(np.array([[0.38], [0.31]]))
        result = sampler.sample(nsamples=5000, burn=1000, initial_state=initial)
        samples_np = self.bkd().to_numpy(result.samples)

        # Compute sample means for p1, p2, p3
        p1_mean = np.mean(samples_np[0, :])
        p2_mean = np.mean(samples_np[1, :])
        p3_mean = np.mean(1 - samples_np[0, :] - samples_np[1, :])

        # Compare to true Dirichlet means
        self.assertLess(
            np.abs(p1_mean - true_mean[0]), 0.05,
            f"p1 mean {p1_mean:.3f} should be close to {true_mean[0]:.3f}"
        )
        self.assertLess(
            np.abs(p2_mean - true_mean[1]), 0.05,
            f"p2 mean {p2_mean:.3f} should be close to {true_mean[1]:.3f}"
        )
        self.assertLess(
            np.abs(p3_mean - true_mean[2]), 0.05,
            f"p3 mean {p3_mean:.3f} should be close to {true_mean[2]:.3f}"
        )


class TestGaussianConjugateMCMC(Generic[Array], unittest.TestCase):
    """Test MCMC against exact Gaussian posterior."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_gaussian_conjugate_metropolis(self) -> None:
        """Test Metropolis-Hastings matches Gaussian posterior mean/cov."""
        np.random.seed(789)

        # Linear model: y = A @ x + noise
        A = self.bkd().asarray(np.array([[1.0, 0.5], [0.5, 1.0], [1.0, 1.0]]))
        prior_mean = self.bkd().asarray(np.zeros((2, 1)))
        prior_cov = self.bkd().asarray(np.eye(2) * 2.0)
        noise_cov = self.bkd().asarray(np.eye(3) * 0.1)

        # Generate synthetic data from true parameters
        true_params = np.array([[1.0], [0.5]])
        A_np = self.bkd().to_numpy(A)
        noise = np.random.randn(3, 1) * 0.316  # sqrt(0.1)
        obs = self.bkd().asarray(A_np @ true_params + noise)

        # Analytical posterior
        conjugate = DenseGaussianConjugatePosterior(
            A, prior_mean, prior_cov, noise_cov, self.bkd()
        )
        conjugate.compute(obs)
        true_post_mean = self.bkd().to_numpy(conjugate.posterior_mean())
        true_post_cov = self.bkd().to_numpy(conjugate.posterior_covariance())

        # Define log posterior for MCMC
        prior_prec = self.bkd().to_numpy(self.bkd().inv(prior_cov))
        noise_prec = self.bkd().to_numpy(self.bkd().inv(noise_cov))
        obs_np = self.bkd().to_numpy(obs)

        def log_posterior(samples: Array) -> Array:
            samples_np = self.bkd().to_numpy(samples)
            nsamples = samples_np.shape[1]
            log_pdf = np.zeros(nsamples)

            for i in range(nsamples):
                x = samples_np[:, i:i+1]
                # Log prior
                diff = x - self.bkd().to_numpy(prior_mean)
                log_prior = -0.5 * float(diff.T @ prior_prec @ diff)
                # Log likelihood
                residual = obs_np - A_np @ x
                log_like = -0.5 * float(residual.T @ noise_prec @ residual)
                log_pdf[i] = log_prior + log_like

            return self.bkd().asarray(log_pdf)

        def gradient(sample: Array) -> Array:
            """Gradient for HMC."""
            sample_np = self.bkd().to_numpy(sample)
            x = sample_np
            # Grad log prior
            grad_prior = -prior_prec @ (x - self.bkd().to_numpy(prior_mean))
            # Grad log likelihood
            residual = obs_np - A_np @ x
            grad_like = A_np.T @ noise_prec @ residual
            return self.bkd().asarray((grad_prior + grad_like).astype(np.float64))

        # Use HMC for better mixing
        sampler = HamiltonianMonteCarlo(
            log_posterior,
            gradient,
            nvars=2,
            bkd=self.bkd(),
            step_size=0.1,
            num_leapfrog_steps=10,
        )

        result = sampler.sample(nsamples=2000, burn=500)
        samples_np = self.bkd().to_numpy(result.samples)

        sample_mean = np.mean(samples_np, axis=1, keepdims=True)
        sample_cov = np.cov(samples_np)

        # Compare to analytical posterior
        np.testing.assert_array_less(
            np.abs(sample_mean - true_post_mean), 0.3,
            "Sample mean should be close to analytical posterior mean"
        )
        np.testing.assert_array_less(
            np.abs(sample_cov - true_post_cov), 0.3,
            "Sample covariance should be close to analytical posterior covariance"
        )

    def test_gaussian_conjugate_adaptive(self) -> None:
        """Test Adaptive Metropolis matches Gaussian posterior."""
        np.random.seed(321)

        # Simple 1D case
        A = self.bkd().asarray(np.array([[1.0]]))
        prior_mean = self.bkd().asarray(np.zeros((1, 1)))
        prior_cov = self.bkd().asarray(np.array([[4.0]]))  # Prior variance = 4
        noise_cov = self.bkd().asarray(np.array([[1.0]]))  # Noise variance = 1

        # Observe y = 2.0
        obs = self.bkd().asarray(np.array([[2.0]]))

        conjugate = DenseGaussianConjugatePosterior(
            A, prior_mean, prior_cov, noise_cov, self.bkd()
        )
        conjugate.compute(obs)
        true_mean = float(self.bkd().to_numpy(conjugate.posterior_mean())[0, 0])
        true_var = float(self.bkd().to_numpy(conjugate.posterior_covariance())[0, 0])

        # For 1D: posterior precision = prior_prec + noise_prec = 0.25 + 1 = 1.25
        # posterior variance = 1/1.25 = 0.8
        # posterior mean = 0.8 * (1 * 2 / 1) = 1.6

        def log_posterior(samples: Array) -> Array:
            samples_np = self.bkd().to_numpy(samples)
            x = samples_np[0, :]
            # Prior: N(0, 4)
            log_prior = -0.5 * x**2 / 4.0
            # Likelihood: N(x, 1) at y=2
            log_like = -0.5 * (x - 2.0)**2 / 1.0
            return self.bkd().asarray(log_prior + log_like)

        sampler = AdaptiveMetropolisSampler(
            log_posterior,
            nvars=1,
            bkd=self.bkd(),
            adaptation_start=100,
            adaptation_interval=50,
        )

        result = sampler.sample(nsamples=3000, burn=500)
        samples_np = self.bkd().to_numpy(result.samples)

        sample_mean = np.mean(samples_np)
        sample_var = np.var(samples_np)

        self.assertLess(
            np.abs(sample_mean - true_mean), 0.1,
            f"Sample mean {sample_mean:.3f} should be close to {true_mean:.3f}"
        )
        self.assertLess(
            np.abs(sample_var - true_var), 0.1,
            f"Sample var {sample_var:.3f} should be close to {true_var:.3f}"
        )


# NumPy backend tests
class TestBetaConjugateMCMCNumpy(TestBetaConjugateMCMC[NDArray[Any]]):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestDirichletConjugateMCMCNumpy(TestDirichletConjugateMCMC[NDArray[Any]]):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestGaussianConjugateMCMCNumpy(TestGaussianConjugateMCMC[NDArray[Any]]):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# PyTorch backend tests
class TestBetaConjugateMCMCTorch(TestBetaConjugateMCMC[torch.Tensor]):
    __test__ = True

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()

    def bkd(self) -> TorchBkd:
        return self._bkd


class TestDirichletConjugateMCMCTorch(TestDirichletConjugateMCMC[torch.Tensor]):
    __test__ = True

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()

    def bkd(self) -> TorchBkd:
        return self._bkd


class TestGaussianConjugateMCMCTorch(TestGaussianConjugateMCMC[torch.Tensor]):
    __test__ = True

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()

    def bkd(self) -> TorchBkd:
        return self._bkd


from pyapprox.util.test_utils import load_tests


if __name__ == "__main__":
    unittest.main()
