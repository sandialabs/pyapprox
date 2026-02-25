"""
Standalone tests comparing MC estimates to exact analytical formulas
for conjugate Gaussian OED.

PERMANENT - no legacy imports.

Tests verify correctness by:
1. Computing MC estimates by sampling from data distribution
2. Comparing to exact analytical formulas
3. Verifying convergence within statistical tolerance

These tests are marked as slow since they require many MC samples.
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.test_utils import load_tests, slow_test, slower_test  # noqa: F401

from pyapprox.expdesign.analytical import (
    ConjugateGaussianOEDExpectedStdDev,
    ConjugateGaussianOEDExpectedEntropicDev,
    ConjugateGaussianOEDExpectedAVaRDev,
    ConjugateGaussianOEDExpectedKLDivergence,
    ConjugateGaussianOEDForLogNormalExpectedStdDev,
    ConjugateGaussianOEDForLogNormalAVaRStdDev,
)
from pyapprox.inverse.conjugate.gaussian import DenseGaussianConjugatePosterior
from pyapprox.inverse.pushforward.gaussian import GaussianPushforward


class TestConjugateMCExactStandalone(Generic[Array], unittest.TestCase):
    """MC vs exact analytical formula comparison tests."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        np.random.seed(42)

        # Problem dimensions
        self._nvars = 3  # Number of parameters
        self._nobs = 4   # Number of observations
        self._nqoi = 1   # Scalar QoI for pushforward

        # Create test problem
        self._prior_mean = self._bkd.zeros((self._nvars, 1))
        self._prior_std = 1.0
        self._prior_cov = self._bkd.asarray(np.eye(self._nvars) * self._prior_std**2)

        # Observation matrix (design matrix)
        np.random.seed(42)
        self._obs_mat = self._bkd.asarray(np.random.randn(self._nobs, self._nvars))

        # Noise covariance (diagonal)
        self._noise_std = 0.3
        self._noise_cov = self._bkd.asarray(np.eye(self._nobs) * self._noise_std**2)

        # QoI matrix for pushforward
        np.random.seed(43)
        self._qoi_mat = self._bkd.asarray(np.random.randn(self._nqoi, self._nvars))

    def _compute_mc_posterior_stdev_samples(
        self, nsamples: int, base_seed: int = 100
    ) -> list:
        """Compute posterior pushforward std dev for many data realizations.

        Uses deterministic seeding per sample for reproducibility.

        Parameters
        ----------
        nsamples : int
            Number of MC samples
        base_seed : int
            Base seed for reproducibility

        Returns
        -------
        list
            List of posterior pushforward standard deviations
        """
        bkd = self._bkd
        stdevs = []

        for ii in range(nsamples):
            # Deterministic seed per sample
            rng = np.random.default_rng(base_seed + ii)

            # Sample theta from prior
            theta_std = rng.standard_normal((self._nvars, 1))
            theta = bkd.asarray(theta_std) * self._prior_std

            # Generate observation: y = A @ theta + noise
            noise = bkd.asarray(rng.standard_normal((self._nobs, 1)) * self._noise_std)
            y = bkd.dot(self._obs_mat, theta) + noise

            # Compute posterior
            posterior = DenseGaussianConjugatePosterior(
                self._obs_mat,
                self._prior_mean,
                self._prior_cov,
                self._noise_cov,
                bkd,
            )
            posterior.compute(y)

            # Pushforward through QoI
            post_push = GaussianPushforward(
                self._qoi_mat,
                posterior.posterior_mean(),
                posterior.posterior_covariance(),
                bkd,
            )
            # Extract scalar std dev
            post_var = float(bkd.to_numpy(post_push.covariance())[0, 0])
            stdevs.append(np.sqrt(post_var))

        return stdevs

    def _create_analytical_utility(self, cls, *args):
        """Create analytical utility and set obs/noise."""
        if args:
            utility = cls(
                self._prior_mean, self._prior_cov, self._qoi_mat, *args, self._bkd
            )
        else:
            utility = cls(
                self._prior_mean, self._prior_cov, self._qoi_mat, self._bkd
            )
        utility.set_observation_matrix(self._obs_mat)
        utility.set_noise_covariance(self._noise_cov)
        return utility

    # ==========================================================================
    # MC vs Exact Tests
    # ==========================================================================

    @slow_test
    def test_expected_stdev_mc_vs_exact(self) -> None:
        """Test MC average of posterior std dev matches exact formula."""
        nsamples = 2000
        rtol = 3e-2  # 3% relative tolerance

        # Compute MC estimate
        stdevs = self._compute_mc_posterior_stdev_samples(nsamples)
        mc_expected_stdev = np.mean(stdevs)

        # Compute exact
        utility = self._create_analytical_utility(ConjugateGaussianOEDExpectedStdDev)
        exact_expected_stdev = utility.value()

        # Compare
        rel_error = abs(mc_expected_stdev - exact_expected_stdev) / exact_expected_stdev
        self.assertLess(
            rel_error, rtol,
            f"MC expected std dev ({mc_expected_stdev:.6f}) does not match "
            f"exact ({exact_expected_stdev:.6f}), relative error: {rel_error:.4f}",
        )

    @slow_test
    def test_expected_entropic_mc_vs_exact(self) -> None:
        """Test MC average of entropic deviation matches exact formula.

        Entropic deviation = lamda * variance / 2.
        """
        nsamples = 2000
        rtol = 3e-2
        lamda = 2.0

        # Compute MC estimate
        stdevs = self._compute_mc_posterior_stdev_samples(nsamples)
        variances = [s**2 for s in stdevs]
        mc_expected = np.mean([lamda * v / 2 for v in variances])

        # Compute exact
        utility = self._create_analytical_utility(
            ConjugateGaussianOEDExpectedEntropicDev, lamda
        )
        exact = utility.value()

        # Compare
        rel_error = abs(mc_expected - exact) / exact
        self.assertLess(
            rel_error, rtol,
            f"MC entropic deviation ({mc_expected:.6f}) does not match "
            f"exact ({exact:.6f}), relative error: {rel_error:.4f}",
        )

    @slow_test
    def test_expected_avar_mc_vs_exact(self) -> None:
        """Test MC average of AVaR deviation matches exact formula.

        Replicates legacy test: for each data realization, compute
        AVaR(Q) - E[Q] where Q is the posterior pushforward QoI.

        For Gaussian Q ~ N(mu, sigma^2):
        AVaR_beta(Q) = mu + sigma * phi(Phi^{-1}(beta)) / (1 - beta)
        So AVaR deviation = sigma * phi(Phi^{-1}(beta)) / (1 - beta)
        """
        from scipy import stats

        nsamples = 2000
        rtol = 3e-2
        beta = 0.5  # Same as legacy test

        bkd = self._bkd
        avardevs = []

        for ii in range(nsamples):
            rng = np.random.default_rng(100 + ii)

            # Sample theta from prior
            theta_std = rng.standard_normal((self._nvars, 1))
            theta = bkd.asarray(theta_std) * self._prior_std

            # Generate observation
            noise = bkd.asarray(rng.standard_normal((self._nobs, 1)) * self._noise_std)
            y = bkd.dot(self._obs_mat, theta) + noise

            # Compute posterior
            posterior = DenseGaussianConjugatePosterior(
                self._obs_mat,
                self._prior_mean,
                self._prior_cov,
                self._noise_cov,
                bkd,
            )
            posterior.compute(y)

            # Pushforward through QoI
            post_push = GaussianPushforward(
                self._qoi_mat,
                posterior.posterior_mean(),
                posterior.posterior_covariance(),
                bkd,
            )

            # Compute AVaR deviation: AVaR(Q) - E[Q]
            # For Gaussian: AVaR_beta = mu + sigma * phi(Phi^{-1}(beta)) / (1 - beta)
            mu = float(bkd.to_numpy(post_push.mean())[0, 0])
            sigma = np.sqrt(float(bkd.to_numpy(post_push.covariance())[0, 0]))
            avar = mu + sigma * stats.norm.pdf(stats.norm.ppf(beta)) / (1 - beta)
            avardevs.append(avar - mu)

        mc_expected = np.mean(avardevs)

        # Compute exact
        utility = self._create_analytical_utility(
            ConjugateGaussianOEDExpectedAVaRDev, beta
        )
        exact = utility.value()

        # Compare
        rel_error = abs(mc_expected - exact) / exact
        self.assertLess(
            rel_error, rtol,
            f"MC AVaR deviation ({mc_expected:.6f}) does not match "
            f"exact ({exact:.6f}), relative error: {rel_error:.4f}",
        )

    @slow_test
    def test_expected_kl_mc_vs_exact(self) -> None:
        """Test MC average of KL divergence matches exact formula.

        NOTE: The exact formula computes E[KL(posterior_pushforward || prior_pushforward)]
        in the pushforward space (QoI space), NOT in the parameter space.
        So MC must compute KL between pushforwards, not between parameter distributions.
        """
        nsamples = 2000
        rtol = 3e-2
        bkd = self._bkd

        # Create prior pushforward
        prior_pushforward = GaussianPushforward(
            self._qoi_mat,
            self._prior_mean,
            self._prior_cov,
            bkd,
        )
        prior_push_cov = prior_pushforward.covariance()
        prior_push_cov_inv = bkd.inv(prior_push_cov)
        prior_push_mean = prior_pushforward.mean()

        # Compute MC estimate of expected KL divergence in pushforward space
        kl_values = []
        for ii in range(nsamples):
            rng = np.random.default_rng(200 + ii)

            # Sample theta from prior
            theta_std = rng.standard_normal((self._nvars, 1))
            theta = bkd.asarray(theta_std) * self._prior_std

            # Generate observation
            noise = bkd.asarray(rng.standard_normal((self._nobs, 1)) * self._noise_std)
            y = bkd.dot(self._obs_mat, theta) + noise

            # Compute posterior
            posterior = DenseGaussianConjugatePosterior(
                self._obs_mat,
                self._prior_mean,
                self._prior_cov,
                self._noise_cov,
                bkd,
            )
            posterior.compute(y)

            # Compute posterior pushforward
            post_pushforward = GaussianPushforward(
                self._qoi_mat,
                posterior.posterior_mean(),
                posterior.posterior_covariance(),
                bkd,
            )
            post_push_cov = post_pushforward.covariance()
            post_push_mean = post_pushforward.mean()

            # Compute KL(posterior_push || prior_push) in pushforward space
            nqoi = self._nqoi
            kl = 0.5 * (
                float(bkd.trace(bkd.dot(prior_push_cov_inv, post_push_cov)))
                - nqoi
            )
            _, log_det_prior_push = bkd.slogdet(prior_push_cov)
            _, log_det_post_push = bkd.slogdet(post_push_cov)
            kl += 0.5 * (float(log_det_prior_push) - float(log_det_post_push))
            diff = prior_push_mean - post_push_mean
            kl += 0.5 * float(bkd.to_numpy(bkd.dot(diff.T, bkd.dot(prior_push_cov_inv, diff)))[0, 0])
            kl_values.append(kl)

        mc_expected_kl = np.mean(kl_values)

        # Compute exact
        utility = self._create_analytical_utility(ConjugateGaussianOEDExpectedKLDivergence)
        exact = utility.value()

        # Compare
        rel_error = abs(mc_expected_kl - exact) / exact
        self.assertLess(
            rel_error, rtol,
            f"MC expected KL ({mc_expected_kl:.6f}) does not match "
            f"exact ({exact:.6f}), relative error: {rel_error:.4f}",
        )

    @slower_test
    def test_lognormal_stdev_mc_vs_exact(self) -> None:
        """Test MC average of lognormal pushforward std dev matches exact."""
        nsamples = 4000  # More samples needed for lognormal
        rtol = 5e-2
        bkd = self._bkd

        # For lognormal, QoI = exp(linear combination of parameters)
        # We compute std of exp(Q @ theta) where Q @ theta | data is Gaussian
        lognormal_stdevs = []

        for ii in range(nsamples):
            rng = np.random.default_rng(300 + ii)

            # Sample theta from prior
            theta_std = rng.standard_normal((self._nvars, 1))
            theta = bkd.asarray(theta_std) * self._prior_std

            # Generate observation
            noise = bkd.asarray(rng.standard_normal((self._nobs, 1)) * self._noise_std)
            y = bkd.dot(self._obs_mat, theta) + noise

            # Compute posterior
            posterior = DenseGaussianConjugatePosterior(
                self._obs_mat,
                self._prior_mean,
                self._prior_cov,
                self._noise_cov,
                bkd,
            )
            posterior.compute(y)

            # Pushforward through QoI (in log space)
            post_push = GaussianPushforward(
                self._qoi_mat,
                posterior.posterior_mean(),
                posterior.posterior_covariance(),
                bkd,
            )

            # For lognormal: if Z ~ N(mu, sigma^2), then
            # X = exp(Z) has std = sqrt(exp(2*mu + sigma^2) * (exp(sigma^2) - 1))
            mu = float(bkd.to_numpy(post_push.mean())[0, 0])
            sigma2 = float(bkd.to_numpy(post_push.covariance())[0, 0])
            lognormal_var = np.exp(2 * mu + sigma2) * (np.exp(sigma2) - 1)
            lognormal_stdevs.append(np.sqrt(lognormal_var))

        mc_expected = np.mean(lognormal_stdevs)

        # Compute exact
        utility = self._create_analytical_utility(
            ConjugateGaussianOEDForLogNormalExpectedStdDev
        )
        exact = utility.value()

        # Compare
        rel_error = abs(mc_expected - exact) / exact
        self.assertLess(
            rel_error, rtol,
            f"MC lognormal expected std ({mc_expected:.6f}) does not match "
            f"exact ({exact:.6f}), relative error: {rel_error:.4f}",
        )

    # ==========================================================================
    # Basic consistency tests (fast)
    # ==========================================================================

    def test_posterior_computes_correctly(self) -> None:
        """Test DenseGaussianConjugatePosterior computes posterior."""
        bkd = self._bkd

        # Generate one observation
        rng = np.random.default_rng(42)
        theta = bkd.asarray(rng.standard_normal((self._nvars, 1)) * self._prior_std)
        noise = bkd.asarray(rng.standard_normal((self._nobs, 1)) * self._noise_std)
        y = bkd.dot(self._obs_mat, theta) + noise

        # Compute posterior
        posterior = DenseGaussianConjugatePosterior(
            self._obs_mat,
            self._prior_mean,
            self._prior_cov,
            self._noise_cov,
            bkd,
        )
        posterior.compute(y)

        # Check shapes
        self.assertEqual(posterior.posterior_mean().shape, (self._nvars, 1))
        self.assertEqual(
            posterior.posterior_covariance().shape,
            (self._nvars, self._nvars),
        )

        # Posterior covariance should be smaller than prior (more info)
        prior_var = float(bkd.to_numpy(self._prior_cov)[0, 0])
        post_var = float(bkd.to_numpy(posterior.posterior_covariance())[0, 0])
        self.assertLess(post_var, prior_var)

    def test_pushforward_computes_correctly(self) -> None:
        """Test GaussianPushforward computes mean and covariance."""
        bkd = self._bkd

        # Create pushforward
        mean = bkd.asarray(np.array([[1.0], [2.0], [3.0]]))
        cov = bkd.asarray(np.eye(3))
        matrix = bkd.asarray(np.array([[1.0, 0.0, 0.0]]))  # Projects to first component

        pf = GaussianPushforward(matrix, mean, cov, bkd)

        # Mean should be matrix @ mean
        expected_mean = bkd.dot(matrix, mean)
        bkd.assert_allclose(pf.mean(), expected_mean, rtol=1e-10)

        # Cov should be matrix @ cov @ matrix.T = 1.0
        expected_cov = bkd.dot(matrix, bkd.dot(cov, matrix.T))
        bkd.assert_allclose(pf.covariance(), expected_cov, rtol=1e-10)

    def test_analytical_utilities_finite(self) -> None:
        """Test all analytical utilities return finite values."""
        utilities = [
            self._create_analytical_utility(ConjugateGaussianOEDExpectedStdDev),
            self._create_analytical_utility(ConjugateGaussianOEDExpectedEntropicDev, 2.0),
            self._create_analytical_utility(ConjugateGaussianOEDExpectedAVaRDev, 0.75),
            self._create_analytical_utility(ConjugateGaussianOEDExpectedKLDivergence),
            self._create_analytical_utility(ConjugateGaussianOEDForLogNormalExpectedStdDev),
            self._create_analytical_utility(ConjugateGaussianOEDForLogNormalAVaRStdDev, 0.5),
        ]

        for utility in utilities:
            value = utility.value()
            self.assertTrue(
                np.isfinite(value),
                f"{type(utility).__name__}.value() returned non-finite: {value}",
            )


class TestConjugateMCExactStandaloneNumpy(
    TestConjugateMCExactStandalone[NDArray[Any]]
):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestConjugateMCExactStandaloneTorch(
    TestConjugateMCExactStandalone[torch.Tensor]
):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
