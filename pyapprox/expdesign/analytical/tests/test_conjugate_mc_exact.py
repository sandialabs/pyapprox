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

import numpy as np

from pyapprox.expdesign.analytical import (
    ConjugateGaussianOEDExpectedAVaRDev,
    ConjugateGaussianOEDExpectedEntropicDev,
    ConjugateGaussianOEDExpectedKLDivergence,
    ConjugateGaussianOEDExpectedStdDev,
    ConjugateGaussianOEDForLogNormalExpectedStdDev,
)
from pyapprox.inverse.conjugate.gaussian import DenseGaussianConjugatePosterior
from pyapprox.inverse.pushforward.gaussian import GaussianPushforward
from pyapprox.util.test_utils import (
    slow_test,
    slower_test,
)


class TestConjugateMCExactStandalone:
    """MC vs exact analytical formula comparison tests."""

    def _setup_data(self, bkd):
        np.random.seed(42)

        # Problem dimensions
        self._nvars = 3  # Number of parameters
        self._nobs = 4  # Number of observations
        self._nqoi = 1  # Scalar QoI for pushforward

        # Create test problem
        self._prior_mean = bkd.zeros((self._nvars, 1))
        self._prior_std = 1.0
        self._prior_cov = bkd.asarray(np.eye(self._nvars) * self._prior_std**2)

        # Observation matrix (design matrix)
        np.random.seed(42)
        self._obs_mat = bkd.asarray(np.random.randn(self._nobs, self._nvars))

        # Noise covariance (diagonal)
        self._noise_std = 0.3
        self._noise_cov = bkd.asarray(np.eye(self._nobs) * self._noise_std**2)

        # QoI matrix for pushforward
        np.random.seed(43)
        self._qoi_mat = bkd.asarray(np.random.randn(self._nqoi, self._nvars))

    def _compute_mc_posterior_stdev_samples(
        self, bkd, nsamples: int, base_seed: int = 100
    ) -> list:
        """Compute posterior pushforward std dev for many data realizations.

        Uses deterministic seeding per sample for reproducibility.

        Parameters
        ----------
        bkd
            Backend instance
        nsamples : int
            Number of MC samples
        base_seed : int
            Base seed for reproducibility

        Returns
        -------
        list
            List of posterior pushforward standard deviations
        """
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

    def _create_analytical_utility(self, cls, bkd, *args):
        """Create analytical utility and set obs/noise."""
        if args:
            utility = cls(
                self._prior_mean, self._prior_cov, self._qoi_mat, *args, bkd
            )
        else:
            utility = cls(self._prior_mean, self._prior_cov, self._qoi_mat, bkd)
        utility.set_observation_matrix(self._obs_mat)
        utility.set_noise_covariance(self._noise_cov)
        return utility

    # ==========================================================================
    # MC vs Exact Tests
    # ==========================================================================

    @slow_test
    def test_expected_stdev_mc_vs_exact(self, bkd) -> None:
        """Test MC average of posterior std dev matches exact formula."""
        self._setup_data(bkd)
        nsamples = 2000
        rtol = 3e-2  # 3% relative tolerance

        # Compute MC estimate
        stdevs = self._compute_mc_posterior_stdev_samples(bkd, nsamples)
        mc_expected_stdev = np.mean(stdevs)

        # Compute exact
        utility = self._create_analytical_utility(
            ConjugateGaussianOEDExpectedStdDev, bkd
        )
        exact_expected_stdev = utility.value()

        # Compare
        rel_error = abs(mc_expected_stdev - exact_expected_stdev) / exact_expected_stdev
        assert rel_error < rtol, (
            f"MC expected std dev ({mc_expected_stdev:.6f}) does not match "
            f"exact ({exact_expected_stdev:.6f}), relative error: {rel_error:.4f}"
        )

    @slow_test
    def test_expected_entropic_mc_vs_exact(self, bkd) -> None:
        """Test MC average of entropic deviation matches exact formula.

        Entropic deviation = lamda * variance / 2.
        """
        self._setup_data(bkd)
        nsamples = 2000
        rtol = 3e-2
        lamda = 2.0

        # Compute MC estimate
        stdevs = self._compute_mc_posterior_stdev_samples(bkd, nsamples)
        variances = [s**2 for s in stdevs]
        mc_expected = np.mean([lamda * v / 2 for v in variances])

        # Compute exact
        utility = self._create_analytical_utility(
            ConjugateGaussianOEDExpectedEntropicDev, bkd, lamda
        )
        exact = utility.value()

        # Compare
        rel_error = abs(mc_expected - exact) / exact
        assert rel_error < rtol, (
            f"MC entropic deviation ({mc_expected:.6f}) does not match "
            f"exact ({exact:.6f}), relative error: {rel_error:.4f}"
        )

    @slow_test
    def test_expected_avar_mc_vs_exact(self, bkd) -> None:
        """Test MC average of AVaR deviation matches exact formula.

        Replicates legacy test: for each data realization, compute
        AVaR(Q) - E[Q] where Q is the posterior pushforward QoI.

        For Gaussian Q ~ N(mu, sigma^2):
        AVaR_beta(Q) = mu + sigma * phi(Phi^{-1}(beta)) / (1 - beta)
        So AVaR deviation = sigma * phi(Phi^{-1}(beta)) / (1 - beta)
        """
        from scipy import stats

        self._setup_data(bkd)
        nsamples = 2000
        rtol = 3e-2
        beta = 0.5  # Same as legacy test

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
            ConjugateGaussianOEDExpectedAVaRDev, bkd, beta
        )
        exact = utility.value()

        # Compare
        rel_error = abs(mc_expected - exact) / exact
        assert rel_error < rtol, (
            f"MC AVaR deviation ({mc_expected:.6f}) does not match "
            f"exact ({exact:.6f}), relative error: {rel_error:.4f}"
        )

    @slow_test
    def test_expected_kl_mc_vs_exact(self, bkd) -> None:
        """Test MC average of KL divergence matches exact formula.

        NOTE: The exact formula computes E[KL(posterior_pushforward ||
        prior_pushforward)]
        in the pushforward space (QoI space), NOT in the parameter space.
        So MC must compute KL between pushforwards, not between parameter distributions.
        """
        self._setup_data(bkd)
        nsamples = 2000
        rtol = 3e-2

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
                float(bkd.trace(bkd.dot(prior_push_cov_inv, post_push_cov))) - nqoi
            )
            _, log_det_prior_push = bkd.slogdet(prior_push_cov)
            _, log_det_post_push = bkd.slogdet(post_push_cov)
            kl += 0.5 * (float(log_det_prior_push) - float(log_det_post_push))
            diff = prior_push_mean - post_push_mean
            kl += 0.5 * float(
                bkd.to_numpy(bkd.dot(diff.T, bkd.dot(prior_push_cov_inv, diff)))[0, 0]
            )
            kl_values.append(kl)

        mc_expected_kl = np.mean(kl_values)

        # Compute exact
        utility = self._create_analytical_utility(
            ConjugateGaussianOEDExpectedKLDivergence, bkd
        )
        exact = utility.value()

        # Compare
        rel_error = abs(mc_expected_kl - exact) / exact
        assert rel_error < rtol, (
            f"MC expected KL ({mc_expected_kl:.6f}) does not match "
            f"exact ({exact:.6f}), relative error: {rel_error:.4f}"
        )

    @slower_test
    def test_lognormal_stdev_mc_vs_exact(self, bkd) -> None:
        """Test MC average of lognormal pushforward std dev matches exact."""
        self._setup_data(bkd)
        nsamples = 4000  # More samples needed for lognormal
        rtol = 5e-2

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
            ConjugateGaussianOEDForLogNormalExpectedStdDev, bkd
        )
        exact = utility.value()

        # Compare
        rel_error = abs(mc_expected - exact) / exact
        assert rel_error < rtol, (
            f"MC lognormal expected std ({mc_expected:.6f}) does not match "
            f"exact ({exact:.6f}), relative error: {rel_error:.4f}"
        )

    # ==========================================================================
    # Basic consistency tests (fast)
    # ==========================================================================

    def test_posterior_computes_correctly(self, bkd) -> None:
        """Test DenseGaussianConjugatePosterior computes posterior."""
        self._setup_data(bkd)

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
        assert posterior.posterior_mean().shape == (self._nvars, 1)
        assert posterior.posterior_covariance().shape == (
            self._nvars,
            self._nvars,
        )

        # Posterior covariance should be smaller than prior (more info)
        prior_var = float(bkd.to_numpy(self._prior_cov)[0, 0])
        post_var = float(bkd.to_numpy(posterior.posterior_covariance())[0, 0])
        assert post_var < prior_var

    def test_pushforward_computes_correctly(self, bkd) -> None:
        """Test GaussianPushforward computes mean and covariance."""
        self._setup_data(bkd)

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

    def test_analytical_utilities_finite(self, bkd) -> None:
        """Test all analytical utilities return finite values."""
        self._setup_data(bkd)
        utilities = [
            self._create_analytical_utility(
                ConjugateGaussianOEDExpectedStdDev, bkd
            ),
            self._create_analytical_utility(
                ConjugateGaussianOEDExpectedEntropicDev, bkd, 2.0
            ),
            self._create_analytical_utility(
                ConjugateGaussianOEDExpectedAVaRDev, bkd, 0.75
            ),
            self._create_analytical_utility(
                ConjugateGaussianOEDExpectedKLDivergence, bkd
            ),
            self._create_analytical_utility(
                ConjugateGaussianOEDForLogNormalExpectedStdDev, bkd
            ),
        ]

        for utility in utilities:
            value = utility.value()
            assert np.isfinite(value), (
                f"{type(utility).__name__}.value() returned non-finite: {value}"
            )
