"""
Standalone tests for conjugate Gaussian OED utilities.

PERMANENT - no legacy imports.

These tests verify correctness using self-consistent checks.
"""

import numpy as np
import pytest

from pyapprox.expdesign.analytical import (
    ConjugateGaussianOEDDataAVaRQoIMeanAVaRDev,
    ConjugateGaussianOEDDataMeanQoIMeanEntropicDev,
    ConjugateGaussianOEDExpectedPushforwardKLDivergence,
    ConjugateGaussianOEDDataMeanQoIMeanStdDev,
    ConjugateGaussianOEDForLogNormalDataAVaRQoIMeanStdDev,
    ConjugateGaussianOEDForLogNormalDataMeanQoIMeanStdDev,
    ConjugateGaussianOEDForLogNormalDataMeanStdDevQoIMeanStdDev,
    ConjugateGaussianOEDForLogNormalDataMeanQoIAVaRStdDev,
    ConjugateGaussianOEDForLogNormalDataMeanQoIMeanStdDev,
)

#TODO: rename this file to better differentiate it from test_conjugate_mc.py


class TestConjugateGaussianOEDStandalone:
    """Standalone tests for conjugate Gaussian OED utilities."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    @pytest.fixture(autouse=True)
    def _setup(self, bkd):
        self._nobs = 3
        self._nvars = 4
        self._nqoi = 1

        # Create test data
        self._prior_mean = bkd.asarray(np.random.randn(self._nvars, 1))
        self._prior_cov = bkd.asarray(np.eye(self._nvars) * 0.5)
        self._obs_mat = bkd.asarray(np.random.randn(self._nobs, self._nvars))
        self._noise_cov = bkd.asarray(
            np.diag(np.random.uniform(0.1, 0.5, self._nobs))
        )
        self._qoi_mat = bkd.asarray(np.random.randn(self._nqoi, self._nvars))

    def _create_utility(self, cls, bkd, *args):
        """Create a utility instance and set observation/noise."""
        if args:
            utility = cls(
                self._prior_mean, self._prior_cov, self._qoi_mat, *args, bkd
            )
        else:
            utility = cls(self._prior_mean, self._prior_cov, self._qoi_mat, bkd)
        utility.set_observation_matrix(self._obs_mat)
        utility.set_noise_covariance(self._noise_cov)
        return utility

    def test_expected_stdev_positive(self, bkd):
        """Test expected std dev is positive."""
        utility = self._create_utility(ConjugateGaussianOEDDataMeanQoIMeanStdDev, bkd)
        value = utility.value()
        assert value > 0.0

    def test_expected_stdev_decreases_with_more_info(self, bkd):
        """Test expected std dev decreases with smaller noise."""
        utility1 = self._create_utility(ConjugateGaussianOEDDataMeanQoIMeanStdDev, bkd)
        value1 = utility1.value()

        # Create utility with smaller noise
        utility2 = ConjugateGaussianOEDDataMeanQoIMeanStdDev(
            self._prior_mean, self._prior_cov, self._qoi_mat, bkd
        )
        utility2.set_observation_matrix(self._obs_mat)
        utility2.set_noise_covariance(self._noise_cov * 0.1)  # Much smaller noise
        value2 = utility2.value()

        # Smaller noise should give smaller posterior variance
        assert value2 < value1

    def test_expected_entropic_dev_positive(self, bkd):
        """Test expected entropic deviation is positive."""
        lamda = 2.0
        utility = self._create_utility(
            ConjugateGaussianOEDDataMeanQoIMeanEntropicDev, bkd, lamda
        )
        value = utility.value()
        assert value > 0.0

    def test_expected_entropic_dev_scales_with_lamda(self, bkd):
        """Test entropic deviation scales linearly with lamda."""
        lamda1 = 1.0
        lamda2 = 2.0

        utility1 = self._create_utility(
            ConjugateGaussianOEDDataMeanQoIMeanEntropicDev, bkd, lamda1
        )
        utility2 = self._create_utility(
            ConjugateGaussianOEDDataMeanQoIMeanEntropicDev, bkd, lamda2
        )

        value1 = utility1.value()
        value2 = utility2.value()

        # Entropic dev = lamda * variance / 2, so should scale linearly
        bkd.assert_allclose(
            bkd.asarray([value2 / value1]),
            bkd.asarray([lamda2 / lamda1]),
            rtol=1e-10,
        )

    def test_expected_avar_dev_positive(self, bkd):
        """Test expected AVaR deviation is positive."""
        beta = 0.75
        utility = self._create_utility(
            ConjugateGaussianOEDDataAVaRQoIMeanAVaRDev, bkd, beta
        )
        value = utility.value()
        assert value > 0.0

    def test_expected_avar_dev_increases_with_beta(self, bkd):
        """Test AVaR deviation increases with beta."""
        beta1 = 0.5
        beta2 = 0.9

        utility1 = self._create_utility(
            ConjugateGaussianOEDDataAVaRQoIMeanAVaRDev, bkd, beta1
        )
        utility2 = self._create_utility(
            ConjugateGaussianOEDDataAVaRQoIMeanAVaRDev, bkd, beta2
        )

        value1 = utility1.value()
        value2 = utility2.value()

        assert value2 > value1

    def test_expected_pushforward_kl_nonnegative(self, bkd):
        """Test expected pushforward KL divergence is non-negative."""
        utility = self._create_utility(
            ConjugateGaussianOEDExpectedPushforwardKLDivergence, bkd
        )
        value = utility.value()
        assert value >= 0.0

    def test_expected_pushforward_kl_zero_limit(self, bkd):
        """Test pushforward KL approaches zero with infinite noise."""
        utility = ConjugateGaussianOEDExpectedPushforwardKLDivergence(
            self._prior_mean, self._prior_cov, self._qoi_mat, bkd
        )
        utility.set_observation_matrix(self._obs_mat)
        utility.set_noise_covariance(self._noise_cov * 1e6)
        value = utility.value()

        assert value < 0.1

    def test_lognormal_expected_stdev_positive(self, bkd):
        """Test lognormal expected std dev is positive."""
        utility = self._create_utility(
            ConjugateGaussianOEDForLogNormalDataMeanQoIMeanStdDev, bkd
        )
        value = utility.value()
        assert value > 0.0

    def test_lognormal_avar_stdev_positive(self, bkd):
        """Test lognormal AVaR std dev is positive."""
        beta = 0.5
        utility = self._create_utility(
            ConjugateGaussianOEDForLogNormalDataAVaRQoIMeanStdDev, bkd, beta
        )
        value = utility.value()
        assert value > 0.0

    def test_lognormal_avar_stdev_increases_with_beta(self, bkd):
        """Test lognormal AVaR std dev increases with beta."""
        beta1 = 0.3
        beta2 = 0.7

        utility1 = self._create_utility(
            ConjugateGaussianOEDForLogNormalDataAVaRQoIMeanStdDev, bkd, beta1
        )
        utility2 = self._create_utility(
            ConjugateGaussianOEDForLogNormalDataAVaRQoIMeanStdDev, bkd, beta2
        )

        value1 = utility1.value()
        value2 = utility2.value()

        assert value2 > value1

    def test_bkd_accessor(self, bkd):
        """Test bkd() returns the backend."""
        utility = self._create_utility(ConjugateGaussianOEDDataMeanQoIMeanStdDev, bkd)
        assert utility.bkd() == bkd

    def test_value_before_noise_raises(self, bkd):
        """Test calling value() before set_noise_covariance raises."""
        utility = ConjugateGaussianOEDDataMeanQoIMeanStdDev(
            self._prior_mean, self._prior_cov, self._qoi_mat, bkd
        )
        utility.set_observation_matrix(self._obs_mat)
        with pytest.raises(ValueError):
            utility.value()

    def test_set_noise_before_obs_raises(self, bkd):
        """Test calling set_noise_covariance before set_observation_matrix raises."""
        utility = ConjugateGaussianOEDDataMeanQoIMeanStdDev(
            self._prior_mean, self._prior_cov, self._qoi_mat, bkd
        )
        with pytest.raises(ValueError):
            utility.set_noise_covariance(self._noise_cov)

    def test_wrong_obs_mat_shape_raises(self, bkd):
        """Test setting obs_mat with wrong number of columns raises."""
        utility = ConjugateGaussianOEDDataMeanQoIMeanStdDev(
            self._prior_mean, self._prior_cov, self._qoi_mat, bkd
        )
        wrong_obs_mat = bkd.asarray(np.random.randn(self._nobs, self._nvars + 1))
        with pytest.raises(ValueError):
            utility.set_observation_matrix(wrong_obs_mat)


class TestConjugateGaussianOEDLogNormalQoIAVaR:
    """Tests for E_y[AVaR_alpha over vector lognormal Std(W_j|y)]."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _build_setup(self, bkd, npred=4):
        """Build a degree-1 basis with general QoI locations."""
        nvars = 2  # degree-1 basis: [1, x_j]
        nobs = 3

        prior_mean = bkd.zeros((nvars, 1))
        prior_cov = bkd.asarray(np.eye(nvars) * 0.5)
        obs_mat = bkd.asarray(np.random.randn(nobs, nvars))
        noise_cov = bkd.asarray(
            np.diag(np.random.uniform(0.1, 0.5, nobs))
        )

        x_vals = np.linspace(-1.5, 1.5, npred)
        qoi_mat = bkd.asarray(
            np.column_stack([np.ones(npred), x_vals])
        )

        return prior_mean, prior_cov, obs_mat, noise_cov, qoi_mat

    def _create_utility(self, bkd, alpha, setup=None, npred=4):
        if setup is None:
            setup = self._build_setup(bkd, npred)
        prior_mean, prior_cov, obs_mat, noise_cov, qoi_mat = setup
        utility = ConjugateGaussianOEDForLogNormalDataMeanQoIAVaRStdDev(
            prior_mean, prior_cov, qoi_mat, alpha, bkd
        )
        utility.set_observation_matrix(obs_mat)
        utility.set_noise_covariance(noise_cov)
        return utility

    def test_lognormal_mean_avar_stdev_positive(self, bkd):
        """Test E_y[AVaR_alpha[Std(W_j|y)]] is positive."""
        utility = self._create_utility(bkd, alpha=0.5)
        assert utility.value() > 0.0

    def test_lognormal_mean_avar_stdev_increases_with_alpha(self, bkd):
        """Test utility increases with alpha (more risk-averse)."""
        prior_mean, prior_cov, obs_mat, noise_cov, qoi_mat = (
            self._build_setup(bkd)
        )
        values = []
        for alpha in [0.0, 0.5, 0.75]:
            utility = ConjugateGaussianOEDForLogNormalDataMeanQoIAVaRStdDev(
                prior_mean, prior_cov, qoi_mat, alpha, bkd
            )
            utility.set_observation_matrix(obs_mat)
            utility.set_noise_covariance(noise_cov)
            values.append(utility.value())

        assert values[1] >= values[0], (
            f"alpha=0.5 ({values[1]}) should be >= alpha=0.0 ({values[0]})"
        )
        assert values[2] >= values[1], (
            f"alpha=0.75 ({values[2]}) should be >= alpha=0.5 ({values[1]})"
        )

    def test_lognormal_mean_avar_stdev_alpha_zero_equals_mean(self, bkd):
        """Test alpha=0 reduces to (1/Q)*sum K_j*exp(nu_j + sigma_tau_j^2/2).

        At alpha=0 (m=Q), AVaR is the plain mean over all QoI components.
        This is a machine-precision identity by the telescoping argument:
        each per-j integral over the full real line yields
        K_j * exp(nu_j + sigma_tau_j^2/2), which equals the per-QoI
        E_y[Std(W_j|y)] from LogNormalExpectedStdDev.
        """
        npred = 4
        prior_mean, prior_cov, obs_mat, noise_cov, qoi_mat = (
            self._build_setup(bkd, npred)
        )

        # Compute via the vector class at alpha=0
        utility = ConjugateGaussianOEDForLogNormalDataMeanQoIAVaRStdDev(
            prior_mean, prior_cov, qoi_mat, 0.0, bkd
        )
        utility.set_observation_matrix(obs_mat)
        utility.set_noise_covariance(noise_cov)
        avar_val = utility.value()

        # Compute per-K_j mean using individual LogNormalExpectedStdDev
        from pyapprox.expdesign.analytical import (
            ConjugateGaussianOEDForLogNormalDataMeanQoIMeanStdDev,
        )

        total = 0.0
        for j in range(npred):
            qoi_row = qoi_mat[j : j + 1]
            u = ConjugateGaussianOEDForLogNormalDataMeanQoIMeanStdDev(
                prior_mean, prior_cov, qoi_row, bkd
            )
            u.set_observation_matrix(obs_mat)
            u.set_noise_covariance(noise_cov)
            total += u.value()
        mean_val = total / npred

        bkd.assert_allclose(
            bkd.asarray([avar_val]),
            bkd.asarray([mean_val]),
            rtol=1e-10,
        )

    def test_lognormal_mean_avar_stdev_finite(self, bkd):
        """Test utility returns finite values for several alpha."""
        setup = self._build_setup(bkd)
        for alpha in [0.0, 0.25, 0.5, 0.75]:
            utility = self._create_utility(bkd, alpha=alpha, setup=setup)
            value = utility.value()
            assert np.isfinite(value), (
                f"Non-finite value at alpha={alpha}: {value}"
            )


class TestConjugateGaussianOEDLogNormalMeanStdDev:
    """Tests for E_y[Std(W|y)] + c * Std_y[Std(W|y)] — safety margin."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    @pytest.fixture(autouse=True)
    def _setup(self, bkd):
        self._nobs = 3
        self._nvars = 4
        self._nqoi = 1
        self._prior_mean = bkd.asarray(np.random.randn(self._nvars, 1))
        self._prior_cov = bkd.asarray(np.eye(self._nvars) * 0.5)
        self._obs_mat = bkd.asarray(np.random.randn(self._nobs, self._nvars))
        self._noise_cov = bkd.asarray(
            np.diag(np.random.uniform(0.1, 0.5, self._nobs))
        )
        self._qoi_mat = bkd.asarray(np.random.randn(self._nqoi, self._nvars))

    def _create_utility(self, bkd, c):
        utility = ConjugateGaussianOEDForLogNormalDataMeanStdDevQoIMeanStdDev(
            self._prior_mean, self._prior_cov, self._qoi_mat, c, bkd
        )
        utility.set_observation_matrix(self._obs_mat)
        utility.set_noise_covariance(self._noise_cov)
        return utility

    def test_positive(self, bkd):
        """Safety margin utility is positive."""
        value = self._create_utility(bkd, c=1.0).value()
        assert value > 0.0

    def test_increases_with_c(self, bkd):
        """Utility increases with safety factor c."""
        v0 = self._create_utility(bkd, c=0.0).value()
        v1 = self._create_utility(bkd, c=1.0).value()
        v2 = self._create_utility(bkd, c=2.0).value()
        assert v1 > v0
        assert v2 > v1

    def test_c_zero_equals_expected_stdev(self, bkd):
        """At c=0, U4 = E_y[Std(W|y)] = U1."""
        u4 = self._create_utility(bkd, c=0.0).value()
        u1_util = ConjugateGaussianOEDForLogNormalDataMeanQoIMeanStdDev(
            self._prior_mean, self._prior_cov, self._qoi_mat, bkd
        )
        u1_util.set_observation_matrix(self._obs_mat)
        u1_util.set_noise_covariance(self._noise_cov)
        u1 = u1_util.value()
        bkd.assert_allclose(
            bkd.asarray([u4]), bkd.asarray([u1]), rtol=1e-10,
        )

    def test_finite_values(self, bkd):
        """Utility is finite for several c values."""
        for c in [0.0, 0.5, 1.0, 5.0]:
            value = self._create_utility(bkd, c).value()
            assert np.isfinite(value), f"Non-finite at c={c}: {value}"
