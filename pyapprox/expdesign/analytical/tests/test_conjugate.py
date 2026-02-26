"""
Standalone tests for conjugate Gaussian OED utilities.

PERMANENT - no legacy imports.

These tests verify correctness using self-consistent checks.
"""

import numpy as np
import pytest

from pyapprox.expdesign.analytical import (
    ConjugateGaussianOEDExpectedAVaRDev,
    ConjugateGaussianOEDExpectedEntropicDev,
    ConjugateGaussianOEDExpectedKLDivergence,
    ConjugateGaussianOEDExpectedStdDev,
    ConjugateGaussianOEDForLogNormalAVaRStdDev,
    ConjugateGaussianOEDForLogNormalExpectedStdDev,
)


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
        utility = self._create_utility(ConjugateGaussianOEDExpectedStdDev, bkd)
        value = utility.value()
        assert value > 0.0

    def test_expected_stdev_decreases_with_more_info(self, bkd):
        """Test expected std dev decreases with smaller noise."""
        utility1 = self._create_utility(ConjugateGaussianOEDExpectedStdDev, bkd)
        value1 = utility1.value()

        # Create utility with smaller noise
        utility2 = ConjugateGaussianOEDExpectedStdDev(
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
            ConjugateGaussianOEDExpectedEntropicDev, bkd, lamda
        )
        value = utility.value()
        assert value > 0.0

    def test_expected_entropic_dev_scales_with_lamda(self, bkd):
        """Test entropic deviation scales linearly with lamda."""
        lamda1 = 1.0
        lamda2 = 2.0

        utility1 = self._create_utility(
            ConjugateGaussianOEDExpectedEntropicDev, bkd, lamda1
        )
        utility2 = self._create_utility(
            ConjugateGaussianOEDExpectedEntropicDev, bkd, lamda2
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
            ConjugateGaussianOEDExpectedAVaRDev, bkd, beta
        )
        value = utility.value()
        assert value > 0.0

    def test_expected_avar_dev_increases_with_beta(self, bkd):
        """Test AVaR deviation increases with beta."""
        beta1 = 0.5
        beta2 = 0.9

        utility1 = self._create_utility(
            ConjugateGaussianOEDExpectedAVaRDev, bkd, beta1
        )
        utility2 = self._create_utility(
            ConjugateGaussianOEDExpectedAVaRDev, bkd, beta2
        )

        value1 = utility1.value()
        value2 = utility2.value()

        assert value2 > value1

    def test_expected_kl_divergence_nonnegative(self, bkd):
        """Test expected KL divergence is non-negative."""
        utility = self._create_utility(
            ConjugateGaussianOEDExpectedKLDivergence, bkd
        )
        value = utility.value()
        assert value >= 0.0

    def test_expected_kl_divergence_zero_limit(self, bkd):
        """Test KL divergence approaches zero with infinite noise."""
        # With very large noise, posterior ~ prior, so KL ~ 0
        utility = ConjugateGaussianOEDExpectedKLDivergence(
            self._prior_mean, self._prior_cov, self._qoi_mat, bkd
        )
        utility.set_observation_matrix(self._obs_mat)
        utility.set_noise_covariance(self._noise_cov * 1e6)  # Huge noise
        value = utility.value()

        assert value < 0.1  # Should be close to zero

    def test_lognormal_expected_stdev_positive(self, bkd):
        """Test lognormal expected std dev is positive."""
        utility = self._create_utility(
            ConjugateGaussianOEDForLogNormalExpectedStdDev, bkd
        )
        value = utility.value()
        assert value > 0.0

    def test_lognormal_avar_stdev_positive(self, bkd):
        """Test lognormal AVaR std dev is positive."""
        beta = 0.5
        utility = self._create_utility(
            ConjugateGaussianOEDForLogNormalAVaRStdDev, bkd, beta
        )
        value = utility.value()
        assert value > 0.0

    def test_lognormal_avar_stdev_increases_with_beta(self, bkd):
        """Test lognormal AVaR std dev increases with beta."""
        beta1 = 0.3
        beta2 = 0.7

        utility1 = self._create_utility(
            ConjugateGaussianOEDForLogNormalAVaRStdDev, bkd, beta1
        )
        utility2 = self._create_utility(
            ConjugateGaussianOEDForLogNormalAVaRStdDev, bkd, beta2
        )

        value1 = utility1.value()
        value2 = utility2.value()

        assert value2 > value1

    def test_bkd_accessor(self, bkd):
        """Test bkd() returns the backend."""
        utility = self._create_utility(ConjugateGaussianOEDExpectedStdDev, bkd)
        assert utility.bkd() == bkd

    def test_value_before_noise_raises(self, bkd):
        """Test calling value() before set_noise_covariance raises."""
        utility = ConjugateGaussianOEDExpectedStdDev(
            self._prior_mean, self._prior_cov, self._qoi_mat, bkd
        )
        utility.set_observation_matrix(self._obs_mat)
        with pytest.raises(ValueError):
            utility.value()

    def test_set_noise_before_obs_raises(self, bkd):
        """Test calling set_noise_covariance before set_observation_matrix raises."""
        utility = ConjugateGaussianOEDExpectedStdDev(
            self._prior_mean, self._prior_cov, self._qoi_mat, bkd
        )
        with pytest.raises(ValueError):
            utility.set_noise_covariance(self._noise_cov)

    def test_wrong_obs_mat_shape_raises(self, bkd):
        """Test setting obs_mat with wrong number of columns raises."""
        utility = ConjugateGaussianOEDExpectedStdDev(
            self._prior_mean, self._prior_cov, self._qoi_mat, bkd
        )
        wrong_obs_mat = bkd.asarray(np.random.randn(self._nobs, self._nvars + 1))
        with pytest.raises(ValueError):
            utility.set_observation_matrix(wrong_obs_mat)
