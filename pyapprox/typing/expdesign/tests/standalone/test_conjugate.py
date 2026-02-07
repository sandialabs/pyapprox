"""
Standalone tests for conjugate Gaussian OED utilities.

PERMANENT - no legacy imports.

These tests verify correctness using self-consistent checks.
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401

from pyapprox.typing.expdesign.analytical import (
    ConjugateGaussianOEDExpectedStdDev,
    ConjugateGaussianOEDExpectedEntropicDev,
    ConjugateGaussianOEDExpectedAVaRDev,
    ConjugateGaussianOEDExpectedKLDivergence,
    ConjugateGaussianOEDForLogNormalExpectedStdDev,
    ConjugateGaussianOEDForLogNormalAVaRStdDev,
)


class TestConjugateGaussianOEDStandalone(Generic[Array], unittest.TestCase):
    """Standalone tests for conjugate Gaussian OED utilities."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        np.random.seed(42)

        self._nobs = 3
        self._nvars = 4
        self._nqoi = 1

        # Create test data
        self._prior_mean = self._bkd.asarray(np.random.randn(self._nvars, 1))
        self._prior_cov = self._bkd.asarray(np.eye(self._nvars) * 0.5)
        self._obs_mat = self._bkd.asarray(np.random.randn(self._nobs, self._nvars))
        self._noise_cov = self._bkd.asarray(
            np.diag(np.random.uniform(0.1, 0.5, self._nobs))
        )
        self._qoi_mat = self._bkd.asarray(np.random.randn(self._nqoi, self._nvars))

    def _create_utility(self, cls, *args):
        """Create a utility instance and set observation/noise."""
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

    def test_expected_stdev_positive(self):
        """Test expected std dev is positive."""
        utility = self._create_utility(ConjugateGaussianOEDExpectedStdDev)
        value = utility.value()
        self.assertGreater(value, 0.0)

    def test_expected_stdev_decreases_with_more_info(self):
        """Test expected std dev decreases with smaller noise."""
        utility1 = self._create_utility(ConjugateGaussianOEDExpectedStdDev)
        value1 = utility1.value()

        # Create utility with smaller noise
        utility2 = ConjugateGaussianOEDExpectedStdDev(
            self._prior_mean, self._prior_cov, self._qoi_mat, self._bkd
        )
        utility2.set_observation_matrix(self._obs_mat)
        utility2.set_noise_covariance(self._noise_cov * 0.1)  # Much smaller noise
        value2 = utility2.value()

        # Smaller noise should give smaller posterior variance
        self.assertLess(value2, value1)

    def test_expected_entropic_dev_positive(self):
        """Test expected entropic deviation is positive."""
        lamda = 2.0
        utility = self._create_utility(ConjugateGaussianOEDExpectedEntropicDev, lamda)
        value = utility.value()
        self.assertGreater(value, 0.0)

    def test_expected_entropic_dev_scales_with_lamda(self):
        """Test entropic deviation scales linearly with lamda."""
        lamda1 = 1.0
        lamda2 = 2.0

        utility1 = self._create_utility(ConjugateGaussianOEDExpectedEntropicDev, lamda1)
        utility2 = self._create_utility(ConjugateGaussianOEDExpectedEntropicDev, lamda2)

        value1 = utility1.value()
        value2 = utility2.value()

        # Entropic dev = lamda * variance / 2, so should scale linearly
        self._bkd.assert_allclose(
            self._bkd.asarray([value2 / value1]),
            self._bkd.asarray([lamda2 / lamda1]),
            rtol=1e-10,
        )

    def test_expected_avar_dev_positive(self):
        """Test expected AVaR deviation is positive."""
        beta = 0.75
        utility = self._create_utility(ConjugateGaussianOEDExpectedAVaRDev, beta)
        value = utility.value()
        self.assertGreater(value, 0.0)

    def test_expected_avar_dev_increases_with_beta(self):
        """Test AVaR deviation increases with beta."""
        beta1 = 0.5
        beta2 = 0.9

        utility1 = self._create_utility(ConjugateGaussianOEDExpectedAVaRDev, beta1)
        utility2 = self._create_utility(ConjugateGaussianOEDExpectedAVaRDev, beta2)

        value1 = utility1.value()
        value2 = utility2.value()

        self.assertGreater(value2, value1)

    def test_expected_kl_divergence_nonnegative(self):
        """Test expected KL divergence is non-negative."""
        utility = self._create_utility(ConjugateGaussianOEDExpectedKLDivergence)
        value = utility.value()
        self.assertGreaterEqual(value, 0.0)

    def test_expected_kl_divergence_zero_limit(self):
        """Test KL divergence approaches zero with infinite noise."""
        # With very large noise, posterior ≈ prior, so KL ≈ 0
        utility = ConjugateGaussianOEDExpectedKLDivergence(
            self._prior_mean, self._prior_cov, self._qoi_mat, self._bkd
        )
        utility.set_observation_matrix(self._obs_mat)
        utility.set_noise_covariance(self._noise_cov * 1e6)  # Huge noise
        value = utility.value()

        self.assertLess(value, 0.1)  # Should be close to zero

    def test_lognormal_expected_stdev_positive(self):
        """Test lognormal expected std dev is positive."""
        utility = self._create_utility(ConjugateGaussianOEDForLogNormalExpectedStdDev)
        value = utility.value()
        self.assertGreater(value, 0.0)

    def test_lognormal_avar_stdev_positive(self):
        """Test lognormal AVaR std dev is positive."""
        beta = 0.5
        utility = self._create_utility(ConjugateGaussianOEDForLogNormalAVaRStdDev, beta)
        value = utility.value()
        self.assertGreater(value, 0.0)

    def test_lognormal_avar_stdev_increases_with_beta(self):
        """Test lognormal AVaR std dev increases with beta."""
        beta1 = 0.3
        beta2 = 0.7

        utility1 = self._create_utility(ConjugateGaussianOEDForLogNormalAVaRStdDev, beta1)
        utility2 = self._create_utility(ConjugateGaussianOEDForLogNormalAVaRStdDev, beta2)

        value1 = utility1.value()
        value2 = utility2.value()

        self.assertGreater(value2, value1)

    def test_bkd_accessor(self):
        """Test bkd() returns the backend."""
        utility = self._create_utility(ConjugateGaussianOEDExpectedStdDev)
        self.assertEqual(utility.bkd(), self._bkd)

    def test_value_before_noise_raises(self):
        """Test calling value() before set_noise_covariance raises."""
        utility = ConjugateGaussianOEDExpectedStdDev(
            self._prior_mean, self._prior_cov, self._qoi_mat, self._bkd
        )
        utility.set_observation_matrix(self._obs_mat)
        with self.assertRaises(ValueError):
            utility.value()

    def test_set_noise_before_obs_raises(self):
        """Test calling set_noise_covariance before set_observation_matrix raises."""
        utility = ConjugateGaussianOEDExpectedStdDev(
            self._prior_mean, self._prior_cov, self._qoi_mat, self._bkd
        )
        with self.assertRaises(ValueError):
            utility.set_noise_covariance(self._noise_cov)

    def test_wrong_obs_mat_shape_raises(self):
        """Test setting obs_mat with wrong number of columns raises."""
        utility = ConjugateGaussianOEDExpectedStdDev(
            self._prior_mean, self._prior_cov, self._qoi_mat, self._bkd
        )
        wrong_obs_mat = self._bkd.asarray(np.random.randn(self._nobs, self._nvars + 1))
        with self.assertRaises(ValueError):
            utility.set_observation_matrix(wrong_obs_mat)


class TestConjugateGaussianOEDStandaloneNumpy(
    TestConjugateGaussianOEDStandalone[NDArray[Any]]
):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestConjugateGaussianOEDStandaloneTorch(
    TestConjugateGaussianOEDStandalone[torch.Tensor]
):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
