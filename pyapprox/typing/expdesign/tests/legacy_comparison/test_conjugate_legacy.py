"""
Legacy comparison tests for conjugate Gaussian OED utilities.

TODO: Delete after legacy removed.

These tests verify that the typing conjugate utilities produce the same
results as the legacy implementation.
"""

import unittest

import numpy as np

from pyapprox.util.backends.numpy import NumpyMixin


class TestConjugateOEDLegacyComparison(unittest.TestCase):
    """Verify typing conjugate OED utilities match legacy."""

    def setUp(self):
        from pyapprox.typing.util.backends.numpy import NumpyBkd
        self._bkd = NumpyBkd()

    def test_expected_stdev_matches_legacy(self):
        """Test expected std dev matches legacy."""
        np.random.seed(42)
        nobs = 3
        nvars = 4
        nqoi = 1

        # Setup shared data
        prior_mean = np.random.randn(nvars, 1)
        prior_cov = np.eye(nvars) * 0.5
        obs_mat = np.random.randn(nobs, nvars)
        noise_cov = np.diag(np.random.uniform(0.1, 0.5, nobs))
        qoi_mat = np.random.randn(nqoi, nvars)

        # Legacy
        from pyapprox.variables.gaussian import DenseCholeskyMultivariateGaussian
        from pyapprox.expdesign.bayesoed_benchmarks import (
            ConjugateGaussianOEDForNormalExpectedStdDev,
        )

        legacy_prior = DenseCholeskyMultivariateGaussian(
            prior_mean, prior_cov, backend=NumpyMixin
        )
        legacy_utility = ConjugateGaussianOEDForNormalExpectedStdDev(
            legacy_prior, qoi_mat
        )
        legacy_utility.set_observation_matrix(obs_mat)
        legacy_utility.set_noise_covariance(noise_cov)
        legacy_result = legacy_utility.value()

        # Typing
        from pyapprox.typing.util.backends.numpy import NumpyBkd
        from pyapprox.typing.expdesign.analytical import (
            ConjugateGaussianOEDExpectedStdDev,
        )

        bkd = NumpyBkd()
        typing_utility = ConjugateGaussianOEDExpectedStdDev(
            bkd.asarray(prior_mean),
            bkd.asarray(prior_cov),
            bkd.asarray(qoi_mat),
            bkd,
        )
        typing_utility.set_observation_matrix(bkd.asarray(obs_mat))
        typing_utility.set_noise_covariance(bkd.asarray(noise_cov))
        typing_result = typing_utility.value()

        self._bkd.assert_allclose(
            self._bkd.asarray(typing_result).reshape(-1),
            self._bkd.asarray(legacy_result).reshape(-1),
            rtol=1e-12,
        )

    def test_expected_entropic_dev_matches_legacy(self):
        """Test expected entropic deviation matches legacy."""
        np.random.seed(42)
        nobs = 3
        nvars = 4
        nqoi = 1
        lamda = 2.0

        # Setup shared data
        prior_mean = np.random.randn(nvars, 1)
        prior_cov = np.eye(nvars) * 0.5
        obs_mat = np.random.randn(nobs, nvars)
        noise_cov = np.diag(np.random.uniform(0.1, 0.5, nobs))
        qoi_mat = np.random.randn(nqoi, nvars)

        # Legacy
        from pyapprox.variables.gaussian import DenseCholeskyMultivariateGaussian
        from pyapprox.expdesign.bayesoed_benchmarks import (
            ConjugateGaussianOEDForNormalExpectedEntropicDev,
        )

        legacy_prior = DenseCholeskyMultivariateGaussian(
            prior_mean, prior_cov, backend=NumpyMixin
        )
        legacy_utility = ConjugateGaussianOEDForNormalExpectedEntropicDev(
            legacy_prior, qoi_mat, lamda
        )
        legacy_utility.set_observation_matrix(obs_mat)
        legacy_utility.set_noise_covariance(noise_cov)
        legacy_result = legacy_utility.value()

        # Typing
        from pyapprox.typing.util.backends.numpy import NumpyBkd
        from pyapprox.typing.expdesign.analytical import (
            ConjugateGaussianOEDExpectedEntropicDev,
        )

        bkd = NumpyBkd()
        typing_utility = ConjugateGaussianOEDExpectedEntropicDev(
            bkd.asarray(prior_mean),
            bkd.asarray(prior_cov),
            bkd.asarray(qoi_mat),
            lamda,
            bkd,
        )
        typing_utility.set_observation_matrix(bkd.asarray(obs_mat))
        typing_utility.set_noise_covariance(bkd.asarray(noise_cov))
        typing_result = typing_utility.value()

        self._bkd.assert_allclose(
            self._bkd.asarray(typing_result).reshape(-1),
            self._bkd.asarray(legacy_result).reshape(-1),
            rtol=1e-12,
        )

    def test_expected_avar_dev_matches_legacy(self):
        """Test expected AVaR deviation matches legacy."""
        np.random.seed(42)
        nobs = 3
        nvars = 4
        nqoi = 1
        beta = 0.75

        # Setup shared data
        prior_mean = np.random.randn(nvars, 1)
        prior_cov = np.eye(nvars) * 0.5
        obs_mat = np.random.randn(nobs, nvars)
        noise_cov = np.diag(np.random.uniform(0.1, 0.5, nobs))
        qoi_mat = np.random.randn(nqoi, nvars)

        # Legacy
        from pyapprox.variables.gaussian import DenseCholeskyMultivariateGaussian
        from pyapprox.expdesign.bayesoed_benchmarks import (
            ConjugateGaussianOEDForNormalExpectedAVaRDev,
        )

        legacy_prior = DenseCholeskyMultivariateGaussian(
            prior_mean, prior_cov, backend=NumpyMixin
        )
        legacy_utility = ConjugateGaussianOEDForNormalExpectedAVaRDev(
            legacy_prior, qoi_mat, beta
        )
        legacy_utility.set_observation_matrix(obs_mat)
        legacy_utility.set_noise_covariance(noise_cov)
        legacy_result = legacy_utility.value()

        # Typing
        from pyapprox.typing.util.backends.numpy import NumpyBkd
        from pyapprox.typing.expdesign.analytical import (
            ConjugateGaussianOEDExpectedAVaRDev,
        )

        bkd = NumpyBkd()
        typing_utility = ConjugateGaussianOEDExpectedAVaRDev(
            bkd.asarray(prior_mean),
            bkd.asarray(prior_cov),
            bkd.asarray(qoi_mat),
            beta,
            bkd,
        )
        typing_utility.set_observation_matrix(bkd.asarray(obs_mat))
        typing_utility.set_noise_covariance(bkd.asarray(noise_cov))
        typing_result = typing_utility.value()

        self._bkd.assert_allclose(
            self._bkd.asarray(typing_result).reshape(-1),
            self._bkd.asarray(legacy_result).reshape(-1),
            rtol=1e-12,
        )

    def test_expected_kl_divergence_matches_legacy(self):
        """Test expected KL divergence matches legacy."""
        np.random.seed(42)
        nobs = 3
        nvars = 4
        nqoi = 1

        # Setup shared data
        prior_mean = np.random.randn(nvars, 1)
        prior_cov = np.eye(nvars) * 0.5
        obs_mat = np.random.randn(nobs, nvars)
        noise_cov = np.diag(np.random.uniform(0.1, 0.5, nobs))
        qoi_mat = np.random.randn(nqoi, nvars)

        # Legacy
        from pyapprox.variables.gaussian import DenseCholeskyMultivariateGaussian
        from pyapprox.expdesign.bayesoed_benchmarks import (
            ConjugateGaussianOEDForNormalExpectedKLDivergence,
        )

        legacy_prior = DenseCholeskyMultivariateGaussian(
            prior_mean, prior_cov, backend=NumpyMixin
        )
        legacy_utility = ConjugateGaussianOEDForNormalExpectedKLDivergence(
            legacy_prior, qoi_mat
        )
        legacy_utility.set_observation_matrix(obs_mat)
        legacy_utility.set_noise_covariance(noise_cov)
        legacy_result = legacy_utility.value()

        # Typing
        from pyapprox.typing.util.backends.numpy import NumpyBkd
        from pyapprox.typing.expdesign.analytical import (
            ConjugateGaussianOEDExpectedKLDivergence,
        )

        bkd = NumpyBkd()
        typing_utility = ConjugateGaussianOEDExpectedKLDivergence(
            bkd.asarray(prior_mean),
            bkd.asarray(prior_cov),
            bkd.asarray(qoi_mat),
            bkd,
        )
        typing_utility.set_observation_matrix(bkd.asarray(obs_mat))
        typing_utility.set_noise_covariance(bkd.asarray(noise_cov))
        typing_result = typing_utility.value()

        self._bkd.assert_allclose(
            self._bkd.asarray(typing_result).reshape(-1),
            self._bkd.asarray(legacy_result).reshape(-1),
            rtol=1e-12,
        )

    def test_lognormal_expected_stdev_matches_legacy(self):
        """Test lognormal expected std dev matches legacy."""
        np.random.seed(42)
        nobs = 3
        nvars = 4
        nqoi = 1

        # Setup shared data
        prior_mean = np.random.randn(nvars, 1)
        prior_cov = np.eye(nvars) * 0.5
        obs_mat = np.random.randn(nobs, nvars)
        noise_cov = np.diag(np.random.uniform(0.1, 0.5, nobs))
        qoi_mat = np.random.randn(nqoi, nvars)

        # Legacy
        from pyapprox.variables.gaussian import DenseCholeskyMultivariateGaussian
        from pyapprox.expdesign.bayesoed_benchmarks import (
            ConjugateGaussianOEDForLogNormalExpectedStdDev,
        )

        legacy_prior = DenseCholeskyMultivariateGaussian(
            prior_mean, prior_cov, backend=NumpyMixin
        )
        legacy_utility = ConjugateGaussianOEDForLogNormalExpectedStdDev(
            legacy_prior, qoi_mat
        )
        legacy_utility.set_observation_matrix(obs_mat)
        legacy_utility.set_noise_covariance(noise_cov)
        legacy_result = legacy_utility.value()

        # Typing
        from pyapprox.typing.util.backends.numpy import NumpyBkd
        from pyapprox.typing.expdesign.analytical import (
            ConjugateGaussianOEDForLogNormalExpectedStdDev,
        )

        bkd = NumpyBkd()
        typing_utility = ConjugateGaussianOEDForLogNormalExpectedStdDev(
            bkd.asarray(prior_mean),
            bkd.asarray(prior_cov),
            bkd.asarray(qoi_mat),
            bkd,
        )
        typing_utility.set_observation_matrix(bkd.asarray(obs_mat))
        typing_utility.set_noise_covariance(bkd.asarray(noise_cov))
        typing_result = typing_utility.value()

        self._bkd.assert_allclose(
            self._bkd.asarray(typing_result).reshape(-1),
            self._bkd.asarray(legacy_result).reshape(-1),
            rtol=1e-12,
        )

    def test_lognormal_avar_stdev_matches_legacy(self):
        """Test lognormal AVaR std dev matches legacy."""
        np.random.seed(42)
        nobs = 3
        nvars = 4
        nqoi = 1
        beta = 0.5

        # Setup shared data
        prior_mean = np.random.randn(nvars, 1)
        prior_cov = np.eye(nvars) * 0.5
        obs_mat = np.random.randn(nobs, nvars)
        noise_cov = np.diag(np.random.uniform(0.1, 0.5, nobs))
        qoi_mat = np.random.randn(nqoi, nvars)

        # Legacy
        from pyapprox.variables.gaussian import DenseCholeskyMultivariateGaussian
        from pyapprox.expdesign.bayesoed_benchmarks import (
            ConjugateGaussianOEDForLogNormalAVaRStdDev,
        )

        legacy_prior = DenseCholeskyMultivariateGaussian(
            prior_mean, prior_cov, backend=NumpyMixin
        )
        legacy_utility = ConjugateGaussianOEDForLogNormalAVaRStdDev(
            legacy_prior, qoi_mat, beta
        )
        legacy_utility.set_observation_matrix(obs_mat)
        legacy_utility.set_noise_covariance(noise_cov)
        legacy_result = legacy_utility.value()

        # Typing
        from pyapprox.typing.util.backends.numpy import NumpyBkd
        from pyapprox.typing.expdesign.analytical import (
            ConjugateGaussianOEDForLogNormalAVaRStdDev,
        )

        bkd = NumpyBkd()
        typing_utility = ConjugateGaussianOEDForLogNormalAVaRStdDev(
            bkd.asarray(prior_mean),
            bkd.asarray(prior_cov),
            bkd.asarray(qoi_mat),
            beta,
            bkd,
        )
        typing_utility.set_observation_matrix(bkd.asarray(obs_mat))
        typing_utility.set_noise_covariance(bkd.asarray(noise_cov))
        typing_result = typing_utility.value()

        self._bkd.assert_allclose(
            self._bkd.asarray(typing_result).reshape(-1),
            self._bkd.asarray(legacy_result).reshape(-1),
            rtol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
