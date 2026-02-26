"""Tests for GaussianCopula."""

import pytest

import numpy as np
from scipy import stats

from pyapprox.probability.copula.correlation.cholesky import (
    CholeskyCorrelationParameterization,
)
from pyapprox.probability.copula.gaussian import GaussianCopula
from pyapprox.util.backends.numpy import NumpyBkd


class TestGaussianCopula:

    def _make_copula(self, bkd, nvars=3):
        # L10=0.5, L20=0.3, L21=0.4
        chol_lower = bkd.asarray([0.5, 0.3, 0.4])
        corr_param = CholeskyCorrelationParameterization(
            chol_lower, nvars, bkd
        )
        return GaussianCopula(corr_param, bkd), corr_param

    def test_logpdf_shape(self, bkd) -> None:
        nvars = 3
        copula, _ = self._make_copula(bkd, nvars)
        np.random.seed(42)
        u = bkd.asarray(
            np.random.uniform(0.01, 0.99, (nvars, 20)).astype(np.float64)
        )
        result = copula.logpdf(u)
        assert result.shape == (1, 20)

    def test_logpdf_identity_correlation(self, bkd) -> None:
        """With Sigma=I, the copula density is 1 (log c = 0)."""
        nvars = 3
        nparams = nvars * (nvars - 1) // 2
        chol_lower = bkd.zeros((nparams,))
        corr_param = CholeskyCorrelationParameterization(
            chol_lower, nvars, bkd
        )
        copula = GaussianCopula(corr_param, bkd)

        np.random.seed(42)
        u = bkd.asarray(
            np.random.uniform(0.01, 0.99, (nvars, 50)).astype(np.float64)
        )
        result = copula.logpdf(u)
        expected = bkd.zeros((1, 50))
        bkd.assert_allclose(result, expected, atol=1e-10)

    def test_logpdf_vs_scipy_mvn(self, bkd) -> None:
        """Compare against scipy: log c(u) = logpdf_MVN(z) - sum logpdf_N(z_i)."""
        nvars = 3
        copula, corr_param = self._make_copula(bkd, nvars)
        np.random.seed(42)
        u_np = np.random.uniform(0.01, 0.99, (nvars, 30))
        u = bkd.asarray(u_np.astype(np.float64))

        # Get the correlation matrix
        sigma_np = bkd.to_numpy(corr_param.correlation_matrix())

        # Compute z = Phi^{-1}(u)
        z_np = stats.norm.ppf(u_np)

        # Expected: log f_MVN(z; 0, Sigma) - sum_i log f_N(z_i; 0, 1)
        mvn = stats.multivariate_normal(mean=np.zeros(nvars), cov=sigma_np)
        expected_np = mvn.logpdf(z_np.T) - np.sum(stats.norm.logpdf(z_np), axis=0)

        actual = copula.logpdf(u)
        actual_np = bkd.to_numpy(actual)[0]

        np_bkd = NumpyBkd()
        np_bkd.assert_allclose(actual_np, expected_np, rtol=1e-8)

    def test_sample_shape_and_range(self, bkd) -> None:
        nvars = 3
        copula, _ = self._make_copula(bkd, nvars)
        np.random.seed(42)
        samples = copula.sample(100)
        assert samples.shape == (nvars, 100)
        # All values should be in (0, 1)
        samples_np = bkd.to_numpy(samples)
        assert np.all(samples_np > 0.0)
        assert np.all(samples_np < 1.0)

    def test_sample_kendall_tau(self, bkd) -> None:
        """For bivariate Gaussian copula, tau = 2/pi * arcsin(rho)."""
        rho = 0.6
        chol_lower = bkd.asarray([rho])
        corr_param = CholeskyCorrelationParameterization(chol_lower, 2, bkd)
        copula = GaussianCopula(corr_param, bkd)

        np.random.seed(42)
        samples = copula.sample(10000)
        samples_np = bkd.to_numpy(samples)

        # Compute empirical Kendall's tau
        tau_stat, _ = stats.kendalltau(samples_np[0], samples_np[1])
        expected_tau = 2.0 / np.pi * np.arcsin(rho)
        np.testing.assert_allclose(tau_stat, expected_tau, atol=0.03)

    def test_input_validation_1d(self, bkd) -> None:
        copula, _ = self._make_copula(bkd)
        u_1d = bkd.asarray([0.5, 0.5, 0.5])
        with pytest.raises(ValueError):
            copula.logpdf(u_1d)

    def test_input_validation_wrong_nvars(self, bkd) -> None:
        copula, _ = self._make_copula(bkd)
        u = bkd.asarray(np.random.uniform(0.01, 0.99, (2, 10)).astype(np.float64))
        with pytest.raises(ValueError):
            copula.logpdf(u)

    def test_kl_divergence_same_is_zero(self, bkd) -> None:
        copula, _ = self._make_copula(bkd)
        kl = copula.kl_divergence(copula)
        bkd.assert_allclose(
            bkd.atleast_1d(kl),
            bkd.zeros((1,)),
            atol=1e-10,
        )

    def test_kl_divergence_positive(self, bkd) -> None:
        nvars = 3
        copula, _ = self._make_copula(bkd, nvars)
        # Create a different copula
        chol_lower2 = bkd.asarray([0.3, 0.1, 0.2])
        corr_param2 = CholeskyCorrelationParameterization(
            chol_lower2, nvars, bkd
        )
        copula2 = GaussianCopula(corr_param2, bkd)

        kl = copula.kl_divergence(copula2)
        assert float(bkd.to_numpy(kl)) > 0.0
