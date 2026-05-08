"""Tests for BivariateGaussianCopula."""

import math

import numpy as np
import pytest
from scipy import stats

from pyapprox.probability.copula.bivariate.gaussian import (
    BivariateGaussianCopula,
)
from pyapprox.probability.copula.bivariate.protocols import (
    BivariateCopulaProtocol,
)
from pyapprox.probability.copula.correlation.cholesky import (
    CholeskyCorrelationParameterization,
)
from pyapprox.probability.copula.gaussian import GaussianCopula

# TODO: Fix typing issues
# TODO: Do not use np.testing.assert_allclose use bkd.assert_allclose

def _scipy_bivariate_normal_cdf(
    z1: np.ndarray, z2: np.ndarray, rho: float
) -> np.ndarray:
    """Reference bivariate normal CDF using scipy (test helper only)."""
    cov = np.array([[1.0, rho], [rho, 1.0]])
    points = np.column_stack([z1.ravel(), z2.ravel()])
    return stats.multivariate_normal.cdf(points, cov=cov)


class TestBivariateGaussianCopula:

    def _make_copula(self, bkd, rho=0.6):
        return BivariateGaussianCopula(rho, bkd)

    def test_satisfies_protocol(self, bkd) -> None:
        copula = self._make_copula(bkd)
        assert isinstance(copula, BivariateCopulaProtocol)

    def test_nparams(self, bkd) -> None:
        copula = self._make_copula(bkd)
        assert copula.nparams() == 1

    def test_logpdf_shape(self, bkd) -> None:
        copula = self._make_copula(bkd)
        np.random.seed(42)
        u = bkd.asarray(np.random.uniform(0.01, 0.99, (2, 20)).astype(np.float64))
        result = copula.logpdf(u)
        assert result.shape == (1, 20)

    def test_logpdf_at_independence(self, bkd) -> None:
        """With rho=0, the copula density is 1 (log c = 0)."""
        copula = BivariateGaussianCopula(0.0, bkd)
        np.random.seed(42)
        u = bkd.asarray(np.random.uniform(0.01, 0.99, (2, 50)).astype(np.float64))
        result = copula.logpdf(u)
        expected = bkd.zeros((1, 50))
        bkd.assert_allclose(result, expected, atol=1e-10)

    def test_logpdf_vs_numerical_cdf_derivative(self, bkd) -> None:
        """Verify c(u1,u2) = d^2C/du1du2 via finite differences on scipy CDF."""
        # TODO: replace with DerivativeChecker
        rho = 0.6
        copula = self._make_copula(bkd, rho)
        np.random.seed(42)
        u_np = np.random.uniform(0.1, 0.9, (2, 15)).astype(np.float64)
        eps = 1e-5

        u = bkd.asarray(u_np)
        logpdf = copula.logpdf(u)
        pdf_np = bkd.to_numpy(bkd.exp(logpdf))[0]

        # d^2C/du1du2 via finite differences on scipy bivariate normal CDF
        stats.norm.ppf(u_np)
        z_pp = stats.norm.ppf(u_np + np.array([[eps], [eps]]))
        z_pm = stats.norm.ppf(u_np + np.array([[eps], [-eps]]))
        z_mp = stats.norm.ppf(u_np + np.array([[-eps], [eps]]))
        z_mm = stats.norm.ppf(u_np + np.array([[-eps], [-eps]]))

        C_pp = _scipy_bivariate_normal_cdf(z_pp[0], z_pp[1], rho)
        C_pm = _scipy_bivariate_normal_cdf(z_pm[0], z_pm[1], rho)
        C_mp = _scipy_bivariate_normal_cdf(z_mp[0], z_mp[1], rho)
        C_mm = _scipy_bivariate_normal_cdf(z_mm[0], z_mm[1], rho)

        fd_pdf = (C_pp - C_pm - C_mp + C_mm) / (4.0 * eps * eps)

        np.testing.assert_allclose(pdf_np, fd_pdf, rtol=1e-4)

    def test_bivariate_matches_multivariate(self, bkd) -> None:
        """BivariateGaussianCopula.logpdf matches GaussianCopula(d=2).logpdf."""
        rho = 0.6
        copula = self._make_copula(bkd, rho)
        chol_lower = bkd.asarray([rho])
        corr_param = CholeskyCorrelationParameterization(chol_lower, 2, bkd)
        mv_copula = GaussianCopula(corr_param, bkd)

        np.random.seed(42)
        u = bkd.asarray(np.random.uniform(0.01, 0.99, (2, 50)).astype(np.float64))

        bivar_logpdf = copula.logpdf(u)
        mv_logpdf = mv_copula.logpdf(u)
        bkd.assert_allclose(bivar_logpdf, mv_logpdf, rtol=1e-10)

    def test_h_function_is_conditional_cdf(self, bkd) -> None:
        """h(u1|u2) = dC(u1,u2)/du2 via finite differences on scipy CDF."""
        rho = 0.6
        copula = self._make_copula(bkd, rho)
        np.random.seed(42)
        # TODO: do not use astype, this will break if we want to use float32
        # use 0., 1. so floats are created and let backend do correct conversion
        u1_np = np.random.uniform(0.1, 0.9, (1, 15)).astype(np.float64)
        u2_np = np.random.uniform(0.1, 0.9, (1, 15)).astype(np.float64)
        eps = 1e-5

        u1 = bkd.asarray(u1_np)
        u2 = bkd.asarray(u2_np)

        h_val = copula.h_function(u1, u2)
        h_np = bkd.to_numpy(h_val)[0]

        # dC/du2 via central differences on scipy CDF
        z1 = stats.norm.ppf(u1_np[0])
        z2_plus = stats.norm.ppf(u2_np[0] + eps)
        z2_minus = stats.norm.ppf(u2_np[0] - eps)

        C_plus = _scipy_bivariate_normal_cdf(z1, z2_plus, rho)
        C_minus = _scipy_bivariate_normal_cdf(z1, z2_minus, rho)
        fd_h = (C_plus - C_minus) / (2.0 * eps)

        np.testing.assert_allclose(h_np, fd_h, rtol=1e-4)

    def test_h_inverse_roundtrip(self, bkd) -> None:
        """h_inverse(h(u1, u2), u2) = u1."""
        copula = self._make_copula(bkd)
        np.random.seed(42)
        u1 = bkd.asarray(
            np.random.uniform(0.05, 0.95, (1, 30)).astype(np.float64)
        )
        u2 = bkd.asarray(
            np.random.uniform(0.05, 0.95, (1, 30)).astype(np.float64)
        )

        v = copula.h_function(u1, u2)
        u1_recovered = copula.h_inverse(v, u2)
        bkd.assert_allclose(u1_recovered, u1, rtol=1e-10)

    def test_sample_shape_and_range(self, bkd) -> None:
        copula = self._make_copula(bkd)
        np.random.seed(42)
        samples = copula.sample(100)
        assert samples.shape == (2, 100)
        samples_np = bkd.to_numpy(samples)
        # TODO: use bkd.all_bool
        assert np.all(samples_np > 0.0)
        assert np.all(samples_np < 1.0)

    def test_sample_kendall_tau(self, bkd) -> None:
        """Empirical Kendall's tau matches theoretical."""
        rho = 0.6
        copula = self._make_copula(bkd, rho)
        np.random.seed(42)
        samples = copula.sample(10000)
        samples_np = bkd.to_numpy(samples)

        tau_stat, _ = stats.kendalltau(samples_np[0], samples_np[1])
        expected_tau = 2.0 / math.pi * math.asin(rho)
        np.testing.assert_allclose(tau_stat, expected_tau, atol=0.03)

    def test_kendall_tau_value(self, bkd) -> None:
        """Analytical Kendall's tau = 2/pi * arcsin(rho)."""
        rho = 0.6
        copula = self._make_copula(bkd, rho)
        tau = copula.kendall_tau()
        expected = 2.0 / math.pi * math.asin(rho)
        bkd.assert_allclose(
            bkd.atleast_1d(tau),
            bkd.asarray([expected]),
            rtol=1e-12,
        )

    def test_input_validation_1d(self, bkd) -> None:
        copula = self._make_copula(bkd)
        u_1d = bkd.asarray([0.5, 0.5])
        with pytest.raises(ValueError):
            copula.logpdf(u_1d)

    def test_input_validation_wrong_nvars(self, bkd) -> None:
        copula = self._make_copula(bkd)
        u = bkd.asarray(np.random.uniform(0.01, 0.99, (3, 10)).astype(np.float64))
        with pytest.raises(ValueError):
            copula.logpdf(u)

    def test_negative_rho(self, bkd) -> None:
        """Negative correlation should work correctly."""
        copula = BivariateGaussianCopula(-0.5, bkd)
        np.random.seed(42)
        u = bkd.asarray(np.random.uniform(0.05, 0.95, (2, 20)).astype(np.float64))
        result = copula.logpdf(u)
        assert result.shape == (1, 20)

        # Cross-validate with multivariate
        chol_lower = bkd.asarray([-0.5])
        corr_param = CholeskyCorrelationParameterization(chol_lower, 2, bkd)
        mv_copula = GaussianCopula(corr_param, bkd)
        mv_logpdf = mv_copula.logpdf(u)
        bkd.assert_allclose(result, mv_logpdf, rtol=1e-10)
