"""Tests for FrankCopula."""

import numpy as np
import pytest
from scipy import stats

from pyapprox.probability.copula.bivariate.frank import (
    FrankCopula,
)
from pyapprox.probability.copula.bivariate.protocols import (
    BivariateCopulaProtocol,
)

# TODO: Fix typing issues
# TODO: Do not use np.testing.assert_allclose use bkd.assert_allclose
# TODO: do not use astype, this will break if we want to use float32
# let backend do correct conversion

def _frank_cdf_reference(u1: np.ndarray, u2: np.ndarray, theta: float) -> np.ndarray:
    """Reference Frank CDF (test helper only)."""
    e1 = np.exp(-theta * u1)
    e2 = np.exp(-theta * u2)
    et = np.exp(-theta)
    return -1.0 / theta * np.log(1.0 + (e1 - 1.0) * (e2 - 1.0) / (et - 1.0))


class TestFrankCopula:

    def _make_copula(self, bkd, theta=5.0):
        return FrankCopula(theta, bkd)

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

    def test_logpdf_vs_numerical_cdf_derivative(self, bkd) -> None:
        """Verify c(u1,u2) = d^2C/du1du2 via finite differences on CDF."""
        #TODO: replace with DerivativeChecker
        theta = 5.0
        copula = self._make_copula(bkd, theta)
        np.random.seed(42)
        u_np = np.random.uniform(0.1, 0.9, (2, 15)).astype(np.float64)
        eps = 1e-5

        u = bkd.asarray(u_np)
        logpdf = copula.logpdf(u)
        pdf_np = bkd.to_numpy(bkd.exp(logpdf))[0]

        C_pp = _frank_cdf_reference(u_np[0] + eps, u_np[1] + eps, theta)
        C_pm = _frank_cdf_reference(u_np[0] + eps, u_np[1] - eps, theta)
        C_mp = _frank_cdf_reference(u_np[0] - eps, u_np[1] + eps, theta)
        C_mm = _frank_cdf_reference(u_np[0] - eps, u_np[1] - eps, theta)

        fd_pdf = (C_pp - C_pm - C_mp + C_mm) / (4.0 * eps * eps)

        np.testing.assert_allclose(pdf_np, fd_pdf, rtol=1e-4)

    def test_h_function_is_conditional_cdf(self, bkd) -> None:
        """h(u1|u2) = dC(u1,u2)/du2 via finite differences."""
        theta = 5.0
        copula = self._make_copula(bkd, theta)
        np.random.seed(42)
        u1_np = np.random.uniform(0.1, 0.9, (1, 15)).astype(np.float64)
        u2_np = np.random.uniform(0.1, 0.9, (1, 15)).astype(np.float64)
        eps = 1e-5

        u1 = bkd.asarray(u1_np)
        u2 = bkd.asarray(u2_np)

        h_val = copula.h_function(u1, u2)
        h_np = bkd.to_numpy(h_val)[0]

        C_plus = _frank_cdf_reference(u1_np[0], u2_np[0] + eps, theta)
        C_minus = _frank_cdf_reference(u1_np[0], u2_np[0] - eps, theta)
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
        assert np.all(samples_np > 0.0)
        assert np.all(samples_np < 1.0)

    def test_sample_kendall_tau(self, bkd) -> None:
        """Empirical Kendall's tau matches theoretical."""
        copula = self._make_copula(bkd)
        np.random.seed(42)
        samples = copula.sample(10000)
        samples_np = bkd.to_numpy(samples)

        tau_stat, _ = stats.kendalltau(samples_np[0], samples_np[1])
        tau_expected = float(bkd.to_numpy(copula.kendall_tau()))
        np.testing.assert_allclose(tau_stat, tau_expected, atol=0.03)

    def test_kendall_tau_value(self, bkd) -> None:
        """Kendall's tau = 1 - 4/theta * (1 - D_1(theta))."""
        theta = 5.0
        copula = self._make_copula(bkd, theta)
        tau = copula.kendall_tau()
        # For theta=5, compute D_1(5) numerically
        from scipy.integrate import quad

        D1, _ = quad(lambda t: t / (np.exp(t) - 1.0), 0, theta)
        D1 /= theta
        expected = 1.0 - 4.0 / theta * (1.0 - D1)
        bkd.assert_allclose(
            bkd.atleast_1d(tau),
            bkd.asarray([expected]),
            rtol=1e-6,
        )

    def test_negative_theta(self, bkd) -> None:
        """Negative theta (negative dependence) should work."""
        copula = FrankCopula(-3.0, bkd)
        np.random.seed(42)
        # TODO: do not use astype, this will break if we want to use float32
        # use 0., 1. so floats are created and let abckend do correct conversion
        u = bkd.asarray(np.random.uniform(0.05, 0.95, (2, 20)).astype(np.float64))
        result = copula.logpdf(u)
        assert result.shape == (1, 20)

        # Kendall's tau should be negative
        tau = copula.kendall_tau()
        assert float(bkd.to_numpy(tau)) < 0.0

    def test_input_validation_1d(self, bkd) -> None:
        copula = self._make_copula(bkd)
        u_1d = bkd.asarray([0.5, 0.5])
        with pytest.raises(ValueError):
            copula.logpdf(u_1d)

    def test_input_validation_wrong_nvars(self, bkd) -> None:
        copula = self._make_copula(bkd)
        # TODO: do not use astype, this will break if we want to use float32
        # use 0., 1. so floats are created and let abckend do correct conversion
        u = bkd.asarray(np.random.uniform(0.01, 0.99, (3, 10)).astype(np.float64))
        with pytest.raises(ValueError):
            copula.logpdf(u)
