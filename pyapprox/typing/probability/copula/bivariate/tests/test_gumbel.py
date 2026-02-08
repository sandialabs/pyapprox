"""Tests for GumbelCopula."""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray
from scipy import stats

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.probability.copula.bivariate.gumbel import (
    GumbelCopula,
)
from pyapprox.typing.probability.copula.bivariate.protocols import (
    BivariateCopulaProtocol,
)
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401


def _gumbel_cdf_reference(u1: np.ndarray, u2: np.ndarray,
                           theta: float) -> np.ndarray:
    """Reference Gumbel CDF (test helper only)."""
    A = (-np.log(u1))**theta + (-np.log(u2))**theta
    return np.exp(-A**(1.0/theta))


class TestGumbelCopula(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self._theta = 2.0
        self._copula = GumbelCopula(self._theta, self._bkd)

    def test_satisfies_protocol(self) -> None:
        self.assertIsInstance(self._copula, BivariateCopulaProtocol)

    def test_nparams(self) -> None:
        self.assertEqual(self._copula.nparams(), 1)

    def test_logpdf_shape(self) -> None:
        np.random.seed(42)
        u = self._bkd.asarray(
            np.random.uniform(0.01, 0.99, (2, 20)).astype(np.float64)
        )
        result = self._copula.logpdf(u)
        self.assertEqual(result.shape, (1, 20))

    def test_logpdf_vs_numerical_cdf_derivative(self) -> None:
        """Verify c(u1,u2) = d^2C/du1du2 via finite differences on CDF."""
        np.random.seed(42)
        u_np = np.random.uniform(0.1, 0.9, (2, 15)).astype(np.float64)
        eps = 1e-5
        theta = self._theta

        u = self._bkd.asarray(u_np)
        logpdf = self._copula.logpdf(u)
        pdf_np = self._bkd.to_numpy(self._bkd.exp(logpdf))[0]

        C_pp = _gumbel_cdf_reference(u_np[0] + eps, u_np[1] + eps, theta)
        C_pm = _gumbel_cdf_reference(u_np[0] + eps, u_np[1] - eps, theta)
        C_mp = _gumbel_cdf_reference(u_np[0] - eps, u_np[1] + eps, theta)
        C_mm = _gumbel_cdf_reference(u_np[0] - eps, u_np[1] - eps, theta)

        fd_pdf = (C_pp - C_pm - C_mp + C_mm) / (4.0 * eps * eps)

        np.testing.assert_allclose(pdf_np, fd_pdf, rtol=1e-4)

    def test_h_function_is_conditional_cdf(self) -> None:
        """h(u1|u2) = dC(u1,u2)/du2 via finite differences."""
        np.random.seed(42)
        u1_np = np.random.uniform(0.1, 0.9, (1, 15)).astype(np.float64)
        u2_np = np.random.uniform(0.1, 0.9, (1, 15)).astype(np.float64)
        eps = 1e-5
        theta = self._theta

        u1 = self._bkd.asarray(u1_np)
        u2 = self._bkd.asarray(u2_np)

        h_val = self._copula.h_function(u1, u2)
        h_np = self._bkd.to_numpy(h_val)[0]

        C_plus = _gumbel_cdf_reference(u1_np[0], u2_np[0] + eps, theta)
        C_minus = _gumbel_cdf_reference(u1_np[0], u2_np[0] - eps, theta)
        fd_h = (C_plus - C_minus) / (2.0 * eps)

        np.testing.assert_allclose(h_np, fd_h, rtol=1e-4)

    def test_h_inverse_roundtrip(self) -> None:
        """h_inverse(h(u1, u2), u2) = u1."""
        np.random.seed(42)
        u1 = self._bkd.asarray(
            np.random.uniform(0.05, 0.95, (1, 30)).astype(np.float64)
        )
        u2 = self._bkd.asarray(
            np.random.uniform(0.05, 0.95, (1, 30)).astype(np.float64)
        )

        v = self._copula.h_function(u1, u2)
        u1_recovered = self._copula.h_inverse(v, u2)
        self._bkd.assert_allclose(u1_recovered, u1, rtol=1e-6)

    def test_sample_shape_and_range(self) -> None:
        np.random.seed(42)
        samples = self._copula.sample(100)
        self.assertEqual(samples.shape, (2, 100))
        samples_np = self._bkd.to_numpy(samples)
        self.assertTrue(np.all(samples_np > 0.0))
        self.assertTrue(np.all(samples_np < 1.0))

    def test_sample_kendall_tau(self) -> None:
        """Empirical Kendall's tau matches theoretical."""
        np.random.seed(42)
        samples = self._copula.sample(10000)
        samples_np = self._bkd.to_numpy(samples)

        tau_stat, _ = stats.kendalltau(samples_np[0], samples_np[1])
        expected_tau = 1.0 - 1.0 / self._theta
        np.testing.assert_allclose(tau_stat, expected_tau, atol=0.03)

    def test_kendall_tau_value(self) -> None:
        """Analytical Kendall's tau = 1 - 1/theta."""
        tau = self._copula.kendall_tau()
        expected = 1.0 - 1.0 / self._theta
        self._bkd.assert_allclose(
            self._bkd.atleast_1d(tau),
            self._bkd.asarray([expected]),
            rtol=1e-12,
        )

    def test_input_validation_1d(self) -> None:
        u_1d = self._bkd.asarray([0.5, 0.5])
        with self.assertRaises(ValueError):
            self._copula.logpdf(u_1d)

    def test_input_validation_wrong_nvars(self) -> None:
        u = self._bkd.asarray(
            np.random.uniform(0.01, 0.99, (3, 10)).astype(np.float64)
        )
        with self.assertRaises(ValueError):
            self._copula.logpdf(u)

    def test_strong_dependence(self) -> None:
        """Large theta produces high dependence."""
        copula = GumbelCopula(5.0, self._bkd)
        np.random.seed(42)
        samples = copula.sample(5000)
        samples_np = self._bkd.to_numpy(samples)
        tau_stat, _ = stats.kendalltau(samples_np[0], samples_np[1])
        expected_tau = 1.0 - 1.0 / 5.0
        np.testing.assert_allclose(tau_stat, expected_tau, atol=0.03)


class TestGumbelCopulaNumpy(TestGumbelCopula[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGumbelCopulaTorch(TestGumbelCopula[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()
