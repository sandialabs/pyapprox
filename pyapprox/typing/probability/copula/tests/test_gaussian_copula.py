"""Tests for GaussianCopula."""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray
from scipy import stats

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.probability.copula.correlation.cholesky import (
    CholeskyCorrelationParameterization,
)
from pyapprox.typing.probability.copula.gaussian import GaussianCopula
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401


class TestGaussianCopula(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        # 3D Gaussian copula with known correlation
        self._nvars = 3
        # L10=0.5, L20=0.3, L21=0.4
        chol_lower = self._bkd.asarray([0.5, 0.3, 0.4])
        self._corr_param = CholeskyCorrelationParameterization(
            chol_lower, self._nvars, self._bkd
        )
        self._copula = GaussianCopula(self._corr_param, self._bkd)

    def test_logpdf_shape(self) -> None:
        np.random.seed(42)
        u = self._bkd.asarray(
            np.random.uniform(0.01, 0.99, (self._nvars, 20)).astype(
                np.float64
            )
        )
        result = self._copula.logpdf(u)
        self.assertEqual(result.shape, (1, 20))

    def test_logpdf_identity_correlation(self) -> None:
        """With Sigma=I, the copula density is 1 (log c = 0)."""
        nparams = self._nvars * (self._nvars - 1) // 2
        chol_lower = self._bkd.zeros((nparams,))
        corr_param = CholeskyCorrelationParameterization(
            chol_lower, self._nvars, self._bkd
        )
        copula = GaussianCopula(corr_param, self._bkd)

        np.random.seed(42)
        u = self._bkd.asarray(
            np.random.uniform(0.01, 0.99, (self._nvars, 50)).astype(
                np.float64
            )
        )
        result = copula.logpdf(u)
        expected = self._bkd.zeros((1, 50))
        self._bkd.assert_allclose(result, expected, atol=1e-10)

    def test_logpdf_vs_scipy_mvn(self) -> None:
        """Compare against scipy: log c(u) = logpdf_MVN(z) - sum logpdf_N(z_i)."""
        np.random.seed(42)
        u_np = np.random.uniform(0.01, 0.99, (self._nvars, 30))
        u = self._bkd.asarray(u_np.astype(np.float64))

        # Get the correlation matrix
        sigma_np = self._bkd.to_numpy(self._corr_param.correlation_matrix())

        # Compute z = Phi^{-1}(u)
        z_np = stats.norm.ppf(u_np)

        # Expected: log f_MVN(z; 0, Sigma) - sum_i log f_N(z_i; 0, 1)
        mvn = stats.multivariate_normal(mean=np.zeros(self._nvars), cov=sigma_np)
        expected_np = mvn.logpdf(z_np.T) - np.sum(
            stats.norm.logpdf(z_np), axis=0
        )

        actual = self._copula.logpdf(u)
        actual_np = self._bkd.to_numpy(actual)[0]

        np_bkd = NumpyBkd()
        np_bkd.assert_allclose(actual_np, expected_np, rtol=1e-8)

    def test_sample_shape_and_range(self) -> None:
        np.random.seed(42)
        samples = self._copula.sample(100)
        self.assertEqual(samples.shape, (self._nvars, 100))
        # All values should be in (0, 1)
        samples_np = self._bkd.to_numpy(samples)
        self.assertTrue(np.all(samples_np > 0.0))
        self.assertTrue(np.all(samples_np < 1.0))

    def test_sample_kendall_tau(self) -> None:
        """For bivariate Gaussian copula, tau = 2/pi * arcsin(rho)."""
        rho = 0.6
        chol_lower = self._bkd.asarray([rho])
        corr_param = CholeskyCorrelationParameterization(
            chol_lower, 2, self._bkd
        )
        copula = GaussianCopula(corr_param, self._bkd)

        np.random.seed(42)
        samples = copula.sample(10000)
        samples_np = self._bkd.to_numpy(samples)

        # Compute empirical Kendall's tau
        tau_stat, _ = stats.kendalltau(samples_np[0], samples_np[1])
        expected_tau = 2.0 / np.pi * np.arcsin(rho)
        np.testing.assert_allclose(tau_stat, expected_tau, atol=0.03)

    def test_input_validation_1d(self) -> None:
        u_1d = self._bkd.asarray([0.5, 0.5, 0.5])
        with self.assertRaises(ValueError):
            self._copula.logpdf(u_1d)

    def test_input_validation_wrong_nvars(self) -> None:
        u = self._bkd.asarray(
            np.random.uniform(0.01, 0.99, (2, 10)).astype(np.float64)
        )
        with self.assertRaises(ValueError):
            self._copula.logpdf(u)

    def test_kl_divergence_same_is_zero(self) -> None:
        kl = self._copula.kl_divergence(self._copula)
        self._bkd.assert_allclose(
            self._bkd.atleast_1d(kl),
            self._bkd.zeros((1,)),
            atol=1e-10,
        )

    def test_kl_divergence_positive(self) -> None:
        # Create a different copula
        chol_lower2 = self._bkd.asarray([0.3, 0.1, 0.2])
        corr_param2 = CholeskyCorrelationParameterization(
            chol_lower2, self._nvars, self._bkd
        )
        copula2 = GaussianCopula(corr_param2, self._bkd)

        kl = self._copula.kl_divergence(copula2)
        self.assertTrue(float(self._bkd.to_numpy(kl)) > 0.0)


class TestGaussianCopulaNumpy(TestGaussianCopula[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGaussianCopulaTorch(TestGaussianCopula[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()
