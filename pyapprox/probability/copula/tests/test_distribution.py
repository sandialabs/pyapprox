"""Tests for CopulaDistribution."""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.probability.copula.correlation.cholesky import (
    CholeskyCorrelationParameterization,
)
from pyapprox.probability.copula.distribution import CopulaDistribution
from pyapprox.probability.copula.gaussian import GaussianCopula
from pyapprox.probability.univariate.gaussian import GaussianMarginal
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401


class TestCopulaDistribution(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self._nvars = 3

        # Create a Gaussian copula
        chol_lower = self._bkd.asarray([0.5, 0.3, 0.4])
        corr_param = CholeskyCorrelationParameterization(
            chol_lower, self._nvars, self._bkd
        )
        self._copula = GaussianCopula(corr_param, self._bkd)

        # Create Gaussian marginals with different means and stdevs
        self._marginals = [
            GaussianMarginal(2.0, 1.5, self._bkd),
            GaussianMarginal(-1.0, 0.5, self._bkd),
            GaussianMarginal(0.0, 2.0, self._bkd),
        ]

        self._dist = CopulaDistribution(self._copula, self._marginals, self._bkd)

    def test_logpdf_decomposes(self) -> None:
        """Verify logpdf = copula.logpdf(u) + sum marginal.logpdf(x_i)."""
        np.random.seed(42)
        x_np = np.random.randn(self._nvars, 20).astype(np.float64)
        x = self._bkd.asarray(x_np)

        # Compute via CopulaDistribution
        actual = self._dist.logpdf(x)

        # Compute manually
        nsamples = x.shape[1]
        u = self._bkd.zeros((self._nvars, nsamples))
        marginal_logpdf_sum = self._bkd.zeros((1, nsamples))
        for i, marginal in enumerate(self._marginals):
            row_2d = self._bkd.reshape(x[i], (1, -1))
            u[i] = marginal.cdf(row_2d)[0]
            marginal_logpdf_sum = marginal_logpdf_sum + marginal.logpdf(row_2d)

        copula_logpdf = self._copula.logpdf(u)
        expected = copula_logpdf + marginal_logpdf_sum

        self._bkd.assert_allclose(actual, expected, rtol=1e-10)

    def test_sample_shape_and_marginal_distribution(self) -> None:
        """Verify samples have correct shape and marginal moments."""
        np.random.seed(42)
        nsamples = 10000
        samples = self._dist.sample(nsamples)

        self.assertEqual(samples.shape, (self._nvars, nsamples))

        # Check marginal means approximately match
        samples_np = self._bkd.to_numpy(samples)
        expected_means = [2.0, -1.0, 0.0]
        expected_stds = [1.5, 0.5, 2.0]
        for i in range(self._nvars):
            empirical_mean = np.mean(samples_np[i])
            empirical_std = np.std(samples_np[i])
            np.testing.assert_allclose(empirical_mean, expected_means[i], atol=0.1)
            np.testing.assert_allclose(empirical_std, expected_stds[i], atol=0.1)

    def test_sample_correlation_structure(self) -> None:
        """For Gaussian copula + Gaussian marginals, correlation matches."""
        np.random.seed(42)
        nsamples = 20000
        samples = self._dist.sample(nsamples)
        samples_np = self._bkd.to_numpy(samples)

        # Empirical correlation
        empirical_corr = np.corrcoef(samples_np)

        # Expected: copula correlation matrix
        expected_corr = self._bkd.to_numpy(
            self._copula.correlation_param().correlation_matrix()
        )

        np_bkd = NumpyBkd()
        np_bkd.assert_allclose(empirical_corr, expected_corr, atol=0.04)


class TestCopulaDistributionNumpy(TestCopulaDistribution[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestCopulaDistributionTorch(TestCopulaDistribution[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()
