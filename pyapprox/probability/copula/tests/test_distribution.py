"""Tests for CopulaDistribution."""

import numpy as np

from pyapprox.probability.copula.correlation.cholesky import (
    CholeskyCorrelationParameterization,
)
from pyapprox.probability.copula.distribution import CopulaDistribution
from pyapprox.probability.copula.gaussian import GaussianCopula
from pyapprox.probability.univariate.gaussian import GaussianMarginal
from pyapprox.util.backends.numpy import NumpyBkd


class TestCopulaDistribution:

    def _make_dist(self, bkd):
        nvars = 3
        # Create a Gaussian copula
        chol_lower = bkd.asarray([0.5, 0.3, 0.4])
        corr_param = CholeskyCorrelationParameterization(
            chol_lower, nvars, bkd
        )
        copula = GaussianCopula(corr_param, bkd)

        # Create Gaussian marginals with different means and stdevs
        marginals = [
            GaussianMarginal(2.0, 1.5, bkd),
            GaussianMarginal(-1.0, 0.5, bkd),
            GaussianMarginal(0.0, 2.0, bkd),
        ]

        return CopulaDistribution(copula, marginals, bkd), copula, marginals, nvars

    def test_logpdf_decomposes(self, bkd) -> None:
        """Verify logpdf = copula.logpdf(u) + sum marginal.logpdf(x_i)."""
        dist, copula, marginals, nvars = self._make_dist(bkd)
        np.random.seed(42)
        x_np = np.random.randn(nvars, 20).astype(np.float64)
        x = bkd.asarray(x_np)

        # Compute via CopulaDistribution
        actual = dist.logpdf(x)

        # Compute manually
        nsamples = x.shape[1]
        u = bkd.zeros((nvars, nsamples))
        marginal_logpdf_sum = bkd.zeros((1, nsamples))
        for i, marginal in enumerate(marginals):
            row_2d = bkd.reshape(x[i], (1, -1))
            u[i] = marginal.cdf(row_2d)[0]
            marginal_logpdf_sum = marginal_logpdf_sum + marginal.logpdf(row_2d)

        copula_logpdf = copula.logpdf(u)
        expected = copula_logpdf + marginal_logpdf_sum

        bkd.assert_allclose(actual, expected, rtol=1e-10)

    def test_sample_shape_and_marginal_distribution(self, bkd) -> None:
        """Verify samples have correct shape and marginal moments."""
        dist, _, _, nvars = self._make_dist(bkd)
        np.random.seed(42)
        nsamples = 10000
        samples = dist.sample(nsamples)

        assert samples.shape == (nvars, nsamples)

        # Check marginal means approximately match
        samples_np = bkd.to_numpy(samples)
        expected_means = [2.0, -1.0, 0.0]
        expected_stds = [1.5, 0.5, 2.0]
        for i in range(nvars):
            empirical_mean = np.mean(samples_np[i])
            empirical_std = np.std(samples_np[i])
            np.testing.assert_allclose(empirical_mean, expected_means[i], atol=0.1)
            np.testing.assert_allclose(empirical_std, expected_stds[i], atol=0.1)

    def test_sample_correlation_structure(self, bkd) -> None:
        """For Gaussian copula + Gaussian marginals, correlation matches."""
        dist, copula, _, _ = self._make_dist(bkd)
        np.random.seed(42)
        nsamples = 20000
        samples = dist.sample(nsamples)
        samples_np = bkd.to_numpy(samples)

        # Empirical correlation
        empirical_corr = np.corrcoef(samples_np)

        # Expected: copula correlation matrix
        expected_corr = bkd.to_numpy(
            copula.correlation_param().correlation_matrix()
        )

        np_bkd = NumpyBkd()
        np_bkd.assert_allclose(empirical_corr, expected_corr, atol=0.04)
