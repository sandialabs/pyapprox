"""
Tests for sample average statistics.

Tests cover:
- Value correctness against numpy equivalents
- Jacobian verification via finite differences
- Dual-backend testing (NumPy and PyTorch)
- Comparison against analytical Gaussian values (from legacy tests)
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from scipy import stats
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401

from pyapprox.typing.expdesign.statistics import (
    SampleAverageMean,
    SampleAverageVariance,
    SampleAverageStdev,
    SampleAverageEntropicRisk,
    SampleAverageSmoothedAVaR,
)
from pyapprox.typing.probability.risk import GaussianAnalyticalRiskMeasures


class TestSampleStatistics(Generic[Array], unittest.TestCase):
    """Base test class for sample statistics."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        np.random.seed(42)
        self._nsamples = 20
        self._nqoi = 3
        self._nvars = 4

        # Create test data: (nqoi, nsamples)
        self._values = self._bkd.asarray(
            np.random.randn(self._nqoi, self._nsamples)
        )
        # Uniform weights summing to 1: (1, nsamples)
        self._weights = self._bkd.asarray(
            np.full((1, self._nsamples), 1.0 / self._nsamples)
        )
        # Random jacobians: (nqoi, nsamples, nvars)
        self._jac_values = self._bkd.asarray(
            np.random.randn(self._nqoi, self._nsamples, self._nvars)
        )

    def _finite_diff_jacobian(
        self, stat, values, weights, eps=1e-6, jac_values=None
    ):
        """Compute Jacobian via finite differences."""
        if jac_values is None:
            jac_values = self._jac_values
        nqoi = values.shape[0]
        nvars = jac_values.shape[2]
        jac_fd = self._bkd.zeros((nqoi, nvars))

        for k in range(nvars):
            # Perturb each variable
            values_plus = values + eps * jac_values[:, :, k]
            values_minus = values - eps * jac_values[:, :, k]

            stat_plus = stat(values_plus, weights)
            stat_minus = stat(values_minus, weights)

            jac_fd[:, k] = (stat_plus[:, 0] - stat_minus[:, 0]) / (2 * eps)

        return jac_fd

    def test_mean_values(self):
        """Test mean matches numpy weighted average."""
        stat = SampleAverageMean(self._bkd)
        result = stat(self._values, self._weights)

        # Expected: weighted mean
        values_np = self._bkd.to_numpy(self._values)
        weights_np = self._bkd.to_numpy(self._weights)
        expected = np.sum(values_np * weights_np, axis=1, keepdims=True)

        self.assertEqual(result.shape, (self._nqoi, 1))
        self.assertTrue(
            self._bkd.allclose(
                result, self._bkd.asarray(expected), rtol=1e-12
            )
        )

    def test_mean_jacobian(self):
        """Test mean Jacobian against finite differences."""
        stat = SampleAverageMean(self._bkd)

        jac_analytical = stat.jacobian(
            self._values, self._jac_values, self._weights
        )
        jac_fd = self._finite_diff_jacobian(
            stat, self._values, self._weights
        )

        self.assertEqual(jac_analytical.shape, (self._nqoi, self._nvars))
        self.assertTrue(
            self._bkd.allclose(jac_analytical, jac_fd, rtol=1e-5, atol=1e-7)
        )

    def test_variance_values(self):
        """Test variance matches numpy weighted variance."""
        stat = SampleAverageVariance(self._bkd)
        result = stat(self._values, self._weights)

        # Expected: weighted variance
        values_np = self._bkd.to_numpy(self._values)
        weights_np = self._bkd.to_numpy(self._weights)
        mean = np.sum(values_np * weights_np, axis=1, keepdims=True)
        diff = values_np - mean
        expected = np.sum(diff**2 * weights_np, axis=1, keepdims=True)

        self.assertEqual(result.shape, (self._nqoi, 1))
        self.assertTrue(
            self._bkd.allclose(
                result, self._bkd.asarray(expected), rtol=1e-12
            )
        )

    def test_variance_jacobian(self):
        """Test variance Jacobian against finite differences."""
        stat = SampleAverageVariance(self._bkd)

        jac_analytical = stat.jacobian(
            self._values, self._jac_values, self._weights
        )
        jac_fd = self._finite_diff_jacobian(
            stat, self._values, self._weights
        )

        self.assertEqual(jac_analytical.shape, (self._nqoi, self._nvars))
        self.assertTrue(
            self._bkd.allclose(jac_analytical, jac_fd, rtol=1e-5, atol=1e-7)
        )

    def test_stdev_values(self):
        """Test stdev is sqrt of variance."""
        var_stat = SampleAverageVariance(self._bkd)
        std_stat = SampleAverageStdev(self._bkd)

        variance = var_stat(self._values, self._weights)
        stdev = std_stat(self._values, self._weights)

        self.assertEqual(stdev.shape, (self._nqoi, 1))
        self.assertTrue(
            self._bkd.allclose(stdev, self._bkd.sqrt(variance), rtol=1e-12)
        )

    def test_stdev_jacobian(self):
        """Test stdev Jacobian against finite differences."""
        stat = SampleAverageStdev(self._bkd)

        jac_analytical = stat.jacobian(
            self._values, self._jac_values, self._weights
        )
        jac_fd = self._finite_diff_jacobian(
            stat, self._values, self._weights
        )

        self.assertEqual(jac_analytical.shape, (self._nqoi, self._nvars))
        self.assertTrue(
            self._bkd.allclose(jac_analytical, jac_fd, rtol=1e-5, atol=1e-7)
        )

    def test_entropic_risk_values(self):
        """Test entropic risk formula."""
        alpha = 2.0
        stat = SampleAverageEntropicRisk(alpha, self._bkd)
        result = stat(self._values, self._weights)

        # Expected: (1/alpha) * log(E[exp(alpha*f)])
        values_np = self._bkd.to_numpy(self._values)
        weights_np = self._bkd.to_numpy(self._weights)
        exp_vals = np.exp(alpha * values_np)
        expected = np.log(np.sum(exp_vals * weights_np, axis=1)) / alpha

        self.assertEqual(result.shape, (self._nqoi, 1))
        self.assertTrue(
            self._bkd.allclose(
                result, self._bkd.asarray(expected[:, None]), rtol=1e-10
            )
        )

    def test_entropic_risk_jacobian(self):
        """Test entropic risk Jacobian against finite differences."""
        alpha = 2.0
        stat = SampleAverageEntropicRisk(alpha, self._bkd)

        jac_analytical = stat.jacobian(
            self._values, self._jac_values, self._weights
        )
        jac_fd = self._finite_diff_jacobian(
            stat, self._values, self._weights
        )

        self.assertEqual(jac_analytical.shape, (self._nqoi, self._nvars))
        self.assertTrue(
            self._bkd.allclose(jac_analytical, jac_fd, rtol=1e-5, atol=1e-7)
        )

    def test_entropic_risk_invalid_alpha(self):
        """Test entropic risk raises for invalid alpha."""
        with self.assertRaises(ValueError):
            SampleAverageEntropicRisk(0.0, self._bkd)
        with self.assertRaises(ValueError):
            SampleAverageEntropicRisk(-1.0, self._bkd)

    def test_avar_values(self):
        """Test AVaR is between mean and max."""
        alpha = 0.5
        stat = SampleAverageSmoothedAVaR(alpha, self._bkd)

        # Use single QoI with uniform weights: (1, nsamples)
        values_single = self._values[0:1, :]
        result = stat(values_single, self._weights)

        # AVaR should be between mean and max
        mean_stat = SampleAverageMean(self._bkd)
        mean_val = mean_stat(values_single, self._weights)
        max_val = self._bkd.max(values_single)

        self.assertEqual(result.shape, (1, 1))
        result_scalar = self._bkd.to_numpy(result)[0, 0]
        mean_scalar = self._bkd.to_numpy(mean_val)[0, 0]
        max_scalar = self._bkd.to_numpy(max_val)

        # AVaR >= mean for tail risk
        self.assertGreaterEqual(result_scalar, mean_scalar - 1e-6)
        # AVaR <= max
        self.assertLessEqual(result_scalar, max_scalar + 1e-6)

    def test_avar_jacobian(self):
        """Test AVaR Jacobian against finite differences."""
        alpha = 0.5
        stat = SampleAverageSmoothedAVaR(alpha, self._bkd)

        # Use single QoI: (1, nsamples)
        values_single = self._values[0:1, :]
        jac_single = self._jac_values[0:1, :, :]

        jac_analytical = stat.jacobian(
            values_single, jac_single, self._weights
        )
        jac_fd = self._finite_diff_jacobian(
            stat, values_single, self._weights, eps=1e-5, jac_values=jac_single
        )

        # AVaR jacobian is less accurate due to projection
        self.assertEqual(jac_analytical.shape, (1, self._nvars))
        self.assertTrue(
            self._bkd.allclose(jac_analytical, jac_fd, rtol=1e-3, atol=1e-5)
        )

    def test_avar_alpha_zero(self):
        """Test AVaR with alpha=0 is approximately mean.

        Note: The smoothed AVaR formula has a smoothing correction term,
        so it won't match exactly but should be very close.
        """
        alpha = 0.0
        stat = SampleAverageSmoothedAVaR(alpha, self._bkd, delta=1000)
        mean_stat = SampleAverageMean(self._bkd)

        values_single = self._values[0:1, :]
        avar_val = stat(values_single, self._weights)
        mean_val = mean_stat(values_single, self._weights)

        # With high delta, smoothed AVaR should be close to mean
        self.assertTrue(
            self._bkd.allclose(avar_val, mean_val, rtol=1e-2)
        )

    def test_shape_validation(self):
        """Test that shape validation raises for invalid inputs."""
        stat = SampleAverageMean(self._bkd)

        # Wrong weights shape
        bad_weights = self._bkd.asarray(
            np.full((2, self._nsamples), 1.0 / self._nsamples)
        )
        with self.assertRaises(ValueError):
            stat(self._values, bad_weights)

    def test_gaussian_mean_analytical(self):
        """Test mean matches analytical Gaussian mean (from legacy test)."""
        mu, sigma = 0.5, 1.0
        risk = GaussianAnalyticalRiskMeasures(mu, sigma)
        exact_value = self._bkd.asarray(risk.mean())

        rv = stats.norm(mu, sigma)
        nsamples = int(1e6)
        values = self._bkd.asarray(rv.rvs(nsamples))[None, :]  # (1, nsamples)
        weights = self._bkd.full((1, nsamples), 1.0 / nsamples)

        stat = SampleAverageMean(self._bkd)
        estimate = stat(values, weights)

        self.assertTrue(
            self._bkd.allclose(estimate[0, 0], exact_value, rtol=1e-2)
        )

    def test_gaussian_variance_analytical(self):
        """Test variance matches analytical Gaussian variance (from legacy)."""
        mu, sigma = 0.5, 1.0
        risk = GaussianAnalyticalRiskMeasures(mu, sigma)
        exact_value = self._bkd.asarray(risk.variance())

        rv = stats.norm(mu, sigma)
        nsamples = int(1e6)
        values = self._bkd.asarray(rv.rvs(nsamples))[None, :]  # (1, nsamples)
        weights = self._bkd.full((1, nsamples), 1.0 / nsamples)

        stat = SampleAverageVariance(self._bkd)
        estimate = stat(values, weights)

        self.assertTrue(
            self._bkd.allclose(estimate[0, 0], exact_value, rtol=1e-2)
        )

    def test_gaussian_stdev_analytical(self):
        """Test stdev matches analytical Gaussian stdev (from legacy test)."""
        mu, sigma = 0.5, 1.0
        risk = GaussianAnalyticalRiskMeasures(mu, sigma)
        exact_value = self._bkd.asarray(risk.stdev())

        rv = stats.norm(mu, sigma)
        nsamples = int(1e6)
        values = self._bkd.asarray(rv.rvs(nsamples))[None, :]  # (1, nsamples)
        weights = self._bkd.full((1, nsamples), 1.0 / nsamples)

        stat = SampleAverageStdev(self._bkd)
        estimate = stat(values, weights)

        self.assertTrue(
            self._bkd.allclose(estimate[0, 0], exact_value, rtol=1e-2)
        )

    def test_gaussian_entropic_analytical(self):
        """Test entropic risk matches analytical Gaussian (from legacy)."""
        mu, sigma = 0.5, 1.0
        alpha = 0.5
        risk = GaussianAnalyticalRiskMeasures(mu, sigma)
        exact_value = self._bkd.asarray(risk.entropic(alpha))

        rv = stats.norm(mu, sigma)
        nsamples = int(1e6)
        values = self._bkd.asarray(rv.rvs(nsamples))[None, :]  # (1, nsamples)
        weights = self._bkd.full((1, nsamples), 1.0 / nsamples)

        stat = SampleAverageEntropicRisk(alpha, self._bkd)
        estimate = stat(values, weights)

        self.assertTrue(
            self._bkd.allclose(estimate[0, 0], exact_value, rtol=1e-2)
        )

    def test_gaussian_avar_analytical(self):
        """Test AVaR matches analytical Gaussian AVaR (from legacy test).

        Uses equidistant samples for higher accuracy (like legacy test).
        """
        mu, sigma, beta = 0, 1, 0.5
        risk = GaussianAnalyticalRiskMeasures(mu, sigma)
        exact_avar = self._bkd.asarray(risk.AVaR(beta))

        rv = stats.norm(mu, sigma)
        nsamples = int(1e6)
        # Use equidistant points for higher accuracy (like legacy)
        samples_np = rv.ppf(np.linspace(1e-6, 1 - 1e-6, nsamples))
        values = self._bkd.asarray(samples_np)[None, :]  # (1, nsamples)
        weights = self._bkd.full((1, nsamples), 1.0 / nsamples)

        stat = SampleAverageSmoothedAVaR(beta, self._bkd, delta=100000)
        estimate = stat(values, weights)

        self.assertTrue(
            self._bkd.allclose(estimate[0, 0], exact_avar, rtol=2e-5)
        )

    def test_avar_multiple_qoi_homogeneity(self):
        """Test AVaR positive homogeneity: R(tZ) = tR(Z) (from legacy)."""
        mu, sigma, beta = 0, 1, 0.5
        risk = GaussianAnalyticalRiskMeasures(mu, sigma)
        exact_avar = self._bkd.asarray(risk.AVaR(beta))

        rv = stats.norm(mu, sigma)
        nsamples = int(1e6)
        samples_np = rv.ppf(np.linspace(1e-6, 1 - 1e-6, nsamples))
        values1 = self._bkd.asarray(samples_np)[None, :]  # (1, nsamples)
        values2 = 2 * values1
        values = self._bkd.vstack([values1, values2])  # (2, nsamples)
        weights = self._bkd.full((1, nsamples), 1.0 / nsamples)

        stat = SampleAverageSmoothedAVaR(beta, self._bkd, delta=100000)
        estimate = stat(values, weights)

        expected = self._bkd.asarray([[exact_avar], [2 * exact_avar]])
        self.assertTrue(
            self._bkd.allclose(estimate, expected, rtol=2e-5)
        )


class TestSampleStatisticsNumpy(TestSampleStatistics[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestSampleStatisticsTorch(TestSampleStatistics[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
