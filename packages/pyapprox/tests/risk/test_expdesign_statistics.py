"""
Tests for sample average statistics.

Tests cover:
- Value correctness against numpy equivalents
- Jacobian verification via finite differences
- Dual-backend testing (NumPy and PyTorch)
- Comparison against analytical Gaussian values (from legacy tests)
"""

import numpy as np
import pytest
from scipy import stats

from pyapprox.risk import (
    SampleAverageEntropicRisk,
    SampleAverageMean,
    SampleAverageSmoothedAVaR,
    SampleAverageStdev,
    SampleAverageVariance,
)
from pyapprox.probability.risk import GaussianAnalyticalRiskMeasures


class TestSampleStatistics:
    """Base test class for sample statistics."""

    def _setup_data(self, bkd):
        np.random.seed(42)
        self._nsamples = 20
        self._nqoi = 3
        self._nvars = 4

        # Create test data: (nqoi, nsamples)
        self._values = bkd.asarray(np.random.randn(self._nqoi, self._nsamples))
        # Uniform weights summing to 1: (1, nsamples)
        self._weights = bkd.asarray(np.full((1, self._nsamples), 1.0 / self._nsamples))
        # Random jacobians: (nqoi, nsamples, nvars)
        self._jac_values = bkd.asarray(
            np.random.randn(self._nqoi, self._nsamples, self._nvars)
        )

    def _finite_diff_jacobian(
        self, bkd, stat, values, weights, eps=1e-6, jac_values=None
    ):
        """Compute Jacobian via finite differences."""
        # TODO: Use DerivativeChecker to check gradients
        if jac_values is None:
            jac_values = self._jac_values
        nqoi = values.shape[0]
        nvars = jac_values.shape[2]
        jac_fd = bkd.zeros((nqoi, nvars))

        for k in range(nvars):
            # Perturb each variable
            values_plus = values + eps * jac_values[:, :, k]
            values_minus = values - eps * jac_values[:, :, k]

            stat_plus = stat(values_plus, weights)
            stat_minus = stat(values_minus, weights)

            jac_fd[:, k] = (stat_plus[:, 0] - stat_minus[:, 0]) / (2 * eps)

        return jac_fd

    def test_mean_values(self, bkd):
        """Test mean matches numpy weighted average."""
        self._setup_data(bkd)
        stat = SampleAverageMean(bkd)
        result = stat(self._values, self._weights)

        # Expected: weighted mean
        values_np = bkd.to_numpy(self._values)
        weights_np = bkd.to_numpy(self._weights)
        expected = np.sum(values_np * weights_np, axis=1, keepdims=True)

        assert result.shape == (self._nqoi, 1)
        assert bkd.allclose(result, bkd.asarray(expected), rtol=1e-12)

    def test_mean_jacobian(self, bkd):
        """Test mean Jacobian against finite differences."""
        self._setup_data(bkd)
        stat = SampleAverageMean(bkd)

        jac_analytical = stat.jacobian(self._values, self._jac_values, self._weights)
        jac_fd = self._finite_diff_jacobian(bkd, stat, self._values, self._weights)

        assert jac_analytical.shape == (self._nqoi, self._nvars)
        assert bkd.allclose(jac_analytical, jac_fd, rtol=1e-5, atol=1e-7)

    def test_variance_values(self, bkd):
        """Test variance matches numpy weighted variance."""
        self._setup_data(bkd)
        stat = SampleAverageVariance(bkd)
        result = stat(self._values, self._weights)

        # Expected: weighted variance
        values_np = bkd.to_numpy(self._values)
        weights_np = bkd.to_numpy(self._weights)
        mean = np.sum(values_np * weights_np, axis=1, keepdims=True)
        diff = values_np - mean
        expected = np.sum(diff**2 * weights_np, axis=1, keepdims=True)

        assert result.shape == (self._nqoi, 1)
        assert bkd.allclose(result, bkd.asarray(expected), rtol=1e-12)

    def test_variance_jacobian(self, bkd):
        """Test variance Jacobian against finite differences."""
        self._setup_data(bkd)
        stat = SampleAverageVariance(bkd)

        jac_analytical = stat.jacobian(self._values, self._jac_values, self._weights)
        jac_fd = self._finite_diff_jacobian(bkd, stat, self._values, self._weights)

        assert jac_analytical.shape == (self._nqoi, self._nvars)
        assert bkd.allclose(jac_analytical, jac_fd, rtol=1e-5, atol=1e-7)

    def test_stdev_values(self, bkd):
        """Test stdev is sqrt of variance."""
        self._setup_data(bkd)
        var_stat = SampleAverageVariance(bkd)
        std_stat = SampleAverageStdev(bkd)

        variance = var_stat(self._values, self._weights)
        stdev = std_stat(self._values, self._weights)

        assert stdev.shape == (self._nqoi, 1)
        assert bkd.allclose(stdev, bkd.sqrt(variance), rtol=1e-12)

    def test_stdev_jacobian(self, bkd):
        """Test stdev Jacobian against finite differences."""
        self._setup_data(bkd)
        stat = SampleAverageStdev(bkd)

        jac_analytical = stat.jacobian(self._values, self._jac_values, self._weights)
        jac_fd = self._finite_diff_jacobian(bkd, stat, self._values, self._weights)

        assert jac_analytical.shape == (self._nqoi, self._nvars)
        assert bkd.allclose(jac_analytical, jac_fd, rtol=1e-5, atol=1e-7)

    def test_entropic_risk_values(self, bkd):
        """Test entropic risk formula."""
        self._setup_data(bkd)
        alpha = 2.0
        stat = SampleAverageEntropicRisk(alpha, bkd)
        result = stat(self._values, self._weights)

        # Expected: (1/alpha) * log(E[exp(alpha*f)])
        values_np = bkd.to_numpy(self._values)
        weights_np = bkd.to_numpy(self._weights)
        exp_vals = np.exp(alpha * values_np)
        expected = np.log(np.sum(exp_vals * weights_np, axis=1)) / alpha

        assert result.shape == (self._nqoi, 1)
        assert bkd.allclose(result, bkd.asarray(expected[:, None]), rtol=1e-10)

    def test_entropic_risk_jacobian(self, bkd):
        """Test entropic risk Jacobian against finite differences."""
        self._setup_data(bkd)
        alpha = 2.0
        stat = SampleAverageEntropicRisk(alpha, bkd)

        jac_analytical = stat.jacobian(self._values, self._jac_values, self._weights)
        jac_fd = self._finite_diff_jacobian(bkd, stat, self._values, self._weights)

        assert jac_analytical.shape == (self._nqoi, self._nvars)
        assert bkd.allclose(jac_analytical, jac_fd, rtol=1e-5, atol=1e-7)

    def test_entropic_risk_invalid_alpha(self, bkd):
        """Test entropic risk raises for invalid alpha."""
        with pytest.raises(ValueError):
            SampleAverageEntropicRisk(0.0, bkd)
        with pytest.raises(ValueError):
            SampleAverageEntropicRisk(-1.0, bkd)

    def test_avar_values(self, bkd):
        """Test AVaR is between mean and max."""
        self._setup_data(bkd)
        alpha = 0.5
        stat = SampleAverageSmoothedAVaR(alpha, bkd)

        # Use single QoI with uniform weights: (1, nsamples)
        values_single = self._values[0:1, :]
        result = stat(values_single, self._weights)

        # AVaR should be between mean and max
        mean_stat = SampleAverageMean(bkd)
        mean_val = mean_stat(values_single, self._weights)
        max_val = bkd.max(values_single)

        assert result.shape == (1, 1)
        result_scalar = bkd.to_numpy(result)[0, 0]
        mean_scalar = bkd.to_numpy(mean_val)[0, 0]
        max_scalar = bkd.to_numpy(max_val)

        # AVaR >= mean for tail risk
        assert result_scalar >= mean_scalar - 1e-6
        # AVaR <= max
        assert result_scalar <= max_scalar + 1e-6

    def test_avar_jacobian(self, bkd):
        """Test AVaR Jacobian against finite differences."""
        self._setup_data(bkd)
        alpha = 0.5
        stat = SampleAverageSmoothedAVaR(alpha, bkd)

        # Use single QoI: (1, nsamples)
        values_single = self._values[0:1, :]
        jac_single = self._jac_values[0:1, :, :]

        jac_analytical = stat.jacobian(values_single, jac_single, self._weights)
        jac_fd = self._finite_diff_jacobian(
            bkd, stat, values_single, self._weights, eps=1e-5, jac_values=jac_single
        )

        # AVaR jacobian is less accurate due to projection
        assert jac_analytical.shape == (1, self._nvars)
        assert bkd.allclose(jac_analytical, jac_fd, rtol=1e-3, atol=1e-5)

    def test_avar_alpha_zero(self, bkd):
        """Test AVaR with alpha=0 is approximately mean.

        Note: The smoothed AVaR formula has a smoothing correction term,
        so it won't match exactly but should be very close.
        """
        self._setup_data(bkd)
        alpha = 0.0
        stat = SampleAverageSmoothedAVaR(alpha, bkd, delta=1000)
        mean_stat = SampleAverageMean(bkd)

        values_single = self._values[0:1, :]
        avar_val = stat(values_single, self._weights)
        mean_val = mean_stat(values_single, self._weights)

        # With high delta, smoothed AVaR should be close to mean
        assert bkd.allclose(avar_val, mean_val, rtol=1e-2)

    def test_shape_validation(self, bkd):
        """Test that shape validation raises for invalid inputs."""
        self._setup_data(bkd)
        stat = SampleAverageMean(bkd)

        # Wrong weights shape
        bad_weights = bkd.asarray(np.full((2, self._nsamples), 1.0 / self._nsamples))
        with pytest.raises(ValueError):
            stat(self._values, bad_weights)

    def test_gaussian_mean_analytical(self, bkd):
        """Test mean matches analytical Gaussian mean (from legacy test)."""
        mu, sigma = 0.5, 1.0
        risk = GaussianAnalyticalRiskMeasures(mu, sigma)
        exact_value = bkd.asarray(risk.mean())

        rv = stats.norm(mu, sigma)
        nsamples = int(1e6)
        values = bkd.asarray(rv.rvs(nsamples))[None, :]  # (1, nsamples)
        weights = bkd.full((1, nsamples), 1.0 / nsamples)

        stat = SampleAverageMean(bkd)
        estimate = stat(values, weights)

        assert bkd.allclose(estimate[0, 0], exact_value, rtol=1e-2)

    def test_gaussian_variance_analytical(self, bkd):
        """Test variance matches analytical Gaussian variance (from legacy)."""
        mu, sigma = 0.5, 1.0
        risk = GaussianAnalyticalRiskMeasures(mu, sigma)
        exact_value = bkd.asarray(risk.variance())

        rv = stats.norm(mu, sigma)
        nsamples = int(1e6)
        values = bkd.asarray(rv.rvs(nsamples))[None, :]  # (1, nsamples)
        weights = bkd.full((1, nsamples), 1.0 / nsamples)

        stat = SampleAverageVariance(bkd)
        estimate = stat(values, weights)

        assert bkd.allclose(estimate[0, 0], exact_value, rtol=1e-2)

    def test_gaussian_stdev_analytical(self, bkd):
        """Test stdev matches analytical Gaussian stdev (from legacy test)."""
        mu, sigma = 0.5, 1.0
        risk = GaussianAnalyticalRiskMeasures(mu, sigma)
        exact_value = bkd.asarray(risk.stdev())

        rv = stats.norm(mu, sigma)
        nsamples = int(1e6)
        values = bkd.asarray(rv.rvs(nsamples))[None, :]  # (1, nsamples)
        weights = bkd.full((1, nsamples), 1.0 / nsamples)

        stat = SampleAverageStdev(bkd)
        estimate = stat(values, weights)

        assert bkd.allclose(estimate[0, 0], exact_value, rtol=1e-2)

    def test_gaussian_entropic_analytical(self, bkd):
        """Test entropic risk matches analytical Gaussian (from legacy)."""
        mu, sigma = 0.5, 1.0
        alpha = 0.5
        risk = GaussianAnalyticalRiskMeasures(mu, sigma)
        exact_value = bkd.asarray(risk.entropic(alpha))

        rv = stats.norm(mu, sigma)
        nsamples = int(1e6)
        values = bkd.asarray(rv.rvs(nsamples))[None, :]  # (1, nsamples)
        weights = bkd.full((1, nsamples), 1.0 / nsamples)

        stat = SampleAverageEntropicRisk(alpha, bkd)
        estimate = stat(values, weights)

        assert bkd.allclose(estimate[0, 0], exact_value, rtol=1e-2)

    def test_gaussian_avar_analytical(self, bkd):
        """Test AVaR matches analytical Gaussian AVaR (from legacy test).

        Uses equidistant samples for higher accuracy (like legacy test).
        """
        mu, sigma, beta = 0, 1, 0.5
        risk = GaussianAnalyticalRiskMeasures(mu, sigma)
        exact_avar = bkd.asarray(risk.AVaR(beta))

        rv = stats.norm(mu, sigma)
        nsamples = int(1e6)
        # Use equidistant points for higher accuracy (like legacy)
        samples_np = rv.ppf(np.linspace(1e-6, 1 - 1e-6, nsamples))
        values = bkd.asarray(samples_np)[None, :]  # (1, nsamples)
        weights = bkd.full((1, nsamples), 1.0 / nsamples)

        stat = SampleAverageSmoothedAVaR(beta, bkd, delta=100000)
        estimate = stat(values, weights)

        assert bkd.allclose(estimate[0, 0], exact_avar, rtol=2e-5)

    def test_avar_multiple_qoi_homogeneity(self, bkd):
        """Test AVaR positive homogeneity: R(tZ) = tR(Z) (from legacy)."""
        mu, sigma, beta = 0, 1, 0.5
        risk = GaussianAnalyticalRiskMeasures(mu, sigma)
        exact_avar = bkd.asarray(risk.AVaR(beta))

        rv = stats.norm(mu, sigma)
        nsamples = int(1e6)
        samples_np = rv.ppf(np.linspace(1e-6, 1 - 1e-6, nsamples))
        values1 = bkd.asarray(samples_np)[None, :]  # (1, nsamples)
        values2 = 2 * values1
        values = bkd.vstack([values1, values2])  # (2, nsamples)
        weights = bkd.full((1, nsamples), 1.0 / nsamples)

        stat = SampleAverageSmoothedAVaR(beta, bkd, delta=100000)
        estimate = stat(values, weights)

        expected = bkd.asarray([[exact_avar], [2 * exact_avar]])
        assert bkd.allclose(estimate, expected, rtol=2e-5)
