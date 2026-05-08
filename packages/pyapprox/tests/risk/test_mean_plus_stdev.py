"""
Tests for SampleAverageMeanPlusStdev statistic.

Tests cover:
- Value correctness (mean + factor * stdev)
- Jacobian verification via finite differences
- Dual-backend testing (NumPy and PyTorch)
"""

#TODO: Why is this file seperate from the other test file which tests all other stats

import numpy as np

from pyapprox.risk import (
    SampleAverageMean,
    SampleAverageMeanPlusStdev,
    SampleAverageStdev,
)


class TestSampleAverageMeanPlusStdev:
    """Base test class for SampleAverageMeanPlusStdev."""

    def _setup_data(self, bkd):
        np.random.seed(42)
        self._nsamples = 20
        self._nqoi = 3
        self._nvars = 4

        # Create test data: (nqoi, nsamples)
        self._values = bkd.asarray(np.random.randn(self._nqoi, self._nsamples))
        # Uniform weights summing to 1: (1, nsamples)
        self._weights = bkd.asarray(
            np.full((1, self._nsamples), 1.0 / self._nsamples)
        )
        # Random jacobians: (nqoi, nsamples, nvars)
        self._jac_values = bkd.asarray(
            np.random.randn(self._nqoi, self._nsamples, self._nvars)
        )

    def _finite_diff_jacobian(self, bkd, stat, values, weights, jac_values, eps=1e-6):
        """Compute Jacobian via finite differences."""
        nqoi = values.shape[0]
        nvars = jac_values.shape[2]
        jac_fd = bkd.zeros((nqoi, nvars))

        for k in range(nvars):
            values_plus = values + eps * jac_values[:, :, k]
            values_minus = values - eps * jac_values[:, :, k]

            stat_plus = stat(values_plus, weights)
            stat_minus = stat(values_minus, weights)

            jac_fd[:, k] = (stat_plus[:, 0] - stat_minus[:, 0]) / (2 * eps)

        return jac_fd

    def test_values_equals_mean_plus_factor_stdev(self, bkd):
        """Test that result equals mean + factor * stdev."""
        self._setup_data(bkd)
        factor = 2.5
        stat = SampleAverageMeanPlusStdev(factor, bkd)
        mean_stat = SampleAverageMean(bkd)
        stdev_stat = SampleAverageStdev(bkd)

        result = stat(self._values, self._weights)
        expected = mean_stat(self._values, self._weights) + factor * stdev_stat(
            self._values, self._weights
        )

        assert result.shape == (self._nqoi, 1)
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_factor_zero_equals_mean(self, bkd):
        """Test that factor=0 gives pure mean."""
        self._setup_data(bkd)
        stat = SampleAverageMeanPlusStdev(0.0, bkd)
        mean_stat = SampleAverageMean(bkd)

        result = stat(self._values, self._weights)
        expected = mean_stat(self._values, self._weights)

        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_jacobian_finite_diff(self, bkd):
        """Test Jacobian against finite differences."""
        self._setup_data(bkd)
        factor = 2.5
        stat = SampleAverageMeanPlusStdev(factor, bkd)

        jac_analytical = stat.jacobian(self._values, self._jac_values, self._weights)
        jac_fd = self._finite_diff_jacobian(
            bkd, stat, self._values, self._weights, self._jac_values
        )

        assert jac_analytical.shape == (self._nqoi, self._nvars)
        bkd.assert_allclose(jac_analytical, jac_fd, rtol=1e-5, atol=1e-7)

    def test_jacobian_equals_mean_plus_factor_stdev_jacobian(self, bkd):
        """Test that jacobian equals mean.jac + factor * stdev.jac."""
        self._setup_data(bkd)
        factor = 3.0
        stat = SampleAverageMeanPlusStdev(factor, bkd)
        mean_stat = SampleAverageMean(bkd)
        stdev_stat = SampleAverageStdev(bkd)

        result = stat.jacobian(self._values, self._jac_values, self._weights)
        expected = mean_stat.jacobian(
            self._values, self._jac_values, self._weights
        ) + factor * stdev_stat.jacobian(self._values, self._jac_values, self._weights)

        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_jacobian_implemented(self, bkd):
        """Test jacobian_implemented returns True."""
        stat = SampleAverageMeanPlusStdev(1.0, bkd)
        assert stat.jacobian_implemented()

    def test_repr(self, bkd):
        """Test string representation."""
        stat = SampleAverageMeanPlusStdev(2.5, bkd)
        assert repr(stat) == "SampleAverageMeanPlusStdev(factor=2.5)"

    def test_single_qoi(self, bkd):
        """Test with single QoI."""
        self._setup_data(bkd)
        factor = 1.5
        stat = SampleAverageMeanPlusStdev(factor, bkd)

        values = self._values[0:1, :]  # (1, nsamples)
        result = stat(values, self._weights)

        assert result.shape == (1, 1)
