"""
Tests for SampleAverageMeanPlusStdev statistic.

Tests cover:
- Value correctness (mean + factor * stdev)
- Jacobian verification via finite differences
- Dual-backend testing (NumPy and PyTorch)
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401

from pyapprox.typing.expdesign.statistics import (
    SampleAverageMean,
    SampleAverageStdev,
    SampleAverageMeanPlusStdev,
)


class TestSampleAverageMeanPlusStdev(Generic[Array], unittest.TestCase):
    """Base test class for SampleAverageMeanPlusStdev."""

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

    def _finite_diff_jacobian(self, stat, values, weights, jac_values, eps=1e-6):
        """Compute Jacobian via finite differences."""
        nqoi = values.shape[0]
        nvars = jac_values.shape[2]
        jac_fd = self._bkd.zeros((nqoi, nvars))

        for k in range(nvars):
            values_plus = values + eps * jac_values[:, :, k]
            values_minus = values - eps * jac_values[:, :, k]

            stat_plus = stat(values_plus, weights)
            stat_minus = stat(values_minus, weights)

            jac_fd[:, k] = (stat_plus[:, 0] - stat_minus[:, 0]) / (2 * eps)

        return jac_fd

    def test_values_equals_mean_plus_factor_stdev(self):
        """Test that result equals mean + factor * stdev."""
        factor = 2.5
        stat = SampleAverageMeanPlusStdev(factor, self._bkd)
        mean_stat = SampleAverageMean(self._bkd)
        stdev_stat = SampleAverageStdev(self._bkd)

        result = stat(self._values, self._weights)
        expected = (
            mean_stat(self._values, self._weights)
            + factor * stdev_stat(self._values, self._weights)
        )

        self.assertEqual(result.shape, (self._nqoi, 1))
        self._bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_factor_zero_equals_mean(self):
        """Test that factor=0 gives pure mean."""
        stat = SampleAverageMeanPlusStdev(0.0, self._bkd)
        mean_stat = SampleAverageMean(self._bkd)

        result = stat(self._values, self._weights)
        expected = mean_stat(self._values, self._weights)

        self._bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_jacobian_finite_diff(self):
        """Test Jacobian against finite differences."""
        factor = 2.5
        stat = SampleAverageMeanPlusStdev(factor, self._bkd)

        jac_analytical = stat.jacobian(
            self._values, self._jac_values, self._weights
        )
        jac_fd = self._finite_diff_jacobian(
            stat, self._values, self._weights, self._jac_values
        )

        self.assertEqual(jac_analytical.shape, (self._nqoi, self._nvars))
        self._bkd.assert_allclose(jac_analytical, jac_fd, rtol=1e-5, atol=1e-7)

    def test_jacobian_equals_mean_plus_factor_stdev_jacobian(self):
        """Test that jacobian equals mean.jac + factor * stdev.jac."""
        factor = 3.0
        stat = SampleAverageMeanPlusStdev(factor, self._bkd)
        mean_stat = SampleAverageMean(self._bkd)
        stdev_stat = SampleAverageStdev(self._bkd)

        result = stat.jacobian(self._values, self._jac_values, self._weights)
        expected = (
            mean_stat.jacobian(self._values, self._jac_values, self._weights)
            + factor
            * stdev_stat.jacobian(self._values, self._jac_values, self._weights)
        )

        self._bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_jacobian_implemented(self):
        """Test jacobian_implemented returns True."""
        stat = SampleAverageMeanPlusStdev(1.0, self._bkd)
        self.assertTrue(stat.jacobian_implemented())

    def test_repr(self):
        """Test string representation."""
        stat = SampleAverageMeanPlusStdev(2.5, self._bkd)
        self.assertEqual(repr(stat), "SampleAverageMeanPlusStdev(factor=2.5)")

    def test_single_qoi(self):
        """Test with single QoI."""
        factor = 1.5
        stat = SampleAverageMeanPlusStdev(factor, self._bkd)

        values = self._values[0:1, :]  # (1, nsamples)
        result = stat(values, self._weights)

        self.assertEqual(result.shape, (1, 1))


class TestSampleAverageMeanPlusStdevNumpy(
    TestSampleAverageMeanPlusStdev[NDArray[Any]]
):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestSampleAverageMeanPlusStdevTorch(
    TestSampleAverageMeanPlusStdev[torch.Tensor]
):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
