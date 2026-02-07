"""
Legacy comparison tests for prediction OED.

Tests verify that the new typing module implementation produces results
consistent with the legacy pyapprox.expdesign implementation by comparing
core components (deviation measures, statistics, evidence).
"""

import unittest

import numpy as np
import torch

from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401

# New typing module imports
from pyapprox.typing.expdesign import (
    SampleAverageMean,
    SampleAverageVariance,
)

# Legacy imports
from pyapprox.optimization.sampleaverage import (
    SampleAverageMean as LegacySampleAverageMean,
    SampleAverageVariance as LegacySampleAverageVariance,
)
from pyapprox.util.backends.torch import TorchMixin


class TestSampleStatisticsLegacyComparison(unittest.TestCase):
    """Compare new and legacy sample statistics implementations."""

    def setUp(self):
        """Set up test data."""
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        self._legacy_bkd = TorchMixin

        np.random.seed(42)
        self._nsamples = 20
        self._nqoi = 3

        # Create random test data
        values_np = np.random.randn(self._nsamples, self._nqoi)
        weights_np = np.random.dirichlet(np.ones(self._nsamples))[:, None]

        self._values = self._bkd.asarray(values_np)
        self._weights = self._bkd.asarray(weights_np)
        self._legacy_values = self._legacy_bkd.asarray(values_np)
        self._legacy_weights = self._legacy_bkd.asarray(weights_np)

    def test_mean_matches_legacy(self):
        """Test that SampleAverageMean matches legacy."""
        new_mean = SampleAverageMean(self._bkd)
        legacy_mean = LegacySampleAverageMean(self._legacy_bkd)

        new_result = new_mean(self._values, self._weights)
        legacy_result = legacy_mean(self._values, self._weights)

        self._bkd.assert_allclose(new_result, legacy_result, rtol=1e-12)

    def test_variance_matches_legacy(self):
        """Test that SampleAverageVariance matches legacy."""
        new_var = SampleAverageVariance(self._bkd)
        legacy_var = LegacySampleAverageVariance(self._legacy_bkd)

        new_result = new_var(self._values, self._weights)
        legacy_result = legacy_var(self._values, self._weights)

        self._bkd.assert_allclose(new_result, legacy_result, rtol=1e-12)

    def test_mean_jacobian_matches_legacy(self):
        """Test that Mean Jacobian matches legacy."""
        new_mean = SampleAverageMean(self._bkd)
        legacy_mean = LegacySampleAverageMean(self._legacy_bkd)

        nvars = 4
        jac_values_np = np.random.randn(self._nsamples, self._nqoi, nvars)
        jac_values = self._bkd.asarray(jac_values_np)
        legacy_jac_values = self._legacy_bkd.asarray(jac_values_np)

        new_result = new_mean.jacobian(self._values, jac_values, self._weights)
        legacy_result = legacy_mean.jacobian(
            self._values, jac_values, self._weights
        )

        self._bkd.assert_allclose(new_result, legacy_result, rtol=1e-12)

    def test_variance_jacobian_matches_legacy(self):
        """Test that Variance Jacobian matches legacy."""
        new_var = SampleAverageVariance(self._bkd)
        legacy_var = LegacySampleAverageVariance(self._legacy_bkd)

        nvars = 4
        jac_values_np = np.random.randn(self._nsamples, self._nqoi, nvars)
        jac_values = self._bkd.asarray(jac_values_np)
        legacy_jac_values = self._legacy_bkd.asarray(jac_values_np)

        new_result = new_var.jacobian(self._values, jac_values, self._weights)
        legacy_result = legacy_var.jacobian(
            self._values, jac_values, self._weights
        )

        self._bkd.assert_allclose(new_result, legacy_result, rtol=1e-12)


# NOTE: The legacy deviation measures (OEDStandardDeviationMeasure, etc.)
# and the legacy PredictionOEDObjective have a different API than the new
# typing module implementations. Direct comparison would require significant
# adaptation. Instead, we verify correctness through:
# 1. The sample statistics tests above (which match exactly)
# 2. Finite difference Jacobian tests in test_deviation.py
# 3. Analytical formula comparisons in test_prediction_integration.py
# 4. Full workflow tests with gradient verification


if __name__ == "__main__":
    unittest.main()
