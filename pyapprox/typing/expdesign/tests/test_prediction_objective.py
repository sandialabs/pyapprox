"""
Tests for PredictionOEDObjective.

Tests cover:
- Objective evaluation correctness
- Jacobian correctness via finite differences
- Different deviation measures (StdDev, Entropic)
- Different risk measures (Mean, Variance)
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

from pyapprox.typing.expdesign.likelihood import GaussianOEDInnerLoopLikelihood
from pyapprox.typing.expdesign.objective import PredictionOEDObjective
from pyapprox.typing.expdesign.deviation import (
    StandardDeviationMeasure,
    EntropicDeviationMeasure,
)
from pyapprox.typing.expdesign.statistics import (
    SampleAverageMean,
    SampleAverageVariance,
)


class TestPredictionOEDObjective(Generic[Array], unittest.TestCase):
    """Base test class for PredictionOEDObjective."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        # Set up test data
        self._nobs = 3
        self._ninner = 15
        self._nouter = 8
        self._npred = 2

        np.random.seed(42)
        self._noise_variances = self._bkd.asarray(
            np.array([0.1, 0.2, 0.15])
        )
        self._outer_shapes = self._bkd.asarray(
            np.random.randn(self._nobs, self._nouter)
        )
        self._inner_shapes = self._bkd.asarray(
            np.random.randn(self._nobs, self._ninner)
        )
        self._latent_samples = self._bkd.asarray(
            np.random.randn(self._nobs, self._nouter)
        )
        self._design_weights = self._bkd.asarray(
            np.random.uniform(0.5, 1.5, (self._nobs, 1))
        )
        # QoI values at inner samples
        self._qoi_vals = self._bkd.asarray(
            np.random.randn(self._ninner, self._npred)
        )

        # Create inner likelihood
        self._likelihood = GaussianOEDInnerLoopLikelihood(
            self._noise_variances, self._bkd
        )

    def _create_objective(
        self,
        deviation_measure,
        risk_measure=None,
        noise_stat=None,
    ):
        """Helper to create PredictionOEDObjective."""
        if risk_measure is None:
            risk_measure = SampleAverageMean(self._bkd)
        if noise_stat is None:
            noise_stat = SampleAverageMean(self._bkd)

        return PredictionOEDObjective(
            self._likelihood,
            self._outer_shapes,
            self._latent_samples,
            self._inner_shapes,
            self._qoi_vals,
            deviation_measure,
            risk_measure,
            noise_stat,
            outer_quad_weights=None,
            inner_quad_weights=None,
            qoi_quad_weights=None,
            bkd=self._bkd,
        )

    def _finite_diff_jacobian(
        self, objective: PredictionOEDObjective[Array], design_weights: Array
    ) -> Array:
        """Compute Jacobian via finite differences."""
        eps = 1e-6
        nobs = design_weights.shape[0]

        jac_fd = self._bkd.zeros((1, nobs))
        for k in range(nobs):
            weights_plus = self._bkd.copy(design_weights)
            weights_minus = self._bkd.copy(design_weights)
            weights_plus[k, 0] = weights_plus[k, 0] + eps
            weights_minus[k, 0] = weights_minus[k, 0] - eps

            val_plus = objective(weights_plus)
            val_minus = objective(weights_minus)

            jac_fd[0, k] = (val_plus[0, 0] - val_minus[0, 0]) / (2 * eps)

        return jac_fd

    # --- Basic Tests ---

    def test_dimension_accessors(self):
        """Test dimension accessor methods."""
        stdev = StandardDeviationMeasure(self._npred, self._bkd)
        objective = self._create_objective(stdev)

        self.assertEqual(objective.nvars(), self._nobs)
        self.assertEqual(objective.nqoi(), 1)
        self.assertEqual(objective.nobs(), self._nobs)
        self.assertEqual(objective.ninner(), self._ninner)
        self.assertEqual(objective.nouter(), self._nouter)
        self.assertEqual(objective.npred(), self._npred)

    # --- StdDev Tests ---

    def test_stdev_jacobian_finite_diff(self):
        """Test StdDev objective Jacobian against finite differences."""
        stdev = StandardDeviationMeasure(self._npred, self._bkd)
        objective = self._create_objective(stdev)

        jac_analytical = objective.jacobian(self._design_weights)
        jac_fd = self._finite_diff_jacobian(objective, self._design_weights)

        self._bkd.assert_allclose(jac_analytical, jac_fd, rtol=1e-4, atol=1e-7)

    def test_stdev_with_variance_risk(self):
        """Test StdDev with variance as risk measure."""
        stdev = StandardDeviationMeasure(self._npred, self._bkd)
        risk = SampleAverageVariance(self._bkd)
        objective = self._create_objective(stdev, risk_measure=risk)

        values = objective(self._design_weights)
        self.assertEqual(values.shape, (1, 1))
        self.assertTrue(self._bkd.all_bool(values >= 0))

    def test_stdev_variance_jacobian_finite_diff(self):
        """Test StdDev + Variance Jacobian against finite differences."""
        stdev = StandardDeviationMeasure(self._npred, self._bkd)
        risk = SampleAverageVariance(self._bkd)
        objective = self._create_objective(stdev, risk_measure=risk)

        jac_analytical = objective.jacobian(self._design_weights)
        jac_fd = self._finite_diff_jacobian(objective, self._design_weights)

        self._bkd.assert_allclose(jac_analytical, jac_fd, rtol=1e-4, atol=1e-7)

    # --- Entropic Tests ---

    def test_entropic_objective(self):
        """Test Entropic deviation objective."""
        entropic = EntropicDeviationMeasure(self._npred, 0.5, self._bkd)
        objective = self._create_objective(entropic)

        values = objective(self._design_weights)
        self.assertEqual(values.shape, (1, 1))

    def test_entropic_jacobian_finite_diff(self):
        """Test Entropic objective Jacobian against finite differences."""
        entropic = EntropicDeviationMeasure(self._npred, 0.5, self._bkd)
        objective = self._create_objective(entropic)

        jac_analytical = objective.jacobian(self._design_weights)
        jac_fd = self._finite_diff_jacobian(objective, self._design_weights)

        self._bkd.assert_allclose(jac_analytical, jac_fd, rtol=1e-4, atol=1e-7)

    # --- Validation Tests ---

    def test_qoi_shape_validation(self):
        """Test that qoi_vals shape is validated."""
        stdev = StandardDeviationMeasure(self._npred, self._bkd)

        # Wrong first dimension
        wrong_qoi = self._bkd.ones((self._ninner + 1, self._npred))
        with self.assertRaises(ValueError):
            PredictionOEDObjective(
                self._likelihood,
                self._outer_shapes,
                self._latent_samples,
                self._inner_shapes,
                wrong_qoi,
                stdev,
                SampleAverageMean(self._bkd),
                SampleAverageMean(self._bkd),
                None,
                None,
                None,
                self._bkd,
            )

    def test_evaluate_alias(self):
        """Test that evaluate() is alias for __call__()."""
        stdev = StandardDeviationMeasure(self._npred, self._bkd)
        objective = self._create_objective(stdev)

        val1 = objective(self._design_weights)
        val2 = objective.evaluate(self._design_weights)

        self._bkd.assert_allclose(val1, val2)

    def test_repr(self):
        """Test string representation."""
        stdev = StandardDeviationMeasure(self._npred, self._bkd)
        objective = self._create_objective(stdev)

        repr_str = repr(objective)
        self.assertIn("PredictionOEDObjective", repr_str)
        self.assertIn("StdDev", repr_str)

    # --- Consistency Tests ---

    def test_different_weights_different_values(self):
        """Test that different weights produce different objective values."""
        stdev = StandardDeviationMeasure(self._npred, self._bkd)
        objective = self._create_objective(stdev)

        weights1 = self._bkd.ones((self._nobs, 1)) * 0.5
        weights2 = self._bkd.ones((self._nobs, 1)) * 2.0

        val1 = objective(weights1)
        val2 = objective(weights2)

        # Values should differ (more weight = less noise = less deviation)
        self.assertFalse(
            self._bkd.allclose(val1, val2, atol=1e-8)
        )


class TestPredictionOEDObjectiveNumpy(
    TestPredictionOEDObjective[NDArray[Any]]
):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestPredictionOEDObjectiveTorch(
    TestPredictionOEDObjective[torch.Tensor]
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
