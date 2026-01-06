"""
Tests for deviation measures.

Tests cover:
- Standard deviation measure
- Entropic deviation measure
- AVaR deviation measure
- Jacobian correctness via finite differences
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
from pyapprox.typing.expdesign.evidence import Evidence
from pyapprox.typing.expdesign.deviation import (
    DeviationMeasure,
    StandardDeviationMeasure,
    EntropicDeviationMeasure,
    AVaRDeviationMeasure,
)


class TestDeviationMeasures(Generic[Array], unittest.TestCase):
    """Base test class for deviation measures."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        # Set up test data
        self._nobs = 3
        self._ninner = 20
        self._nouter = 5
        self._npred = 2

        np.random.seed(42)
        self._noise_variances = self._bkd.asarray(
            np.array([0.1, 0.2, 0.15])
        )
        self._shapes_inner = self._bkd.asarray(
            np.random.randn(self._nobs, self._ninner)
        )
        self._observations = self._bkd.asarray(
            np.random.randn(self._nobs, self._nouter)
        )
        self._design_weights = self._bkd.asarray(
            np.random.uniform(0.5, 1.5, (self._nobs, 1))
        )
        # QoI values at inner samples
        self._qoi_vals = self._bkd.asarray(
            np.random.randn(self._ninner, self._npred)
        )

        # Create inner likelihood and evidence
        self._likelihood = GaussianOEDInnerLoopLikelihood(
            self._noise_variances, self._bkd
        )
        self._likelihood.set_shapes(self._shapes_inner)
        self._likelihood.set_observations(self._observations)

        quad_weights = self._bkd.ones((self._ninner,)) / self._ninner
        self._evidence = Evidence(self._likelihood, quad_weights, self._bkd)

    def _setup_deviation(self, deviation: DeviationMeasure[Array]) -> None:
        """Helper to set up deviation measure with evidence and qoi data."""
        deviation.set_evidence(self._evidence)
        deviation.set_qoi_data(self._qoi_vals)

    def _finite_diff_jacobian(
        self, deviation: DeviationMeasure[Array], design_weights: Array
    ) -> Array:
        """Compute Jacobian via finite differences."""
        eps = 1e-6
        nobs = design_weights.shape[0]
        nout = self._npred * self._nouter

        jac_fd = self._bkd.zeros((nout, nobs))
        for k in range(nobs):
            weights_plus = self._bkd.copy(design_weights)
            weights_minus = self._bkd.copy(design_weights)
            weights_plus[k, 0] = weights_plus[k, 0] + eps
            weights_minus[k, 0] = weights_minus[k, 0] - eps

            val_plus = deviation(weights_plus)
            val_minus = deviation(weights_minus)

            jac_fd[:, k] = (val_plus[0] - val_minus[0]) / (2 * eps)

        return jac_fd

    # --- StandardDeviationMeasure Tests ---

    def test_stdev_shape(self):
        """Test StandardDeviationMeasure output shape."""
        stdev = StandardDeviationMeasure(self._npred, self._bkd)
        self._setup_deviation(stdev)

        values = stdev(self._design_weights)
        self.assertEqual(values.shape, (1, self._npred * self._nouter))

    def test_stdev_positive(self):
        """Test that standard deviation values are non-negative."""
        stdev = StandardDeviationMeasure(self._npred, self._bkd)
        self._setup_deviation(stdev)

        values = stdev(self._design_weights)
        self.assertTrue(self._bkd.all_bool(values >= 0))

    def test_stdev_jacobian_shape(self):
        """Test StandardDeviationMeasure Jacobian shape."""
        stdev = StandardDeviationMeasure(self._npred, self._bkd)
        self._setup_deviation(stdev)

        jac = stdev.jacobian(self._design_weights)
        self.assertEqual(
            jac.shape, (self._npred * self._nouter, self._nobs)
        )

    def test_stdev_jacobian_finite_diff(self):
        """Test StandardDeviationMeasure Jacobian against finite differences."""
        stdev = StandardDeviationMeasure(self._npred, self._bkd)
        self._setup_deviation(stdev)

        jac_analytical = stdev.jacobian(self._design_weights)
        jac_fd = self._finite_diff_jacobian(stdev, self._design_weights)

        self._bkd.assert_allclose(jac_analytical, jac_fd, rtol=1e-4, atol=1e-7)

    def test_stdev_constant_qoi(self):
        """Test standard deviation with constant QoI (should be near zero)."""
        stdev = StandardDeviationMeasure(self._npred, self._bkd)
        stdev.set_evidence(self._evidence)

        # Set constant QoI values
        constant_qoi = self._bkd.ones((self._ninner, self._npred)) * 5.0
        stdev.set_qoi_data(constant_qoi)

        values = stdev(self._design_weights)
        # Values should be near zero (clamped to sqrt(1e-16) = 1e-8 plus
        # numerical precision effects)
        self.assertTrue(
            self._bkd.all_bool(values < 1e-6)
        )

    # --- EntropicDeviationMeasure Tests ---

    def test_entropic_shape(self):
        """Test EntropicDeviationMeasure output shape."""
        entropic = EntropicDeviationMeasure(self._npred, 0.5, self._bkd)
        self._setup_deviation(entropic)

        values = entropic(self._design_weights)
        self.assertEqual(values.shape, (1, self._npred * self._nouter))

    def test_entropic_alpha_validation(self):
        """Test that alpha must be positive."""
        with self.assertRaises(ValueError):
            EntropicDeviationMeasure(self._npred, 0.0, self._bkd)
        with self.assertRaises(ValueError):
            EntropicDeviationMeasure(self._npred, -1.0, self._bkd)

    def test_entropic_jacobian_shape(self):
        """Test EntropicDeviationMeasure Jacobian shape."""
        entropic = EntropicDeviationMeasure(self._npred, 0.5, self._bkd)
        self._setup_deviation(entropic)

        jac = entropic.jacobian(self._design_weights)
        self.assertEqual(
            jac.shape, (self._npred * self._nouter, self._nobs)
        )

    def test_entropic_jacobian_finite_diff(self):
        """Test EntropicDeviationMeasure Jacobian against finite differences."""
        entropic = EntropicDeviationMeasure(self._npred, 0.5, self._bkd)
        self._setup_deviation(entropic)

        jac_analytical = entropic.jacobian(self._design_weights)
        jac_fd = self._finite_diff_jacobian(entropic, self._design_weights)

        self._bkd.assert_allclose(jac_analytical, jac_fd, rtol=1e-4, atol=1e-7)

    def test_entropic_small_alpha_limit(self):
        """Test entropic deviation with small alpha approaches variance/2."""
        # For small alpha: entropic_risk ≈ E[qoi] + alpha * Var[qoi] / 2
        # So entropic_deviation ≈ alpha * Var[qoi] / 2
        stdev = StandardDeviationMeasure(self._npred, self._bkd)
        self._setup_deviation(stdev)
        stdev_vals = stdev(self._design_weights)
        var_vals = stdev_vals ** 2

        # Small alpha
        alpha = 0.01
        entropic = EntropicDeviationMeasure(self._npred, alpha, self._bkd)
        self._setup_deviation(entropic)
        entropic_vals = entropic(self._design_weights)

        # Approximate: entropic_dev ≈ alpha * var / 2
        expected = alpha * var_vals / 2.0

        # Allow reasonable tolerance for approximation
        self._bkd.assert_allclose(entropic_vals, expected, rtol=0.1)

    def test_entropic_constant_qoi(self):
        """Test entropic deviation with constant QoI (should be zero)."""
        entropic = EntropicDeviationMeasure(self._npred, 0.5, self._bkd)
        entropic.set_evidence(self._evidence)

        # Set constant QoI values
        constant_qoi = self._bkd.ones((self._ninner, self._npred)) * 5.0
        entropic.set_qoi_data(constant_qoi)

        values = entropic(self._design_weights)
        expected = self._bkd.zeros((1, self._npred * self._nouter))
        self._bkd.assert_allclose(values, expected, atol=1e-10)

    # --- AVaRDeviationMeasure Tests ---

    def test_avar_shape(self):
        """Test AVaRDeviationMeasure output shape."""
        avar = AVaRDeviationMeasure(self._npred, 0.8, self._bkd, delta=100)
        self._setup_deviation(avar)

        values = avar(self._design_weights)
        self.assertEqual(values.shape, (1, self._npred * self._nouter))

    def test_avar_positive_for_positive_qoi(self):
        """Test AVaR deviation is positive for positive-centered QoI."""
        avar = AVaRDeviationMeasure(self._npred, 0.8, self._bkd, delta=100)
        avar.set_evidence(self._evidence)

        # Set positive QoI values (mean 0 but with variance)
        positive_qoi = self._bkd.asarray(
            np.abs(np.random.randn(self._ninner, self._npred))
        )
        avar.set_qoi_data(positive_qoi)

        values = avar(self._design_weights)
        # AVaR deviation should be non-negative for positive data
        # (AVaR focuses on upper tail, which is above mean)
        self.assertTrue(self._bkd.all_bool(values >= -1e-10))

    def test_avar_constant_qoi(self):
        """Test AVaR deviation with constant QoI (should be near zero)."""
        # Use large delta for better accuracy with constant data
        avar = AVaRDeviationMeasure(
            self._npred, 0.8, self._bkd, delta=1000
        )
        avar.set_evidence(self._evidence)

        # Set constant QoI values
        constant_qoi = self._bkd.ones((self._ninner, self._npred)) * 5.0
        avar.set_qoi_data(constant_qoi)

        values = avar(self._design_weights)
        expected = self._bkd.zeros((1, self._npred * self._nouter))
        # Allow some tolerance due to smoothing parameter
        self._bkd.assert_allclose(values, expected, atol=1e-2)

    # --- Common Tests ---

    def test_set_evidence_validation(self):
        """Test that set_evidence validates input type."""
        stdev = StandardDeviationMeasure(self._npred, self._bkd)
        with self.assertRaises(TypeError):
            stdev.set_evidence("not an evidence")  # type: ignore

    def test_qoi_shape_validation(self):
        """Test that set_qoi_data validates shape."""
        stdev = StandardDeviationMeasure(self._npred, self._bkd)
        stdev.set_evidence(self._evidence)

        wrong_shape = self._bkd.ones((self._ninner + 1, self._npred))
        with self.assertRaises(ValueError):
            stdev.set_qoi_data(wrong_shape)

    def test_evaluate_before_setup_raises(self):
        """Test that evaluation before setup raises error."""
        stdev = StandardDeviationMeasure(self._npred, self._bkd)

        with self.assertRaises(RuntimeError):
            stdev(self._design_weights)

    def test_label(self):
        """Test that deviation measures have labels."""
        stdev = StandardDeviationMeasure(self._npred, self._bkd)
        entropic = EntropicDeviationMeasure(self._npred, 0.5, self._bkd)
        avar = AVaRDeviationMeasure(self._npred, 0.8, self._bkd)

        self.assertEqual(stdev.label(), "StdDev")
        self.assertEqual(entropic.label(), "Entropic")
        self.assertEqual(avar.label(), "AVaRDev")


class TestDeviationMeasuresNumpy(TestDeviationMeasures[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestDeviationMeasuresTorch(TestDeviationMeasures[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def test_avar_jacobian_autodiff(self):
        """Test AVaR Jacobian using PyTorch autodiff."""
        avar = AVaRDeviationMeasure(self._npred, 0.8, self._bkd, delta=100)
        self._setup_deviation(avar)

        jac_autodiff = avar.jacobian(self._design_weights)

        # Verify against finite differences
        jac_fd = self._finite_diff_jacobian(avar, self._design_weights)

        # Use tolerances similar to legacy code
        self._bkd.assert_allclose(jac_autodiff, jac_fd, rtol=1e-3, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
