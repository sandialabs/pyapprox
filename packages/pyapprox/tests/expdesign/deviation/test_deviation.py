"""
Tests for deviation measures.

Tests cover:
- Standard deviation measure
- Entropic deviation measure
- AVaR deviation measure
- Jacobian correctness via finite differences
"""

import numpy as np
import pytest

from pyapprox.expdesign.deviation import (
    AVaRDeviationMeasure,
    EntropicDeviationMeasure,
    StandardDeviationMeasure,
)
from pyapprox.expdesign.evidence import Evidence
from pyapprox.expdesign.likelihood import GaussianOEDInnerLoopLikelihood


class TestDeviationMeasures:
    """Base test class for deviation measures."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    @pytest.fixture(autouse=True)
    def _setup(self, bkd):
        # Set up test data
        self._nobs = 3
        self._ninner = 20
        self._nouter = 5
        self._npred = 2

        self._noise_variances = bkd.asarray(np.array([0.1, 0.2, 0.15]))
        self._shapes_inner = bkd.asarray(
            np.random.randn(self._nobs, self._ninner)
        )
        self._observations = bkd.asarray(
            np.random.randn(self._nobs, self._nouter)
        )
        self._design_weights = bkd.asarray(
            np.random.uniform(0.5, 1.5, (self._nobs, 1))
        )
        # QoI values at inner samples
        self._qoi_vals = bkd.asarray(np.random.randn(self._ninner, self._npred))

        # Create inner likelihood and evidence
        self._likelihood = GaussianOEDInnerLoopLikelihood(
            self._noise_variances, bkd
        )
        self._likelihood.set_shapes(self._shapes_inner)
        self._likelihood.set_observations(self._observations)

        quad_weights = bkd.ones((self._ninner,)) / self._ninner
        self._evidence = Evidence(self._likelihood, quad_weights, bkd)

    def _setup_deviation(self, deviation):
        """Helper to set up deviation measure with evidence and qoi data."""
        deviation.set_evidence(self._evidence)
        deviation.set_qoi_data(self._qoi_vals)

    def _finite_diff_jacobian(self, bkd, deviation, design_weights):
        """Compute Jacobian via finite differences."""
        #TODO: use DerivativeChecker
        eps = 1e-6
        nobs = design_weights.shape[0]
        nout = self._npred * self._nouter

        jac_fd = bkd.zeros((nout, nobs))
        for k in range(nobs):
            weights_plus = bkd.copy(design_weights)
            weights_minus = bkd.copy(design_weights)
            weights_plus[k, 0] = weights_plus[k, 0] + eps
            weights_minus[k, 0] = weights_minus[k, 0] - eps

            val_plus = deviation(weights_plus)
            val_minus = deviation(weights_minus)

            jac_fd[:, k] = (val_plus[0] - val_minus[0]) / (2 * eps)

        return jac_fd

    # --- StandardDeviationMeasure Tests ---

    def test_stdev_positive(self, bkd):
        """Test that standard deviation values are non-negative."""
        stdev = StandardDeviationMeasure(self._npred, bkd)
        self._setup_deviation(stdev)

        values = stdev(self._design_weights)
        assert bkd.all_bool(values >= 0)

    def test_stdev_jacobian_finite_diff(self, bkd):
        """Test StandardDeviationMeasure Jacobian against finite differences."""
        stdev = StandardDeviationMeasure(self._npred, bkd)
        self._setup_deviation(stdev)

        jac_analytical = stdev.jacobian(self._design_weights)
        jac_fd = self._finite_diff_jacobian(bkd, stdev, self._design_weights)

        bkd.assert_allclose(jac_analytical, jac_fd, rtol=1e-4, atol=1e-7)

    def test_stdev_constant_qoi(self, bkd):
        """Test standard deviation with constant QoI (should be near zero)."""
        stdev = StandardDeviationMeasure(self._npred, bkd)
        stdev.set_evidence(self._evidence)

        # Set constant QoI values
        constant_qoi = bkd.ones((self._ninner, self._npred)) * 5.0
        stdev.set_qoi_data(constant_qoi)

        values = stdev(self._design_weights)
        # Values should be near zero (clamped to sqrt(1e-16) = 1e-8 plus
        # numerical precision effects)
        assert bkd.all_bool(values < 1e-6)

    # --- EntropicDeviationMeasure Tests ---

    def test_entropic_alpha_validation(self, bkd):
        """Test that alpha must be positive."""
        with pytest.raises(ValueError):
            EntropicDeviationMeasure(self._npred, 0.0, bkd)
        with pytest.raises(ValueError):
            EntropicDeviationMeasure(self._npred, -1.0, bkd)

    def test_entropic_jacobian_finite_diff(self, bkd):
        """Test EntropicDeviationMeasure Jacobian against finite differences."""
        entropic = EntropicDeviationMeasure(self._npred, 0.5, bkd)
        self._setup_deviation(entropic)

        jac_analytical = entropic.jacobian(self._design_weights)
        jac_fd = self._finite_diff_jacobian(bkd, entropic, self._design_weights)

        bkd.assert_allclose(jac_analytical, jac_fd, rtol=1e-4, atol=1e-7)

    def test_entropic_small_alpha_limit(self, bkd):
        """Test entropic deviation with small alpha approaches variance/2."""
        # For small alpha: entropic_risk ~ E[qoi] + alpha * Var[qoi] / 2
        # So entropic_deviation ~ alpha * Var[qoi] / 2
        stdev = StandardDeviationMeasure(self._npred, bkd)
        self._setup_deviation(stdev)
        stdev_vals = stdev(self._design_weights)
        var_vals = stdev_vals**2

        # Small alpha
        alpha = 0.01
        entropic = EntropicDeviationMeasure(self._npred, alpha, bkd)
        self._setup_deviation(entropic)
        entropic_vals = entropic(self._design_weights)

        # Approximate: entropic_dev ~ alpha * var / 2
        expected = alpha * var_vals / 2.0

        # Allow reasonable tolerance for approximation
        bkd.assert_allclose(entropic_vals, expected, rtol=0.1)

    def test_entropic_constant_qoi(self, bkd):
        """Test entropic deviation with constant QoI (should be zero)."""
        entropic = EntropicDeviationMeasure(self._npred, 0.5, bkd)
        entropic.set_evidence(self._evidence)

        # Set constant QoI values
        constant_qoi = bkd.ones((self._ninner, self._npred)) * 5.0
        entropic.set_qoi_data(constant_qoi)

        values = entropic(self._design_weights)
        expected = bkd.zeros((1, self._npred * self._nouter))
        bkd.assert_allclose(values, expected, atol=1e-10)

    # --- AVaRDeviationMeasure Tests ---

    def test_avar_positive_for_positive_qoi(self, bkd):
        """Test AVaR deviation is positive for positive-centered QoI."""
        avar = AVaRDeviationMeasure(self._npred, 0.8, bkd, delta=100)
        avar.set_evidence(self._evidence)

        # Set positive QoI values (mean 0 but with variance)
        positive_qoi = bkd.asarray(
            np.abs(np.random.randn(self._ninner, self._npred))
        )
        avar.set_qoi_data(positive_qoi)

        values = avar(self._design_weights)
        # AVaR deviation should be non-negative for positive data
        # (AVaR focuses on upper tail, which is above mean)
        assert bkd.all_bool(values >= -1e-10)

    @pytest.mark.slow_on("TorchBkd")
    def test_avar_constant_qoi(self, bkd):
        """Test AVaR deviation with constant QoI (should be near zero)."""
        # Use large delta for better accuracy with constant data
        avar = AVaRDeviationMeasure(self._npred, 0.8, bkd, delta=1000)
        avar.set_evidence(self._evidence)

        # Set constant QoI values
        constant_qoi = bkd.ones((self._ninner, self._npred)) * 5.0
        avar.set_qoi_data(constant_qoi)

        values = avar(self._design_weights)
        expected = bkd.zeros((1, self._npred * self._nouter))
        # Allow some tolerance due to smoothing parameter
        bkd.assert_allclose(values, expected, atol=1e-2)

    # --- Common Tests ---

    def test_set_evidence_validation(self, bkd):
        """Test that set_evidence validates input type."""
        stdev = StandardDeviationMeasure(self._npred, bkd)
        with pytest.raises(TypeError):
            stdev.set_evidence("not an evidence")  # type: ignore

    def test_qoi_shape_validation(self, bkd):
        """Test that set_qoi_data validates shape."""
        stdev = StandardDeviationMeasure(self._npred, bkd)
        stdev.set_evidence(self._evidence)

        wrong_shape = bkd.ones((self._ninner + 1, self._npred))
        with pytest.raises(ValueError):
            stdev.set_qoi_data(wrong_shape)

    def test_evaluate_before_setup_raises(self, bkd):
        """Test that evaluation before setup raises error."""
        stdev = StandardDeviationMeasure(self._npred, bkd)

        with pytest.raises(RuntimeError):
            stdev(self._design_weights)

    def test_label(self, bkd):
        """Test that deviation measures have labels."""
        stdev = StandardDeviationMeasure(self._npred, bkd)
        entropic = EntropicDeviationMeasure(self._npred, 0.5, bkd)
        avar = AVaRDeviationMeasure(self._npred, 0.8, bkd)

        assert stdev.label() == "StdDev"
        assert entropic.label() == "Entropic"
        assert avar.label() == "AVaRDev"

    def test_avar_jacobian_autodiff(self, bkd):
        """Test AVaR Jacobian using autodiff (Torch only)."""
        from pyapprox.util.backends.torch import TorchBkd

        if not isinstance(bkd, TorchBkd):
            pytest.skip("Torch-only test")

        # torch.compile donated buffers conflict with autograd jacobian
        import torch._functorch.config as _ftconfig

        _ftconfig.donated_buffer = False
        avar = AVaRDeviationMeasure(self._npred, 0.8, bkd, delta=100)
        self._setup_deviation(avar)

        jac_autodiff = avar.jacobian(self._design_weights)

        # Verify against finite differences
        jac_fd = self._finite_diff_jacobian(bkd, avar, self._design_weights)

        # Use tolerances similar to legacy code
        bkd.assert_allclose(jac_autodiff, jac_fd, rtol=1e-3, atol=1e-6)
