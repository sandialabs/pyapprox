"""
Tests for PredictionOEDObjective.

Tests cover:
- Objective evaluation correctness
- Jacobian correctness via finite differences
- Different deviation measures (StdDev, Entropic)
- Different risk measures (Mean, Variance)
"""

import numpy as np
import pytest

from pyapprox.expdesign.deviation import (
    EntropicDeviationMeasure,
    StandardDeviationMeasure,
)
from pyapprox.expdesign.likelihood import GaussianOEDInnerLoopLikelihood
from pyapprox.expdesign.objective import PredictionOEDObjective
from pyapprox.expdesign.statistics import (
    SampleAverageMean,
    SampleAverageVariance,
)


class TestPredictionOEDObjective:
    """Base test class for PredictionOEDObjective."""

    @pytest.fixture(autouse=True)
    def _setup(self, bkd):
        # Set up test data
        self._nobs = 3
        self._ninner = 15
        self._nouter = 8
        self._npred = 2

        np.random.seed(42)
        self._noise_variances = bkd.asarray(np.array([0.1, 0.2, 0.15]))
        self._outer_shapes = bkd.asarray(
            np.random.randn(self._nobs, self._nouter)
        )
        self._inner_shapes = bkd.asarray(
            np.random.randn(self._nobs, self._ninner)
        )
        self._latent_samples = bkd.asarray(
            np.random.randn(self._nobs, self._nouter)
        )
        self._design_weights = bkd.asarray(
            np.random.uniform(0.5, 1.5, (self._nobs, 1))
        )
        # QoI values at inner samples
        self._qoi_vals = bkd.asarray(np.random.randn(self._ninner, self._npred))

        # Create inner likelihood
        self._likelihood = GaussianOEDInnerLoopLikelihood(
            self._noise_variances, bkd
        )

    def _create_objective(
        self,
        bkd,
        deviation_measure,
        risk_measure=None,
        noise_stat=None,
    ):
        """Helper to create PredictionOEDObjective."""
        if risk_measure is None:
            risk_measure = SampleAverageMean(bkd)
        if noise_stat is None:
            noise_stat = SampleAverageMean(bkd)

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
            bkd=bkd,
        )

    def _finite_diff_jacobian(self, bkd, objective, design_weights):
        """Compute Jacobian via finite differences."""
        eps = 1e-6
        nobs = design_weights.shape[0]

        jac_fd = bkd.zeros((1, nobs))
        for k in range(nobs):
            weights_plus = bkd.copy(design_weights)
            weights_minus = bkd.copy(design_weights)
            weights_plus[k, 0] = weights_plus[k, 0] + eps
            weights_minus[k, 0] = weights_minus[k, 0] - eps

            val_plus = objective(weights_plus)
            val_minus = objective(weights_minus)

            jac_fd[0, k] = (val_plus[0, 0] - val_minus[0, 0]) / (2 * eps)

        return jac_fd

    # --- Basic Tests ---

    def test_dimension_accessors(self, bkd):
        """Test dimension accessor methods."""
        stdev = StandardDeviationMeasure(self._npred, bkd)
        objective = self._create_objective(bkd, stdev)

        assert objective.nvars() == self._nobs
        assert objective.nqoi() == 1
        assert objective.nobs() == self._nobs
        assert objective.ninner() == self._ninner
        assert objective.nouter() == self._nouter
        assert objective.npred() == self._npred

    # --- StdDev Tests ---

    def test_stdev_jacobian_finite_diff(self, bkd):
        """Test StdDev objective Jacobian against finite differences."""
        stdev = StandardDeviationMeasure(self._npred, bkd)
        objective = self._create_objective(bkd, stdev)

        jac_analytical = objective.jacobian(self._design_weights)
        jac_fd = self._finite_diff_jacobian(bkd, objective, self._design_weights)

        bkd.assert_allclose(jac_analytical, jac_fd, rtol=1e-4, atol=1e-7)

    def test_stdev_with_variance_risk(self, bkd):
        """Test StdDev with variance as risk measure."""
        stdev = StandardDeviationMeasure(self._npred, bkd)
        risk = SampleAverageVariance(bkd)
        objective = self._create_objective(bkd, stdev, risk_measure=risk)

        values = objective(self._design_weights)
        assert values.shape == (1, 1)
        assert bkd.all_bool(values >= 0)

    def test_stdev_variance_jacobian_finite_diff(self, bkd):
        """Test StdDev + Variance Jacobian against finite differences."""
        stdev = StandardDeviationMeasure(self._npred, bkd)
        risk = SampleAverageVariance(bkd)
        objective = self._create_objective(bkd, stdev, risk_measure=risk)

        jac_analytical = objective.jacobian(self._design_weights)
        jac_fd = self._finite_diff_jacobian(bkd, objective, self._design_weights)

        bkd.assert_allclose(jac_analytical, jac_fd, rtol=1e-4, atol=1e-7)

    # --- Entropic Tests ---

    def test_entropic_objective(self, bkd):
        """Test Entropic deviation objective."""
        entropic = EntropicDeviationMeasure(self._npred, 0.5, bkd)
        objective = self._create_objective(bkd, entropic)

        values = objective(self._design_weights)
        assert values.shape == (1, 1)

    def test_entropic_jacobian_finite_diff(self, bkd):
        """Test Entropic objective Jacobian against finite differences."""
        entropic = EntropicDeviationMeasure(self._npred, 0.5, bkd)
        objective = self._create_objective(bkd, entropic)

        jac_analytical = objective.jacobian(self._design_weights)
        jac_fd = self._finite_diff_jacobian(bkd, objective, self._design_weights)

        bkd.assert_allclose(jac_analytical, jac_fd, rtol=1e-4, atol=1e-7)

    # --- Validation Tests ---

    def test_qoi_shape_validation(self, bkd):
        """Test that qoi_vals shape is validated."""
        stdev = StandardDeviationMeasure(self._npred, bkd)

        # Wrong first dimension
        wrong_qoi = bkd.ones((self._ninner + 1, self._npred))
        with pytest.raises(ValueError):
            PredictionOEDObjective(
                self._likelihood,
                self._outer_shapes,
                self._latent_samples,
                self._inner_shapes,
                wrong_qoi,
                stdev,
                SampleAverageMean(bkd),
                SampleAverageMean(bkd),
                None,
                None,
                None,
                bkd,
            )

    def test_evaluate_alias(self, bkd):
        """Test that evaluate() is alias for __call__()."""
        stdev = StandardDeviationMeasure(self._npred, bkd)
        objective = self._create_objective(bkd, stdev)

        val1 = objective(self._design_weights)
        val2 = objective.evaluate(self._design_weights)

        bkd.assert_allclose(val1, val2)

    def test_repr(self, bkd):
        """Test string representation."""
        stdev = StandardDeviationMeasure(self._npred, bkd)
        objective = self._create_objective(bkd, stdev)

        repr_str = repr(objective)
        assert "PredictionOEDObjective" in repr_str
        assert "StdDev" in repr_str

    # --- Consistency Tests ---

    def test_different_weights_different_values(self, bkd):
        """Test that different weights produce different objective values."""
        stdev = StandardDeviationMeasure(self._npred, bkd)
        objective = self._create_objective(bkd, stdev)

        weights1 = bkd.ones((self._nobs, 1)) * 0.5
        weights2 = bkd.ones((self._nobs, 1)) * 2.0

        val1 = objective(weights1)
        val2 = objective(weights2)

        # Values should differ (more weight = less noise = less deviation)
        assert not bkd.allclose(val1, val2, atol=1e-8)

    def test_avar_factory_creates_working_objective(self, bkd):
        """Test create_prediction_oed_objective with avar deviation."""
        from pyapprox.expdesign.deviation import AVaRDeviationMeasure
        from pyapprox.expdesign.objective import (
            create_prediction_oed_objective,
        )

        objective = create_prediction_oed_objective(
            self._noise_variances,
            self._outer_shapes,
            self._inner_shapes,
            self._latent_samples,
            self._qoi_vals,
            bkd,
            deviation_type="avar",
            risk_type="mean",
            alpha=0.8,
            delta=100,
        )
        # Evaluate — verifies the factory created a fully wired objective
        val = objective(self._design_weights)
        assert val.shape == (1, 1)

        # Compare against manually constructed AVaR objective
        avar = AVaRDeviationMeasure(self._npred, 0.8, bkd, delta=100)
        manual_obj = self._create_objective(bkd, avar)
        val_manual = manual_obj(self._design_weights)
        bkd.assert_allclose(val, val_manual)
