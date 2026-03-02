"""
Standalone tests for prediction OED gradients.

PERMANENT - no legacy imports.

Tests gradient verification using DerivativeChecker for combinations of:
- Deviation measures: StdDev, Entropic
- Risk measures: Mean, Variance
"""

import numpy as np
import pytest

from pyapprox.expdesign import (
    create_prediction_oed_objective,
)
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)


class TestPredictionOEDGradientsStandalone:
    """Standalone tests for prediction OED gradients using DerivativeChecker."""

    def _setup_data(self, bkd):
        self._nobs = 3
        self._ninner = 30
        self._nouter = 20
        self._npred = 2
        np.random.seed(42)

    def _create_test_data(self, bkd):
        """Create test data for gradient verification."""
        noise_variances = bkd.asarray(np.array([0.1, 0.15, 0.2]))
        outer_shapes = bkd.asarray(np.random.randn(self._nobs, self._nouter))
        inner_shapes = bkd.asarray(np.random.randn(self._nobs, self._ninner))
        latent_samples = bkd.asarray(np.random.randn(self._nobs, self._nouter))
        qoi_vals = bkd.asarray(np.random.randn(self._ninner, self._npred))
        return noise_variances, outer_shapes, inner_shapes, latent_samples, qoi_vals

    def _create_derivative_checker_function(self, bkd, obj):
        """Wrap objective for DerivativeChecker compatibility.

        DerivativeChecker expects:
        - __call__(samples): (nvars, nsamples) -> (nqoi, nsamples)
        - jacobian(sample): (nvars, 1) -> (nqoi, nvars)

        PredictionOEDObjective has:
        - __call__(weights): (nobs, 1) -> (1, 1)
        - jacobian(weights): (nobs, 1) -> (1, nobs)
        """

        def value_fun(samples):
            # samples: (nvars, nsamples) = (nobs, nsamples)
            nsamples = samples.shape[1]
            results = []
            for ii in range(nsamples):
                w = samples[:, ii : ii + 1]  # (nobs, 1)
                val = obj(w)  # (1, 1)
                results.append(val[0, 0])
            return bkd.reshape(bkd.stack(results), (1, nsamples))

        def jacobian_fun(sample):
            # sample: (nvars, 1) = (nobs, 1)
            jac = obj.jacobian(sample)  # (1, nobs)
            return jac  # (nqoi=1, nvars=nobs)

        return FunctionWithJacobianFromCallable(
            nqoi=1,  # scalar objective
            nvars=obj.nvars(),
            fun=value_fun,
            jacobian=jacobian_fun,
            bkd=bkd,
        )

    @pytest.mark.parametrize(
        "deviation_type,risk_type",
        [
            ("stdev", "mean"),
            ("stdev", "variance"),
            ("entropic", "mean"),
            ("entropic", "variance"),
        ],
    )
    def test_jacobian_derivative_checker(self, bkd, deviation_type, risk_type):
        """Test Jacobian using DerivativeChecker for all combinations."""
        self._setup_data(bkd)
        noise_variances, outer_shapes, inner_shapes, latent_samples, qoi_vals = (
            self._create_test_data(bkd)
        )

        # Handle entropic alpha parameter
        extra_kwargs = {}
        if deviation_type == "entropic":
            extra_kwargs["alpha"] = 0.5

        objective = create_prediction_oed_objective(
            noise_variances,
            outer_shapes,
            inner_shapes,
            latent_samples,
            qoi_vals,
            bkd,
            deviation_type=deviation_type,
            risk_type=risk_type,
            **extra_kwargs,
        )

        wrapped = self._create_derivative_checker_function(bkd, objective)
        checker = DerivativeChecker(wrapped)

        weights = bkd.asarray(np.random.uniform(0.5, 1.5, (self._nobs, 1)))
        errors = checker.check_derivatives(weights)

        # Jacobian errors should show second-order convergence
        ratio = checker.error_ratio(errors[0])
        assert float(bkd.to_numpy(ratio)) <= 1e-5, (
            f"DerivativeChecker failed for {deviation_type}/{risk_type}: ratio={ratio}"
        )

    def test_jacobian_shape(self, bkd):
        """Test Jacobian has correct shape."""
        self._setup_data(bkd)
        noise_variances, outer_shapes, inner_shapes, latent_samples, qoi_vals = (
            self._create_test_data(bkd)
        )

        objective = create_prediction_oed_objective(
            noise_variances,
            outer_shapes,
            inner_shapes,
            latent_samples,
            qoi_vals,
            bkd,
            deviation_type="stdev",
            risk_type="mean",
        )

        weights = bkd.ones((self._nobs, 1)) / self._nobs
        jac = objective.jacobian(weights)

        jac_np = bkd.to_numpy(jac)
        # Shape should be (1, nobs) for scalar objective
        assert jac_np.shape == (1, self._nobs)

    def test_jacobian_is_finite(self, bkd):
        """Test Jacobian values are finite."""
        self._setup_data(bkd)
        noise_variances, outer_shapes, inner_shapes, latent_samples, qoi_vals = (
            self._create_test_data(bkd)
        )

        objective = create_prediction_oed_objective(
            noise_variances,
            outer_shapes,
            inner_shapes,
            latent_samples,
            qoi_vals,
            bkd,
            deviation_type="stdev",
            risk_type="mean",
        )

        weights = bkd.asarray(np.random.uniform(0.5, 1.5, (self._nobs, 1)))
        jac = objective.jacobian(weights)

        jac_np = bkd.to_numpy(jac)
        assert np.all(np.isfinite(jac_np))

    @pytest.mark.parametrize(
        "deviation_type",
        [
            "stdev",
            "entropic",
        ],
    )
    def test_jacobian_nonzero_for_deviation(self, bkd, deviation_type):
        """Test Jacobian is non-zero for different deviation types."""
        self._setup_data(bkd)
        noise_variances, outer_shapes, inner_shapes, latent_samples, qoi_vals = (
            self._create_test_data(bkd)
        )

        extra_kwargs = {}
        if deviation_type == "entropic":
            extra_kwargs["alpha"] = 0.5

        objective = create_prediction_oed_objective(
            noise_variances,
            outer_shapes,
            inner_shapes,
            latent_samples,
            qoi_vals,
            bkd,
            deviation_type=deviation_type,
            risk_type="mean",
            **extra_kwargs,
        )

        weights = bkd.asarray(np.random.uniform(0.5, 1.5, (self._nobs, 1)))
        jac = objective.jacobian(weights)

        jac_np = bkd.to_numpy(jac)
        # At least some gradient components should be non-zero
        assert not np.allclose(jac_np, 0.0)

    def test_jacobian_changes_with_weights(self, bkd):
        """Test Jacobian values change with different weights."""
        self._setup_data(bkd)
        noise_variances, outer_shapes, inner_shapes, latent_samples, qoi_vals = (
            self._create_test_data(bkd)
        )

        objective = create_prediction_oed_objective(
            noise_variances,
            outer_shapes,
            inner_shapes,
            latent_samples,
            qoi_vals,
            bkd,
            deviation_type="stdev",
            risk_type="mean",
        )

        weights1 = bkd.ones((self._nobs, 1)) * 0.5
        weights2 = bkd.ones((self._nobs, 1)) * 2.0

        jac1 = objective.jacobian(weights1)
        jac2 = objective.jacobian(weights2)

        jac1_np = bkd.to_numpy(jac1)
        jac2_np = bkd.to_numpy(jac2)

        # Jacobians should be different at different weights
        assert not np.allclose(jac1_np, jac2_np)

    def test_objective_value_and_jacobian_consistency(self, bkd):
        """Test objective value and Jacobian are computed consistently."""
        self._setup_data(bkd)
        noise_variances, outer_shapes, inner_shapes, latent_samples, qoi_vals = (
            self._create_test_data(bkd)
        )

        objective = create_prediction_oed_objective(
            noise_variances,
            outer_shapes,
            inner_shapes,
            latent_samples,
            qoi_vals,
            bkd,
            deviation_type="stdev",
            risk_type="mean",
        )

        weights = bkd.asarray(np.random.uniform(0.5, 1.5, (self._nobs, 1)))

        # Compute multiple times - should be deterministic
        val1 = objective(weights)
        val2 = objective(weights)
        jac1 = objective.jacobian(weights)
        jac2 = objective.jacobian(weights)

        bkd.assert_allclose(val1, val2, rtol=1e-12)
        bkd.assert_allclose(jac1, jac2, rtol=1e-12)
