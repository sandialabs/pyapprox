"""
Tests for ExactGaussianProcess hyperparameter optimization.

Tests that fit() performs hyperparameter optimization, custom optimizer
configuration, and fixed hyperparameter handling.
"""

import numpy as np
import pytest

from pyapprox.optimization.minimize.scipy.trust_constr import (
    ScipyTrustConstrOptimizer,
)
from pyapprox.surrogates.gaussianprocess import (
    ConstantMean,
    ExactGaussianProcess,
)
from pyapprox.surrogates.kernels.matern import Matern52Kernel


class TestExactGPOptimization:
    """
    Test class for ExactGaussianProcess hyperparameter optimization.
    """

    def test_fit_optimizes_hyperparameters(self, bkd) -> None:
        """Test that fit() optimizes hyperparameters with default optimizer."""
        np.random.seed(42)
        nvars = 2
        n_train = 10
        n_test = 5

        X_train_np = np.random.randn(nvars, n_train)
        y_train_np = np.sin(X_train_np[0, :] + X_train_np[1, :])[None, :]
        X_train = bkd.array(X_train_np)
        y_train = bkd.array(y_train_np)

        X_test_np = np.random.randn(nvars, n_test)
        X_test = bkd.array(X_test_np)

        # Use suboptimal long length scales initially
        kernel = Matern52Kernel([3.0, 3.0], (0.1, 10.0), nvars, bkd)

        gp = ExactGaussianProcess(kernel, nvars, bkd, nugget=0.1)

        # Store initial hyperparameters (before fit)
        params_before = bkd.to_numpy(gp.hyp_list().get_values()).copy()

        # fit() should optimize hyperparameters automatically
        gp.fit(X_train, y_train)

        # Get final hyperparameters
        params_after = bkd.to_numpy(gp.hyp_list().get_values())

        # Hyperparameters should have changed
        params_diff = np.abs(params_after - params_before)
        max_change = float(np.max(params_diff))
        assert max_change > 1e-6, "Hyperparameters should change during fit()"

        # GP should still make predictions
        mean = gp.predict(X_test)
        assert mean.shape == (1, n_test)
        assert bkd.all_bool(bkd.isfinite(mean))

    def test_fit_with_constant_mean(self, bkd) -> None:
        """Test fit() optimizes both kernel and mean hyperparameters."""
        np.random.seed(42)
        nvars = 2
        n_train = 10
        n_test = 5

        X_train_np = np.random.randn(nvars, n_train)
        y_train_np = np.sin(X_train_np[0, :] + X_train_np[1, :])[None, :]
        X_train = bkd.array(X_train_np)
        y_train = bkd.array(y_train_np)

        X_test_np = np.random.randn(nvars, n_test)
        X_test = bkd.array(X_test_np)

        kernel = Matern52Kernel([2.0, 2.0], (0.1, 10.0), nvars, bkd)

        # Use ConstantMean with initial guess
        constant_mean = ConstantMean(0.0, (-5.0, 5.0), bkd)

        gp = ExactGaussianProcess(
            kernel, nvars, bkd, mean_function=constant_mean, nugget=0.1
        )

        # fit() should optimize both kernel and mean parameters
        gp.fit(X_train, y_train)

        # Should have 3 hyperparameters (2 length scales + 1 constant)
        assert gp.hyp_list().nparams() == 3

        # GP should make predictions
        mean = gp.predict(X_test)
        assert mean.shape == (1, n_test)

    def test_fit_improves_predictions(self, bkd) -> None:
        """Test that fit() optimization improves prediction accuracy."""
        nvars = 2

        # Create synthetic data with known length scales
        np.random.seed(123)
        X_train_np = np.random.uniform(-2, 2, (nvars, 30))

        # True function with characteristic length scale ~0.5
        y_train_np = np.sin(2 * X_train_np[0, :]) * np.cos(2 * X_train_np[1, :])
        y_train_np = y_train_np[None, :]

        X_train = bkd.array(X_train_np)
        y_train = bkd.array(y_train_np)

        # Test points
        X_test_np = np.random.uniform(-2, 2, (nvars, 10))
        y_test_np = np.sin(2 * X_test_np[0, :]) * np.cos(2 * X_test_np[1, :])
        y_test_np = y_test_np[None, :]

        X_test = bkd.array(X_test_np)
        y_test = bkd.array(y_test_np)

        # Start with poor initial hyperparameters (long length scales)
        kernel = Matern52Kernel(
            [5.0, 5.0],  # Too long
            (0.1, 10.0),
            nvars,
            bkd,
        )

        gp = ExactGaussianProcess(kernel, nvars, bkd, nugget=0.01)

        # First fit WITHOUT optimization to get baseline error
        gp.hyp_list().set_all_inactive()
        gp.fit(X_train, y_train)
        mean_before = gp.predict(X_test)
        error_before = mean_before - y_test
        abs_error_before = bkd.abs(error_before)
        mse_before = float(
            bkd.sum(abs_error_before**2) / abs_error_before.shape[1]
        )

        # Reset hyperparameters and re-enable optimization
        # Note: kernel uses LogHyperParameter, so set_values expects log-space
        # values. log(5.0) = 1.609... which is within log-space bounds [-2.3, 2.3]
        import math

        kernel._hyp_list.set_values(bkd.array([math.log(5.0), math.log(5.0)]))
        gp.hyp_list().set_all_active()

        # Now fit WITH optimization
        gp.fit(X_train, y_train)

        # Predict after optimization
        mean_after = gp.predict(X_test)
        error_after = mean_after - y_test
        abs_error_after = bkd.abs(error_after)
        mse_after = float(bkd.sum(abs_error_after**2) / abs_error_after.shape[1])

        # MSE should improve
        assert mse_after <= mse_before * 1.1, \
            f"MSE got worse: {mse_before} -> {mse_after}"

    def test_fit_with_custom_optimizer(self, bkd) -> None:
        """Test fit() with custom optimizer via set_optimizer()."""
        np.random.seed(42)
        nvars = 2
        n_train = 10
        n_test = 5

        X_train_np = np.random.randn(nvars, n_train)
        y_train_np = np.sin(X_train_np[0, :] + X_train_np[1, :])[None, :]
        X_train = bkd.array(X_train_np)
        y_train = bkd.array(y_train_np)

        X_test_np = np.random.randn(nvars, n_test)
        X_test = bkd.array(X_test_np)

        kernel = Matern52Kernel([3.0, 3.0], (0.1, 10.0), nvars, bkd)

        gp = ExactGaussianProcess(kernel, nvars, bkd, nugget=0.1)

        # Set custom optimizer
        custom_optimizer = ScipyTrustConstrOptimizer(
            maxiter=500, gtol=1e-8, verbosity=0
        )
        gp.set_optimizer(custom_optimizer)

        # Verify optimizer is set
        assert gp.optimizer() is not None

        # fit() should use the custom optimizer
        gp.fit(X_train, y_train)

        # Should still make predictions
        mean = gp.predict(X_test)
        assert mean.shape == (1, n_test)

    def test_fit_skips_optimization_when_all_inactive(self, bkd) -> None:
        """Test that fit() skips optimization when all hyperparameters are inactive."""
        np.random.seed(42)
        nvars = 2
        n_train = 10
        n_test = 5

        X_train_np = np.random.randn(nvars, n_train)
        y_train_np = np.sin(X_train_np[0, :] + X_train_np[1, :])[None, :]
        X_train = bkd.array(X_train_np)
        y_train = bkd.array(y_train_np)

        X_test_np = np.random.randn(nvars, n_test)
        X_test = bkd.array(X_test_np)

        kernel = Matern52Kernel([1.0, 1.0], (0.1, 10.0), nvars, bkd)

        gp = ExactGaussianProcess(kernel, nvars, bkd, nugget=0.1)

        # Set all hyperparameters as inactive (fixed)
        gp.hyp_list().set_all_inactive()

        # Store initial values
        params_before = bkd.to_numpy(gp.hyp_list().get_values()).copy()

        # fit() should only store data (no optimization)
        gp.fit(X_train, y_train)

        # Hyperparameters should NOT change
        params_after = bkd.to_numpy(gp.hyp_list().get_values())
        np.testing.assert_array_equal(
            params_before,
            params_after,
            "Hyperparameters should not change when all are inactive",
        )

        # GP should still make predictions
        mean = gp.predict(X_test)
        assert mean.shape == (1, n_test)

    def test_shared_optimizer_is_cloned(self, bkd) -> None:
        """Test that optimizer is cloned so shared optimizer is safe."""
        np.random.seed(42)
        nvars = 2
        n_train = 10
        n_test = 5

        X_train_np = np.random.randn(nvars, n_train)
        y_train_np = np.sin(X_train_np[0, :] + X_train_np[1, :])[None, :]
        X_train = bkd.array(X_train_np)
        y_train = bkd.array(y_train_np)

        X_test_np = np.random.randn(nvars, n_test)
        X_test = bkd.array(X_test_np)

        # Create shared optimizer
        shared_optimizer = ScipyTrustConstrOptimizer(maxiter=500, verbosity=0)

        # Create two GPs with same optimizer
        kernel1 = Matern52Kernel([2.0, 2.0], (0.1, 10.0), nvars, bkd)
        gp1 = ExactGaussianProcess(kernel1, nvars, bkd, nugget=0.1)
        gp1.set_optimizer(shared_optimizer)

        kernel2 = Matern52Kernel([2.0, 2.0], (0.1, 10.0), nvars, bkd)
        gp2 = ExactGaussianProcess(kernel2, nvars, bkd, nugget=0.1)
        gp2.set_optimizer(shared_optimizer)

        # Fitting gp1 should not affect gp2's optimizer
        gp1.fit(X_train, y_train)

        # The shared optimizer should still be unbound
        assert not shared_optimizer.is_bound()

        # gp2 should also be able to fit
        gp2.fit(X_train, y_train)

        # Both should make predictions
        mean1 = gp1.predict(X_test)
        mean2 = gp2.predict(X_test)
        assert mean1.shape == (1, n_test)
        assert mean2.shape == (1, n_test)

    def test_set_optimizer_validates_protocol(self, bkd) -> None:
        """Test that set_optimizer validates the protocol."""
        nvars = 2
        kernel = Matern52Kernel([1.0, 1.0], (0.1, 10.0), nvars, bkd)
        gp = ExactGaussianProcess(kernel, nvars, bkd, nugget=0.1)

        # Should raise TypeError for invalid optimizer
        with pytest.raises(TypeError):
            gp.set_optimizer("not an optimizer")  # type: ignore
