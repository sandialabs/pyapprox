"""
Tests for Gaussian Process mean functions.

This module tests mean function implementations, focusing on
Jacobian accuracy using DerivativeChecker.
"""

import numpy as np

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.surrogates.gaussianprocess.mean_functions import (
    ConstantMean,
    ZeroMean,
)


class TestMeanFunctions:
    """
    Test class for mean functions.
    """

    # ========== ZeroMean Tests ==========

    def test_zero_mean_evaluation(self, bkd) -> None:
        """Test ZeroMean evaluation."""
        np.random.seed(42)
        nvars = 2
        n_points = 10

        X_np = np.random.randn(nvars, n_points)
        X = bkd.array(X_np)

        mean = ZeroMean(bkd)

        m = mean(X)

        # Check shape - (1, n_points) per convention
        assert m.shape == (1, n_points)

        # Check all zeros
        assert bkd.all_bool(m == 0.0)

    def test_zero_mean_hyperparameters(self, bkd) -> None:
        """Test ZeroMean has no hyperparameters."""
        mean = ZeroMean(bkd)

        hyp_list = mean.hyp_list()

        # No hyperparameters
        assert hyp_list.nparams() == 0
        assert hyp_list.nactive_params() == 0

    def test_zero_mean_jacobian_shape(self, bkd) -> None:
        """Test ZeroMean Jacobian shape."""
        np.random.seed(42)
        nvars = 2
        n_points = 10

        X_np = np.random.randn(nvars, n_points)
        X = bkd.array(X_np)

        mean = ZeroMean(bkd)

        jac = mean.jacobian_wrt_params(X)

        # Should have shape (0, 1, n_points) since no parameters
        assert jac.shape == (0, 1, n_points)

    # ========== ConstantMean Tests ==========

    def test_constant_mean_evaluation(self, bkd) -> None:
        """Test ConstantMean evaluation."""
        np.random.seed(42)
        nvars = 2
        n_points = 10

        X_np = np.random.randn(nvars, n_points)
        X = bkd.array(X_np)

        constant_value = 2.5
        mean = ConstantMean(constant_value, (-10.0, 10.0), bkd)

        m = mean(X)

        # Check shape - (1, n_points) per convention
        assert m.shape == (1, n_points)

        # Check all equal to constant using backend
        expected = bkd.full((1, n_points), constant_value)
        bkd.assert_allclose(m, expected)

    def test_constant_mean_hyperparameters(self, bkd) -> None:
        """Test ConstantMean hyperparameters."""
        constant_value = 1.5
        mean = ConstantMean(constant_value, (-10.0, 10.0), bkd)

        hyp_list = mean.hyp_list()

        # One hyperparameter
        assert hyp_list.nparams() == 1
        assert hyp_list.nactive_params() == 1

        # Check value using backend
        values = hyp_list.get_active_values()
        expected = bkd.array([constant_value])
        bkd.assert_allclose(values, expected, rtol=1e-10)

    def test_constant_mean_jacobian_shape(self, bkd) -> None:
        """Test ConstantMean Jacobian shape."""
        np.random.seed(42)
        nvars = 2
        n_points = 10

        X_np = np.random.randn(nvars, n_points)
        X = bkd.array(X_np)

        mean = ConstantMean(1.0, (-10.0, 10.0), bkd)

        jac = mean.jacobian_wrt_params(X)

        # Should have shape (1, 1, n_points) - one parameter
        assert jac.shape == (1, 1, n_points)

    def test_constant_mean_jacobian_values(self, bkd) -> None:
        """Test ConstantMean Jacobian values."""
        np.random.seed(42)
        nvars = 2
        n_points = 10

        X_np = np.random.randn(nvars, n_points)
        X = bkd.array(X_np)

        mean = ConstantMean(2.0, (-10.0, 10.0), bkd)

        jac = mean.jacobian_wrt_params(X)

        # For constant mean m(x) = c, we have dm/dc = 1
        # So Jacobian should be all ones
        expected = bkd.ones((1, 1, n_points))
        bkd.assert_allclose(jac, expected)

    def test_constant_mean_jacobian_finite_difference(self, bkd) -> None:
        """
        Test ConstantMean Jacobian using finite differences.

        Uses DerivativeChecker to validate analytical Jacobian.
        """
        np.random.seed(42)
        nvars = 2
        n_points = 10

        X_np = np.random.randn(nvars, n_points)
        X = bkd.array(X_np)

        mean = ConstantMean(2.0, (-10.0, 10.0), bkd)

        # Create a wrapper class that implements the protocol
        class MeanFunctionWrapper:
            """Wrapper to make mean function compatible with DerivativeChecker."""

            def __init__(self, mean_func, X_data, bkd_inst):
                self._mean = mean_func
                self._X = X_data
                self._bkd = bkd_inst

            def bkd(self):
                return self._bkd

            def nvars(self) -> int:
                """Number of parameters being optimized."""
                return self._mean.hyp_list().nactive_params()

            def nqoi(self) -> int:
                """Number of outputs - number of samples in this case."""
                return self._X.shape[1]

            def __call__(self, params):
                """
                Evaluate mean function with given parameters.

                Parameters
                ----------
                params : Array, shape (nparams,) or (nparams, 1)
                    Hyperparameters.

                Returns
                -------
                Array, shape (nqoi, 1) = (n_points, 1)
                    Mean function values.
                """
                # Flatten params if needed
                params_flat = self._bkd.reshape(params, (-1,))

                # Update hyperparameters
                self._mean.hyp_list().set_active_values(params_flat)

                # Evaluate mean (returns shape (1, n_points))
                # Transpose to (n_points, 1) for DerivativeChecker
                return self._mean(self._X).T

            def jacobian(self, params):
                """
                Compute Jacobian of mean function.

                Parameters
                ----------
                params : Array, shape (nparams,) or (nparams, 1)
                    Hyperparameters.

                Returns
                -------
                Array, shape (nqoi, nvars) = (n_points, nparams)
                    Jacobian dm/dtheta.
                """
                # Flatten params if needed
                params_flat = self._bkd.reshape(params, (-1,))

                # Update hyperparameters
                self._mean.hyp_list().set_active_values(params_flat)

                # Get Jacobian: shape (nparams, 1, n_points)
                jac = self._mean.jacobian_wrt_params(self._X)

                # Reshape to (n_points, nparams) for DerivativeChecker
                nparams = jac.shape[0]
                n_pts = jac.shape[2]
                jac = self._bkd.reshape(jac, (nparams, n_pts))
                # Transpose to (n_points, nparams)
                jac = jac.T

                return jac

            def jvp(self, params, v):
                """
                Compute Jacobian-vector product.

                Parameters
                ----------
                params : Array, shape (nparams,) or (nparams, 1)
                    Hyperparameters.
                v : Array, shape (nparams,) or (nparams, 1)
                    Vector for JVP.

                Returns
                -------
                Array, shape (nqoi, 1) = (n_points, 1)
                    JVP result.
                """
                # Compute Jacobian
                J = self.jacobian(params)  # (n_points, nparams)

                # Reshape v to (nparams, 1) using backend
                v_reshaped = self._bkd.reshape(v, (-1, 1))

                # JVP = J @ v
                return J @ v_reshaped

        # Create wrapper
        wrapper = MeanFunctionWrapper(mean, X, bkd)

        # Create derivative checker
        checker = DerivativeChecker(wrapper)

        # Get current hyperparameters
        params = mean.hyp_list().get_active_values()

        # Use logarithmically-spaced step sizes
        fd_eps = bkd.flip(bkd.logspace(-14, 0, 15))

        # Check gradient accuracy
        errors = checker.check_derivatives(
            params[:, None],  # Shape: (nactive, 1)
            fd_eps=fd_eps,
            relative=True,
            verbosity=0,
        )

        # Get gradient error
        grad_error = errors[0]

        # Minimum error should be small
        min_error = float(bkd.min(grad_error))
        assert min_error < 1e-6, \
            f"Minimum gradient relative error {min_error} exceeds threshold"

    def test_constant_mean_updates(self, bkd) -> None:
        """Test ConstantMean updates when hyperparameters change."""
        np.random.seed(42)
        nvars = 2
        n_points = 10

        X_np = np.random.randn(nvars, n_points)
        X = bkd.array(X_np)

        mean = ConstantMean(1.0, (-10.0, 10.0), bkd)

        # Initial evaluation
        m1 = mean(X)
        expected1 = bkd.full((1, n_points), 1.0)
        bkd.assert_allclose(m1, expected1)

        # Update hyperparameter
        new_value = bkd.array([3.5])
        mean.hyp_list().set_active_values(new_value)

        # Evaluate again
        m2 = mean(X)
        expected2 = bkd.full((1, n_points), 3.5)
        bkd.assert_allclose(m2, expected2)
