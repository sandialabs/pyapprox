"""
Tests for Gaussian Process mean functions.

This module tests mean function implementations, focusing on
Jacobian accuracy using DerivativeChecker.
"""

import unittest
from typing import Generic, Any
import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.protocols import Backend, Array
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.surrogates.gaussianprocess.mean_functions import (
    ZeroMean,
    ConstantMean
)
from pyapprox.typing.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker
)


class TestMeanFunctions(Generic[Array], unittest.TestCase):
    """
    Base test class for mean functions.

    Derived classes must implement the bkd() method.
    """

    __test__ = False

    def setUp(self) -> None:
        """Set up test environment."""
        np.random.seed(42)
        self.nvars = 2
        self.n_points = 10

        # Create test points
        X_np = np.random.randn(self.nvars, self.n_points)
        self.X = self.bkd().array(X_np)

    def bkd(self) -> Backend[Array]:
        """Override in derived classes."""
        raise NotImplementedError

    # ========== ZeroMean Tests ==========

    def test_zero_mean_evaluation(self) -> None:
        """Test ZeroMean evaluation."""
        mean = ZeroMean(self.bkd())

        m = mean(self.X)

        # Check shape
        self.assertEqual(m.shape, (self.n_points, 1))

        # Check all zeros
        self.assertTrue(
            self.bkd().all_bool(m == 0.0)
        )

    def test_zero_mean_hyperparameters(self) -> None:
        """Test ZeroMean has no hyperparameters."""
        mean = ZeroMean(self.bkd())

        hyp_list = mean.hyp_list()

        # No hyperparameters
        self.assertEqual(hyp_list.nparams(), 0)
        self.assertEqual(hyp_list.nactive_params(), 0)

    def test_zero_mean_jacobian_shape(self) -> None:
        """Test ZeroMean Jacobian shape."""
        mean = ZeroMean(self.bkd())

        jac = mean.jacobian_wrt_params(self.X)

        # Should have shape (0, n_points, 1) since no parameters
        self.assertEqual(jac.shape, (0, self.n_points, 1))

    # ========== ConstantMean Tests ==========

    def test_constant_mean_evaluation(self) -> None:
        """Test ConstantMean evaluation."""
        constant_value = 2.5
        mean = ConstantMean(constant_value, (-10.0, 10.0), self.bkd())

        m = mean(self.X)

        # Check shape
        self.assertEqual(m.shape, (self.n_points, 1))

        # Check all equal to constant
        m_np = self.bkd().to_numpy(m)
        self.assertTrue(np.allclose(m_np, constant_value))

    def test_constant_mean_hyperparameters(self) -> None:
        """Test ConstantMean hyperparameters."""
        constant_value = 1.5
        mean = ConstantMean(constant_value, (-10.0, 10.0), self.bkd())

        hyp_list = mean.hyp_list()

        # One hyperparameter
        self.assertEqual(hyp_list.nparams(), 1)
        self.assertEqual(hyp_list.nactive_params(), 1)

        # Check value
        values = hyp_list.get_active_values()
        values_np = self.bkd().to_numpy(values)
        self.assertAlmostEqual(values_np[0], constant_value, places=10)

    def test_constant_mean_jacobian_shape(self) -> None:
        """Test ConstantMean Jacobian shape."""
        mean = ConstantMean(1.0, (-10.0, 10.0), self.bkd())

        jac = mean.jacobian_wrt_params(self.X)

        # Should have shape (1, n_points, 1) - one parameter
        self.assertEqual(jac.shape, (1, self.n_points, 1))

    def test_constant_mean_jacobian_values(self) -> None:
        """Test ConstantMean Jacobian values."""
        mean = ConstantMean(2.0, (-10.0, 10.0), self.bkd())

        jac = mean.jacobian_wrt_params(self.X)

        # For constant mean m(x) = c, we have ∂m/∂c = 1
        # So Jacobian should be all ones
        jac_np = self.bkd().to_numpy(jac)
        self.assertTrue(np.allclose(jac_np, 1.0))

    def test_constant_mean_jacobian_finite_difference(self) -> None:
        """
        Test ConstantMean Jacobian using finite differences.

        Uses DerivativeChecker to validate analytical Jacobian.
        """
        mean = ConstantMean(2.0, (-10.0, 10.0), self.bkd())

        # Create a wrapper class that implements the protocol
        class MeanFunctionWrapper:
            """Wrapper to make mean function compatible with DerivativeChecker."""

            def __init__(self, mean_func, X, bkd):
                self._mean = mean_func
                self._X = X
                self._bkd = bkd

            def bkd(self):
                return self._bkd

            def nvars(self):
                """Number of parameters being optimized."""
                return self._mean.hyp_list().nactive_params()

            def nqoi(self):
                """Number of outputs - number of samples in this case."""
                return self._X.shape[1]

            def __call__(self, params: Array) -> Array:
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
                # Flatten params if needed (DerivativeChecker may pass (nparams, 1))
                params_flat = self._bkd.reshape(params, (-1,))

                # Update hyperparameters
                self._mean.hyp_list().set_active_values(params_flat)

                # Evaluate mean (returns shape (n_points, 1))
                return self._mean(self._X)

            def jacobian(self, params: Array) -> Array:
                """
                Compute Jacobian of mean function.

                Parameters
                ----------
                params : Array, shape (nparams,) or (nparams, 1)
                    Hyperparameters.

                Returns
                -------
                Array, shape (nqoi, nvars) = (n_points, nparams)
                    Jacobian ∂m/∂θ.
                """
                # Flatten params if needed
                params_flat = self._bkd.reshape(params, (-1,))

                # Update hyperparameters
                self._mean.hyp_list().set_active_values(params_flat)

                # Get Jacobian: shape (nparams, n_points, 1)
                jac = self._mean.jacobian_wrt_params(self._X)

                # Reshape to (n_points, nparams) for DerivativeChecker
                # First reshape to (nparams, n_points) by removing last dim
                nparams = jac.shape[0]
                n_points = jac.shape[1]
                jac = self._bkd.reshape(jac, (nparams, n_points))
                # Then transpose to (n_points, nparams)
                jac = jac.T

                return jac

            def jvp(self, params: Array, v: Array) -> Array:
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

                # Flatten v and ensure same dtype as J
                v_np = self._bkd.to_numpy(v).flatten()
                v_reshaped = self._bkd.array(v_np.reshape(-1, 1))  # (nparams, 1)

                # JVP = J @ v
                return J @ v_reshaped

        # Create wrapper
        wrapper = MeanFunctionWrapper(mean, self.X, self.bkd())

        # Create derivative checker
        checker = DerivativeChecker(wrapper)

        # Get current hyperparameters
        params = mean.hyp_list().get_active_values()

        # Use logarithmically-spaced step sizes
        fd_eps = self.bkd().flip(self.bkd().logspace(-14, 0, 15))

        # Check gradient accuracy
        errors = checker.check_derivatives(
            params[:, None],  # Shape: (nactive, 1)
            fd_eps=fd_eps,
            relative=True,
            verbosity=0
        )

        # Get gradient error
        grad_error = errors[0]

        # Minimum error should be small
        min_error = float(self.bkd().min(grad_error))
        self.assertLess(min_error, 1e-6,
                       f"Minimum gradient relative error {min_error} exceeds threshold")

    def test_constant_mean_updates(self) -> None:
        """Test ConstantMean updates when hyperparameters change."""
        mean = ConstantMean(1.0, (-10.0, 10.0), self.bkd())

        # Initial evaluation
        m1 = mean(self.X)
        m1_np = self.bkd().to_numpy(m1)
        self.assertTrue(np.allclose(m1_np, 1.0))

        # Update hyperparameter
        new_value = self.bkd().array([3.5])
        mean.hyp_list().set_active_values(new_value)

        # Evaluate again
        m2 = mean(self.X)
        m2_np = self.bkd().to_numpy(m2)
        self.assertTrue(np.allclose(m2_np, 3.5))


# ========== Backend-Specific Test Classes ==========

class TestMeanFunctionsNumpy(TestMeanFunctions[NDArray[Any]]):
    """Test mean functions with NumPy backend."""

    def bkd(self) -> NumpyBkd:
        """Return NumPy backend."""
        if not hasattr(self, '_bkd'):
            self._bkd = NumpyBkd()
        return self._bkd


class TestMeanFunctionsTorch(TestMeanFunctions[torch.Tensor]):
    """Test mean functions with PyTorch backend."""

    def bkd(self) -> TorchBkd:
        """Return PyTorch backend."""
        if not hasattr(self, '_bkd'):
            # Set default dtype to float64 for consistency
            torch.set_default_dtype(torch.float64)
            self._bkd = TorchBkd()
        return self._bkd


from pyapprox.typing.util.test_utils import load_tests


if __name__ == "__main__":
    # Run tests
    suite = unittest.TestLoader().loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit with error code if tests failed
    import sys
    sys.exit(0 if result.wasSuccessful() else 1)
