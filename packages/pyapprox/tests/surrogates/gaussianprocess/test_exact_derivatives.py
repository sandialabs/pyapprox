"""
Tests for ExactGaussianProcess derivative computations.

Tests Jacobian computation using derivative checker with finite differences.
"""

import numpy as np

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.surrogates.gaussianprocess import ExactGaussianProcess
from pyapprox.surrogates.kernels.matern import Matern52Kernel


class TestExactGPDerivatives:
    """
    Test class for ExactGaussianProcess derivative computations.
    """

    def test_jacobian_derivative_checker(self, bkd) -> None:
        """Test GP Jacobian using DerivativeChecker."""
        np.random.seed(42)
        nvars = 2
        n_train = 10

        # Create training data
        X_train_np = np.random.randn(nvars, n_train)
        y_train_np = np.sin(X_train_np[0, :] + X_train_np[1, :])[
            None, :
        ]  # Shape: (1, n_train)

        X_train = bkd.array(X_train_np)
        y_train = bkd.array(y_train_np)

        # Create kernel
        kernel = Matern52Kernel([1.0, 1.0], (0.1, 10.0), nvars, bkd)

        gp = ExactGaussianProcess(kernel, nvars, bkd, nugget=0.01)

        gp.fit(X_train, y_train)

        # Create derivative checker
        checker = DerivativeChecker(gp)

        # Test point for Jacobian evaluation
        x0 = bkd.array([[0.5], [0.5]])

        # Use logarithmically-spaced step sizes from 1 down to 1e-14
        # This follows the standard approach: 15 values from 10^0 to 10^(-14)
        fd_eps = bkd.flip(bkd.logspace(-14, 0, 15))

        # Check Jacobian accuracy
        errors = checker.check_derivatives(
            x0, fd_eps=fd_eps, relative=True, verbosity=0
        )

        # Get Jacobian error (first element of errors list)
        jac_error = errors[0]

        # All errors should be finite
        assert bkd.all_bool(bkd.isfinite(jac_error)), \
            "Jacobian errors contain non-finite values"

        # Minimum error should be small, showing accuracy with optimal step size
        min_error = float(bkd.min(jac_error))
        assert min_error < 1e-6, \
            f"Minimum Jacobian relative error {min_error} exceeds threshold"

        # Test that error ratio (min/max) is very small, indicating good convergence
        # A small ratio means errors decrease consistently with smaller step sizes
        error_ratio = float(checker.error_ratio(jac_error))
        assert error_ratio < 1e-6, \
            f"Error ratio {error_ratio:.2e} suggests poor convergence"
