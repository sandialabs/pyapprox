"""
Tests for ExactGaussianProcess derivative computations.

Tests Jacobian computation using derivative checker with finite differences.
"""
import unittest
from typing import Generic, Any
import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.util.backends.protocols import Backend, Array
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.surrogates.kernels.matern import Matern52Kernel
from pyapprox.surrogates.gaussianprocess import ExactGaussianProcess
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker
)


class TestExactGPDerivatives(Generic[Array], unittest.TestCase):
    """
    Base test class for ExactGaussianProcess derivative computations.

    Derived classes must implement the bkd() method.
    """

    __test__ = False

    def setUp(self) -> None:
        """Set up test environment."""
        np.random.seed(42)
        self.nvars = 2
        self.n_train = 10

        # Create training data
        X_train_np = np.random.randn(self.nvars, self.n_train)
        y_train_np = np.sin(X_train_np[0, :] + X_train_np[1, :])[None, :]  # Shape: (1, n_train)

        self.X_train = self.bkd().array(X_train_np)
        self.y_train = self.bkd().array(y_train_np)

        # Create kernel
        self.kernel = Matern52Kernel(
            [1.0, 1.0],
            (0.1, 10.0),
            self.nvars,
            self.bkd()
        )

    def bkd(self) -> Backend[Array]:
        """Override in derived classes."""
        raise NotImplementedError

    def test_jacobian_derivative_checker(self) -> None:
        """Test GP Jacobian using DerivativeChecker."""
        gp = ExactGaussianProcess(
            self.kernel,
            self.nvars,
            self.bkd(),
            nugget=0.01
        )

        gp.fit(self.X_train, self.y_train)

        # Create derivative checker
        checker = DerivativeChecker(gp)

        # Test point for Jacobian evaluation
        x0 = self.bkd().array([[0.5], [0.5]])

        # Use logarithmically-spaced step sizes from 1 down to 1e-14
        # This follows the standard approach: 15 values from 10^0 to 10^(-14)
        fd_eps = self.bkd().flip(self.bkd().logspace(-14, 0, 15))

        # Check Jacobian accuracy
        errors = checker.check_derivatives(
            x0,
            fd_eps=fd_eps,
            relative=True,
            verbosity=0
        )

        # Get Jacobian error (first element of errors list)
        jac_error = errors[0]

        # All errors should be finite
        self.assertTrue(self.bkd().all_bool(self.bkd().isfinite(jac_error)),
                       "Jacobian errors contain non-finite values")

        # Minimum error should be small, showing accuracy with optimal step size
        min_error = float(self.bkd().min(jac_error))
        self.assertLess(min_error, 1e-6,
                       f"Minimum Jacobian relative error {min_error} exceeds threshold")

        # Test that error ratio (min/max) is very small, indicating good convergence
        # A small ratio means errors decrease consistently with smaller step sizes
        error_ratio = float(checker.error_ratio(jac_error))
        self.assertLess(error_ratio, 1e-6,
                       f"Error ratio {error_ratio:.2e} suggests poor convergence")


# NumPy implementation
class TestExactGPDerivativesNumpy(TestExactGPDerivatives[NDArray[Any]]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# PyTorch implementation
class TestExactGPDerivativesTorch(TestExactGPDerivatives[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> TorchBkd:
        return self._bkd


from pyapprox.util.test_utils import load_tests  # noqa: F401


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = load_tests(loader, [], None)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
