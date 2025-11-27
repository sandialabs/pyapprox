import unittest
from typing import Generic, Any

from numpy.typing import NDArray
import numpy as np
import torch

from pyapprox.typing.util.backend import Array, Backend
from pyapprox.typing.util.numpy import NumpyBkd
from pyapprox.typing.util.torch import TorchBkd
from pyapprox.typing.util.abstracttestcase import AbstractTestCase
from pyapprox.typing.interface.functions.hessian import (
    FunctionWithJacobianApplyHessianFromCallable,
)
from pyapprox.typing.interface.functions.derivative_checker import (
    DerivativeChecker,
)


class TestDerivativeChecker(Generic[Array], AbstractTestCase):
    def bkd(self) -> Backend[Array]:
        """
        Override this method in derived classes to provide the specific backend.
        """
        raise NotImplementedError(
            "Derived classes must implement this method."
        )

    def test_derivative_checker(self) -> None:
        """
        Test the derivative checker for a simple quadratic function.
        """
        bkd = self.bkd()

        # Define the value function
        def value_function(x: Array) -> Array:
            print(x.shape)
            return bkd.asarray([[x[0] ** 3 + x[1] ** 2]])

        def jacobian_function(x: Array) -> Array:
            return bkd.asarray([[3 * x[0] ** 2, 2 * x[1]]])

        def hvp_function(x: Array, v: Array) -> Array:
            return bkd.asarray([[6 * x[0] * v[0], 2 * v[1]]]).T

        # Wrap the function using FunctionWithJacobianApplyHessianFromCallable
        function_object = FunctionWithJacobianApplyHessianFromCallable(
            nvars=2,
            fun=value_function,
            jacobian=jacobian_function,
            hvp=hvp_function,
            bkd=bkd,
        )

        # Initialize DerivativeChecker
        checker = DerivativeChecker(function_object)

        # Define a sample point
        sample = bkd.asarray([[2.0, 1.0]]).T

        # Check derivatives
        errors = checker.check_derivatives(sample, verbosity=1)

        # Assert that the gradient errors are below a tolerance
        self.assertTrue(errors[0].min() / errors[0].max() < 1e-7)

        # Assert that the Hessian errors are below a tolerance
        self.assertTrue(errors[1].min() / errors[1].max() < 1e-7)


# Derived test class for NumPy backend
class TestDerivativeCheckerNumpy(
    TestDerivativeChecker[NDArray[Any]], unittest.TestCase
):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# Derived test class for PyTorch backend
class TestDerivativeCheckerTorch(
    TestDerivativeChecker[torch.Tensor], unittest.TestCase
):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


if __name__ == "__main__":
    unittest.main()
