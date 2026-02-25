import unittest
from typing import Generic, Any

from numpy.typing import NDArray
import numpy as np
import torch

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.interface.functions.fromcallable.hessian import (
    FunctionWithJacobianAndHVPFromCallable,
)
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)


class TestDerivativeChecker(Generic[Array], unittest.TestCase):
    __test__ = False

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
            return bkd.reshape(x[0] ** 3 + x[1] ** 2, (1, x.shape[1]))

        def jacobian_function(x: Array) -> Array:
            return bkd.stack([3 * x[0] ** 2, 2 * x[1]], axis=1)

        def hvp_function(x: Array, v: Array) -> Array:
            return bkd.stack([6 * x[0] * v[0], 2 * v[1]], axis=0)

        # Wrap the function using FunctionWithJacobianAndHVPFromCallable
        function_object = FunctionWithJacobianAndHVPFromCallable(
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
        errors = checker.check_derivatives(sample)

        # Assert that the gradient errors are below a tolerance
        self.assertLessEqual(checker.error_ratio(errors[0]), 1e-5)

        # Assert that the Hessian errors are below a tolerance
        self.assertLessEqual(checker.error_ratio(errors[1]), 1e-5)


# Derived test class for NumPy backend
class TestDerivativeCheckerNumpy(TestDerivativeChecker[NDArray[Any]]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# Derived test class for PyTorch backend
class TestDerivativeCheckerTorch(TestDerivativeChecker[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


from pyapprox.util.test_utils import load_tests


if __name__ == "__main__":
    unittest.main()
