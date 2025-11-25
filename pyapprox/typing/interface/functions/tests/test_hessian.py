import unittest
from typing import Generic, Any

from numpy.typing import NDArray
import torch
import numpy as np

from pyapprox.typing.interface.functions.function import FunctionFromCallable
from pyapprox.typing.interface.functions.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.typing.interface.functions.hessian import (
    FunctionWithJacobianApplyHessianFromCallable,
)
from pyapprox.typing.util.backend import Array, Backend
from pyapprox.typing.util.numpy import NumpyBkd
from pyapprox.typing.util.torch import TorchBkd
from pyapprox.typing.util.abstracttestcase import AbstractTestCase


class TestHessian(Generic[Array], AbstractTestCase):
    def bkd(self) -> Backend[Array]:
        """
        Override this method in derived classes to provide the specific backend.
        """
        raise NotImplementedError(
            "Derived classes must implement this method."
        )

    def setUp(self) -> None:
        self.nqoi = 1
        self.nvars = 1
        self.samples = (
            self.bkd().linspace(0, 10, 100).reshape(1, -1)
        )  # Shape (nvars, npts)
        self.vec = self.bkd().ones((self.nvars,))  # Vector for Hessian tests

        # Define the function
        self.function = FunctionFromCallable(
            nqoi=self.nqoi,
            nvars=self.nvars,
            fun=self.example_function,
            bkd=self.bkd(),
        )

        # Define the function with Jacobian
        self.function_with_jacobian = FunctionWithJacobianFromCallable(
            nqoi=self.nqoi,
            nvars=self.nvars,
            fun=self.example_function,
            jacobian=self.example_jacobian,
            bkd=self.bkd(),
        )

        # Define the function with Hessian
        self.function_with_hessian = (
            FunctionWithJacobianApplyHessianFromCallable(
                nvars=self.nvars,
                fun=self.example_function,
                jacobian=self.example_jacobian,
                hvp=self.example_hvp,
                bkd=self.bkd(),
            )
        )

    def example_function(self, samples: Array) -> Array:
        """
        Example function: Z = sin(x)
        """
        return self.bkd().sin(samples)

    def example_jacobian(self, samples: Array) -> Array:
        """
        Example Jacobian: d(sin(x))/dx = cos(x)
        """
        return self.bkd().cos(samples)

    def example_hvp(self, samples: Array, vec: Array) -> Array:
        """
        Example Hessian-vector product: d^2(sin(x))/dx^2 * vec = -sin(x) * vec
        """
        return -self.bkd().sin(samples) * vec

    def test_function_call(self) -> None:
        values = self.function(self.samples)
        self.assertEqual(values.shape, (self.nqoi, self.samples.shape[1]))
        self.assertTrue(
            self.bkd().allclose(values, self.bkd().sin(self.samples))
        )

    def test_jacobian(self) -> None:
        jacobian = self.function_with_jacobian.jacobian(self.samples)
        self.assertEqual(jacobian.shape, (self.nqoi, self.nvars))
        self.assertTrue(
            self.bkd().allclose(jacobian, self.bkd().cos(self.samples))
        )

    def test_hvp(self) -> None:
        hvp = self.function_with_hessian.apply_hessian(self.samples, self.vec)
        self.assertEqual(hvp.shape, (self.nvars,))
        self.assertTrue(
            self.bkd().allclose(hvp, -self.bkd().sin(self.samples) * self.vec)
        )


# Derived test class for NumPy backend
class TestHessianNumpy(TestHessian[NDArray[Any]], unittest.TestCase):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# Derived test class for PyTorch backend
class TestHessianTorch(TestHessian[torch.Tensor], unittest.TestCase):
    def setUp(self) -> None:
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:  # -> TorchBkd:
        return self._bkd


if __name__ == "__main__":
    unittest.main()
