import unittest
from typing import Generic, Any

from numpy.typing import NDArray
import torch

from pyapprox.typing.interface.functions.fromcallable.function import (
    FunctionFromCallable,
)
from pyapprox.typing.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.typing.interface.functions.fromcallable.hessian import (
    FunctionWithJacobianAndHVPFromCallable,
)
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd


class TestFunction1D(Generic[Array], unittest.TestCase):
    def bkd(self) -> Backend[Array]:
        """
        Override this method in derived classes to provide the specific
        backend.
        """
        raise NotImplementedError(
            "Derived classes must implement this method."
        )

    def setUp(self) -> None:
        self.nqoi = 1
        self.nvars = 1
        self.samples = self.bkd().reshape(
            self.bkd().linspace(0, 10, 100), (1, -1)
        )  # Shape (nvars, npts)
        self.vec = self.bkd().ones((self.nvars, 1))  # Vector for Hessian tests

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
        self.function_with_hessian = FunctionWithJacobianAndHVPFromCallable(
            nvars=self.nvars,
            fun=self.example_function,
            jacobian=self.example_jacobian,
            hvp=self.example_hvp,
            bkd=self.bkd(),
        )

    def example_function(self, samples: Array) -> Array:
        """
        Example function: Z = sin(x)
        """
        return self.bkd().sin(samples)

    def example_jacobian(self, sample: Array) -> Array:
        """
        Example Jacobian: d(sin(x))/dx = cos(x)
        """
        return self.bkd().cos(sample)

    def example_hvp(self, sample: Array, vec: Array) -> Array:
        """
        Example Hessian-vector product: d^2(sin(x))/dx^2 * vec = -sin(x) * vec
        """
        return -self.bkd().sin(sample) * vec

    def test_function_call(self) -> None:
        values = self.function(self.samples)
        self.assertEqual(values.shape, (self.nqoi, self.samples.shape[1]))
        self.bkd().assert_allclose(values, self.bkd().sin(self.samples))

    def test_jacobian(self) -> None:
        sample = self.samples[:, :1]
        jacobian = self.function_with_jacobian.jacobian(sample)
        self.assertEqual(jacobian.shape, (self.nqoi, self.nvars))
        self.bkd().assert_allclose(jacobian, self.bkd().cos(sample))

    def test_hvp(self) -> None:
        sample = self.samples[:, :1]
        hvp = self.function_with_hessian.hvp(sample, self.vec)
        self.assertEqual(hvp.shape, (self.nvars, 1))
        self.bkd().assert_allclose(hvp, -self.bkd().sin(sample) * self.vec)


class TestFunction3D(Generic[Array], unittest.TestCase):
    def bkd(self) -> Backend[Array]:
        """
        Override this method in derived classes to provide the specific
        backend.
        """
        raise NotImplementedError(
            "Derived classes must implement this method."
        )

    def setUp(self) -> None:
        self.nqoi = 1
        self.nvars = 3
        self.samples = self.bkd().stack(
            [
                self.bkd().linspace(0, 10, 100),
                self.bkd().linspace(10, 20, 100),
                self.bkd().linspace(20, 30, 100),
            ]
        )  # Shape (3, npts)
        self.vec = self.bkd().ones((self.nvars, 1))  # Vector for Hessian tests

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
        self.function_with_hessian = FunctionWithJacobianAndHVPFromCallable(
            nvars=self.nvars,
            fun=self.example_function,
            jacobian=self.example_jacobian,
            hvp=self.example_hvp,
            bkd=self.bkd(),
        )

    def example_function(self, samples: Array) -> Array:
        """
        Example function: Z = sum(sin(x_i)) for i in 1, 2, 3
        """
        return self.bkd().reshape(
            self.bkd().sum(self.bkd().sin(samples), axis=0), (1, -1)
        )

    def example_jacobian(self, sample: Array) -> Array:
        """
        Example Jacobian: d(sum(sin(x_i)))/dx_i = cos(x_i)
        """
        return self.bkd().cos(sample).T

    def example_hvp(self, sample: Array, vec: Array) -> Array:
        """
        Example Hessian-vector product:
        d^2(sum(sin(x_i)))/dx_i^2 * vec = -sin(x_i) * vec
        """
        return -self.bkd().sin(sample) * vec

    def test_function_call(self) -> None:
        values = self.function(self.samples)
        self.assertEqual(values.shape, (self.nqoi, self.samples.shape[1]))
        self.bkd().assert_allclose(values, self.example_function(self.samples))

    def test_jacobian(self) -> None:
        sample = self.samples[:, :1]
        jacobian = self.function_with_jacobian.jacobian(sample)
        self.assertEqual(jacobian.shape, (self.nqoi, self.nvars))
        self.bkd().assert_allclose(jacobian, self.example_jacobian(sample))

    def test_hvp(self) -> None:
        sample = self.samples[:, :1]
        hvp = self.function_with_hessian.hvp(sample, self.vec)
        self.assertEqual(hvp.shape, (self.nvars, 1))
        self.bkd().assert_allclose(hvp, self.example_hvp(sample, self.vec))


# Derived test class for NumPy backend
class TestFunction1DNumpy(TestFunction1D[NDArray[Any]], unittest.TestCase):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestFunction3DNumpy(TestFunction3D[NDArray[Any]], unittest.TestCase):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# Derived test class for PyTorch backend
class TestFunction1DTorch(TestFunction1D[torch.Tensor], unittest.TestCase):
    def setUp(self) -> None:
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


class TestFunction3DTorch(TestFunction3D[torch.Tensor], unittest.TestCase):
    def setUp(self) -> None:
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


# Custom test loader to exclude the base class
def load_tests(
    loader: unittest.TestLoader, tests, pattern: str
) -> unittest.TestSuite:
    """
    Custom test loader to exclude the base class Function1D.
    """
    test_suite = unittest.TestSuite()
    for test_class in [
        TestFunction1DNumpy,
        TestFunction1DTorch,
        TestFunction3DNumpy,
        TestFunction3DTorch,
    ]:
        test_suite.addTests(loader.loadTestsFromTestCase(test_class))
    return test_suite


# Main block to explicitly run tests using the custom loader
if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = load_tests(loader, [], None)
    runner = unittest.TextTestRunner()
    runner.run(suite)
