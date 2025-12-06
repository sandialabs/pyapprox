import unittest
from typing import Generic, Any

import torch
import numpy as np
from numpy.typing import NDArray

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.interface.functions.numpy.numpy_function_factory import (
    numpy_function_wrapper_factory,
)


class ExampleFunction(Generic[Array]):
    def __init__(self, backend: Backend[Array]):
        self._bkd = backend

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return 2

    def nqoi(self) -> int:
        return 1

    def __call__(self, samples: Array) -> Array:
        return samples**2


class ExampleFunctionWithJacobian(Generic[Array]):
    def __init__(self, backend: Backend[Array]):
        self._bkd = backend

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return 2

    def nqoi(self) -> int:
        return 1

    def __call__(self, samples: Array) -> Array:
        return samples**2

    def jacobian(self, sample: Array) -> Array:
        return 2 * sample


class ExampleFunctionWithJacobianAndHVP(Generic[Array]):
    def __init__(self, backend: Backend[Array]):
        self._bkd = backend

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return 2

    def nqoi(self) -> int:
        return 1

    def __call__(self, samples: Array) -> Array:
        return samples**2

    def jacobian(self, sample: Array) -> Array:
        return 2 * sample

    def hvp(self, sample: Array, vec: Array) -> Array:
        return 2 * vec


class TestFunctionWrappers(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        """
        Override this method in derived classes to provide the specific
        backend.
        """
        raise NotImplementedError(
            "Derived classes must implement this method."
        )

    def test_numpy_function_wrapper(self) -> None:
        """
        Test the NumpyFunctionWrapper class using the factory.
        """
        bkd = self.bkd()

        # Create an example function
        function = ExampleFunction(bkd)

        # Wrap the function using the factory
        wrapper = numpy_function_wrapper_factory(function)

        # Define samples
        samples = np.array([1.0, 2.0])

        # Evaluate the function
        result = wrapper(samples)

        # Assert that the result matches the expected values
        expected_result = np.array([1.0, 4.0])
        np.testing.assert_allclose(result, expected_result)

    def test_numpy_function_with_jacobian_wrapper(self) -> None:
        """
        Test the NumpyFunctionWithJacobianWrapper class using the factory.
        """
        bkd = self.bkd()

        # Create an example function with Jacobian
        function = ExampleFunctionWithJacobian(bkd)

        # Wrap the function using the factory
        wrapper = numpy_function_wrapper_factory(function)

        # Define samples
        samples = np.array([1.0, 2.0])

        # Evaluate the function
        result = wrapper(samples)

        # Assert that the result matches the expected values
        expected_result = np.array([1.0, 4.0])
        np.testing.assert_allclose(result, expected_result)

        # Compute the Jacobian
        jacobian_result = wrapper.jacobian(samples)

        # Assert that the Jacobian matches the expected values
        expected_jacobian = np.array([2.0, 4.0])
        np.testing.assert_allclose(jacobian_result, expected_jacobian)

    def test_numpy_function_with_jacobian_and_hvp_wrapper(self) -> None:
        """
        Test the NumpyFunctionWithJacobianAndHVPWrapper class using the factory.
        """
        bkd = self.bkd()

        # Create an example function with Jacobian and HVP
        function = ExampleFunctionWithJacobianAndHVP(bkd)

        # Wrap the function using the factory
        wrapper = numpy_function_wrapper_factory(function)

        # Define samples
        samples = np.array([1.0, 2.0])

        # Evaluate the function
        result = wrapper(samples)

        # Assert that the result matches the expected values
        expected_result = np.array([1.0, 4.0])
        np.testing.assert_allclose(result, expected_result)

        # Compute the Jacobian
        jacobian_result = wrapper.jacobian(samples)

        # Assert that the Jacobian matches the expected values
        expected_jacobian = np.array([2.0, 4.0])
        np.testing.assert_allclose(jacobian_result, expected_jacobian)

        # Compute the Hessian-vector product
        vec = np.array([0.5, 0.5])
        hvp_result = wrapper.hvp(samples, vec)

        # Assert that the HVP matches the expected values
        expected_hvp = np.array([1.0, 1.0])
        np.testing.assert_allclose(hvp_result, expected_hvp)


class TestFunctionWrappersNumpy(TestFunctionWrappers[NDArray[Any]]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestFunctionWrappersTorch(TestFunctionWrappers[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


from pyapprox.typing.util.test_utils import load_tests


if __name__ == "__main__":
    unittest.main()
