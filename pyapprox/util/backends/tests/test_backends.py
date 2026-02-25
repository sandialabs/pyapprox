import unittest
from typing import Any, Generic, Union

import torch
from numpy.typing import NDArray

from pyapprox.util.backends.numpy import NumpyBkd

# Define the Backend and Array types
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd


# Example function using the backend
def foo(x: Array, backend: Backend[Array]) -> Union[Array]:
    """
    Example function that computes the dot product of an identity matrix
    and the input array using the specified backend.

    Args:
        x (Array): Input array.
        backend (Backend[Array]): Backend for array operations.

    Returns:
        Union[Array, float]: Result of the dot product.
    """
    identity = backend.eye(
        x.shape[0],
        x.shape[1] if x.ndim > 1 else None,
        dtype=x.dtype,
    )
    return identity @ x


# Base test class
class TestBackend(Generic[Array]):
    __test__ = False

    def get_backend(self) -> Backend[Array]:
        """
        Override this method in derived classes to provide the specific backend.
        """
        raise NotImplementedError("Derived classes must implement this method.")

    def test_2d_array(self) -> None:
        """
        Test the foo function with a 2D array.
        """
        backend = self.get_backend()
        array_2d = backend.array([[1, 2], [3, 4]], dtype=backend.double_dtype())
        result = foo(array_2d, backend)
        expected = (
            array_2d  # Identity matrix dot product should return the original array
        )
        backend.assert_allclose(result, expected)

    def test_1d_array(self) -> None:
        """
        Test the foo function with a 1D array.
        """
        backend = self.get_backend()
        array_1d = backend.array([1, 2], dtype=backend.double_dtype())
        result = foo(array_1d, backend)
        expected = (
            array_1d  # Identity matrix dot product should return the original array
        )
        backend.assert_allclose(result, expected)


# Derived test class for NumPy backend
class TestNumpyBackend(TestBackend[NDArray[Any]], unittest.TestCase):
    def get_backend(self) -> NumpyBkd:
        return NumpyBkd()


# Derived test class for PyTorch
class TestTorchBackend(TestBackend[torch.Tensor], unittest.TestCase):
    # Base test class TestBackend must be typed on Generic[Array]
    # and the derived class must return Backend[torch.Tensor]
    def get_backend(self) -> Backend[torch.Tensor]:  # -> TorchBkd:
        return TorchBkd()


# Run the tests
if __name__ == "__main__":
    unittest.main()


# TODO:
# complete tests of all functions in backend protocol:
# use __test__ = False pattern typing/interface/functions/fromcallable/tests/ and
# load_tests to avoid running base class.
