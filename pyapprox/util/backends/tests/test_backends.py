from typing import Union

from pyapprox.util.backends.protocols import Array, Backend


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
class TestBackend:

    def test_2d_array(self, bkd) -> None:
        """
        Test the foo function with a 2D array.
        """
        array_2d = bkd.array([[1, 2], [3, 4]], dtype=bkd.double_dtype())
        result = foo(array_2d, bkd)
        expected = (
            array_2d  # Identity matrix dot product should return the original array
        )
        bkd.assert_allclose(result, expected)

    def test_1d_array(self, bkd) -> None:
        """
        Test the foo function with a 1D array.
        """
        array_1d = bkd.array([1, 2], dtype=bkd.double_dtype())
        result = foo(array_1d, bkd)
        expected = (
            array_1d  # Identity matrix dot product should return the original array
        )
        bkd.assert_allclose(result, expected)


# TODO:
# complete tests of all functions in backend protocol:
# use __test__ = False pattern typing/interface/functions/fromcallable/tests/ and
# load_tests to avoid running base class.
