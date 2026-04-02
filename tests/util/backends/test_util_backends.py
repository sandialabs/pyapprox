from typing import Union

import pytest

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


class TestTorchBkdDtype:
    """Test that TorchBkd respects device and dtype configuration."""

    @pytest.mark.parametrize(
        "dtype_name",
        ["float32", "float64"],
    )
    def test_tensor_creation_dtype(self, dtype_name):
        torch = pytest.importorskip("torch")
        from pyapprox.util.backends.torch import TorchBkd

        dtype = getattr(torch, dtype_name)
        bkd = TorchBkd(dtype=dtype)

        assert bkd.array([[1.0, 2.0]]).dtype == dtype
        assert bkd.zeros((2, 3)).dtype == dtype
        assert bkd.ones((2, 3)).dtype == dtype
        assert bkd.eye(3).dtype == dtype
        assert bkd.full((2,), 5.0).dtype == dtype
        assert bkd.empty((2,)).dtype == dtype
        assert bkd.linspace(0.0, 1.0, 5).dtype == dtype
        assert bkd.logspace(0.0, 1.0, 5).dtype == dtype

    def test_default_dtype_matches_init(self):
        torch = pytest.importorskip("torch")
        from pyapprox.util.backends.torch import TorchBkd

        bkd32 = TorchBkd(dtype=torch.float32)
        assert bkd32.default_dtype() == torch.float32

        bkd64 = TorchBkd(dtype=torch.float64)
        assert bkd64.default_dtype() == torch.float64

    def test_default_constructor_uses_torch_default(self):
        torch = pytest.importorskip("torch")
        from pyapprox.util.backends.torch import TorchBkd

        torch.set_default_dtype(torch.float64)
        bkd = TorchBkd()
        assert bkd.default_dtype() == torch.float64
        assert bkd.zeros((2,)).dtype == torch.float64

    def test_explicit_dtype_overrides_default(self):
        torch = pytest.importorskip("torch")
        from pyapprox.util.backends.torch import TorchBkd

        bkd = TorchBkd(dtype=torch.float32)
        # Explicit dtype arg to method should override backend default
        result = bkd.zeros((2,), dtype=torch.float64)
        assert result.dtype == torch.float64

    def test_arange_device(self):
        torch = pytest.importorskip("torch")
        from pyapprox.util.backends.torch import TorchBkd

        bkd = TorchBkd()
        result = bkd.arange(5)
        assert result.device == torch.device("cpu")

    def test_asarray_device(self):
        torch = pytest.importorskip("torch")
        from pyapprox.util.backends.torch import TorchBkd

        bkd = TorchBkd()
        result = bkd.asarray([1.0, 2.0])
        assert result.device == torch.device("cpu")

    def test_tril_indices_device(self):
        torch = pytest.importorskip("torch")
        from pyapprox.util.backends.torch import TorchBkd

        bkd = TorchBkd()
        row_idx, col_idx = bkd.tril_indices(3)
        assert row_idx.device == torch.device("cpu")
        assert col_idx.device == torch.device("cpu")


class TestTorchBkdMPS:
    """Test TorchBkd with MPS device (skipped if unavailable)."""

    def test_mps_tensor_creation(self, torch_mps_bkd):
        result = torch_mps_bkd.zeros((2, 3))
        assert result.device.type == "mps"
        assert result.dtype.is_floating_point

    def test_mps_to_numpy(self, torch_mps_bkd):
        import numpy as np

        tensor = torch_mps_bkd.ones((3,))
        arr = torch_mps_bkd.to_numpy(tensor)
        assert isinstance(arr, np.ndarray)
        np.testing.assert_allclose(arr, np.ones(3), rtol=1e-6)
