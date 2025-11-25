from typing import Any, Optional, Union, Sequence, List, Tuple
from numpy.typing import NDArray
import numpy as np

from pyapprox.typing.util.backend import Backend


# Implement the NumPy backend
class NumpyBkd(Backend[NDArray[Any]]):  # Specify NDArray type
    @staticmethod
    def dot(Amat: NDArray[Any], Bmat: NDArray[Any]) -> NDArray[Any]:
        # for typing consistancy make sure dot always returns an array
        return np.asarray(np.dot(Amat, Bmat))

    @staticmethod
    def eye(
        nrows: int, ncols: Optional[int] = None, dtype: Optional[Any] = None
    ) -> NDArray[Any]:
        return np.eye(nrows, ncols, dtype=dtype)

    @staticmethod
    def inv(matrix: NDArray[Any]) -> NDArray[Any]:
        return np.linalg.inv(matrix)

    @staticmethod
    def asarray(
        array: Union[Sequence[Any], NDArray[Any], float, int],
        dtype: Optional[Any] = None,
    ) -> NDArray[Any]:
        return np.asarray(array, dtype=dtype)

    @staticmethod
    def array(
        array: Union[Sequence[Any], NDArray[Any], float, int],
        dtype: Optional[Any] = None,
    ) -> NDArray[Any]:
        return np.array(array, dtype=dtype)

    @staticmethod
    def assert_allclose(
        actual: np.ndarray,
        desired: np.ndarray,
        rtol: float = 1e-7,
        atol: float = 0,
        equal_nan: bool = True,
        err_msg: Optional[str] = None,
    ) -> None:
        np.testing.assert_allclose(
            actual,
            desired,
            rtol,
            atol,
            equal_nan,
            err_msg="" if err_msg is None else err_msg,
            verbose=True,
            strict=True,
        )

    @staticmethod
    def double_dtype() -> Any:
        return np.float64

    @staticmethod
    def complex_dtype() -> Any:
        return np.complex128

    @staticmethod
    def int64_dtype() -> Any:
        return np.int64

    @staticmethod
    def stack(
        arrays: Union[List[NDArray[Any]], Tuple[NDArray[Any], ...]],
        axis: int = 0,
    ) -> NDArray[Any]:
        return np.stack(arrays, axis=axis)

    @staticmethod
    def linspace(start: float, stop: float, num: int) -> NDArray[Any]:
        return np.linspace(start, stop, num)

    @staticmethod
    def logspace(start: float, stop: float, num: int) -> NDArray[Any]:
        return np.logspace(start, stop, num)

    @staticmethod
    def to_numpy(mat: NDArray[Any]) -> NDArray[Any]:
        return mat

    @staticmethod
    def flatten(array: NDArray[Any]) -> NDArray[Any]:
        return array.flatten()

    @staticmethod
    def meshgrid(
        *arrays: NDArray[Any], indexing: str = "xy"
    ) -> Tuple[NDArray[Any], ...]:
        return np.meshgrid(*arrays, indexing=indexing)

    @staticmethod
    def reshape(array: NDArray[Any], newshape: Sequence[int]) -> NDArray[Any]:
        return np.reshape(array, newshape)

    @staticmethod
    def sum(
        array: NDArray[Any], axis: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> NDArray[Any]:
        return np.sum(array, axis=axis)

    @staticmethod
    def sin(array: NDArray[Any]) -> NDArray[Any]:
        return np.sin(array)

    @staticmethod
    def cos(array: NDArray[Any]) -> NDArray[Any]:
        return np.cos(array)

    @staticmethod
    def full(
        shape: Tuple[int, ...], fill_value: float, dtype: Optional[Any] = None
    ) -> NDArray[Any]:
        return np.full(shape, fill_value, dtype=dtype)

    @staticmethod
    def zeros(
        shape: Tuple[int, ...], dtype: Optional[Any] = None
    ) -> NDArray[Any]:
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def ones(
        shape: Tuple[int, ...], dtype: Optional[Any] = None
    ) -> NDArray[Any]:
        return np.ones(shape, dtype=dtype)

    @staticmethod
    def arange(
        start: Union[int, float],
        stop: Union[int, float],
        step: Union[int, float] = 1,
        dtype: Optional[Any] = None,
    ) -> NDArray[Any]:
        return np.arange(start, stop, step, dtype=dtype)
