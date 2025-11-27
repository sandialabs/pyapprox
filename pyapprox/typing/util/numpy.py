from typing import Any, Optional, Union, Sequence, List, Tuple, overload, cast

from numpy.typing import NDArray
import numpy as np
import scipy

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
    def hstack(
        arrays: Union[List[NDArray[Any]], Tuple[NDArray[Any], ...]],
    ) -> NDArray[Any]:
        return np.hstack(arrays)

    @staticmethod
    def vstack(
        arrays: Union[List[NDArray[Any]], Tuple[NDArray[Any], ...]],
    ) -> NDArray[Any]:
        return np.vstack(arrays)

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
        arrays: Tuple[NDArray[Any], ...], indexing: str = "xy"
    ) -> Tuple[NDArray[Any], ...]:
        return np.meshgrid(*arrays, indexing=indexing)

    @staticmethod
    def reshape(array: NDArray[Any], newshape: Sequence[int]) -> NDArray[Any]:
        return np.reshape(array, newshape)

    @staticmethod
    def sum(
        array: NDArray[Any],
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> NDArray[Any]:
        return np.asarray(np.sum(array, axis=axis, keepdims=keepdims))

    @staticmethod
    def sin(array: NDArray[Any]) -> NDArray[Any]:
        return cast(NDArray[Any], np.sin(array))

    @staticmethod
    def cos(array: NDArray[Any]) -> NDArray[Any]:
        return cast(NDArray[Any], np.cos(array))

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
    def arange(*args: Any, **kwargs: Any) -> NDArray[Any]:
        return np.arange(*args, **kwargs)

    @staticmethod
    def prod(
        array: NDArray[Any],
        axis: Optional[int] = None,
        keepdims: bool = False,
    ) -> NDArray[Any]:
        return np.asarray(np.prod(array, axis=axis, keepdims=keepdims))

    @staticmethod
    def any_bool(
        array: NDArray[Any],
        keepdims: bool = False,
    ) -> bool:
        return np.any(array, axis=None, keepdims=keepdims)

    @staticmethod
    def any_array(
        array: NDArray[Any],
        axis: int,
        keepdims: bool = False,
    ) -> NDArray[Any]:
        return np.any(array, axis=axis, keepdims=keepdims)

    @staticmethod
    def log(array: NDArray[Any]) -> NDArray[Any]:
        return cast(NDArray[Any], np.log(array))

    @staticmethod
    def exp(array: NDArray[Any]) -> NDArray[Any]:
        return np.asarray(np.exp(array))

    @staticmethod
    def copy(array: NDArray[Any]) -> NDArray[Any]:
        return array.copy()

    @staticmethod
    def erf(array: NDArray[Any]) -> NDArray[Any]:
        return scipy.special.erf(array)

    @staticmethod
    def erfinv(array: NDArray[Any]) -> NDArray[Any]:
        return scipy.special.erfinv(array)

    @staticmethod
    def isfinite(array: NDArray[Any]) -> NDArray[Any]:
        return cast(NDArray[Any], np.isfinite(array))

    @staticmethod
    def nonzero(condition: NDArray[Any]) -> NDArray[Any]:
        return cast(NDArray[Any], np.nonzero(condition))

    @staticmethod
    def norm(
        array: NDArray[Any],
        ord: Optional[Union[int, float, str]] = None,
        axis: Optional[Union[int, Tuple[int, int]]] = None,
        keepdims: bool = False,
    ) -> NDArray[Any]:
        return np.asarray(
            np.linalg.norm(array, ord=ord, axis=axis, keepdims=keepdims)
        )

    @staticmethod
    def sign(array: NDArray[Any]) -> NDArray[Any]:
        return cast(NDArray[Any], np.sign(array))

    @staticmethod
    def sqrt(array: NDArray[Any]) -> NDArray[Any]:
        return np.sqrt(array)

    @staticmethod
    def solve(Amat: NDArray[Any], Bmat: NDArray[Any]) -> NDArray[Any]:
        return np.linalg.solve(Amat, Bmat)

    @staticmethod
    def flip(
        array: NDArray[Any], axis: Optional[Tuple[int]] = None
    ) -> NDArray[Any]:
        return np.flip(array, axis=axis)

    @staticmethod
    def min(
        array: NDArray[Any], axis: Optional[int] = None, keepdims: bool = False
    ) -> NDArray[Any]:
        return np.min(array, axis=axis, keepdims=keepdims)

    @staticmethod
    def max(
        array: NDArray[Any], axis: Optional[int] = None, keepdims: bool = False
    ) -> NDArray[Any]:
        return np.max(array, axis=axis, keepdims=keepdims)
