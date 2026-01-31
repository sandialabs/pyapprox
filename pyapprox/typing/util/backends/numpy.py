from typing import Any, Optional, Union, Sequence, List, Tuple, overload, cast

from numpy.typing import NDArray
import numpy as np
import scipy

from pyapprox.typing.util.backends.protocols import Backend


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
    def concatenate(
        arrays: Union[List[NDArray[Any]], Tuple[NDArray[Any], ...]],
        axis: int = 0,
    ) -> NDArray[Any]:
        return np.concatenate(arrays, axis=axis)

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
    def ravel(array: NDArray[Any]) -> NDArray[Any]:
        return np.ravel(array)

    @staticmethod
    def meshgrid(
        *arrays: NDArray[Any], indexing: str = "xy"
    ) -> Tuple[NDArray[Any], ...]:
        return np.meshgrid(*arrays, indexing=indexing)  # type: ignore

    @staticmethod
    def reshape(array: NDArray[Any], newshape: Sequence[int]) -> NDArray[Any]:
        return np.reshape(array, newshape)

    @staticmethod
    def transpose(array: NDArray[Any], axes: Optional[Sequence[int]] = None) -> NDArray[Any]:
        return np.transpose(array, axes)

    @staticmethod
    def sum(
        array: NDArray[Any],
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> NDArray[Any]:
        return np.asarray(np.sum(array, axis=axis, keepdims=keepdims))

    @staticmethod
    def cumsum(array: NDArray[Any], axis: Optional[int] = None) -> NDArray[Any]:
        return np.cumsum(array, axis=axis)

    @staticmethod
    def sin(array: NDArray[Any]) -> NDArray[Any]:
        return cast(NDArray[Any], np.sin(array))

    @staticmethod
    def cos(array: NDArray[Any]) -> NDArray[Any]:
        return cast(NDArray[Any], np.cos(array))

    @staticmethod
    def arccos(array: NDArray[Any]) -> NDArray[Any]:
        return cast(NDArray[Any], np.arccos(array))

    @staticmethod
    def arctan(array: NDArray[Any]) -> NDArray[Any]:
        return cast(NDArray[Any], np.arctan(array))

    @staticmethod
    def arctan2(y: NDArray[Any], x: NDArray[Any]) -> NDArray[Any]:
        return cast(NDArray[Any], np.arctan2(y, x))

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
    def empty(
        shape: Tuple[int, ...], dtype: Optional[Any] = None
    ) -> NDArray[Any]:
        return np.empty(shape, dtype=dtype)

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
        return cast(bool, np.any(array, axis=None, keepdims=keepdims))

    @staticmethod
    def any_array(
        array: NDArray[Any],
        axis: int,
        keepdims: bool = False,
    ) -> NDArray[Any]:
        return cast(NDArray[Any], np.any(array, axis=axis, keepdims=keepdims))

    @staticmethod
    def all_bool(
        array: NDArray[Any],
        keepdims: bool = False,
    ) -> bool:
        return cast(bool, np.all(array, axis=None, keepdims=keepdims))

    @staticmethod
    def all_array(
        array: NDArray[Any],
        axis: int,
        keepdims: bool = False,
    ) -> NDArray[Any]:
        return cast(NDArray[Any], np.all(array, axis=axis, keepdims=keepdims))

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
    def nonzero(condition: NDArray[Any]) -> Tuple[NDArray[Any], ...]:
        return cast(Tuple[NDArray[Any], ...], np.nonzero(condition))

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
    def abs(array: NDArray[Any]) -> NDArray[Any]:
        return cast(NDArray[Any], np.abs(array))

    @staticmethod
    def round(array: NDArray[Any]) -> NDArray[Any]:
        return cast(NDArray[Any], np.round(array))

    @staticmethod
    def floor(array: NDArray[Any]) -> NDArray[Any]:
        return cast(NDArray[Any], np.floor(array))

    @staticmethod
    def ceil(array: NDArray[Any]) -> NDArray[Any]:
        return cast(NDArray[Any], np.ceil(array))

    @staticmethod
    def sqrt(array: NDArray[Any]) -> NDArray[Any]:
        return cast(NDArray[Any], np.sqrt(array))

    @staticmethod
    def solve(Amat: NDArray[Any], Bmat: NDArray[Any]) -> NDArray[Any]:
        return np.linalg.solve(Amat, Bmat)

    @staticmethod
    def flip(
        array: NDArray[Any], axis: Optional[Tuple[int]] = None
    ) -> NDArray[Any]:
        return np.flip(array, axis=axis)

    @staticmethod
    def sort(
        array: NDArray[Any], axis: int = -1
    ) -> NDArray[Any]:
        return np.sort(array, axis=axis)

    @staticmethod
    def argsort(
        array: NDArray[Any], axis: int = -1
    ) -> NDArray[Any]:
        return np.argsort(array, axis=axis)

    @staticmethod
    def searchsorted(
        sorted_array: NDArray[Any], values: NDArray[Any], side: str = "left"
    ) -> NDArray[Any]:
        return np.searchsorted(sorted_array, values, side=side)

    @staticmethod
    def clip(
        array: NDArray[Any], a_min: Any, a_max: Any
    ) -> NDArray[Any]:
        return np.clip(array, a_min, a_max)

    @staticmethod
    def min(
        array: NDArray[Any], axis: Optional[int] = None, keepdims: bool = False
    ) -> NDArray[Any]:
        return np.asarray(np.min(array, axis=axis, keepdims=keepdims))

    @staticmethod
    def max(
        array: NDArray[Any], axis: Optional[int] = None, keepdims: bool = False
    ) -> NDArray[Any]:
        return np.asarray(np.max(array, axis=axis, keepdims=keepdims))

    @staticmethod
    def einsum(subscripts: str, *operands: NDArray[Any]) -> NDArray[Any]:
        return np.einsum(subscripts, *operands)

    @staticmethod
    def moveaxis(
        array: NDArray[Any],
        source: Union[int, tuple[int, ...]],
        destination: Union[int, tuple[int, ...]],
    ) -> NDArray[Any]:
        return np.moveaxis(array, source, destination)

    @staticmethod
    def diag(array: NDArray[Any], k: int = 0) -> NDArray[Any]:
        return np.diag(array, k)

    @staticmethod
    def outer(a: NDArray[Any], b: NDArray[Any]) -> NDArray[Any]:
        return np.outer(a, b)

    @staticmethod
    def diff(array: NDArray[Any], n: int = 1, axis: int = -1) -> NDArray[Any]:
        return np.diff(array, n=n, axis=axis)

    @staticmethod
    def allclose(
        array1: NDArray[Any],
        array2: NDArray[Any],
        rtol: float = 1e-05,
        atol: float = 1e-08,
        equal_nan: bool = False,
    ) -> bool:
        return np.allclose(
            array1, array2, rtol=rtol, atol=atol, equal_nan=equal_nan
        )

    @staticmethod
    def cholesky(array: NDArray[Any]) -> NDArray[Any]:
        return np.linalg.cholesky(array)

    @staticmethod
    def solve_triangular(
        matrix: NDArray[Any],
        rhs: NDArray[Any],
        lower: bool = False,
        unit_diagonal: bool = False,
    ) -> NDArray[Any]:
        return scipy.linalg.solve_triangular(
            matrix, rhs, lower=lower, unit_diagonal=unit_diagonal
        )

    @staticmethod
    def cholesky_solve(
        matrix: NDArray[Any],
        rhs: NDArray[Any],
        lower: bool = True,
    ) -> NDArray[Any]:
        return scipy.linalg.cho_solve((matrix, lower), rhs)

    @staticmethod
    def eigvalsh(array: NDArray[Any]) -> NDArray[Any]:
        return np.linalg.eigvalsh(array)

    @staticmethod
    def trace(array: NDArray[Any]) -> NDArray[Any]:
        return np.asarray(np.linalg.trace(array))

    @staticmethod
    def atleast_1d(array: NDArray[Any]) -> NDArray[Any]:
        return np.atleast_1d(array)

    @staticmethod
    def atleast_2d(array: NDArray[Any]) -> NDArray[Any]:
        return np.atleast_2d(array)

    @staticmethod
    def tile(array: NDArray[Any], reps: Tuple[int, ...]) -> NDArray[Any]:
        return np.tile(array, reps=reps)

    @staticmethod
    def isnan(array: NDArray[Any]) -> NDArray[Any]:
        return np.isnan(array)

    @staticmethod
    def isinf(array: NDArray[Any]) -> NDArray[Any]:
        return np.isinf(array)

    @staticmethod
    def get_diagonal(
        array: NDArray[Any], offset: int = 0, axis1: int = 0, axis2: int = 1
    ) -> NDArray[Any]:
        return np.diagonal(array, offset, axis1=axis1, axis2=axis2)

    @staticmethod
    def cdist(
        XA: NDArray[Any], XB: NDArray[Any], p: float = 2.0
    ) -> NDArray[Any]:
        if p == 1.0:
            metric = "manhattan"
        elif p == 2.0:
            metric = "euclidean"
        else:
            raise ValueError("p must be either 1.0 or 2.0")
        return scipy.spatial.distance.cdist(XA, XB, metric)

    @staticmethod
    def tril(array: NDArray[Any], k: int = 0) -> NDArray[Any]:
        return np.tril(array, k)

    @staticmethod
    def triu(array: NDArray[Any], k: int = 0) -> NDArray[Any]:
        return np.triu(array, k)

    @staticmethod
    def kron(a: NDArray[Any], b: NDArray[Any]) -> NDArray[Any]:
        return np.kron(a, b)

    @staticmethod
    def repeat(
        array: NDArray[Any], repeats: int, axis: Optional[int] = None
    ) -> NDArray[Any]:
        return np.repeat(array, repeats, axis=axis)

    @staticmethod
    def eigh(array: NDArray[Any]) -> Tuple[NDArray[Any], NDArray[Any]]:
        return np.linalg.eigh(array)

    @staticmethod
    def gammaln(array: NDArray[Any]) -> NDArray[Any]:
        return scipy.special.gammaln(array)

    @staticmethod
    def digamma(array: NDArray[Any]) -> NDArray[Any]:
        return scipy.special.digamma(array)

    @staticmethod
    def argmin(
        array: NDArray[Any], axis: Optional[int] = None
    ) -> NDArray[Any]:
        return np.asarray(np.argmin(array, axis=axis))

    @staticmethod
    def argmax(
        array: NDArray[Any], axis: Optional[int] = None
    ) -> NDArray[Any]:
        return np.asarray(np.argmax(array, axis=axis))

    @staticmethod
    def int64_dtype() -> Any:
        return np.int64

    @staticmethod
    def where(
        condition: NDArray[Any],
        x: Union[NDArray[Any], float, None] = None,
        y: Union[NDArray[Any], float, None] = None,
    ) -> NDArray[Any]:
        if x is None and y is None:
            return np.where(condition)
        return np.where(condition, x, y)

    @staticmethod
    def lstsq(
        a: NDArray[Any],
        b: NDArray[Any],
        rcond: Optional[float] = None,
    ) -> NDArray[Any]:
        result, _, _, _ = np.linalg.lstsq(a, b, rcond=rcond)
        return result

    @staticmethod
    def ones_like(
        array: NDArray[Any], dtype: Optional[Any] = None
    ) -> NDArray[Any]:
        return np.ones_like(array, dtype=dtype)

    @staticmethod
    def zeros_like(
        array: NDArray[Any], dtype: Optional[Any] = None
    ) -> NDArray[Any]:
        return np.zeros_like(array, dtype=dtype)

    @staticmethod
    def default_dtype() -> Any:
        return np.float64

    @staticmethod
    def slogdet(array: NDArray[Any]) -> Tuple[NDArray[Any], NDArray[Any]]:
        return np.linalg.slogdet(array)

    @staticmethod
    def mean(
        array: NDArray[Any],
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> NDArray[Any]:
        return np.asarray(np.mean(array, axis=axis, keepdims=keepdims))

    @staticmethod
    def var(
        array: NDArray[Any],
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
        ddof: int = 0,
    ) -> NDArray[Any]:
        return np.asarray(np.var(array, axis=axis, keepdims=keepdims, ddof=ddof))

    @staticmethod
    def std(
        array: NDArray[Any],
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
        ddof: int = 0,
    ) -> NDArray[Any]:
        return np.asarray(np.std(array, axis=axis, keepdims=keepdims, ddof=ddof))

    @staticmethod
    def cov(
        array: NDArray[Any],
        rowvar: bool = True,
        ddof: int = 1,
    ) -> NDArray[Any]:
        result = np.cov(array, rowvar=rowvar, ddof=ddof)
        # Ensure result is always 2D (np.cov returns scalar for 1 variable)
        return np.atleast_2d(np.asarray(result))

    @staticmethod
    def maximum(x1: NDArray[Any], x2: NDArray[Any]) -> NDArray[Any]:
        return np.maximum(x1, x2)

    @staticmethod
    def minimum(x1: NDArray[Any], x2: NDArray[Any]) -> NDArray[Any]:
        return np.minimum(x1, x2)

    @staticmethod
    def qr(
        array: NDArray[Any], mode: str = "reduced"
    ) -> Tuple[NDArray[Any], NDArray[Any]]:
        return np.linalg.qr(array, mode=mode)

    @staticmethod
    def split(
        array: NDArray[Any],
        indices_or_sections: NDArray[Any],
        axis: int = 0,
    ) -> List[NDArray[Any]]:
        return np.split(array, indices_or_sections, axis=axis)

    @staticmethod
    def lu(
        array: NDArray[Any],
    ) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
        """Compute LU factorization with partial pivoting.

        Returns P, L, U such that A = P @ L @ U.

        Parameters
        ----------
        array : NDArray
            Matrix to factorize.

        Returns
        -------
        P : NDArray
            Permutation matrix.
        L : NDArray
            Lower triangular matrix with unit diagonal.
        U : NDArray
            Upper triangular matrix.
        """
        return scipy.linalg.lu(array)

    @staticmethod
    def index_update(
        array: NDArray[Any],
        index: Union[int, Tuple[int, ...]],
        value: Any,
    ) -> NDArray[Any]:
        """Return a copy of array with array[index] = value.

        This provides a functional interface for array updates that
        is compatible with autograd frameworks.

        Parameters
        ----------
        array : NDArray
            Array to update.
        index : int or tuple of int
            Index to update.
        value : Any
            Value to set at index.

        Returns
        -------
        NDArray
            Copy of array with updated value.
        """
        result = array.copy()
        result[index] = value
        return result
