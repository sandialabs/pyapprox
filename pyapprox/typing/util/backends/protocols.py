from typing import (
    Protocol,
    runtime_checkable,
    Union,
    Any,
    TypeVar,
    Generic,
    Optional,
    Sequence,
    Tuple,
    List,
    overload,
)

# from typing_extensions import SupportsIndex
from numpy.typing import NDArray


# Define generic types for arrays
Array = TypeVar("Array", bound="ArrayProtocol")


@runtime_checkable
class ArrayProtocol(Protocol):
    @property
    def shape(self) -> Any: ...

    @property
    def ndim(self) -> Any: ...

    @property
    def dtype(self) -> Any: ...

    @property
    def T(self: Array) -> Array: ...

    def __truediv__(self: Array, other: Union[float, Array]) -> Array: ...

    def __rtruediv__(self: Array, other: Union[float, Array]) -> Array: ...

    def __mul__(self: Array, other: Union[float, Array]) -> Array: ...

    def __rmul__(self: Array, other: Union[float, Array]) -> Array: ...

    def __add__(self: Array, other: Union[float, Array]) -> Array: ...

    def __radd__(self: Array, other: Union[float, Array]) -> Array: ...

    def __sub__(self: Array, other: Union[float, Array]) -> Array: ...

    def __rsub__(self: Array, other: Union[float, Array]) -> Array: ...

    def __pow__(self: Array, other: Union[float, Array]) -> Array: ...

    def __rpow__(self: Array, other: Union[float, Array]) -> Array: ...

    def __gt__(
        self: Array, other: Union[float, Array]
    ) -> Union[bool, Array]: ...

    def __lt__(
        self: Array, other: Union[float, Array]
    ) -> Union[bool, Array]: ...

    def __ge__(
        self: Array, other: Union[float, Array]
    ) -> Union[bool, Array]: ...

    def __le__(
        self: Array, other: Union[float, Array]
    ) -> Union[bool, Array]: ...

    def __matmul__(self: Array, other: Array) -> Array: ...

    def __eq__(self: Array, other: object) -> Union[bool, Array]: ...  # type: ignore

    def __ne__(self: Array, other: object) -> Union[bool, Array]: ...  # type: ignore

    def __setitem__(self: Array, index: Any, value: Any) -> None: ...

    def __neg__(self: Array) -> Array: ...

    def __getitem__(self: Array, index: Any) -> Any: ...

    def __iter__(self: Array) -> Any: ...

    def __len__(self) -> int: ...

    def item(self) -> Any: ...

    def __invert__(self) -> "ArrayProtocol": ...


# Define the Backend protocol without dtype
@runtime_checkable
class Backend(Protocol, Generic[Array]):
    @staticmethod
    def dot(Amat: Array, Bmat: Array) -> Array: ...

    @staticmethod
    def eye(
        nrows: int, ncols: Optional[int] = None, dtype: Optional[Any] = None
    ) -> Array: ...

    @staticmethod
    def inv(matrix: Array) -> Array: ...

    @staticmethod
    def asarray(
        array: Union[Sequence[Any], Array, float, int],
        dtype: Optional[Any] = None,
    ) -> Array: ...

    @staticmethod
    def array(
        array: Union[Sequence[Any], Array, float, int],
        dtype: Optional[Any] = None,
    ) -> Array: ...

    @staticmethod
    def assert_allclose(
        actual: Array,
        desired: Array,
        rtol: float = 1e-7,
        atol: float = 0,
        equal_nan: bool = True,
        err_msg: Optional[str] = None,
    ) -> None: ...

    @staticmethod
    def double_dtype() -> Any: ...

    @staticmethod
    def complex_dtype() -> Any: ...

    @staticmethod
    def int64_dtype() -> Any: ...

    @staticmethod
    def stack(
        arrays: Union[List[Array], Tuple[Array, ...]], axis: int = 0
    ) -> Array: ...

    @staticmethod
    def concatenate(
        arrays: Union[List[Array], Tuple[Array, ...]], axis: int = 0
    ) -> Array: ...

    @staticmethod
    def hstack(
        arrays: Union[List[Array], Tuple[Array, ...]],
    ) -> Array: ...

    @staticmethod
    def vstack(
        arrays: Union[List[Array], Tuple[Array, ...]],
    ) -> Array: ...

    @staticmethod
    def linspace(start: float, stop: float, num: int) -> Array: ...

    @staticmethod
    def logspace(start: float, stop: float, num: int) -> Array: ...

    @staticmethod
    def to_numpy(array: Array) -> NDArray[Any]: ...

    @staticmethod
    def flatten(array: Array) -> Array: ...

    @staticmethod
    def meshgrid(
        arrays: Tuple[Array, ...], indexing: str = "xy"
    ) -> Tuple[Array, ...]: ...

    @staticmethod
    def reshape(array: Array, newshape: Sequence[int]) -> Array: ...

    @staticmethod
    def transpose(array: Array, axes: Optional[Sequence[int]] = None) -> Array: ...

    @staticmethod
    def sum(
        array: Array,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> Array: ...

    @staticmethod
    def sin(array: Array) -> Array: ...

    @staticmethod
    def cos(array: Array) -> Array: ...

    @staticmethod
    def arccos(array: Array) -> Array: ...

    @staticmethod
    def arctan(array: Array) -> Array: ...

    @staticmethod
    def arctan2(y: Array, x: Array) -> Array: ...

    @staticmethod
    def full(
        shape: Tuple[int, ...], fill_value: float, dtype: Optional[Any] = None
    ) -> Array: ...

    @staticmethod
    def zeros(
        shape: Tuple[int, ...], dtype: Optional[Any] = None
    ) -> Array: ...

    @staticmethod
    def ones(shape: Tuple[int, ...], dtype: Optional[Any] = None) -> Array: ...

    # @staticmethod
    # def arange(
    #     start: Union[int, float],
    #     stop: Union[int, float],
    #     step: Union[int, float] = 1,
    #     dtype: Optional[Any] = None,
    # ) -> Array: ...

    # Overload 1: arange(stop)
    @staticmethod
    @overload
    def arange(stop: Union[int, float], /) -> Array:
        """
        Overload for when only `stop` is provided.
        """
        ...

    # Overload 2: arange(start, stop)
    @staticmethod
    @overload
    def arange(start: Union[int, float], stop: Union[int, float], /) -> Array:
        """
        Overload for when `start` and `stop` are provided.
        """
        ...

    # Overload 3: arange(start, stop, step)
    @staticmethod
    @overload
    def arange(
        start: Union[int, float],
        stop: Union[int, float],
        step: Union[int, float],
        /,
    ) -> Array:
        """
        Overload for when `start`, `stop`, and `step` are provided.
        """
        ...

    # Overload 4: Allow keyword arguments including dtype
    @staticmethod
    @overload
    def arange(
        *args: Union[int, float],
        dtype: Optional[Any] = None,
    ) -> Array:
        """
        Overload for when keyword arguments like `dtype` are provided.
        """
        ...

    # Runtime implementation
    @staticmethod
    def arange(*args: Any, **kwargs: Any) -> Array:
        """
        Runtime implementation of arange.

        Handles all variations of arguments using *args and **kwargs.
        """
        ...

    @staticmethod
    def prod(
        array: Array,
        axis: Optional[int] = None,
        keepdims: bool = False,
    ) -> Array: ...

    @staticmethod
    def any_bool(
        array: Array,
        keepdims: bool = False,
    ) -> bool:
        """
        Overload for any when `axis` is None. Returns a scalar boolean.
        """
        ...

    @staticmethod
    def any_array(
        array: Array,
        axis: int,
        keepdims: bool = False,
    ) -> Array:
        """
        Overload for any when `axis` is specified. Returns an array.
        """
        ...

    @staticmethod
    def all_bool(
        array: Array,
        keepdims: bool = False,
    ) -> bool:
        """
        Overload for all when `axis` is None. Returns a scalar boolean.
        """
        ...

    @staticmethod
    def all_array(
        array: Array,
        axis: int,
        keepdims: bool = False,
    ) -> Array:
        """
        Overload for all when `axis` is specified. Returns an array.
        """
        ...

    @staticmethod
    def log(array: Array) -> Array: ...

    @staticmethod
    def exp(array: Array) -> Array: ...

    @staticmethod
    def copy(array: Array) -> Array: ...

    @staticmethod
    def erf(array: Array) -> Array: ...

    @staticmethod
    def erfinv(array: Array) -> Array: ...

    @staticmethod
    def isfinite(array: Array) -> Array: ...

    @staticmethod
    def nonzero(condition: Array) -> Tuple[Array, ...]: ...

    @staticmethod
    def norm(
        array: Array,
        ord: Optional[Union[int, float, str]] = None,
        axis: Optional[Union[int, Tuple[int, int]]] = None,
        keepdims: bool = False,
    ) -> Array: ...

    @staticmethod
    def sign(array: Array) -> Array: ...

    @staticmethod
    def abs(array: Array) -> Array: ...

    @staticmethod
    def round(array: Array) -> Array: ...

    @staticmethod
    def sqrt(array: Array) -> Array: ...

    @staticmethod
    def solve(Amat: Array, Bmat: Array) -> Array: ...

    @staticmethod
    def flip(array: Array, axis: Optional[Tuple[int]] = None) -> Array: ...

    @staticmethod
    def sort(array: Array, axis: int = -1) -> Array: ...

    @staticmethod
    def min(
        array: Array, axis: Optional[int] = None, keepdims: bool = False
    ) -> Array: ...

    @staticmethod
    def max(
        array: Array, axis: Optional[int] = None, keepdims: bool = False
    ) -> Array: ...

    @staticmethod
    def einsum(subscripts: str, *operands: Array) -> Array: ...

    @staticmethod
    def moveaxis(
        array: Array,
        source: Union[int, tuple[int, ...]],
        destination: Union[int, tuple[int, ...]],
    ) -> Array: ...

    @staticmethod
    def diag(array: Array, k: int = 0) -> Array: ...

    @staticmethod
    def diff(array: Array, n: int = 1, axis: int = -1) -> Array: ...

    @staticmethod
    def allclose(
        array1: Array,
        array2: Array,
        rtol: float = 1e-05,
        atol: float = 1e-08,
        equal_nan: bool = False,
    ) -> bool: ...

    @staticmethod
    def cholesky(array: Array) -> Array: ...

    @staticmethod
    def solve_triangular(
        matrix: Array,
        rhs: Array,
        lower: bool = True,
        unit_diagonal: bool = False,
    ) -> Array: ...

    @staticmethod
    def cholesky_solve(
        matrix: Array,
        rhs: Array,
        lower: bool = True,
    ) -> Array: ...

    @staticmethod
    def trace(array: Array) -> Array: ...

    @staticmethod
    def atleast_1d(array: Array) -> Array: ...

    @staticmethod
    def atleast_2d(array: Array) -> Array: ...

    @staticmethod
    def tile(array: Array, reps: Tuple[int, ...]) -> Array: ...

    @staticmethod
    def kron(a: Array, b: Array) -> Array:
        """Compute Kronecker product of two matrices."""
        ...

    @staticmethod
    def repeat(array: Array, repeats: int, axis: Optional[int] = None) -> Array:
        """Repeat elements of an array."""
        ...

    @staticmethod
    def isnan(array: Array) -> Array: ...

    @staticmethod
    def get_diagonal(
        array: Array, offset: int = 0, axis1: int = 0, axis2: int = 1
    ) -> Array: ...

    @staticmethod
    def cdist(XA: Array, XB: Array, p: float = 2.0) -> Array: ...

    @staticmethod
    def tril(array: Array, k: int = 0) -> Array: ...

    @staticmethod
    def triu(array: Array, k: int = 0) -> Array: ...

    @staticmethod
    def eigh(array: Array) -> Tuple[Array, Array]:
        """Compute eigenvalues and eigenvectors of symmetric matrix."""
        ...

    @staticmethod
    def gammaln(array: Array) -> Array:
        """Compute log of gamma function."""
        ...

    @staticmethod
    def digamma(array: Array) -> Array:
        """Compute digamma function (psi function).

        The digamma function is the logarithmic derivative of the gamma function:
        psi(x) = d/dx ln(Gamma(x)) = Gamma'(x) / Gamma(x)
        """
        ...

    @staticmethod
    def argmin(array: Array, axis: Optional[int] = None) -> Array:
        """Return indices of minimum values along an axis."""
        ...

    @staticmethod
    def argmax(array: Array, axis: Optional[int] = None) -> Array:
        """Return indices of maximum values along an axis."""
        ...

    @staticmethod
    def int64_dtype() -> Any:
        """Return the int64 dtype for this backend."""
        ...

    @staticmethod
    def where(
        condition: Array,
        x: Union[Array, float, None] = None,
        y: Union[Array, float, None] = None,
    ) -> Array:
        """Return elements chosen from x or y depending on condition.

        Parameters
        ----------
        condition : Array
            Boolean array of condition.
        x : Array or scalar, optional
            Values where condition is True.
        y : Array or scalar, optional
            Values where condition is False.

        Returns
        -------
        Array
            If x and y provided: array of elements from x where condition is
            True, elements from y otherwise.
            If only condition provided: tuple of arrays of indices where True.
        """
        ...

    @staticmethod
    def lstsq(
        a: Array,
        b: Array,
        rcond: Optional[float] = None,
    ) -> Array:
        """Solve least squares problem min ||ax - b||_2.

        Parameters
        ----------
        a : Array
            Coefficient matrix. Shape: (m, n)
        b : Array
            Ordinate values. Shape: (m,) or (m, k)
        rcond : float, optional
            Cutoff for small singular values.

        Returns
        -------
        Array
            Least squares solution. Shape: (n,) or (n, k)
        """
        ...

    @staticmethod
    def ones_like(array: Array, dtype: Optional[Any] = None) -> Array:
        """Return an array of ones with the same shape as input."""
        ...

    @staticmethod
    def zeros_like(array: Array, dtype: Optional[Any] = None) -> Array:
        """Return an array of zeros with the same shape as input."""
        ...

    @staticmethod
    def default_dtype() -> Any:
        """Return the default floating point dtype for this backend."""
        ...

    @staticmethod
    def slogdet(array: Array) -> Tuple[Array, Array]:
        """Compute sign and log-determinant of a matrix.

        Parameters
        ----------
        array : Array
            Square matrix. Shape: (n, n)

        Returns
        -------
        sign : Array
            Sign of determinant (1, 0, or -1).
        logdet : Array
            Natural log of absolute value of determinant.
        """
        ...

    @staticmethod
    def mean(
        array: Array,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> Array:
        """Compute arithmetic mean along the specified axis.

        Parameters
        ----------
        array : Array
            Input array.
        axis : int or tuple of ints, optional
            Axis or axes along which to compute means.
            If None, compute over flattened array.
        keepdims : bool
            If True, reduced axes are kept with size 1.

        Returns
        -------
        Array
            Mean values.
        """
        ...

    @staticmethod
    def var(
        array: Array,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
        ddof: int = 0,
    ) -> Array:
        """Compute variance along the specified axis.

        Parameters
        ----------
        array : Array
            Input array.
        axis : int or tuple of ints, optional
            Axis or axes along which to compute variance.
            If None, compute over flattened array.
        keepdims : bool
            If True, reduced axes are kept with size 1.
        ddof : int
            Delta degrees of freedom. Default: 0 (population variance).
            Use 1 for sample variance.

        Returns
        -------
        Array
            Variance values.
        """
        ...

    @staticmethod
    def std(
        array: Array,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
        ddof: int = 0,
    ) -> Array:
        """Compute standard deviation along the specified axis.

        Parameters
        ----------
        array : Array
            Input array.
        axis : int or tuple of ints, optional
            Axis or axes along which to compute standard deviation.
            If None, compute over flattened array.
        keepdims : bool
            If True, reduced axes are kept with size 1.
        ddof : int
            Delta degrees of freedom. Default: 0 (population std).
            Use 1 for sample std.

        Returns
        -------
        Array
            Standard deviation values.
        """
        ...

    @staticmethod
    def maximum(x1: Array, x2: Array) -> Array:
        """Element-wise maximum of two arrays."""
        ...

    @staticmethod
    def minimum(x1: Array, x2: Array) -> Array:
        """Element-wise minimum of two arrays."""
        ...

    @staticmethod
    def qr(array: Array, mode: str = "reduced") -> Tuple[Array, Array]:
        """Compute QR factorization.

        Parameters
        ----------
        array : Array
            Matrix to factorize. Shape: (m, n)
        mode : str
            Factorization mode. 'reduced' (default) or 'complete'.

        Returns
        -------
        Q : Array
            Orthogonal matrix.
        R : Array
            Upper triangular matrix.
        """
        ...

    @staticmethod
    def lu(array: Array) -> Tuple[Array, Array, Array]:
        """Compute LU factorization with partial pivoting.

        Returns P, L, U such that A = P @ L @ U.

        Parameters
        ----------
        array : Array
            Matrix to factorize.

        Returns
        -------
        P : Array
            Permutation matrix.
        L : Array
            Lower triangular matrix with unit diagonal.
        U : Array
            Upper triangular matrix.
        """
        ...
