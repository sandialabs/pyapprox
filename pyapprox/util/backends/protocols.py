from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
    runtime_checkable,
)

# from typing_extensions import SupportsIndex
from numpy.typing import NDArray

if TYPE_CHECKING:
    import torch

# Define generic types for arrays
Array = TypeVar("Array", bound="ArrayProtocol")

# Input type for asarray/array — must be accepted by all backends
ArrayLike = Union[Sequence[Any], float, int, "NDArray[Any]", "torch.Tensor"]


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

    def __gt__(self: Array, other: Union[float, Array]) -> Union[bool, Array]: ...

    def __lt__(self: Array, other: Union[float, Array]) -> Union[bool, Array]: ...

    def __ge__(self: Array, other: Union[float, Array]) -> Union[bool, Array]: ...

    def __le__(self: Array, other: Union[float, Array]) -> Union[bool, Array]: ...

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
    def multidot(arrays: List[Array]) -> Array:
        """Compute the dot product of multiple matrices."""
        ...

    def eye(
        self, nrows: int, ncols: Optional[int] = None, dtype: Optional[Any] = None
    ) -> Array: ...

    @staticmethod
    def inv(matrix: Array) -> Array: ...

    @staticmethod
    def pinv(matrix: Array) -> Array: ...

    def asarray(
        self,
        array: ArrayLike,
        dtype: Optional[Any] = None,
    ) -> Array: ...

    def array(
        self,
        array: ArrayLike,
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

    def linspace(self, start: float, stop: float, num: int) -> Array: ...

    def logspace(self, start: float, stop: float, num: int) -> Array: ...

    @staticmethod
    def to_numpy(array: Array) -> NDArray[Any]: ...

    @staticmethod
    def to_float(array: "float | Array") -> float:
        """Convert a scalar array or Python scalar to a Python float.

        Parameters
        ----------
        array : float | Array
            A Python scalar, or a single-element array.

        Returns
        -------
        float
            The scalar value.

        Raises
        ------
        ValueError
            If the array has more than one element.
        """
        ...

    @staticmethod
    def to_int(array: "int | Array") -> int:
        """Convert a scalar array or Python scalar to a Python int.

        Parameters
        ----------
        array : int | Array
            A Python scalar, or a single-element array.

        Returns
        -------
        int
            The scalar value.

        Raises
        ------
        ValueError
            If the array has more than one element.
        """
        ...

    @staticmethod
    def flatten(array: Array) -> Array: ...

    @staticmethod
    def ravel(array: Array) -> Array:
        """Flatten array to 1D (contiguous view when possible).

        Parameters
        ----------
        array : Array
            Input array.

        Returns
        -------
        Array
            Flattened 1D array.
        """
        ...

    @staticmethod
    def meshgrid(*arrays: Array, indexing: str = "xy") -> Tuple[Array, ...]:
        """Create coordinate grids from 1D arrays.

        Parameters
        ----------
        *arrays : Array
            1D arrays representing grid coordinates.
        indexing : str
            Cartesian ('xy') or matrix ('ij') indexing. Default: 'xy'.

        Returns
        -------
        Tuple[Array, ...]
            Coordinate grids, one per input array.
        """
        ...

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
    def cumsum(array: Array, axis: Optional[int] = None) -> Array: ...

    @staticmethod
    def sin(array: Array) -> Array: ...

    @staticmethod
    def cos(array: Array) -> Array: ...

    @staticmethod
    def arcsin(array: Array) -> Array: ...

    @staticmethod
    def arccos(array: Array) -> Array: ...

    @staticmethod
    def arctan(array: Array) -> Array: ...

    @staticmethod
    def arctan2(y: Array, x: Array) -> Array: ...

    @staticmethod
    def sinh(array: Array) -> Array: ...

    @staticmethod
    def cosh(array: Array) -> Array: ...

    @staticmethod
    def tanh(array: Array) -> Array: ...

    @staticmethod
    def arcsinh(array: Array) -> Array: ...

    @staticmethod
    def arccosh(array: Array) -> Array: ...

    @staticmethod
    def arctanh(array: Array) -> Array: ...

    def full(
        self, shape: Tuple[int, ...], fill_value: float, dtype: Optional[Any] = None
    ) -> Array: ...

    def zeros(self, shape: Tuple[int, ...], dtype: Optional[Any] = None) -> Array: ...

    def ones(self, shape: Tuple[int, ...], dtype: Optional[Any] = None) -> Array: ...

    def empty(self, shape: Tuple[int, ...], dtype: Optional[Any] = None) -> Array: ...

    # @staticmethod
    # def arange(
    #     start: Union[int, float],
    #     stop: Union[int, float],
    #     step: Union[int, float] = 1,
    #     dtype: Optional[Any] = None,
    # ) -> Array: ...

    # Overload 1: arange(stop)
    @overload
    def arange(self, stop: Union[int, float], /) -> Array:
        """
        Overload for when only `stop` is provided.
        """
        ...

    # Overload 2: arange(start, stop)
    @overload
    def arange(
        self, start: Union[int, float], stop: Union[int, float], /
    ) -> Array:
        """
        Overload for when `start` and `stop` are provided.
        """
        ...

    # Overload 3: arange(start, stop, step)
    @overload
    def arange(
        self,
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
    @overload
    def arange(
        self,
        *args: Union[int, float],
        dtype: Optional[Any] = None,
    ) -> Array:
        """
        Overload for when keyword arguments like `dtype` are provided.
        """
        ...

    # Runtime implementation
    def arange(self, *args: Any, **kwargs: Any) -> Array:
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
    def floor(array: Array) -> Array: ...

    @staticmethod
    def ceil(array: Array) -> Array: ...

    @staticmethod
    def sqrt(array: Array) -> Array: ...

    @staticmethod
    def solve(Amat: Array, Bmat: Array) -> Array: ...

    @staticmethod
    def solve_sparse(Amat, bvec: Array) -> Array:
        """Solve A @ x = b where A is a sparse matrix.

        Parameters
        ----------
        Amat : sparse matrix
            Sparse system matrix (scipy CSR/CSC or torch sparse CSR).
        bvec : Array
            Right-hand side vector. Shape: (n,).

        Returns
        -------
        Array
            Solution vector. Shape: (n,).
        """
        ...

    @staticmethod
    def flip(array: Array, axis: Optional[Tuple[int]] = None) -> Array: ...

    @staticmethod
    def sort(array: Array, axis: int = -1) -> Array: ...

    @staticmethod
    def argsort(array: Array, axis: int = -1) -> Array:
        """Return indices that would sort the array.

        Parameters
        ----------
        array : Array
            Input array.
        axis : int
            Axis along which to sort. Default is -1 (last axis).

        Returns
        -------
        Array
            Array of indices that sort the input array.
        """
        ...

    @staticmethod
    def searchsorted(
        sorted_array: Array, values: Array, side: str = "left"
    ) -> Array: ...

    @staticmethod
    def clip(array: Array, a_min: Any, a_max: Any) -> Array: ...

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
    def outer(a: Array, b: Array) -> Array: ...

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
    def eigvalsh(array: Array) -> Array:
        """
        Compute eigenvalues of a Hermitian/symmetric matrix.

        Parameters
        ----------
        array : Array
            Hermitian/symmetric matrix, shape (n, n).

        Returns
        -------
        Array
            Eigenvalues in ascending order, shape (n,).
        """
        ...

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
    def isinf(array: Array) -> Array: ...

    @staticmethod
    def unique(array: Array) -> Array:
        """Find unique elements in array.

        Parameters
        ----------
        array : Array
            Input array.

        Returns
        -------
        Array
            Sorted unique elements.
        """
        ...

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
    def svd(array: Array, full_matrices: bool = True) -> Tuple[Array, Array, Array]:
        """Compute singular value decomposition: U, S, Vh = svd(A)."""
        ...

    @staticmethod
    def rank(array: Array) -> int:
        """Compute matrix rank."""
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

    def default_dtype(self) -> Any:
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
    def det(array: Array) -> Array:
        """Compute determinant of a matrix.

        Parameters
        ----------
        array : Array
            Square matrix. Shape: (n, n)

        Returns
        -------
        det : Array
            Determinant (scalar array).
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
    def cov(
        array: Array,
        rowvar: bool = True,
        ddof: int = 1,
    ) -> Array:
        """Estimate covariance matrix.

        Parameters
        ----------
        array : Array
            Input array containing observations.
            If rowvar=True (default), rows are variables, columns are observations.
            If rowvar=False, columns are variables, rows are observations.
        rowvar : bool
            If True, each row is a variable; if False, each column is a variable.
        ddof : int
            Delta degrees of freedom. Default: 1 (unbiased sample covariance).

        Returns
        -------
        Array
            Covariance matrix. Shape: (nvars, nvars), or scalar if 1D input.
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
    def split(
        array: Array,
        indices_or_sections: Array,
        axis: int = 0,
    ) -> List[Array]:
        """Split array into multiple sub-arrays.

        Parameters
        ----------
        array : Array
            Array to split.
        indices_or_sections : Array
            Indices at which to split.
        axis : int
            Axis along which to split.

        Returns
        -------
        List[Array]
            List of sub-arrays.
        """
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

    @staticmethod
    def index_update(
        array: Array,
        index: Union[int, Tuple[int, ...]],
        value: Any,
    ) -> Array:
        """Return a copy of array with array[index] = value.

        This provides a functional interface for array updates that
        is compatible with autograd frameworks.

        Parameters
        ----------
        array : Array
            Array to update.
        index : int or tuple of int
            Index to update.
        value : Any
            Value to set at index.

        Returns
        -------
        Array
            Copy of array with updated value.
        """
        ...

    @staticmethod
    def block(blocks: List[List[Array]]) -> Array:
        """Create a block matrix from nested list of arrays.

        Assembles arrays from a nested list into a single array.
        Equivalent to np.block.

        Parameters
        ----------
        blocks : list of list of Array
            Nested list of arrays. Each inner list forms a row of blocks.

        Returns
        -------
        Array
            Block matrix assembled from input arrays.
        """
        ...

    @staticmethod
    def array_type() -> type:
        """Return the array type class for this backend.

        Returns
        -------
        type
            The array type class (e.g., np.ndarray for NumPy, torch.Tensor for PyTorch).
        """
        ...

    def tril_indices(self, n: int, k: int = 0) -> Tuple[Array, Array]:
        """Return indices for lower-triangular part of an (n, n) array.

        Parameters
        ----------
        n : int
            Size of the arrays for which the indices are returned.
        k : int, optional
            Diagonal offset. k=0 (default) is the main diagonal,
            k<0 is below it, and k>0 is above.

        Returns
        -------
        rows : Array
            Row indices of lower triangular elements.
        cols : Array
            Column indices of lower triangular elements.
        """
        ...
