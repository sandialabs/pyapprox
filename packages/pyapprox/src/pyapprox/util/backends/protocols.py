from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    List,
    Literal,
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

from pyapprox.util.backends.precision import Precision

if TYPE_CHECKING:
    import torch
    from scipy.sparse import spmatrix

# Define generic types for arrays
Array = TypeVar("Array", bound="ArrayProtocol")
Array_co = TypeVar("Array_co", bound="ArrayProtocol", covariant=True)
Array_contra = TypeVar("Array_contra", bound="ArrayProtocol", contravariant=True)
# What a kernel compiled by ``Backend.jit`` returns.
KernelResult = TypeVar("KernelResult")

# Input type for asarray/array — must be accepted by all backends
ArrayLike = Union[
    Sequence[Any], float, int, "NDArray[Any]", "torch.Tensor", "ArrayProtocol"
]

# Index expression for index_update: every form that both backends
# accept in ``array[index] = value`` (fancy indexing included). The
# parameter is the backend's array type (integer or boolean arrays).
IndexLike = Union[int, slice, Array, Tuple[Union[int, slice, Array], ...]]


@runtime_checkable
class ArrayProtocol(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def ndim(self) -> int: ...

    @property
    def dtype(self) -> Any: ...

    @property
    def T(self: Array) -> Array: ...

    def __truediv__(self: Array, other: Union[float, Array]) -> Array:
        """Elementwise division. With both operands integer the result
        dtype is undefined (NumPy gives float64, torch its global default
        dtype): convert integers to a floating dtype first."""
        ...

    def __rtruediv__(self: Array, other: Union[float, Array]) -> Array: ...

    def __mul__(self: Array, other: Union[float, Array]) -> Array: ...

    def __rmul__(self: Array, other: Union[float, Array]) -> Array: ...

    def __add__(self: Array, other: Union[float, Array]) -> Array: ...

    def __radd__(self: Array, other: Union[float, Array]) -> Array: ...

    def __sub__(self: Array, other: Union[float, Array]) -> Array: ...

    def __rsub__(self: Array, other: Union[float, Array]) -> Array: ...

    def __pow__(self: Array, other: Union[float, Array]) -> Array: ...

    def __rpow__(self: Array, other: Union[float, Array]) -> Array: ...

    def __gt__(self: Array, other: Union[float, Array]) -> Array: ...

    def __lt__(self: Array, other: Union[float, Array]) -> Array: ...

    def __ge__(self: Array, other: Union[float, Array]) -> Array: ...

    def __le__(self: Array, other: Union[float, Array]) -> Array: ...

    def __matmul__(self: Array, other: Array) -> Array: ...

    # Elementwise, like the other comparisons: a == b is a boolean array.
    # That breaks object.__eq__'s contract (any object in, bool out), so
    # the override is ignored, as the NumPy and JAX stubs ignore it.
    def __eq__(self: Array, other: Union[float, Array]) -> Array: ...  # type: ignore[override]

    def __ne__(self: Array, other: Union[float, Array]) -> Array: ...  # type: ignore[override]

    # No __setitem__: JAX arrays are immutable and an in-place write can
    # break torch autograd, so generic code updates arrays through
    # Backend.index_update and Backend.index_add.

    def __neg__(self: Array) -> Array: ...

    def __getitem__(self: Array, index: Any) -> Array:
        """NumPy indexing: integers, slices, ``None``, integer-array
        gathers and boolean masks. Two cases are not portable and
        generic code must not use them: an out-of-range index (JAX
        clamps it rather than raising ``IndexError``) and a negative
        slice step (torch refuses it; use ``Backend.flip``)."""
        ...

    def __iter__(self: Array) -> Any: ...

    def __len__(self) -> int: ...

    def item(self) -> Any: ...

    def __float__(self) -> float: ...

    def __int__(self) -> int: ...

    def __invert__(self: Array) -> Array: ...

    def __and__(self: Array, other: Union[bool, Array]) -> Array: ...

    def __rand__(self: Array, other: Union[bool, Array]) -> Array: ...

    def __or__(self: Array, other: Union[bool, Array]) -> Array: ...

    def __ror__(self: Array, other: Union[bool, Array]) -> Array: ...

    def __xor__(self: Array, other: Union[bool, Array]) -> Array: ...

    def __rxor__(self: Array, other: Union[bool, Array]) -> Array: ...


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
    def inv(matrix: Array) -> Array:
        """Inverse of a square matrix.

        Raises
        ------
        pyapprox.util.backends.errors.LinAlgError
            If the matrix is singular.
        """
        ...

    @staticmethod
    def pinv(matrix: Array, *, rtol: Optional[float] = None) -> Array:
        """Moore-Penrose pseudo-inverse of an ``(m, n)`` matrix.

        Singular values at most ``rtol`` times the largest are treated as
        zero. ``None`` is the Array API cutoff ``max(m, n) * eps`` of the
        matrix's dtype, computed identically on every backend, so the
        result does not depend on the library's own default.
        """
        ...

    def asarray(
        self,
        array: ArrayLike,
        dtype: Optional[Any] = None,
    ) -> Array:
        """``array`` as an array of this backend, sharing memory with it
        where the library allows. An explicit ``dtype`` wins. Without one,
        an existing array keeps its dtype, and Python data is coerced by
        kind: float to ``real_dtype()``, complex to ``complex_dtype()``,
        int to ``index_dtype()``, bool kept."""
        ...

    def array(
        self,
        array: ArrayLike,
        dtype: Optional[Any] = None,
    ) -> Array:
        """A new array holding a copy of ``array``, never sharing memory
        with it; on an autodiff backend the copy is detached. Without
        ``dtype``, floating input takes the default dtype and other
        input keeps its kind."""
        ...

    @staticmethod
    def assert_allclose(
        actual: Array,
        desired: Array,
        rtol: float = 1e-7,
        atol: float = 0,
        equal_nan: bool = True,
        err_msg: Optional[str] = None,
    ) -> None: ...

    def precision(self) -> Precision:
        """The precision the backend is configured with."""
        ...

    def device(self) -> str:
        """The device arrays are created on, e.g. ``"cpu"`` or
        ``"cuda:0"``."""
        ...

    def is_compatible(self, other: object) -> bool:
        """Whether ``other`` is a backend of the same class, precision and
        device, so arrays pass between the two unconverted. Options that
        change how code runs but not what arrays look like (torch's
        ``compile``) do not count; equality (``==``) also compares them."""
        ...

    def replace(
        self, *, precision: Optional[Precision] = None, device: Optional[str] = None
    ) -> "Backend[Array]":
        """A backend of the same class with the given precision or device
        changed and every other option kept, e.g. a double-precision
        sibling for one sensitive substep of a single-precision model."""
        ...

    def from_other(self, array: ArrayProtocol, src: "Backend[Any]") -> Array:
        """``array``, an array of backend ``src``, as an array of this
        backend: same values and kind, in this backend's precision and on
        its device. Returned unchanged when ``src`` is compatible
        (:meth:`is_compatible`). Between backends of one library the
        conversion keeps the autograd graph; between libraries it goes
        through NumPy and is not differentiable through.
        """
        ...

    def real_dtype(self) -> Any:
        """The real working dtype: float32 or float64, from
        :meth:`precision`."""
        ...

    def complex_dtype(self) -> Any:
        """The complex working dtype, paired with :meth:`real_dtype`
        (complex64 or complex128); never configured on its own."""
        ...

    def index_dtype(self) -> Any:
        """The dtype of indices and counts: int64 on every backend. Code
        may assume an integral kind, never a width."""
        ...

    def eps(self) -> float:
        """Machine epsilon of :meth:`real_dtype`."""
        ...

    def result_type(self, *arrays: Array) -> Any:
        """The dtype NumPy's promotion rules give ``arrays`` combined."""
        ...

    def isdtype(self, array: Array, kind: str) -> bool:
        """Whether ``array``'s dtype is of ``kind``, one of the Array API
        kinds ``"real floating"``, ``"complex floating"``, ``"integral"``
        (signed or unsigned, not bool) and ``"bool"``. Reads the dtype
        only, so it never syncs with the device.

        Raises
        ------
        ValueError
            If ``kind`` is not one of those four.
        """
        ...

    def real_dtype_of(self, dtype: Any) -> Any:
        """The real floating dtype of ``dtype``'s width: float32 for
        float32 and complex64, float64 for float64 and complex128, so
        code holding data of one width finds its partner without
        consulting the configuration.

        Raises
        ------
        ValueError
            If ``dtype`` is not floating or complex.
        """
        ...

    def complex_dtype_of(self, dtype: Any) -> Any:
        """The complex dtype of ``dtype``'s width: complex64 for float32
        and complex64, complex128 for float64 and complex128.

        Raises
        ------
        ValueError
            If ``dtype`` is not floating or complex.
        """
        ...

    @staticmethod
    def real(array: Array) -> Array:
        """The real part; ``array`` itself for real input."""
        ...

    @staticmethod
    def imag(array: Array) -> Array:
        """The imaginary part; zeros of ``array``'s dtype for real
        input."""
        ...

    def double_dtype(self) -> Any:
        """The same as :meth:`real_dtype`."""
        ...

    def int64_dtype(self) -> Any:
        """The same as :meth:`index_dtype`."""
        ...

    @staticmethod
    def is_integer_dtype(array: Array) -> bool:
        """Whether ``array`` has a signed or unsigned integer dtype;
        boolean is not integer."""
        ...

    @staticmethod
    def is_floating_dtype(array: Array) -> bool:
        """Whether ``array`` has a real floating dtype; complex is not
        floating."""
        ...

    @staticmethod
    def tracks_gradient(array: Array) -> bool:
        """Whether ``array`` is being differentiated through.

        True only while a backend that supports automatic
        differentiation is recording operations on this array. Code
        that memoizes results keyed on array *values* must consult
        this: cached results are expressions wired to the array that
        produced them, so serving them for a different array of equal
        values yields a graph with no path back to the current input,
        and a derivative that is silently zero.
        """
        ...

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
            If the array does not have exactly one element.
        TypeError
            If the array is complex: drop the imaginary part explicitly.
        """
        ...

    @staticmethod
    def to_int(array: "int | Array") -> int:
        """Convert a scalar array or Python scalar to a Python int,
        truncating a floating value toward zero, as ``int()`` does.

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
            If the array does not have exactly one element.
        TypeError
            If the array is complex.
        """
        ...

    @staticmethod
    def flatten(array: Array) -> Array:
        """A 1-D copy of ``array`` in row-major order; never shares memory
        with it (``ravel`` may)."""
        ...

    @staticmethod
    def squeeze(array: Array) -> Array:
        """Remove single-dimensional entries from the shape of an array."""
        ...

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
    def meshgrid(
        *arrays: Array, indexing: Literal["xy", "ij"] = "xy"
    ) -> Tuple[Array, ...]:
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
    def transpose(array: Array, axes: Optional[Sequence[int]] = None) -> Array:
        """Permute the axes; reverse them all if ``axes`` is ``None``."""
        ...

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

    def full_like(self, a: Array, fill_value: float) -> Array: ...

    def zeros(self, shape: Tuple[int, ...], dtype: Optional[Any] = None) -> Array: ...

    def ones(self, shape: Tuple[int, ...], dtype: Optional[Any] = None) -> Array: ...

    def empty(self, shape: Tuple[int, ...], dtype: Optional[Any] = None) -> Array:
        """An array of ``shape`` whose values are unspecified."""
        ...

    def arange(
        self,
        start: Union[int, float],
        /,
        stop: Optional[Union[int, float]] = None,
        step: Union[int, float] = 1,
        *,
        dtype: Optional[Any] = None,
    ) -> Array:
        """Evenly spaced values in ``[start, stop)``; ``[0, start)`` when
        ``stop`` is ``None`` (the Array API signature).

        Without ``dtype``: the index dtype if every bound is an int, the
        default floating dtype otherwise.
        """
        ...

    @staticmethod
    def prod(
        array: Array,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> Array:
        """Product over ``axis`` (all axes if ``None``, none if ``()``)."""
        ...

    @staticmethod
    def any(
        array: Array,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> Array:
        """Test whether any element along the given axes is nonzero.

        The Array API ``any``: the result is always a boolean array, 0-d
        when every axis is reduced, so it stays on the backend's device
        and inside a compiled graph. Convert it with ``to_bool`` only
        where Python control flow needs a ``bool``.

        Parameters
        ----------
        array : Array
            Input of any dtype; nonzero entries count as True.
        axis : int, tuple of int, or None
            Axes to reduce. None reduces all axes; ``()`` reduces none.
            Negative axes count from the end.
        keepdims : bool
            If True, reduced axes are kept with length one.

        Returns
        -------
        Array
            Boolean array. An empty reduction is False.
        """
        ...

    @staticmethod
    def all(
        array: Array,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> Array:
        """Test whether every element along the given axes is nonzero.

        The Array API ``all``; see ``any`` for the arguments and the
        result. An empty reduction is True.
        """
        ...

    @staticmethod
    def to_bool(array: Array) -> bool:
        """Convert a one-element array to a Python ``bool``.

        This is where a value leaves the array world: on a GPU it waits
        for the device, and under ``torch.compile`` it breaks the graph.
        Call it only at a Python control-flow decision, after reducing
        with ``any`` or ``all``.

        Raises
        ------
        ValueError
            If the array does not hold exactly one element.
        """
        ...

    @staticmethod
    def log(array: Array) -> Array: ...

    @staticmethod
    def exp(array: Array) -> Array: ...

    @staticmethod
    def expm1(array: Array) -> Array:
        """exp(x) - 1, accurate for small x."""
        ...

    @staticmethod
    def log1p(array: Array) -> Array:
        """log(1 + x), accurate for small x."""
        ...

    @staticmethod
    def copy(array: Array) -> Array: ...

    @staticmethod
    def erf(array: Array) -> Array: ...

    @staticmethod
    def erfinv(array: Array) -> Array: ...

    @staticmethod
    def ndtr(array: Array) -> Array:
        """Standard normal CDF (Phi). Equivalent to scipy.special.ndtr."""
        ...

    @staticmethod
    def log_ndtr(array: Array) -> Array:
        """Log of standard normal CDF. Equivalent to scipy.special.log_ndtr."""
        ...

    @staticmethod
    def equal(a: Array, b: "float | int | Array") -> Array: ...

    @staticmethod
    def isfinite(array: Array) -> Array: ...

    @staticmethod
    def nonzero(condition: Array) -> Tuple[Array, ...]: ...

    @staticmethod
    def norm(
        array: Array,
        ord: Optional[Union[int, float, Literal["fro", "nuc"]]] = None,
        axis: Optional[Union[int, Tuple[int, int]]] = None,
        keepdims: bool = False,
    ) -> Array:
        """Vector or matrix norm, as ``numpy.linalg.norm``.

        ``array`` is floating or complex; the result is real.
        """
        ...

    @staticmethod
    def sign(array: Array) -> Array: ...

    @staticmethod
    def abs(array: Array) -> Array: ...

    @staticmethod
    def conj(array: Array) -> Array:
        """Return the elementwise complex conjugate; real arrays unchanged.

        Differentiable, so a gradient graph through ``array`` survives.
        """
        ...

    @staticmethod
    def round(array: Array) -> Array: ...

    @staticmethod
    def floor(array: Array) -> Array: ...

    @staticmethod
    def floor_divide(array: Array, divisor: "int | Array") -> Array: ...

    @staticmethod
    def ceil(array: Array) -> Array: ...

    @staticmethod
    def sqrt(array: Array) -> Array: ...

    @staticmethod
    def solve(Amat: Array, Bmat: Array) -> Array:
        """Solve ``Amat @ x = Bmat``; ``Bmat`` is ``(n,)`` or ``(n, k)``.

        Raises
        ------
        pyapprox.util.backends.errors.LinAlgError
            If ``Amat`` is singular.
        """
        ...

    @staticmethod
    def solve_sparse(Amat: spmatrix, bvec: Array) -> Array:
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

        Raises
        ------
        NotImplementedError
            On a backend without a sparse solver.
        """
        ...

    @staticmethod
    def flip(array: Array, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Array:
        """Reverse the order of elements along ``axis``; every axis if
        ``None``."""
        ...

    @staticmethod
    def sort(array: Array, axis: int = -1) -> Array:
        """Sorted copy along ``axis``, stable on every backend.

        Raises
        ------
        TypeError
            If ``array`` is complex: complex values have no order.
        """
        ...

    @staticmethod
    def quantile(array: Array, level: float, axis: int = 0) -> Array:
        """Return the ``level`` quantile along ``axis``.

        Parameters
        ----------
        array : Array
            Input array.
        level : float
            Quantile level in [0, 1].
        axis : int
            Axis to reduce. Defaults to 0, the sample axis of a stack of
            observations.

        Returns
        -------
        Array
            The quantile, with ``axis`` removed.
        """
        ...

    @staticmethod
    def argsort(array: Array, axis: int = -1) -> Array:
        """Return indices that would sort the array, stably: equal values
        keep their order, on every backend.

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

        Raises
        ------
        TypeError
            If ``array`` is complex: complex values have no order.
        """
        ...

    @staticmethod
    def searchsorted(
        sorted_array: Array, values: Array, side: Literal["left", "right"] = "left"
    ) -> Array: ...

    @staticmethod
    def clip(array: Array, a_min: Any, a_max: Any) -> Array: ...

    @staticmethod
    def min(
        array: Array,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> Array:
        """Minimum over ``axis`` (all axes if ``None``, none if ``()``).

        Raises
        ------
        TypeError
            If ``array`` is complex: complex values have no order; reduce
            ``abs`` or the real part instead.
        ValueError
            If an axis reduced over has length zero.
        """
        ...

    @staticmethod
    def max(
        array: Array,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> Array:
        """Maximum over ``axis`` (all axes if ``None``, none if ``()``).

        Raises
        ------
        TypeError
            If ``array`` is complex: complex values have no order; reduce
            ``abs`` or the real part instead.
        ValueError
            If an axis reduced over has length zero.
        """
        ...

    @staticmethod
    def logsumexp(
        array: Array,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> Array:
        """Log of sum of exponentials over ``axis`` (all axes if
        ``None``, none if ``()``). Numerically stable."""
        ...

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
    def outer(a: Array, b: Array) -> Array:
        """Outer product of ``a`` and ``b``, each flattened first:
        shape ``(a.size, b.size)``."""
        ...

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
    def cholesky(array: Array) -> Array:
        """Lower Cholesky factor of a symmetric positive definite matrix.

        Raises
        ------
        pyapprox.util.backends.errors.LinAlgError
            If the matrix is not positive definite.
        """
        ...

    @staticmethod
    def solve_triangular(
        matrix: Array,
        rhs: Array,
        *,
        lower: bool,
        unit_diagonal: bool = False,
    ) -> Array:
        """Solve ``matrix @ x = rhs`` for triangular ``matrix``.

        ``lower`` has no default: the libraries disagree on it (SciPy
        ``False``, torch ``True``), so every call says which triangle.
        ``rhs`` is ``(n,)`` or ``(n, k)``; ``x`` has its shape.
        """
        ...

    @staticmethod
    def cholesky_solve(
        matrix: Array,
        rhs: Array,
        lower: bool = True,
    ) -> Array:
        """Solve ``A @ x = rhs`` given the Cholesky factor ``matrix`` of
        ``A``. ``rhs`` is ``(n,)`` or ``(n, k)``; ``x`` has its shape."""
        ...

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
    def cond(
        array: Array,
        p: Optional[Union[int, float, Literal["fro", "nuc"]]] = None,
    ) -> Array:
        """
        Compute the condition number of a matrix.

        Parameters
        ----------
        array : Array
            Matrix, shape (n, n).
        p : optional
            Norm used for the condition number (default: 2-norm, the ratio of
            largest to smallest singular value).

        Returns
        -------
        Array
            The (scalar) condition number.
        """
        ...

    @staticmethod
    def trace(array: Array) -> Array:
        """Sum of the diagonal over the last two axes, batched over any
        leading axes (``numpy.linalg.trace``)."""
        ...

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
    def cdist(XA: Array, XB: Array, p: float = 2.0) -> Array:
        """Distances between the rows of ``XA`` and of ``XB``: Manhattan
        for ``p=1``, Euclidean for ``p=2``; any other ``p`` raises
        ``ValueError``."""
        ...

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
        """Index of the first minimum along ``axis`` (of the flattened
        array if ``None``), in the index dtype.

        Raises
        ------
        TypeError
            If ``array`` is complex.
        ValueError
            If the axis searched has length zero.
        """
        ...

    @staticmethod
    def argmax(array: Array, axis: Optional[int] = None) -> Array:
        """Index of the first maximum along ``axis`` (of the flattened
        array if ``None``), in the index dtype.

        Raises
        ------
        TypeError
            If ``array`` is complex.
        ValueError
            If the axis searched has length zero.
        """
        ...

    @overload
    @staticmethod
    def where(condition: Array) -> Tuple[Array, ...]: ...

    @overload
    @staticmethod
    def where(
        condition: Array,
        x: Union[Array, float],
        y: Union[Array, float],
    ) -> Array: ...

    @staticmethod
    def where(
        condition: Array,
        x: Union[Array, float, None] = None,
        y: Union[Array, float, None] = None,
    ) -> Union[Array, Tuple[Array, ...]]:
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
        Array or Tuple[Array, ...]
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
        """The same as :meth:`real_dtype`."""
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
            Square floating or complex matrix. Shape: (n, n)

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
            Floating or complex input array.
        axis : int or tuple of ints, optional
            Axis or axes along which to compute means.
            If None, compute over flattened array; if ``()``, over none.
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
        """Compute variance along the specified axis; real for complex
        input.

        Parameters
        ----------
        array : Array
            Floating or complex input array.
        axis : int or tuple of ints, optional
            Axis or axes along which to compute variance.
            If None, compute over flattened array; if ``()``, over none.
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
        """Compute standard deviation along the specified axis; real for
        complex input.

        Parameters
        ----------
        array : Array
            Floating or complex input array.
        axis : int or tuple of ints, optional
            Axis or axes along which to compute standard deviation.
            If None, compute over flattened array; if ``()``, over none.
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
    def qr(
        array: Array, mode: Literal["reduced", "complete"] = "reduced"
    ) -> Tuple[Array, Array]:
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
    def lu_factor(array: Array) -> Tuple[Array, Array]:
        """Compact LU factorization with pivots.

        Parameters
        ----------
        array : Array
            Square matrix to factorize, shape ``(n, n)``, or a stack of
            them, shape ``(k, n, n)``, each factored on its own.

        Returns
        -------
        LU : Array
            Compact LU factor (L and U packed in one matrix), stacked like
            ``array``.
        pivots : Array
            Pivot indices, one row per matrix of a stack: 0-based, in the
            index dtype, as SciPy's ``lu_factor`` (row ``j`` was swapped
            with row ``pivots[j]``).
        """
        ...

    @staticmethod
    def lu_solve(
        LU: Array, pivots: Array, rhs: Array, transpose: bool = False
    ) -> Array:
        """Solve a linear system given a compact LU factorization.

        Parameters
        ----------
        LU : Array
            Compact LU factor from lu_factor.
        pivots : Array
            Pivot indices from lu_factor.
        rhs : Array
            Right-hand side. Shape: (n,) or (n, k); for a stack of
            factors, (k, n, ncols), one right-hand side block per matrix.
        transpose : bool
            If True, solve :math:`A^T x = b` instead of :math:`A x = b`,
            with :math:`b` the right-hand side. This is the plain transpose, without
            conjugation, for complex matrices too.

        Returns
        -------
        Array
            Solution. Same shape as rhs.
        """
        ...

    @staticmethod
    def index_update(
        array: Array,
        index: IndexLike[Array],
        value: Any,
    ) -> Array:
        """Return a copy of array with array[index] = value.

        This provides a functional interface for array updates that
        is compatible with autograd frameworks.

        Parameters
        ----------
        array : Array
            Array to update.
        index : int, slice, index Array, or tuple of those
            Any index expression valid in ``array[index] = value``.
        value : Any
            Value to set at index.

        Returns
        -------
        Array
            Copy of array with updated value.
        """
        ...

    @staticmethod
    def index_add(array: Array, index: Array, values: Array) -> Array:
        """Return a copy of array with ``values[k]`` added at
        ``array[index[k]]`` along the first axis.

        Repeated indices accumulate: every ``values[k]`` whose ``index[k]``
        is ``i`` is added to row ``i``. The input is not modified, so the
        operation is compatible with autograd frameworks and with
        libraries whose arrays are immutable.

        Parameters
        ----------
        array : Array
            Array to add into. Shape: ``(n, ...)``.
        index : Array
            Integer row indices. Shape: ``(m,)``.
        values : Array
            Rows to add. Shape: ``(m, ...)``, trailing dimensions matching
            ``array``'s.

        Returns
        -------
        Array
            Copy of array with the values added. Shape: ``(n, ...)``.

        Raises
        ------
        ValueError
            If ``index`` is not 1-D or ``values`` has a different number of
            rows than ``index``.
        """
        ...

    def jit(
        self,
        kernel: Callable[..., KernelResult],
        static_argnames: Sequence[str] = (),
        dynamic: bool = False,
    ) -> Callable[..., KernelResult]:
        """``kernel`` with this backend bound as its ``bkd`` keyword,
        compiled if this backend compiles.

        A kernel is a module-level function or a staticmethod, pure, that
        takes the backend as the keyword ``bkd``: everything that changes
        between calls is an argument, and the arguments that select a
        code path (orders, flags, shapes) are named in ``static_argnames``.
        Call ``self._bkd.jit(kernel)(...)`` at each use: the backend caches
        the result per ``(kernel, static_argnames, dynamic)``, so later
        calls are a lookup, and nothing compiled is stored on the calling
        object, which keeps it picklable. A composite kernel calls smaller
        kernels directly and is compiled as one; called while a kernel is
        being traced, ``jit`` returns the uncompiled bound kernel, so
        nesting is harmless.

        Parameters
        ----------
        kernel : Callable[..., KernelResult]
            Module-level function or staticmethod with a ``bkd`` keyword.
        static_argnames : Sequence[str]
            Arguments compiled as constants: one compilation per distinct
            value. Must be hashable.
        dynamic : bool
            Compile for varying shapes instead of once per shape, where
            the compiler supports it.

        Returns
        -------
        Callable[..., KernelResult]
            The kernel with ``bkd`` bound; compiled by the backend's
            compiler, or uncompiled on a backend without one or with
            compilation off.

        Raises
        ------
        TypeError
            If ``kernel`` is a lambda, a local function or a bound method:
            each is a new object per call (recompiled every call) and
            bakes captured values in as constants.
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
