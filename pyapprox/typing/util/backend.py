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

    def __pow__(self: Array, other: Union[float, int]) -> Array: ...

    def __setitem__(self: Array, index: Any, value: Any) -> None: ...

    def __neg__(self: Array) -> Array: ...

    def __getitem__(self: Array, index: Any) -> Any: ...

    def __iter__(self: Array) -> Any: ...

    def __len__(self) -> int: ...


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

    # @staticmethod
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
        *arrays: Array, indexing: str = "xy"
    ) -> Tuple[Array, ...]: ...

    @staticmethod
    def reshape(array: Array, newshape: Sequence[int]) -> Array: ...

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
    @overload
    @staticmethod
    def arange(stop: Union[int, float], /) -> Array:
        """
        Overload for when only `stop` is provided.
        """
        ...

    # Overload 2: arange(start, stop)
    @overload
    @staticmethod
    def arange(start: Union[int, float], stop: Union[int, float], /) -> Array:
        """
        Overload for when `start` and `stop` are provided.
        """
        ...

    # Overload 3: arange(start, stop, step)
    @overload
    @staticmethod
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
    @overload
    @staticmethod
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
    def any(
        array: Array, axis: Optional[int] = None, keepdims: bool = False
    ) -> Array: ...

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


def validate_backend(obj: Any) -> None:
    """
    Validate that the given object is an instance of the Backend protocol.

    Parameters
    ----------
    obj : Any
        The object to validate.

    Raises
    ------
    TypeError
        If the object is not an instance of Backend.
    """
    if not isinstance(obj, Backend):
        raise TypeError(
            f"Invalid backend type: expected an instance of Backend, "
            f"got {type(obj).__name__}. Object details: {obj}"
        )


def validate_backends(backends: Sequence[Any]) -> None:
    """
    Validate that all backends in the sequence have the same class name and are valid backends.

    Parameters
    ----------
    backends : Sequence[Any]
        A sequence of backend objects or classes.

    Raises
    ------
    ValueError
        If the sequence is empty or the backends are inconsistent.
    TypeError
        If any backend is not a valid backend.
    """
    if len(backends) == 0:
        raise ValueError("The sequence of backends cannot be empty.")

    # Validate each backend
    for backend in backends:
        validate_backend(backend)

    # Extract class names for comparison
    class_names = [
        (
            type(backend).__name__
            if not isinstance(backend, type)
            else backend.__name__
        )
        for backend in backends
    ]

    # Check if all class names are the same
    if len(set(class_names)) != 1:
        raise ValueError(
            f"Inconsistent backends: expected all backends to have the same class name, "
            f"got {set(class_names)}."
        )
