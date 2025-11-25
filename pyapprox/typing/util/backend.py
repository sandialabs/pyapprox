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
    # def asarray(array: Array, dtype: Optional[Any] = None) -> Array: ...

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
