from typing import Protocol, Generic, runtime_checkable, Union, Any

from pyapprox.typing.util.backend import Array, Backend


@runtime_checkable
class FunctionWithJacobianProtocol(Protocol, Generic[Array]):
    def bkd(self) -> Backend[Array]: ...

    def nvars(self) -> int: ...

    def nqoi(self) -> int: ...

    def __call__(self, samples: Array) -> Array: ...

    def jacobian(self, sample: Array) -> Array: ...


@runtime_checkable
class FunctionWithJVPProtocol(Protocol, Generic[Array]):
    def bkd(self) -> Backend[Array]: ...

    def nvars(self) -> int: ...

    def nqoi(self) -> int: ...

    def __call__(self, samples: Array) -> Array: ...

    def jvp(self, sample: Array, vec: Array) -> Array: ...


FunctionWithJacobianOrJVPProtocol = Union[
    FunctionWithJacobianProtocol[Array],
    FunctionWithJVPProtocol[Array],
]


def function_has_jacobian_or_jvp(function: Any) -> bool:
    if not isinstance(
        function, FunctionWithJacobianProtocol
    ) and not isinstance(function, FunctionWithJVPProtocol):
        return False
    return True
