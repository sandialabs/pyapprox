from typing import Protocol, runtime_checkable, Generic, Union, Any

from pyapprox.typing.util.backend import Array, Backend


@runtime_checkable
class FunctionWithJacobianHVPProtocol(Protocol, Generic[Array]):
    def bkd(self) -> Backend[Array]: ...

    def nvars(self) -> int: ...

    def nqoi(self) -> int: ...

    def __call__(self, samples: Array) -> Array: ...

    def jacobian(self, sample: Array) -> Array: ...

    def hvp(self, sample: Array, vec: Array) -> Array: ...


@runtime_checkable
class FunctionWithJVPHVPProtocol(Protocol, Generic[Array]):
    def bkd(self) -> Backend[Array]: ...

    def nvars(self) -> int: ...

    def nqoi(self) -> int: ...

    def __call__(self, samples: Array) -> Array: ...

    def jvp(self, sample: Array, vec: Array) -> Array: ...

    def hvp(self, sample: Array, vec: Array) -> Array: ...


FunctionWithHVPAndJacobianOrJVPProtocol = Union[
    FunctionWithJacobianHVPProtocol[Array],
    FunctionWithJVPHVPProtocol[Array],
]


def function_has_hvp_and_jacobian_or_jvp(function: Any) -> bool:
    if not isinstance(
        function, FunctionWithJacobianHVPProtocol
    ) and not isinstance(function, FunctionWithJVPHVPProtocol):
        return False
    return True
