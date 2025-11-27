from typing import Protocol, runtime_checkable, Generic

from pyapprox.typing.util.backend import Array, Backend
from pyapprox.typing.interface.functions.jacobian_protocols import (
    FunctionWithJacobianProtocol,
    FunctionWithJVPProtocol,
)


@runtime_checkable
class DerivativeCheckerFunctionWithJacobianProtocol(Protocol, Generic[Array]):
    """
    Protocol for function objects used in derivative checks.

    Defines the required components for the function object.
    """

    def bkd(self) -> Backend[Array]: ...

    def nvars(self) -> int: ...

    def __call__(self, samples: Array) -> Array: ...

    def jacobian(self, sample: Array) -> Array: ...


@runtime_checkable
class DerivativeCheckerFunctionWithJVPProtocol(Protocol, Generic[Array]):
    """
    Protocol for function objects used in derivative checks.

    Defines the required components for the function object.
    """

    def bkd(self) -> Backend[Array]: ...

    def nvars(self) -> int: ...

    def __call__(self, samples: Array) -> Array: ...

    def jvp(self, sample: Array, vec: Array) -> Array: ...
