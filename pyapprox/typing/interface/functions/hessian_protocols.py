from typing import Protocol, runtime_checkable

from pyapprox.typing.util.backend import Array
from pyapprox.typing.interface.functions.function import FunctionProtocol
from pyapprox.typing.interface.functions.jacobian_protocols import (
    JacobianProtocol,
    ApplyJacobianProtocol,
)


class ApplyHessianProtocol(Protocol):
    """
    Protocol for functions with Hessian functionality.
    """

    def apply_hessian(self, sample: Array, vec: Array) -> Array: ...


@runtime_checkable
class FunctionWithJacobianApplyHessianProtocol(
    FunctionProtocol, JacobianProtocol, ApplyHessianProtocol, Protocol
):
    pass


@runtime_checkable
class FunctionWithApplyJacobianApplyHessianProtocol(
    FunctionProtocol, ApplyJacobianProtocol, ApplyHessianProtocol, Protocol
):
    pass


def validate_hvp(nvars: int, hvp: Array) -> None:
    if hvp.shape != (nvars, 1):
        raise ValueError(
            f"Hvp shape mismatch: expected " f"({nvars, 1}), got {hvp.shape}"
        )
