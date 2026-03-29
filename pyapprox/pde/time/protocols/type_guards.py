"""TypeGuards for narrowing time stepping residual protocols."""

from typing import TypeGuard

from pyapprox.pde.time.protocols.time_stepping import (
    AdjointEnabledTimeSteppingResidualProtocol,
    HVPEnabledTimeSteppingResidualProtocol,
)
from pyapprox.util.backends.protocols import Array


def is_hvp_enabled(
    residual: AdjointEnabledTimeSteppingResidualProtocol[Array],
) -> TypeGuard[HVPEnabledTimeSteppingResidualProtocol[Array]]:
    """Return True if residual supports Hessian-vector products.

    Checks for the four core HVP methods. The prev_* methods are only
    present on cross-step schemes (e.g. Crank-Nicolson) and are checked
    separately via has_prev_state_hessian().
    """
    return (
        hasattr(residual, "state_state_hvp")
        and hasattr(residual, "state_param_hvp")
        and hasattr(residual, "param_state_hvp")
        and hasattr(residual, "param_param_hvp")
    )
