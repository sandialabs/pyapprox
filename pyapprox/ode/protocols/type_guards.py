"""TypeGuards for narrowing time stepping residual protocols."""

from typing import TypeGuard

from pyapprox.ode.protocols.time_stepping import (
    AdjointEnabledTimeSteppingResidualProtocol,
    HVPEnabledTimeSteppingResidualProtocol,
    PrevStepHVPEnabledTimeSteppingResidualProtocol,
)
from pyapprox.util.backends.protocols import Array


def is_hvp_enabled(
    residual: AdjointEnabledTimeSteppingResidualProtocol[Array],
) -> TypeGuard[HVPEnabledTimeSteppingResidualProtocol[Array]]:
    """Return True if residual supports the four core HVP methods.

    The prev_* methods are checked separately via is_prev_step_hvp_enabled().
    """
    return (
        hasattr(residual, "state_state_hvp")
        and hasattr(residual, "state_param_hvp")
        and hasattr(residual, "param_state_hvp")
        and hasattr(residual, "param_param_hvp")
    )


def is_prev_step_hvp_enabled(
    residual: HVPEnabledTimeSteppingResidualProtocol[Array],
) -> TypeGuard[PrevStepHVPEnabledTimeSteppingResidualProtocol[Array]]:
    """Return True if residual supports cross-step HVP methods.

    These are only present on schemes where R_{n+1} depends on f(y_n),
    such as Crank-Nicolson.
    """
    return (
        hasattr(residual, "prev_state_state_hvp")
        and hasattr(residual, "prev_state_param_hvp")
        and hasattr(residual, "prev_param_state_hvp")
    )
