"""TypeGuards for narrowing time stepping residual protocols."""

from typing import TypeGuard

from pyapprox.ode.protocols.time_stepping import (
    AdjointEnabledTimeSteppingResidualProtocol,
    HVPEnabledTimeSteppingResidualProtocol,
)
from pyapprox.util.backends.protocols import Array


def is_hvp_enabled(
    residual: AdjointEnabledTimeSteppingResidualProtocol[Array],
) -> TypeGuard[HVPEnabledTimeSteppingResidualProtocol[Array]]:
    """Return True if residual supports all HVP methods (same-step + cross-step)."""
    return (
        hasattr(residual, "state_state_hvp")
        and hasattr(residual, "state_param_hvp")
        and hasattr(residual, "param_state_hvp")
        and hasattr(residual, "param_param_hvp")
        and hasattr(residual, "prev_state_state_hvp")
        and hasattr(residual, "prev_state_param_hvp")
        and hasattr(residual, "prev_param_state_hvp")
    )
