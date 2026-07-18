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
    return isinstance(residual, HVPEnabledTimeSteppingResidualProtocol)
