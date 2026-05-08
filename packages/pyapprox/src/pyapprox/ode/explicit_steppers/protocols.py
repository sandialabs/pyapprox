"""
Protocols for explicit time stepping residuals.

This module re-exports protocols from the unified protocols module.

For new code, prefer importing from ``pyapprox.ode.protocols``.
"""

# Re-export from unified protocols
from pyapprox.ode.protocols import (
    AdjointEnabledTimeSteppingResidualProtocol,
    HVPEnabledTimeSteppingResidualProtocol,
    ODEResidualProtocol,
    ODEResidualWithHVPProtocol,
    ODEResidualWithParamJacobianProtocol,
    TimeSteppingResidualBase,
    TimeSteppingResidualProtocol,
)

# Legacy alias for backward compatibility
ExplicitODEResidualProtocol = ODEResidualProtocol

__all__ = [
    "ODEResidualProtocol",
    "ODEResidualWithParamJacobianProtocol",
    "ODEResidualWithHVPProtocol",
    "TimeSteppingResidualProtocol",
    "AdjointEnabledTimeSteppingResidualProtocol",
    "HVPEnabledTimeSteppingResidualProtocol",
    "TimeSteppingResidualBase",
    "ExplicitODEResidualProtocol",
]
