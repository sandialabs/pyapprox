"""
Protocols for explicit time stepping residuals.

This module re-exports protocols from the unified protocols module.

For new code, prefer importing from ``pyapprox.pde.time.protocols``.
"""

# Re-export from unified protocols
from pyapprox.pde.time.protocols import (
    ODEResidualProtocol,
    ODEResidualWithParamJacobianProtocol,
    ODEResidualWithHVPProtocol,
    TimeSteppingResidualProtocol,
    AdjointEnabledTimeSteppingResidualProtocol,
    HVPEnabledTimeSteppingResidualProtocol,
    TimeSteppingResidualBase,
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
