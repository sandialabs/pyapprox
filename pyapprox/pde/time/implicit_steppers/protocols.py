"""
Protocols for implicit time stepping residuals.

This module re-exports protocols from the unified protocols module
for backward compatibility.

For new code, prefer importing from ``pyapprox.pde.time.protocols``.
"""

# Re-export from unified protocols for backward compatibility
from pyapprox.pde.time.protocols import (
    AdjointEnabledTimeSteppingResidualProtocol,
    HVPEnabledTimeSteppingResidualProtocol,
    ODEResidualProtocol,
    ODEResidualWithHVPProtocol,
    ODEResidualWithParamJacobianProtocol,
    TimeSteppingResidualBase,
    TimeSteppingResidualProtocol,
)

# Legacy aliases for backward compatibility
ImplicitODEResidualProtocol = ODEResidualProtocol
ImplicitTimeSteppingResidualProtocol = TimeSteppingResidualProtocol
AdjointEnabledImplicitTimeSteppingResidualProtocol = (
    AdjointEnabledTimeSteppingResidualProtocol
)
ImplicitTimeSteppingResidualBase = TimeSteppingResidualBase

__all__ = [
    # New names
    "ODEResidualProtocol",
    "ODEResidualWithParamJacobianProtocol",
    "ODEResidualWithHVPProtocol",
    "TimeSteppingResidualProtocol",
    "AdjointEnabledTimeSteppingResidualProtocol",
    "HVPEnabledTimeSteppingResidualProtocol",
    "TimeSteppingResidualBase",
    # Legacy aliases
    "ImplicitODEResidualProtocol",
    "ImplicitTimeSteppingResidualProtocol",
    "AdjointEnabledImplicitTimeSteppingResidualProtocol",
    "ImplicitTimeSteppingResidualBase",
]
