"""Implicit time stepping residuals with adjoint support."""

from pyapprox.pde.time.implicit_steppers.backward_euler import (
    BackwardEulerResidual,
)
from pyapprox.pde.time.implicit_steppers.crank_nicolson import (
    CrankNicolsonResidual,
)
from pyapprox.pde.time.implicit_steppers.protocols import (
    ODEResidualProtocol,
    ODEResidualWithParamJacobianProtocol,
    ODEResidualWithHVPProtocol,
    TimeSteppingResidualProtocol,
    AdjointEnabledTimeSteppingResidualProtocol,
    HVPEnabledTimeSteppingResidualProtocol,
    TimeSteppingResidualBase,
    # Legacy aliases
    ImplicitODEResidualProtocol,
    ImplicitTimeSteppingResidualProtocol,
    AdjointEnabledImplicitTimeSteppingResidualProtocol,
    ImplicitTimeSteppingResidualBase,
)

__all__ = [
    # Steppers
    "BackwardEulerResidual",
    "CrankNicolsonResidual",
    # Protocols
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
