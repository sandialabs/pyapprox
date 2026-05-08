"""Implicit time stepping residuals with adjoint support."""

from pyapprox.ode.implicit_steppers.backward_euler import (
    BackwardEulerAdjoint,
    BackwardEulerHVP,
    BackwardEulerStepper,
)
from pyapprox.ode.implicit_steppers.crank_nicolson import (
    CrankNicolsonAdjoint,
    CrankNicolsonHVP,
    CrankNicolsonStepper,
)
from pyapprox.ode.implicit_steppers.protocols import (
    AdjointEnabledImplicitTimeSteppingResidualProtocol,
    AdjointEnabledTimeSteppingResidualProtocol,
    HVPEnabledTimeSteppingResidualProtocol,
    # Legacy aliases
    ImplicitODEResidualProtocol,
    ImplicitTimeSteppingResidualBase,
    ImplicitTimeSteppingResidualProtocol,
    ODEResidualProtocol,
    ODEResidualWithHVPProtocol,
    ODEResidualWithParamJacobianProtocol,
    TimeSteppingResidualBase,
    TimeSteppingResidualProtocol,
)

__all__ = [
    # Steppers
    "BackwardEulerStepper",
    "BackwardEulerAdjoint",
    "BackwardEulerHVP",
    "CrankNicolsonStepper",
    "CrankNicolsonAdjoint",
    "CrankNicolsonHVP",
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
