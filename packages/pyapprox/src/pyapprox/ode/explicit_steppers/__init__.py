"""Explicit time stepping residuals with adjoint support."""

from pyapprox.ode.explicit_steppers.forward_euler import (
    ForwardEulerAdjoint,
    ForwardEulerHVP,
    ForwardEulerStepper,
)
from pyapprox.ode.explicit_steppers.heun import (
    HeunAdjoint,
    HeunHVP,
    HeunStepper,
)
from pyapprox.ode.explicit_steppers.protocols import (
    AdjointEnabledTimeSteppingResidualProtocol,
    # Legacy alias
    ExplicitODEResidualProtocol,
    HVPEnabledTimeSteppingResidualProtocol,
    ODEResidualProtocol,
    ODEResidualWithHVPProtocol,
    ODEResidualWithParamJacobianProtocol,
    TimeSteppingResidualBase,
    TimeSteppingResidualProtocol,
)

__all__ = [
    # Steppers
    "ForwardEulerStepper",
    "ForwardEulerAdjoint",
    "ForwardEulerHVP",
    "HeunStepper",
    "HeunAdjoint",
    "HeunHVP",
    # Protocols
    "ODEResidualProtocol",
    "ODEResidualWithParamJacobianProtocol",
    "ODEResidualWithHVPProtocol",
    "TimeSteppingResidualProtocol",
    "AdjointEnabledTimeSteppingResidualProtocol",
    "HVPEnabledTimeSteppingResidualProtocol",
    "TimeSteppingResidualBase",
    # Legacy alias
    "ExplicitODEResidualProtocol",
]
