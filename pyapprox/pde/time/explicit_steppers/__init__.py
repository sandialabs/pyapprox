"""Explicit time stepping residuals with adjoint support."""

from pyapprox.pde.time.explicit_steppers.forward_euler import (
    ForwardEulerResidual,
)
from pyapprox.pde.time.explicit_steppers.heun import (
    HeunResidual,
)
from pyapprox.pde.time.explicit_steppers.protocols import (
    ODEResidualProtocol,
    ODEResidualWithParamJacobianProtocol,
    ODEResidualWithHVPProtocol,
    TimeSteppingResidualProtocol,
    AdjointEnabledTimeSteppingResidualProtocol,
    HVPEnabledTimeSteppingResidualProtocol,
    TimeSteppingResidualBase,
    # Legacy alias
    ExplicitODEResidualProtocol,
)

__all__ = [
    # Steppers
    "ForwardEulerResidual",
    "HeunResidual",
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
