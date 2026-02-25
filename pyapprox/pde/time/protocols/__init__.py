"""
Protocols for time integration with adjoint and HVP support.

This package defines the protocol hierarchy for ODE residuals and time stepping
residuals used in time integration with sensitivity computation.

Protocol Hierarchy
------------------
ODE Residuals (user-defined):
    ODEResidualProtocol
        -> ODEResidualWithParamJacobianProtocol
            -> ODEResidualWithHVPProtocol

Time Stepping Residuals (framework):
    TimeSteppingResidualProtocol
        -> AdjointEnabledTimeSteppingResidualProtocol
            -> HVPEnabledTimeSteppingResidualProtocol

Base Classes:
    TimeSteppingResidualBase
        Abstract base for all time steppers.

Time Handling
-------------
Time is incorporated via stateful `set_time()` methods rather than explicit
arguments. This keeps the Newton solver generic (just solves R(y)=0).

- ODE residual: `set_time(time)` sets current time
- Time stepping residual: `set_time(time, deltat, prev_state)` sets context
- Integrator calls set_time before each evaluation
"""

from .base import TimeSteppingResidualBase
from .ode_residual import (
    ODEResidualProtocol,
    ODEResidualWithHVPProtocol,
    ODEResidualWithParamJacobianProtocol,
)
from .time_stepping import (
    AdjointEnabledTimeSteppingResidualProtocol,
    HVPEnabledTimeSteppingResidualProtocol,
    TimeSteppingResidualProtocol,
)

__all__ = [
    # ODE Residual Protocols
    "ODEResidualProtocol",
    "ODEResidualWithParamJacobianProtocol",
    "ODEResidualWithHVPProtocol",
    # Time Stepping Residual Protocols
    "TimeSteppingResidualProtocol",
    "AdjointEnabledTimeSteppingResidualProtocol",
    "HVPEnabledTimeSteppingResidualProtocol",
    # Base Classes
    "TimeSteppingResidualBase",
]
