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
        -> SensitivityStepperProtocol
            -> AdjointEnabledTimeSteppingResidualProtocol
                -> HVPEnabledTimeSteppingResidualProtocol

Time Handling
-------------
Newton-facing methods read from bound state via ``bind(ctx)``.
Post-hoc methods (adjoint, HVP, param_jacobian) take their
``StepContext`` as explicit parameters — no mutable state reads.
"""

from .jacobian_operator import JacobianOperatorProtocol
from .mass_system import MassSystemProtocol
from .ode_residual import (
    ImplicitODEResidualProtocol,
    ImplicitODEResidualWithHVPProtocol,
    ImplicitODEResidualWithParamJacobianProtocol,
    ODEResidualProtocol,
    ODEResidualWithHVPProtocol,
    ODEResidualWithParamJacobianProtocol,
)
from .residual_form_system import ResidualFormSystemProtocol
from .spatial_operator import SpatialOperatorProtocol
from .time_stepping import (
    AdjointEnabledTimeSteppingResidualProtocol,
    HVPEnabledTimeSteppingResidualProtocol,
    SensitivityStepperProtocol,
    TimeSteppingResidualProtocol,
)

__all__ = [
    # ODE Residual Protocols
    "ODEResidualProtocol",
    "ODEResidualWithParamJacobianProtocol",
    "ODEResidualWithHVPProtocol",
    # Implicit ODE Residual Protocols
    "ImplicitODEResidualProtocol",
    "ImplicitODEResidualWithParamJacobianProtocol",
    "ImplicitODEResidualWithHVPProtocol",
    # Time Stepping Residual Protocols
    "TimeSteppingResidualProtocol",
    "SensitivityStepperProtocol",
    "AdjointEnabledTimeSteppingResidualProtocol",
    "HVPEnabledTimeSteppingResidualProtocol",
    # Spatial Operator Protocol
    "SpatialOperatorProtocol",
    # Operator-valued Jacobian and implicit-system Protocols
    "JacobianOperatorProtocol",
    "ResidualFormSystemProtocol",
    "MassSystemProtocol",
    # Base Classes and Mixins
]
