"""Time integration adapters for Galerkin physics.

This module provides adapters that allow Galerkin physics (with mass matrices)
to be used with the time steppers in typing.pde.time.
"""

from pyapprox.ode.config import TimeIntegrationConfig
from pyapprox.pde.galerkin.time_integration.constrained_residual import (
    ConstrainedTimeStepResidual,
)
from pyapprox.pde.galerkin.time_integration.explicit_adapter import (
    GalerkinExplicitODEAdapter,
)
from pyapprox.pde.galerkin.time_integration.galerkin_model import (
    GalerkinModel,
)
from pyapprox.pde.galerkin.time_integration.physics_adapter import (
    GalerkinPhysicsToODEResidualAdapter,
    GalerkinPhysicsToODEResidualWithParamJacobianAdapter,
    GalerkinPhysicsToODEResidualWithSetParamAdapter,
    create_galerkin_physics_ode_residual,
)
from pyapprox.pde.galerkin.time_integration.stokes_time_stepper import (
    StokesTimeStepResidual,
)

__all__ = [
    "GalerkinPhysicsToODEResidualAdapter",
    "GalerkinPhysicsToODEResidualWithSetParamAdapter",
    "GalerkinPhysicsToODEResidualWithParamJacobianAdapter",
    "create_galerkin_physics_ode_residual",
    "GalerkinExplicitODEAdapter",
    "ConstrainedTimeStepResidual",
    "StokesTimeStepResidual",
    "GalerkinModel",
    "TimeIntegrationConfig",
]
