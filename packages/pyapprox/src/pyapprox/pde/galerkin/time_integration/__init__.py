"""Time integration adapters for Galerkin physics.

This module provides adapters that allow Galerkin physics (with mass matrices)
to be used with the time steppers in typing.pde.time.

The parameterized adapter tiers and their factory live in
``pyapprox.pde.models.galerkin``.
"""

from pyapprox.ode.config import TimeIntegrationConfig
from pyapprox.pde.galerkin.time_integration.bc_time_residual_adapter import (
    GalerkinBCEnforcingForwardResidual,
    create_galerkin_bc_enforcing_residual,
)
from pyapprox.pde.galerkin.time_integration.galerkin_model import (
    GalerkinModel,
)
from pyapprox.pde.galerkin.time_integration.physics_adapter import (
    GalerkinPhysicsToODEResidualAdapter,
)
from pyapprox.pde.galerkin.time_integration.stokes_time_stepper import (
    StokesTimeStepResidual,
)

__all__ = [
    "GalerkinPhysicsToODEResidualAdapter",
    "GalerkinBCEnforcingForwardResidual",
    "StokesTimeStepResidual",
    "GalerkinModel",
    "TimeIntegrationConfig",
    "create_galerkin_bc_enforcing_residual",
]
