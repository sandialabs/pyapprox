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
    GalerkinSteadyModel,
)
from pyapprox.pde.galerkin.time_integration.physics_adapter import (
    GalerkinODEResidualProtocol,
    GalerkinPhysicsToODEResidualAdapter,
)
from pyapprox.pde.galerkin.time_integration.skfem_discretization import (
    skfem_discretization,
    skfem_ode_residual,
)
from pyapprox.pde.galerkin.time_integration.transient_model import (
    GalerkinTransientModel,
)

__all__ = [
    "GalerkinODEResidualProtocol",
    "GalerkinPhysicsToODEResidualAdapter",
    "skfem_discretization",
    "skfem_ode_residual",
    "GalerkinBCEnforcingForwardResidual",
    "GalerkinSteadyModel",
    "GalerkinTransientModel",
    "TimeIntegrationConfig",
    "create_galerkin_bc_enforcing_residual",
]
