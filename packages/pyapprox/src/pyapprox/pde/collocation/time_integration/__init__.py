"""Time integration bridge for spectral collocation methods."""

from pyapprox.pde.collocation.time_integration.bc_time_residual_adapter import (
    BCEnforcingAdjointResidual,
    BCEnforcingForwardResidual,
    BCEnforcingHVPResidual,
    create_bc_enforcing_residual,
)
from pyapprox.pde.collocation.time_integration.collocation_model import (
    CollocationModel,
    TimeIntegrationConfig,
)
from pyapprox.pde.collocation.time_integration.physics_adapter import (
    PhysicsToODEResidualAdapter,
)

__all__ = [
    "PhysicsToODEResidualAdapter",
    "TimeIntegrationConfig",
    "CollocationModel",
    "BCEnforcingForwardResidual",
    "BCEnforcingAdjointResidual",
    "BCEnforcingHVPResidual",
    "create_bc_enforcing_residual",
]
