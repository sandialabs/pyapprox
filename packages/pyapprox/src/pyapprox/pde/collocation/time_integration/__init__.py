"""Time integration bridge for spectral collocation methods.

The parameterized adapter tiers and their factory live in
``pyapprox.pde.models.collocation``.
"""

from pyapprox.ode.config import TimeIntegrationConfig
from pyapprox.pde.collocation.time_integration.bc_time_residual_adapter import (
    BCEnforcingAdjointResidual,
    BCEnforcingForwardResidual,
    BCEnforcingHVPResidual,
    create_bc_enforcing_residual,
)
from pyapprox.pde.collocation.time_integration.collocation_model import (
    CollocationModel,
)
from pyapprox.pde.collocation.time_integration.physics_adapter import (
    CollocationPhysicsToODEResidualAdapter,
)

__all__ = [
    "CollocationPhysicsToODEResidualAdapter",
    "TimeIntegrationConfig",
    "CollocationModel",
    "BCEnforcingForwardResidual",
    "BCEnforcingAdjointResidual",
    "BCEnforcingHVPResidual",
    "create_bc_enforcing_residual",
]
