"""Time integration bridge for spectral collocation methods."""

from pyapprox.typing.pde.collocation.time_integration.physics_adapter import (
    PhysicsToODEResidualAdapter,
)
from pyapprox.typing.pde.collocation.time_integration.collocation_model import (
    TimeIntegrationConfig,
    CollocationModel,
)
from pyapprox.typing.pde.collocation.time_integration.bc_time_residual_adapter import (
    BCEnforcingTimeResidual,
)

__all__ = [
    "PhysicsToODEResidualAdapter",
    "TimeIntegrationConfig",
    "CollocationModel",
    "BCEnforcingTimeResidual",
]
