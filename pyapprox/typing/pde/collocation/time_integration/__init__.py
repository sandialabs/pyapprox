"""Time integration bridge for spectral collocation methods."""

from pyapprox.typing.pde.collocation.time_integration.physics_adapter import (
    PhysicsToODEResidualAdapter,
)
from pyapprox.typing.pde.collocation.time_integration.collocation_model import (
    TimeIntegrationConfig,
    CollocationModel,
)

__all__ = [
    "PhysicsToODEResidualAdapter",
    "TimeIntegrationConfig",
    "CollocationModel",
]
