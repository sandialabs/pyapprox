"""Time integration adapters for Galerkin physics.

This module provides adapters that allow Galerkin physics (with mass matrices)
to be used with the time steppers in typing.pde.time.
"""

from pyapprox.typing.pde.galerkin.time_integration.physics_adapter import (
    GalerkinPhysicsODEAdapter,
)
from pyapprox.typing.pde.galerkin.time_integration.explicit_adapter import (
    GalerkinExplicitODEAdapter,
)
from pyapprox.typing.pde.galerkin.time_integration.stokes_time_stepper import (
    StokesTimeStepResidual,
)
from pyapprox.typing.pde.galerkin.time_integration.galerkin_model import (
    GalerkinModel,
)
from pyapprox.typing.pde.time.config import TimeIntegrationConfig

__all__ = [
    "GalerkinPhysicsODEAdapter",
    "GalerkinExplicitODEAdapter",
    "StokesTimeStepResidual",
    "GalerkinModel",
    "TimeIntegrationConfig",
]
