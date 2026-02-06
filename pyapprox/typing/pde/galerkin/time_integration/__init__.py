"""Time integration adapters for Galerkin physics.

This module provides adapters that allow Galerkin physics (with mass matrices)
to be used with the time steppers in typing.pde.time.
"""

from pyapprox.typing.pde.galerkin.time_integration.physics_adapter import (
    GalerkinPhysicsODEAdapter,
)
from pyapprox.typing.pde.galerkin.time_integration.stokes_time_stepper import (
    StokesTimeStepResidual,
)

__all__ = [
    "GalerkinPhysicsODEAdapter",
    "StokesTimeStepResidual",
]
