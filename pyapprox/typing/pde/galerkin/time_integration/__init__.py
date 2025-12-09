"""Time integration adapters for Galerkin physics.

This module provides adapters that allow Galerkin physics (with mass matrices)
to be used with the time steppers in typing.pde.time.
"""

from pyapprox.typing.pde.galerkin.time_integration.physics_adapter import (
    GalerkinPhysicsODEAdapter,
)

__all__ = [
    "GalerkinPhysicsODEAdapter",
]
