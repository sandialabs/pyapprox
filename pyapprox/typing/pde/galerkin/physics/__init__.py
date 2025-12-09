"""Physics implementations for Galerkin finite element methods."""

from pyapprox.typing.pde.galerkin.physics.base import AbstractGalerkinPhysics
from pyapprox.typing.pde.galerkin.physics.advection_diffusion import (
    LinearAdvectionDiffusionReaction,
)

__all__ = [
    "AbstractGalerkinPhysics",
    "LinearAdvectionDiffusionReaction",
]
