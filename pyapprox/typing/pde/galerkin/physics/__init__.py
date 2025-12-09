"""Physics implementations for Galerkin finite element methods."""

from pyapprox.typing.pde.galerkin.physics.base import AbstractGalerkinPhysics
from pyapprox.typing.pde.galerkin.physics.advection_diffusion import (
    LinearAdvectionDiffusionReaction,
)
from pyapprox.typing.pde.galerkin.physics.helmholtz import Helmholtz

__all__ = [
    "AbstractGalerkinPhysics",
    "LinearAdvectionDiffusionReaction",
    "Helmholtz",
]
