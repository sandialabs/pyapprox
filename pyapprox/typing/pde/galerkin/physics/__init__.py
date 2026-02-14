"""Physics implementations for Galerkin finite element methods."""

from pyapprox.typing.pde.galerkin.physics.base import AbstractGalerkinPhysics
from pyapprox.typing.pde.galerkin.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
    LinearAdvectionDiffusionReaction,
)
from pyapprox.typing.pde.galerkin.physics.bilaplacian import BiLaplacianPrior
from pyapprox.typing.pde.galerkin.physics.burgers import BurgersPhysics
from pyapprox.typing.pde.galerkin.physics.helmholtz import Helmholtz
from pyapprox.typing.pde.galerkin.physics.hyperelasticity import (
    HyperelasticityPhysics,
)
from pyapprox.typing.pde.galerkin.physics.linear_elasticity import LinearElasticity
from pyapprox.typing.pde.galerkin.physics.stokes import StokesPhysics

__all__ = [
    "AbstractGalerkinPhysics",
    "AdvectionDiffusionReaction",
    "BiLaplacianPrior",
    "BurgersPhysics",
    "HyperelasticityPhysics",
    "LinearAdvectionDiffusionReaction",
    "Helmholtz",
    "LinearElasticity",
    "StokesPhysics",
]
