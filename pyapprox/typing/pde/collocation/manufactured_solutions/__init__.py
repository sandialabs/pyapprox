"""Manufactured solutions for PDE verification.

This module provides manufactured solution classes for verifying
spectral collocation PDE solvers using the Method of Manufactured Solutions (MMS).
"""

from pyapprox.typing.pde.collocation.manufactured_solutions.base import (
    ManufacturedSolution,
    ScalarSolutionMixin,
    VectorSolutionMixin,
)
from pyapprox.typing.pde.collocation.manufactured_solutions.mixins import (
    DiffusionMixin,
    ReactionMixin,
    AdvectionMixin,
)
from pyapprox.typing.pde.collocation.manufactured_solutions.advection_diffusion import (
    ManufacturedAdvectionDiffusionReaction,
)
from pyapprox.typing.pde.collocation.manufactured_solutions.linear_elasticity import (
    ManufacturedLinearElasticityEquations,
)
from pyapprox.typing.pde.collocation.manufactured_solutions.helmholtz import (
    ManufacturedHelmholtz,
)
from pyapprox.typing.pde.collocation.manufactured_solutions.burgers import (
    ManufacturedBurgers1D,
)
from pyapprox.typing.pde.collocation.manufactured_solutions.shallow_ice import (
    ManufacturedShallowIce,
)
from pyapprox.typing.pde.collocation.manufactured_solutions.shallow_wave import (
    ManufacturedShallowWave,
)
from pyapprox.typing.pde.collocation.manufactured_solutions.reaction_diffusion import (
    ManufacturedTwoSpeciesReactionDiffusion,
)
from pyapprox.typing.pde.collocation.manufactured_solutions.shallow_shelf import (
    ManufacturedShallowShelfVelocityEquations,
    ManufacturedShallowShelfVelocityAndDepthEquations,
)
from pyapprox.typing.pde.collocation.manufactured_solutions.stokes import (
    ManufacturedStokes,
)
from pyapprox.typing.pde.collocation.manufactured_solutions.hyperelasticity import (
    ManufacturedHyperelasticityEquations,
)

__all__ = [
    # Base classes
    "ManufacturedSolution",
    "ScalarSolutionMixin",
    "VectorSolutionMixin",
    # Mixins
    "DiffusionMixin",
    "ReactionMixin",
    "AdvectionMixin",
    # Scalar manufactured solutions
    "ManufacturedAdvectionDiffusionReaction",
    "ManufacturedHelmholtz",
    "ManufacturedBurgers1D",
    "ManufacturedShallowIce",
    # Vector manufactured solutions
    "ManufacturedLinearElasticityEquations",
    "ManufacturedShallowWave",
    "ManufacturedTwoSpeciesReactionDiffusion",
    "ManufacturedShallowShelfVelocityEquations",
    "ManufacturedShallowShelfVelocityAndDepthEquations",
    "ManufacturedStokes",
    # Hyperelasticity
    "ManufacturedHyperelasticityEquations",
]
