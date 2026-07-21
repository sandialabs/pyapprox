"""Manufactured solutions for PDE verification.

This module provides manufactured solution classes for verifying
spectral collocation PDE solvers using the Method of Manufactured Solutions (MMS).
"""

from pyapprox.pde.manufactured.advection_diffusion import (
    ManufacturedAdvectionDiffusionReaction,
)
from pyapprox.pde.manufactured.base import (
    ManufacturedSolution,
    ScalarSolutionMixin,
    VectorSolutionMixin,
)
from pyapprox.pde.manufactured.burgers import (
    ManufacturedBurgers1D,
)
from pyapprox.pde.manufactured.helmholtz import (
    ManufacturedHelmholtz,
)
from pyapprox.pde.manufactured.hyperelasticity import (
    ManufacturedHyperelasticityEquations,
)
from pyapprox.pde.manufactured.linear_elasticity import (
    ManufacturedLinearElasticityEquations,
)
from pyapprox.pde.manufactured.mixins import (
    AdvectionMixin,
    DiffusionMixin,
    ReactionMixin,
)
from pyapprox.pde.manufactured.reaction_diffusion import (
    ManufacturedTwoSpeciesReactionDiffusion,
)
from pyapprox.pde.manufactured.shallow_ice import (
    ManufacturedShallowIce,
)
from pyapprox.pde.manufactured.shallow_shelf import (
    ManufacturedShallowShelfVelocityAndDepthEquations,
    ManufacturedShallowShelfVelocityEquations,
)
from pyapprox.pde.manufactured.shallow_wave import (
    ManufacturedShallowWave,
)
from pyapprox.pde.manufactured.stokes import (
    ManufacturedStokes,
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
