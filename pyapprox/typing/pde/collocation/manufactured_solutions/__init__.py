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

__all__ = [
    # Base classes
    "ManufacturedSolution",
    "ScalarSolutionMixin",
    "VectorSolutionMixin",
    # Mixins
    "DiffusionMixin",
    "ReactionMixin",
    "AdvectionMixin",
    # Manufactured solutions
    "ManufacturedAdvectionDiffusionReaction",
    "ManufacturedLinearElasticityEquations",
]
