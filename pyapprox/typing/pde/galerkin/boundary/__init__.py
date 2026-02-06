"""Boundary condition implementations for Galerkin finite element methods.

This module provides concrete implementations of boundary conditions
that satisfy the protocols defined in protocols.boundary.
"""

from pyapprox.typing.pde.galerkin.boundary.implementations import (
    DirichletBC,
    NeumannBC,
    RobinBC,
    BoundaryConditionSet,
)
from pyapprox.typing.pde.galerkin.boundary.manufactured import (
    ManufacturedSolutionBC,
    canonical_boundary_normal,
)

__all__ = [
    "DirichletBC",
    "NeumannBC",
    "RobinBC",
    "BoundaryConditionSet",
    "ManufacturedSolutionBC",
    "canonical_boundary_normal",
]
