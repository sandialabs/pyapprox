"""Boundary condition implementations for Galerkin finite element methods.

This module provides concrete implementations of boundary conditions
that satisfy the protocols defined in protocols.boundary.
"""

from pyapprox.pde.galerkin.boundary.implementations import (
    CallableDirichletBC,
    DirichletBC,
    DirectDirichletBC,
    NeumannBC,
    RobinBC,
    BoundaryConditionSet,
)
from pyapprox.pde.galerkin.boundary.manufactured import (
    ManufacturedSolutionBC,
    canonical_boundary_normal,
)

__all__ = [
    "CallableDirichletBC",
    "DirichletBC",
    "DirectDirichletBC",
    "NeumannBC",
    "RobinBC",
    "BoundaryConditionSet",
    "ManufacturedSolutionBC",
    "canonical_boundary_normal",
]
