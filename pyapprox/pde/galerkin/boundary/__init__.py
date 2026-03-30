"""Boundary condition implementations for Galerkin finite element methods.

This module provides concrete implementations of boundary conditions
that satisfy the protocols defined in protocols.boundary.
"""

from pyapprox.util.optional_deps import package_available

__all__: list[str]

if package_available("skfem"):
    from pyapprox.pde.galerkin.boundary.implementations import (
        BoundaryConditionSet,
        CallableDirichletBC,
        DirectDirichletBC,
        DirichletBC,
        NeumannBC,
        RobinBC,
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
else:
    __all__ = []
