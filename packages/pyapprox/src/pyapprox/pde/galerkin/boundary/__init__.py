"""Boundary condition implementations for Galerkin finite element methods.

This module provides concrete implementations of boundary conditions
that satisfy the role protocols in ``pyapprox.pde.boundary``, and the
pointwise flux laws that define their natural-boundary data.
"""

from pyapprox.pde.galerkin.boundary.flux_law import (
    AdvectiveFlux,
    DiffusiveFlux,
    FluxLawProviderProtocol,
    LinearElasticTraction,
    NeoHookeanTraction,
    NormalFluxLawProtocol,
    PK1Traction,
    PointwiseField,
    SumFlux,
)
from pyapprox.util.optional_deps import package_available

__all__: list[str] = [
    "AdvectiveFlux",
    "DiffusiveFlux",
    "FluxLawProviderProtocol",
    "LinearElasticTraction",
    "NeoHookeanTraction",
    "NormalFluxLawProtocol",
    "PK1Traction",
    "PointwiseField",
    "SumFlux",
]

if package_available("skfem"):
    from pyapprox.pde.galerkin.boundary.implementations import (
        BoundaryConditionSet,
        CallableDirichletBC,
        DirectDirichletBC,
        DirichletBC,
        NeumannBC,
        NormalDependentDataProtocol,
        RobinBC,
    )

    __all__ += [
        "CallableDirichletBC",
        "DirichletBC",
        "DirectDirichletBC",
        "NeumannBC",
        "NormalDependentDataProtocol",
        "RobinBC",
        "BoundaryConditionSet",
    ]
