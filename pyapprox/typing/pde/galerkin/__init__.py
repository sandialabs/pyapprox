"""Galerkin finite element methods for PDEs.

This module provides a framework for solving PDEs using Galerkin finite
element methods with scikit-fem (skfem):
- Backend abstraction for NumPy/PyTorch compatibility
- Protocol-based interfaces for extensibility
- Support for 1D, 2D and 3D domains
- Integration with time stepping from typing.pde.time

The key difference from the collocation module is that Galerkin uses
the weak formulation with mass matrices: M*du/dt = F(u,t) instead of
the strong form du/dt = f(u,t).
"""

from pyapprox.typing.pde.galerkin.protocols import (
    # Mesh protocols
    GalerkinMeshProtocol,
    StructuredMeshProtocol,
    # Basis protocols
    GalerkinBasisProtocol,
    VectorBasisProtocol,
    # Physics protocols
    GalerkinPhysicsProtocol,
    # Boundary condition protocols
    BoundaryConditionProtocol,
    DirichletBCProtocol,
    NeumannBCProtocol,
    RobinBCProtocol,
)

from pyapprox.typing.pde.galerkin.mesh import (
    StructuredMesh1D,
    StructuredMesh2D,
    StructuredMesh3D,
    UnstructuredMesh2D,
)

from pyapprox.typing.pde.galerkin.basis import (
    LagrangeBasis,
    VectorLagrangeBasis,
)

from pyapprox.typing.pde.galerkin.physics import (
    GalerkinBCMixin,
    GalerkinPhysicsBase,
    ScalarMassAssembler,
    AdvectionDiffusionReaction,
    LinearAdvectionDiffusionReaction,
    BurgersPhysics,
    Helmholtz,
    CompositeLinearElasticity,
    LinearElasticity,
    HyperelasticityPhysics,
    CompositeHyperelasticityPhysics,
    EulerBernoulliBeamAnalytical,
    EulerBernoulliBeamFEM,
    StokesPhysics,
)

from pyapprox.typing.pde.galerkin.bilaplacian import BiLaplacianPrior

from pyapprox.typing.pde.galerkin.boundary import (
    DirectDirichletBC,
    CallableDirichletBC,
)

from pyapprox.typing.pde.galerkin.time_integration import (
    GalerkinPhysicsODEAdapter,
)

from pyapprox.typing.pde.galerkin.solvers import (
    SteadyStateSolver,
)

__all__ = [
    # Mesh protocols
    "GalerkinMeshProtocol",
    "StructuredMeshProtocol",
    # Basis protocols
    "GalerkinBasisProtocol",
    "VectorBasisProtocol",
    # Physics protocols
    "GalerkinPhysicsProtocol",
    # Boundary condition protocols
    "BoundaryConditionProtocol",
    "DirichletBCProtocol",
    "NeumannBCProtocol",
    "RobinBCProtocol",
    # Mesh implementations
    "StructuredMesh1D",
    "StructuredMesh2D",
    "StructuredMesh3D",
    "UnstructuredMesh2D",
    # Basis implementations
    "LagrangeBasis",
    "VectorLagrangeBasis",
    # Physics infrastructure
    "GalerkinBCMixin",
    "GalerkinPhysicsBase",
    "ScalarMassAssembler",
    # Physics implementations
    "AdvectionDiffusionReaction",
    "LinearAdvectionDiffusionReaction",
    "BurgersPhysics",
    "Helmholtz",
    "CompositeLinearElasticity",
    "LinearElasticity",
    "HyperelasticityPhysics",
    "CompositeHyperelasticityPhysics",
    "EulerBernoulliBeamAnalytical",
    "EulerBernoulliBeamFEM",
    "StokesPhysics",
    "BiLaplacianPrior",
    # Boundary condition implementations
    "DirectDirichletBC",
    "CallableDirichletBC",
    # Time integration adapter
    "GalerkinPhysicsODEAdapter",
    # Solvers
    "SteadyStateSolver",
]
