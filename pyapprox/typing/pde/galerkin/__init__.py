"""Galerkin finite element methods for PDEs.

This module provides a framework for solving PDEs using Galerkin finite
element methods with scikit-fem (skfem):
- Backend abstraction for NumPy/PyTorch compatibility
- Protocol-based interfaces for extensibility
- Support for 1D and 2D domains
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
    # Physics protocols (3-level hierarchy)
    GalerkinPhysicsProtocol,
    GalerkinPhysicsWithParamJacobianProtocol,
    GalerkinPhysicsWithHVPProtocol,
    # Boundary condition protocols
    BoundaryConditionProtocol,
    DirichletBCProtocol,
    NeumannBCProtocol,
    RobinBCProtocol,
)

from pyapprox.typing.pde.galerkin.mesh import (
    StructuredMesh1D,
    StructuredMesh2D,
)

from pyapprox.typing.pde.galerkin.basis import (
    LagrangeBasis,
)

from pyapprox.typing.pde.galerkin.physics import (
    AbstractGalerkinPhysics,
    LinearAdvectionDiffusionReaction,
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
    # Physics protocols (3-level hierarchy)
    "GalerkinPhysicsProtocol",
    "GalerkinPhysicsWithParamJacobianProtocol",
    "GalerkinPhysicsWithHVPProtocol",
    # Boundary condition protocols
    "BoundaryConditionProtocol",
    "DirichletBCProtocol",
    "NeumannBCProtocol",
    "RobinBCProtocol",
    # Mesh implementations
    "StructuredMesh1D",
    "StructuredMesh2D",
    # Basis implementations
    "LagrangeBasis",
    # Physics implementations
    "AbstractGalerkinPhysics",
    "LinearAdvectionDiffusionReaction",
    # Time integration adapter
    "GalerkinPhysicsODEAdapter",
    # Solvers
    "SteadyStateSolver",
]
