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

from pyapprox.pde.galerkin.postprocessing import (
    strain_from_displacement_2d,
    stress_from_strain_2d,
    von_mises_stress_2d,
)
from pyapprox.pde.galerkin.protocols import (
    # Boundary condition protocols
    BoundaryConditionProtocol,
    DirichletBCProtocol,
    # Basis protocols
    GalerkinBasisProtocol,
    # Mesh protocols
    GalerkinMeshProtocol,
    # Physics protocols
    GalerkinPhysicsProtocol,
    NeumannBCProtocol,
    RobinBCProtocol,
    StructuredMeshProtocol,
    VectorBasisProtocol,
)
from pyapprox.pde.galerkin.solvers import (
    SteadyStateSolver,
)
from pyapprox.pde.galerkin.time_integration import (
    GalerkinPhysicsODEAdapter,
)
from pyapprox.util.optional_deps import package_available

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
    # Time integration adapter
    "GalerkinPhysicsODEAdapter",
    # Solvers
    "SteadyStateSolver",
    # Postprocessing
    "von_mises_stress_2d",
    "strain_from_displacement_2d",
    "stress_from_strain_2d",
]

if package_available("skfem"):
    from pyapprox.pde.galerkin.basis import (
        LagrangeBasis,
        VectorLagrangeBasis,
    )
    from pyapprox.pde.galerkin.bilaplacian import BiLaplacianPrior
    from pyapprox.pde.galerkin.boundary import (
        CallableDirichletBC,
        DirectDirichletBC,
    )
    from pyapprox.pde.galerkin.mesh import (
        StructuredMesh1D,
        StructuredMesh2D,
        StructuredMesh3D,
        UnstructuredMesh2D,
    )
    from pyapprox.pde.galerkin.physics import (
        AdvectionDiffusionReaction,
        BurgersPhysics,
        CompositeHyperelasticityPhysics,
        CompositeLinearElasticity,
        EulerBernoulliBeamAnalytical,
        EulerBernoulliBeamFEM,
        GalerkinBCMixin,
        GalerkinPhysicsBase,
        Helmholtz,
        HyperelasticityPhysics,
        LinearAdvectionDiffusionReaction,
        LinearElasticity,
        ScalarMassAssembler,
        StokesPhysics,
    )

    __all__ += [
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
    ]
