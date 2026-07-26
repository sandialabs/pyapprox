"""Protocols for spectral collocation methods.

This module defines all protocol interfaces used in the collocation
discretization of PDEs.
"""

from pyapprox.optimization.linear_solvers.protocols import (
    IterativeSolverProtocol,
    LinearSolverProtocol,
    MatrixFreeSolverProtocol,
    PreconditionerProtocol,
    PreconditionerWithSetupProtocol,
)
from pyapprox.pde.collocation.protocols.basis import (
    BasisProtocol,
    BasisWithQuadratureProtocol,
    DerivativeMatrix1DProtocol,
    NodesGenerator1DProtocol,
    TensorProductBasisProtocol,
)
from pyapprox.pde.collocation.protocols.boundary import (
    BCPhysicalSensitivities,
    BoundaryConditionProtocol,
    BoundaryConditionWithNormalOperatorProtocol,
    BoundaryConditionWithParamJacobianProtocol,
    DirichletBCProtocol,
    FluxProviderProtocol,
    NormalOperatorProtocol,
    RobinBCProtocol,
)
from pyapprox.pde.collocation.protocols.mesh import (
    MeshProtocol,
    MeshWithTransformProtocol,
    TransformProtocol,
)
from pyapprox.pde.collocation.protocols.operators import (
    DifferentialOperatorProtocol,
    FieldProtocol,
    FieldWithJacobianProtocol,
)
from pyapprox.pde.collocation.protocols.physics import (
    PhysicsProtocol,
    PhysicsWithStateStateHVPProtocol,
)

__all__ = [
    # Mesh
    "MeshProtocol",
    "TransformProtocol",
    "MeshWithTransformProtocol",
    # Basis (extensibility protocols)
    "NodesGenerator1DProtocol",
    "DerivativeMatrix1DProtocol",
    "TensorProductBasisProtocol",
    # Basis (high-level)
    "BasisProtocol",
    "BasisWithQuadratureProtocol",
    # Operators
    "FieldProtocol",
    "FieldWithJacobianProtocol",
    "DifferentialOperatorProtocol",
    # Physics
    "PhysicsProtocol",
    "PhysicsWithStateStateHVPProtocol",
    # Parameterization
    # Solvers
    "LinearSolverProtocol",
    "IterativeSolverProtocol",
    "MatrixFreeSolverProtocol",
    "PreconditionerProtocol",
    "PreconditionerWithSetupProtocol",
    # Boundary conditions
    "BCPhysicalSensitivities",
    "BoundaryConditionProtocol",
    "BoundaryConditionWithNormalOperatorProtocol",
    "BoundaryConditionWithParamJacobianProtocol",
    "DirichletBCProtocol",
    "RobinBCProtocol",
    "NormalOperatorProtocol",
    "FluxProviderProtocol",
]
