"""Protocols for spectral collocation methods.

This module defines all protocol interfaces used in the collocation
discretization of PDEs.
"""

from pyapprox.pde.collocation.protocols.mesh import (
    MeshProtocol,
    TransformProtocol,
    MeshWithTransformProtocol,
)
from pyapprox.pde.collocation.protocols.basis import (
    NodesGenerator1DProtocol,
    DerivativeMatrix1DProtocol,
    TensorProductBasisProtocol,
    BasisProtocol,
    BasisWithQuadratureProtocol,
)
from pyapprox.pde.collocation.protocols.operators import (
    FieldProtocol,
    FieldWithJacobianProtocol,
    DifferentialOperatorProtocol,
)
from pyapprox.pde.collocation.protocols.physics import (
    PhysicsProtocol,
    PhysicsWithParamJacobianProtocol,
    PhysicsWithHVPProtocol,
)
from pyapprox.optimization.linear_solvers.protocols import (
    LinearSolverProtocol,
    IterativeSolverProtocol,
    MatrixFreeSolverProtocol,
    PreconditionerProtocol,
    PreconditionerWithSetupProtocol,
)
from pyapprox.pde.collocation.protocols.boundary import (
    BCDofClassification,
    BoundaryConditionProtocol,
    BoundaryConditionWithParamJacobianProtocol,
    DirichletBCProtocol,
    RobinBCProtocol,
    NormalOperatorProtocol,
    FluxProviderProtocol,
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
    # Physics (3-level hierarchy)
    "PhysicsProtocol",
    "PhysicsWithParamJacobianProtocol",
    "PhysicsWithHVPProtocol",
    # Solvers
    "LinearSolverProtocol",
    "IterativeSolverProtocol",
    "MatrixFreeSolverProtocol",
    "PreconditionerProtocol",
    "PreconditionerWithSetupProtocol",
    # Boundary conditions
    "BCDofClassification",
    "BoundaryConditionProtocol",
    "BoundaryConditionWithParamJacobianProtocol",
    "DirichletBCProtocol",
    "RobinBCProtocol",
    "NormalOperatorProtocol",
    "FluxProviderProtocol",
]
