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
    BCDofClassification,
    BoundaryConditionProtocol,
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
    PhysicsWithHVPProtocol,
    PhysicsWithParamJacobianProtocol,
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
