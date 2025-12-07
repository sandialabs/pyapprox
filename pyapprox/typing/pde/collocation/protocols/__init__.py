"""Protocols for spectral collocation methods.

This module defines all protocol interfaces used in the collocation
discretization of PDEs.
"""

from pyapprox.typing.pde.collocation.protocols.mesh import (
    MeshProtocol,
    TransformProtocol,
    MeshWithTransformProtocol,
)
from pyapprox.typing.pde.collocation.protocols.basis import (
    BasisProtocol,
    BasisWithQuadratureProtocol,
)
from pyapprox.typing.pde.collocation.protocols.operators import (
    FieldProtocol,
    FieldWithJacobianProtocol,
    DifferentialOperatorProtocol,
)
from pyapprox.typing.pde.collocation.protocols.physics import (
    PhysicsProtocol,
    PhysicsWithParamJacobianProtocol,
    PhysicsWithHVPProtocol,
)
from pyapprox.typing.pde.collocation.protocols.solvers import (
    LinearSolverProtocol,
    IterativeSolverProtocol,
    MatrixFreeSolverProtocol,
    PreconditionerProtocol,
    PreconditionerWithSetupProtocol,
)
from pyapprox.typing.pde.collocation.protocols.boundary import (
    BoundaryConditionProtocol,
    BoundaryConditionWithParamJacobianProtocol,
    DirichletBCProtocol,
    RobinBCProtocol,
)

__all__ = [
    # Mesh
    "MeshProtocol",
    "TransformProtocol",
    "MeshWithTransformProtocol",
    # Basis
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
    "BoundaryConditionProtocol",
    "BoundaryConditionWithParamJacobianProtocol",
    "DirichletBCProtocol",
    "RobinBCProtocol",
]
