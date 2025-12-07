"""Spectral collocation methods for PDEs.

This module provides a framework for solving PDEs using spectral
collocation methods with:
- Backend abstraction for NumPy/PyTorch compatibility
- Protocol-based interfaces for extensibility
- Support for 1D, 2D, and 3D domains
- Integration with time stepping from typing.pde.time
"""

from pyapprox.typing.pde.collocation.protocols import (
    MeshProtocol,
    TransformProtocol,
    MeshWithTransformProtocol,
    BasisProtocol,
    BasisWithQuadratureProtocol,
    FieldProtocol,
    FieldWithJacobianProtocol,
    PhysicsProtocol,
    PhysicsWithParamJacobianProtocol,
    PhysicsWithHVPProtocol,
    LinearSolverProtocol,
    IterativeSolverProtocol,
    BoundaryConditionProtocol,
)

__all__ = [
    # Mesh protocols
    "MeshProtocol",
    "TransformProtocol",
    "MeshWithTransformProtocol",
    # Basis protocols
    "BasisProtocol",
    "BasisWithQuadratureProtocol",
    # Operator protocols
    "FieldProtocol",
    "FieldWithJacobianProtocol",
    # Physics protocols (3-level hierarchy)
    "PhysicsProtocol",
    "PhysicsWithParamJacobianProtocol",
    "PhysicsWithHVPProtocol",
    # Solver protocols
    "LinearSolverProtocol",
    "IterativeSolverProtocol",
    # Boundary condition protocols
    "BoundaryConditionProtocol",
]
