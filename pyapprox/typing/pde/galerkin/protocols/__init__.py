"""Protocols for Galerkin finite element methods.

This module defines all protocol interfaces used in the Galerkin
discretization of PDEs using scikit-fem.
"""

from pyapprox.typing.pde.galerkin.protocols.mesh import (
    GalerkinMeshProtocol,
    StructuredMeshProtocol,
)
from pyapprox.typing.pde.galerkin.protocols.basis import (
    GalerkinBasisProtocol,
    VectorBasisProtocol,
)
from pyapprox.typing.pde.galerkin.protocols.physics import (
    GalerkinPhysicsProtocol,
)
from pyapprox.typing.pde.galerkin.protocols.boundary import (
    BoundaryConditionProtocol,
    DirichletBCProtocol,
    NeumannBCProtocol,
    RobinBCProtocol,
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
]
