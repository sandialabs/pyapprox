"""Mesh implementations for Galerkin finite element methods."""

from pyapprox.typing.pde.galerkin.mesh.structured import (
    StructuredMesh1D,
    StructuredMesh2D,
    StructuredMesh3D,
)
from pyapprox.typing.pde.galerkin.mesh.unstructured import (
    UnstructuredMesh2D,
)
from pyapprox.typing.pde.galerkin.mesh.obstructed import (
    ObstructedMesh2D,
)

__all__ = [
    "StructuredMesh1D",
    "StructuredMesh2D",
    "StructuredMesh3D",
    "UnstructuredMesh2D",
    "ObstructedMesh2D",
]
