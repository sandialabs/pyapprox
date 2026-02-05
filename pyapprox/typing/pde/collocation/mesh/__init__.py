"""Mesh module for spectral collocation methods."""

from pyapprox.typing.pde.collocation.mesh.base import (
    MeshData,
    MeshDataWithTransform,
    compute_cartesian_product,
    compute_boundary_indices_1d,
    compute_boundary_indices_2d,
    compute_boundary_indices_3d,
)
from pyapprox.typing.pde.collocation.mesh.cartesian_1d import (
    CartesianMesh1D,
    create_uniform_mesh_1d,
)
from pyapprox.typing.pde.collocation.mesh.cartesian_2d import (
    CartesianMesh2D,
    create_uniform_mesh_2d,
)
from pyapprox.typing.pde.collocation.mesh.cartesian_3d import (
    CartesianMesh3D,
    create_uniform_mesh_3d,
)
from pyapprox.typing.pde.collocation.mesh.transforms import (
    AffineTransform1D,
    AffineTransform2D,
    AffineTransform3D,
)
from pyapprox.typing.pde.collocation.mesh.transformed import (
    TransformedMesh1D,
    TransformedMesh2D,
    TransformedMesh3D,
)

__all__ = [
    # Data structures
    "MeshData",
    "MeshDataWithTransform",
    # Utilities
    "compute_cartesian_product",
    "compute_boundary_indices_1d",
    "compute_boundary_indices_2d",
    "compute_boundary_indices_3d",
    # Mesh classes
    "CartesianMesh1D",
    "CartesianMesh2D",
    "CartesianMesh3D",
    # Factory functions
    "create_uniform_mesh_1d",
    "create_uniform_mesh_2d",
    "create_uniform_mesh_3d",
    # Transforms
    "AffineTransform1D",
    "AffineTransform2D",
    "AffineTransform3D",
    # Transformed meshes
    "TransformedMesh1D",
    "TransformedMesh2D",
    "TransformedMesh3D",
]
