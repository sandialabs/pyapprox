"""Mesh module for spectral collocation methods."""

from pyapprox.typing.pde.collocation.mesh.base import (
    MeshData,
    MeshDataWithTransform,
    compute_cartesian_product,
    compute_boundary_indices_1d,
    compute_boundary_indices_2d,
    compute_boundary_indices_3d,
)

__all__ = [
    "MeshData",
    "MeshDataWithTransform",
    "compute_cartesian_product",
    "compute_boundary_indices_1d",
    "compute_boundary_indices_2d",
    "compute_boundary_indices_3d",
]
