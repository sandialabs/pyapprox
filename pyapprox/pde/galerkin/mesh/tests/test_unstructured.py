"""Tests for UnstructuredMesh2D."""

import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import os
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.pde.galerkin.basis import VectorLagrangeBasis
from pyapprox.pde.galerkin.mesh.unstructured import UnstructuredMesh2D
from pyapprox.pde.galerkin.protocols.mesh import GalerkinMeshProtocol
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array
from pyapprox.util.backends.torch import TorchBkd
# Path to the beam mesh JSON
_BEAM_MESH_PATH = os.path.join(
    os.path.expanduser("~"),
    "MrHyDE",
    "materials",
    "beam",
    "meshes",
    "composite_beam_2d_mi0_5_mo0_5.json",
)


class TestUnstructuredMesh2D:

    def setup_method(self):
        if not os.path.exists(_BEAM_MESH_PATH):
            self.skipTest(f"Beam mesh not found at {_BEAM_MESH_PATH}")

    def test_load_geometry(self, numpy_bkd):
        bkd = numpy_bkd
        mesh = UnstructuredMesh2D(_BEAM_MESH_PATH, bkd)
        assert mesh.nnodes() == 10673
        assert mesh.nelements() == 10262
        assert mesh.ndim() == 2

    def test_rescale_to_origin(self, numpy_bkd):
        bkd = numpy_bkd
        mesh = UnstructuredMesh2D(_BEAM_MESH_PATH, bkd, rescale_origin=(0.0, 0.0))
        nodes_np = bkd.to_numpy(mesh.nodes())
        x_min, y_min = nodes_np[0].min(), nodes_np[1].min()
        x_max, y_max = nodes_np[0].max(), nodes_np[1].max()
        bkd.assert_allclose(
            bkd.asarray([x_min, y_min]),
            bkd.asarray([0.0, 0.0]),
            atol=1e-12,
        )
        bkd.assert_allclose(
            bkd.asarray([x_max, y_max]),
            bkd.asarray([100.0, 30.0]),
            atol=1e-12,
        )

    def test_boundary_nodes_left_edge(self, numpy_bkd):
        bkd = numpy_bkd
        mesh = UnstructuredMesh2D(_BEAM_MESH_PATH, bkd, rescale_origin=(0.0, 0.0))
        left_nodes = mesh.boundary_nodes("left_edge")
        left_np = bkd.to_numpy(left_nodes)
        nodes_np = bkd.to_numpy(mesh.nodes())
        # All left_edge nodes should have x ≈ 0
        x_coords = nodes_np[0, left_np]
        bkd.assert_allclose(
            bkd.asarray(x_coords),
            bkd.asarray(np.zeros_like(x_coords)),
            atol=1e-10,
        )
        # Should span full height
        y_coords = nodes_np[1, left_np]
        bkd.assert_allclose(
            bkd.asarray([y_coords.min()]),
            bkd.asarray([0.0]),
            atol=1e-10,
        )
        bkd.assert_allclose(
            bkd.asarray([y_coords.max()]),
            bkd.asarray([30.0]),
            atol=1e-10,
        )

    def test_boundary_names(self, numpy_bkd):
        bkd = numpy_bkd
        mesh = UnstructuredMesh2D(_BEAM_MESH_PATH, bkd)
        names = mesh.boundary_names()
        assert "left_edge" in names
        assert "right_edge" in names
        assert "bottom_edge" in names
        assert "top_edge" in names

    def test_invalid_boundary_raises(self, numpy_bkd):
        bkd = numpy_bkd
        mesh = UnstructuredMesh2D(_BEAM_MESH_PATH, bkd)
        with pytest.raises(ValueError):
            mesh.boundary_nodes("nonexistent")

    def test_subdomain_elements(self, numpy_bkd):
        bkd = numpy_bkd
        mesh = UnstructuredMesh2D(_BEAM_MESH_PATH, bkd)
        top = mesh.subdomain_elements("top_layer")
        assert len(top) == 2000
        bottom = mesh.subdomain_elements("bottom_layer")
        assert len(bottom) == 2000
        core = mesh.subdomain_elements("inner_core")
        assert len(core) == 6262
        # All elements should be covered
        total = len(top) + len(bottom) + len(core)
        assert total == mesh.nelements()

    def test_subdomain_names(self, numpy_bkd):
        bkd = numpy_bkd
        mesh = UnstructuredMesh2D(_BEAM_MESH_PATH, bkd)
        names = mesh.subdomain_names()
        assert "top_layer" in names
        assert "bottom_layer" in names
        assert "inner_core" in names

    def test_invalid_subdomain_raises(self, numpy_bkd):
        bkd = numpy_bkd
        mesh = UnstructuredMesh2D(_BEAM_MESH_PATH, bkd)
        with pytest.raises(ValueError):
            mesh.subdomain_elements("nonexistent")

    def test_satisfies_galerkin_mesh_protocol(self, numpy_bkd):
        bkd = numpy_bkd
        mesh = UnstructuredMesh2D(_BEAM_MESH_PATH, bkd)
        assert isinstance(mesh, GalerkinMeshProtocol)

    def test_vector_lagrange_basis_construction(self, numpy_bkd):
        bkd = numpy_bkd
        mesh = UnstructuredMesh2D(_BEAM_MESH_PATH, bkd, rescale_origin=(0.0, 0.0))
        basis = VectorLagrangeBasis(mesh, degree=1)
        # 2D vector basis: 2 DOFs per node
        assert basis.ndofs() == 2 * mesh.nnodes()

    def test_elements_shape(self, numpy_bkd):
        bkd = numpy_bkd
        mesh = UnstructuredMesh2D(_BEAM_MESH_PATH, bkd)
        elems = mesh.elements()
        elems_np = bkd.to_numpy(elems)
        assert elems_np.shape == (4, 10262)

    def test_nodes_shape(self, numpy_bkd):
        bkd = numpy_bkd
        mesh = UnstructuredMesh2D(_BEAM_MESH_PATH, bkd)
        nodes = mesh.nodes()
        nodes_np = bkd.to_numpy(nodes)
        assert nodes_np.shape == (2, 10673)
