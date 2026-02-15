"""Tests for UnstructuredMesh2D."""

import os
import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.pde.galerkin.mesh.unstructured import UnstructuredMesh2D
from pyapprox.typing.pde.galerkin.basis import VectorLagrangeBasis
from pyapprox.typing.pde.galerkin.protocols.mesh import GalerkinMeshProtocol
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401

# Path to the beam mesh JSON
_BEAM_MESH_PATH = os.path.join(
    os.path.expanduser("~"),
    "MrHyDE",
    "materials",
    "beam",
    "meshes",
    "composite_beam_2d_mi0_5_mo0_5.json",
)


class TestUnstructuredMesh2D(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        if not os.path.exists(_BEAM_MESH_PATH):
            self.skipTest(f"Beam mesh not found at {_BEAM_MESH_PATH}")

    def test_load_geometry(self):
        mesh = UnstructuredMesh2D(_BEAM_MESH_PATH, self._bkd)
        self.assertEqual(mesh.nnodes(), 10673)
        self.assertEqual(mesh.nelements(), 10262)
        self.assertEqual(mesh.ndim(), 2)

    def test_rescale_to_origin(self):
        mesh = UnstructuredMesh2D(
            _BEAM_MESH_PATH, self._bkd, rescale_origin=(0.0, 0.0)
        )
        nodes_np = self._bkd.to_numpy(mesh.nodes())
        x_min, y_min = nodes_np[0].min(), nodes_np[1].min()
        x_max, y_max = nodes_np[0].max(), nodes_np[1].max()
        self._bkd.assert_allclose(
            self._bkd.asarray([x_min, y_min]),
            self._bkd.asarray([0.0, 0.0]),
            atol=1e-12,
        )
        self._bkd.assert_allclose(
            self._bkd.asarray([x_max, y_max]),
            self._bkd.asarray([100.0, 30.0]),
            atol=1e-12,
        )

    def test_boundary_nodes_left_edge(self):
        mesh = UnstructuredMesh2D(
            _BEAM_MESH_PATH, self._bkd, rescale_origin=(0.0, 0.0)
        )
        left_nodes = mesh.boundary_nodes("left_edge")
        left_np = self._bkd.to_numpy(left_nodes)
        nodes_np = self._bkd.to_numpy(mesh.nodes())
        # All left_edge nodes should have x ≈ 0
        x_coords = nodes_np[0, left_np]
        self._bkd.assert_allclose(
            self._bkd.asarray(x_coords),
            self._bkd.asarray(np.zeros_like(x_coords)),
            atol=1e-10,
        )
        # Should span full height
        y_coords = nodes_np[1, left_np]
        self._bkd.assert_allclose(
            self._bkd.asarray([y_coords.min()]),
            self._bkd.asarray([0.0]),
            atol=1e-10,
        )
        self._bkd.assert_allclose(
            self._bkd.asarray([y_coords.max()]),
            self._bkd.asarray([30.0]),
            atol=1e-10,
        )

    def test_boundary_names(self):
        mesh = UnstructuredMesh2D(_BEAM_MESH_PATH, self._bkd)
        names = mesh.boundary_names()
        self.assertIn("left_edge", names)
        self.assertIn("right_edge", names)
        self.assertIn("bottom_edge", names)
        self.assertIn("top_edge", names)

    def test_invalid_boundary_raises(self):
        mesh = UnstructuredMesh2D(_BEAM_MESH_PATH, self._bkd)
        with self.assertRaises(ValueError):
            mesh.boundary_nodes("nonexistent")

    def test_subdomain_elements(self):
        mesh = UnstructuredMesh2D(_BEAM_MESH_PATH, self._bkd)
        top = mesh.subdomain_elements("top_layer")
        self.assertEqual(len(top), 2000)
        bottom = mesh.subdomain_elements("bottom_layer")
        self.assertEqual(len(bottom), 2000)
        core = mesh.subdomain_elements("inner_core")
        self.assertEqual(len(core), 6262)
        # All elements should be covered
        total = len(top) + len(bottom) + len(core)
        self.assertEqual(total, mesh.nelements())

    def test_subdomain_names(self):
        mesh = UnstructuredMesh2D(_BEAM_MESH_PATH, self._bkd)
        names = mesh.subdomain_names()
        self.assertIn("top_layer", names)
        self.assertIn("bottom_layer", names)
        self.assertIn("inner_core", names)

    def test_invalid_subdomain_raises(self):
        mesh = UnstructuredMesh2D(_BEAM_MESH_PATH, self._bkd)
        with self.assertRaises(ValueError):
            mesh.subdomain_elements("nonexistent")

    def test_satisfies_galerkin_mesh_protocol(self):
        mesh = UnstructuredMesh2D(_BEAM_MESH_PATH, self._bkd)
        self.assertIsInstance(mesh, GalerkinMeshProtocol)

    def test_vector_lagrange_basis_construction(self):
        mesh = UnstructuredMesh2D(
            _BEAM_MESH_PATH, self._bkd, rescale_origin=(0.0, 0.0)
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        # 2D vector basis: 2 DOFs per node
        self.assertEqual(basis.ndofs(), 2 * mesh.nnodes())

    def test_elements_shape(self):
        mesh = UnstructuredMesh2D(_BEAM_MESH_PATH, self._bkd)
        elems = mesh.elements()
        elems_np = self._bkd.to_numpy(elems)
        self.assertEqual(elems_np.shape, (4, 10262))

    def test_nodes_shape(self):
        mesh = UnstructuredMesh2D(_BEAM_MESH_PATH, self._bkd)
        nodes = mesh.nodes()
        nodes_np = self._bkd.to_numpy(nodes)
        self.assertEqual(nodes_np.shape, (2, 10673))


class TestUnstructuredMesh2DNumpy(TestUnstructuredMesh2D[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestUnstructuredMesh2DTorch(TestUnstructuredMesh2D[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()
