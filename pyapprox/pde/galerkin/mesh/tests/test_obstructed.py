"""Tests for ObstructedMesh2D."""

import unittest

import numpy as np

from pyapprox.pde.galerkin.mesh.obstructed import ObstructedMesh2D
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401


class TestObstructedMesh2D(unittest.TestCase):
    """Tests for ObstructedMesh2D (NumPy only — skfem is NumPy-based)."""

    def setUp(self):
        self._bkd = NumpyBkd()
        self._xintervals = np.array([0, 2 / 7, 3 / 7, 4 / 7, 5 / 7, 1.0])
        self._yintervals = np.linspace(0, 1, 5)
        self._obstruction_indices = np.array([3, 6, 13], dtype=int)

    def _create_mesh(self, nrefine=0):
        return ObstructedMesh2D(
            self._xintervals,
            self._yintervals,
            self._obstruction_indices,
            self._bkd,
            nrefine=nrefine,
        )

    def test_construction(self):
        """Test mesh can be constructed."""
        mesh = self._create_mesh()
        self.assertIsNotNone(mesh)

    def test_ndim(self):
        """Test mesh has 2 spatial dimensions."""
        mesh = self._create_mesh()
        self.assertEqual(mesh.ndim(), 2)

    def test_element_count(self):
        """Test correct number of elements after obstruction removal.

        Full grid: (6-1) * (5-1) = 20 cells.
        With 3 obstructions removed: 17 elements.
        """
        mesh = self._create_mesh()
        self.assertEqual(mesh.nelements(), 17)

    def test_node_count(self):
        """Test correct number of nodes.

        Full grid: 6 * 5 = 30 vertices.
        """
        mesh = self._create_mesh()
        self.assertEqual(mesh.nnodes(), 30)

    def test_nodes_shape(self):
        """Test nodes array has shape (2, nnodes)."""
        mesh = self._create_mesh()
        nodes = mesh.nodes()
        self.assertEqual(nodes.shape, (2, mesh.nnodes()))

    def test_elements_shape(self):
        """Test elements array has shape (4, nelements)."""
        mesh = self._create_mesh()
        elems = mesh.elements()
        self.assertEqual(elems.shape, (4, mesh.nelements()))

    def test_nodes_in_domain(self):
        """Test all nodes are within [0,1] x [0,1]."""
        mesh = self._create_mesh()
        nodes_np = self._bkd.to_numpy(mesh.nodes())
        self.assertTrue(np.all(nodes_np[0, :] >= -1e-12))
        self.assertTrue(np.all(nodes_np[0, :] <= 1.0 + 1e-12))
        self.assertTrue(np.all(nodes_np[1, :] >= -1e-12))
        self.assertTrue(np.all(nodes_np[1, :] <= 1.0 + 1e-12))

    def test_boundary_names(self):
        """Test all expected boundary names are present."""
        mesh = self._create_mesh()
        names = mesh.boundary_names()
        for name in ["left", "right", "bottom", "top", "obs0", "obs1", "obs2"]:
            self.assertIn(name, names)

    def test_boundary_nodes_left(self):
        """Test left boundary nodes have x=0."""
        mesh = self._create_mesh()
        left_nodes = self._bkd.to_numpy(mesh.boundary_nodes("left"))
        all_nodes = self._bkd.to_numpy(mesh.nodes())
        for idx in left_nodes:
            self.assertAlmostEqual(all_nodes[0, idx], 0.0, places=10)

    def test_boundary_nodes_right(self):
        """Test right boundary nodes have x=1."""
        mesh = self._create_mesh()
        right_nodes = self._bkd.to_numpy(mesh.boundary_nodes("right"))
        all_nodes = self._bkd.to_numpy(mesh.nodes())
        for idx in right_nodes:
            self.assertAlmostEqual(all_nodes[0, idx], 1.0, places=10)

    def test_boundary_nodes_bottom(self):
        """Test bottom boundary nodes have y=0."""
        mesh = self._create_mesh()
        bottom_nodes = self._bkd.to_numpy(mesh.boundary_nodes("bottom"))
        all_nodes = self._bkd.to_numpy(mesh.nodes())
        for idx in bottom_nodes:
            self.assertAlmostEqual(all_nodes[1, idx], 0.0, places=10)

    def test_boundary_nodes_top(self):
        """Test top boundary nodes have y=1."""
        mesh = self._create_mesh()
        top_nodes = self._bkd.to_numpy(mesh.boundary_nodes("top"))
        all_nodes = self._bkd.to_numpy(mesh.nodes())
        for idx in top_nodes:
            self.assertAlmostEqual(all_nodes[1, idx], 1.0, places=10)

    def test_refinement_increases_elements(self):
        """Test refinement increases number of elements."""
        mesh0 = self._create_mesh(nrefine=0)
        mesh1 = self._create_mesh(nrefine=1)
        self.assertGreater(mesh1.nelements(), mesh0.nelements())

    def test_refinement_increases_nodes(self):
        """Test refinement increases number of nodes."""
        mesh0 = self._create_mesh(nrefine=0)
        mesh1 = self._create_mesh(nrefine=1)
        self.assertGreater(mesh1.nnodes(), mesh0.nnodes())

    def test_skfem_mesh_accessible(self):
        """Test skfem mesh object is accessible."""
        mesh = self._create_mesh()
        sk_mesh = mesh.skfem_mesh()
        self.assertIsNotNone(sk_mesh)
        self.assertEqual(sk_mesh.nelements, mesh.nelements())

    def test_invalid_boundary_raises(self):
        """Test invalid boundary name raises ValueError."""
        mesh = self._create_mesh()
        with self.assertRaises(ValueError):
            mesh.boundary_nodes("invalid_name")

    def test_obstruction_nodes_within_cell(self):
        """Test obstacle boundary nodes lie within the obstruction cell."""
        mesh = self._create_mesh()
        all_nodes = self._bkd.to_numpy(mesh.nodes())
        obs0_nodes = self._bkd.to_numpy(mesh.boundary_nodes("obs0"))
        self.assertGreater(len(obs0_nodes), 0)
        # obs0 corresponds to obstruction_indices[0]=3
        # which is row 0 (y-index 0), col 3 (x-index 3)
        # Cell 3: x in [4/7, 5/7], y in [0, 1/4]
        for idx in obs0_nodes:
            x, y = all_nodes[0, idx], all_nodes[1, idx]
            self.assertGreaterEqual(x, 4 / 7 - 1e-8)
            self.assertLessEqual(x, 5 / 7 + 1e-8)
            self.assertGreaterEqual(y, 0.0 - 1e-8)
            self.assertLessEqual(y, 0.25 + 1e-8)

    def test_repr(self):
        """Test repr string is informative."""
        mesh = self._create_mesh()
        r = repr(mesh)
        self.assertIn("ObstructedMesh2D", r)
        self.assertIn("nobstructions=3", r)


if __name__ == "__main__":
    unittest.main()
