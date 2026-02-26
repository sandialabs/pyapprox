"""Tests for ObstructedMesh2D."""


import pytest
import numpy as np

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.pde.galerkin.mesh.obstructed import ObstructedMesh2D
class TestObstructedMesh2D:
    """Tests for ObstructedMesh2D (NumPy only — skfem is NumPy-based)."""

    def setup_method(self):
        bkd = NumpyBkd()
        self._xintervals = np.array([0, 2 / 7, 3 / 7, 4 / 7, 5 / 7, 1.0])
        self._yintervals = np.linspace(0, 1, 5)
        self._obstruction_indices = np.array([3, 6, 13], dtype=int)

    def _create_mesh(self, bkd, nrefine=0) :
        return ObstructedMesh2D(
            self._xintervals,
            self._yintervals,
            self._obstruction_indices,
            bkd,
            nrefine=nrefine,
        )

    def test_construction(self, numpy_bkd):
        """Test mesh can be constructed."""
        bkd = numpy_bkd
        mesh = self._create_mesh(bkd)
        assert mesh is not None

    def test_ndim(self, numpy_bkd):
        """Test mesh has 2 spatial dimensions."""
        bkd = numpy_bkd
        mesh = self._create_mesh(bkd)
        assert mesh.ndim() == 2

    def test_element_count(self, numpy_bkd):
        """Test correct number of elements after obstruction removal.

        Full grid: (6-1) * (5-1) = 20 cells.
        With 3 obstructions removed: 17 elements.
        """
        bkd = numpy_bkd
        mesh = self._create_mesh(bkd)
        assert mesh.nelements() == 17

    def test_node_count(self, numpy_bkd):
        """Test correct number of nodes.

        Full grid: 6 * 5 = 30 vertices.
        """
        bkd = numpy_bkd
        mesh = self._create_mesh(bkd)
        assert mesh.nnodes() == 30

    def test_nodes_shape(self, numpy_bkd):
        """Test nodes array has shape (2, nnodes)."""
        bkd = numpy_bkd
        mesh = self._create_mesh(bkd)
        nodes = mesh.nodes()
        assert nodes.shape == (2, mesh.nnodes())

    def test_elements_shape(self, numpy_bkd):
        """Test elements array has shape (4, nelements)."""
        bkd = numpy_bkd
        mesh = self._create_mesh(bkd)
        elems = mesh.elements()
        assert elems.shape == (4, mesh.nelements())

    def test_nodes_in_domain(self, numpy_bkd):
        """Test all nodes are within [0,1] x [0,1]."""
        bkd = numpy_bkd
        mesh = self._create_mesh(bkd)
        nodes_np = bkd.to_numpy(mesh.nodes())
        assert np.all(nodes_np[0, :] >= -1e-12)
        assert np.all(nodes_np[0, :] <= 1.0 + 1e-12)
        assert np.all(nodes_np[1, :] >= -1e-12)
        assert np.all(nodes_np[1, :] <= 1.0 + 1e-12)

    def test_boundary_names(self, numpy_bkd):
        """Test all expected boundary names are present."""
        bkd = numpy_bkd
        mesh = self._create_mesh(bkd)
        names = mesh.boundary_names()
        for name in ["left", "right", "bottom", "top", "obs0", "obs1", "obs2"]:
            assert name in names

    def test_boundary_nodes_left(self, numpy_bkd):
        """Test left boundary nodes have x=0."""
        bkd = numpy_bkd
        mesh = self._create_mesh(bkd)
        left_nodes = bkd.to_numpy(mesh.boundary_nodes("left"))
        all_nodes = bkd.to_numpy(mesh.nodes())
        for idx in left_nodes:
            assert abs(all_nodes[0, idx] - 0.0) < 10**(-10)

    def test_boundary_nodes_right(self, numpy_bkd):
        """Test right boundary nodes have x=1."""
        bkd = numpy_bkd
        mesh = self._create_mesh(bkd)
        right_nodes = bkd.to_numpy(mesh.boundary_nodes("right"))
        all_nodes = bkd.to_numpy(mesh.nodes())
        for idx in right_nodes:
            assert abs(all_nodes[0, idx] - 1.0) < 10**(-10)

    def test_boundary_nodes_bottom(self, numpy_bkd):
        """Test bottom boundary nodes have y=0."""
        bkd = numpy_bkd
        mesh = self._create_mesh(bkd)
        bottom_nodes = bkd.to_numpy(mesh.boundary_nodes("bottom"))
        all_nodes = bkd.to_numpy(mesh.nodes())
        for idx in bottom_nodes:
            assert abs(all_nodes[1, idx] - 0.0) < 10**(-10)

    def test_boundary_nodes_top(self, numpy_bkd):
        """Test top boundary nodes have y=1."""
        bkd = numpy_bkd
        mesh = self._create_mesh(bkd)
        top_nodes = bkd.to_numpy(mesh.boundary_nodes("top"))
        all_nodes = bkd.to_numpy(mesh.nodes())
        for idx in top_nodes:
            assert abs(all_nodes[1, idx] - 1.0) < 10**(-10)

    def test_refinement_increases_elements(self, numpy_bkd):
        """Test refinement increases number of elements."""
        bkd = numpy_bkd
        mesh0 = self._create_mesh(bkd, nrefine=0)
        mesh1 = self._create_mesh(bkd, nrefine=1)
        assert mesh1.nelements() > mesh0.nelements()

    def test_refinement_increases_nodes(self, numpy_bkd):
        """Test refinement increases number of nodes."""
        bkd = numpy_bkd
        mesh0 = self._create_mesh(bkd, nrefine=0)
        mesh1 = self._create_mesh(bkd, nrefine=1)
        assert mesh1.nnodes() > mesh0.nnodes()

    def test_skfem_mesh_accessible(self, numpy_bkd):
        """Test skfem mesh object is accessible."""
        bkd = numpy_bkd
        mesh = self._create_mesh(bkd)
        sk_mesh = mesh.skfem_mesh()
        assert sk_mesh is not None
        assert sk_mesh.nelements == mesh.nelements()

    def test_invalid_boundary_raises(self, numpy_bkd):
        """Test invalid boundary name raises ValueError."""
        bkd = numpy_bkd
        mesh = self._create_mesh(bkd)
        with pytest.raises(ValueError):
            mesh.boundary_nodes("invalid_name")

    def test_obstruction_nodes_within_cell(self, numpy_bkd):
        """Test obstacle boundary nodes lie within the obstruction cell."""
        bkd = numpy_bkd
        mesh = self._create_mesh(bkd)
        all_nodes = bkd.to_numpy(mesh.nodes())
        obs0_nodes = bkd.to_numpy(mesh.boundary_nodes("obs0"))
        assert len(obs0_nodes) > 0
        # obs0 corresponds to obstruction_indices[0]=3
        # which is row 0 (y-index 0), col 3 (x-index 3)
        # Cell 3: x in [4/7, 5/7], y in [0, 1/4]
        for idx in obs0_nodes:
            x, y = all_nodes[0, idx], all_nodes[1, idx]
            assert x >= 4 / 7 - 1e-8
            assert x <= 5 / 7 + 1e-8
            assert y >= 0.0 - 1e-8
            assert y <= 0.25 + 1e-8

    def test_repr(self, numpy_bkd):
        """Test repr string is informative."""
        bkd = numpy_bkd
        mesh = self._create_mesh(bkd)
        r = repr(mesh)
        assert "ObstructedMesh2D" in r
        assert "nobstructions=3" in r
