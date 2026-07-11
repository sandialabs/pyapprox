"""Tests for PeriodicStructuredMesh1D (endpoint-identified topology)."""

import numpy as np
import pytest
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.mesh import PeriodicStructuredMesh1D
from pyapprox.pde.galerkin.protocols.mesh import GalerkinMeshProtocol
from pyapprox.util.backends.numpy import NumpyBkd


class TestPeriodicStructuredMesh1D:
    def test_satisfies_mesh_protocol(self):
        bkd = NumpyBkd()
        mesh = PeriodicStructuredMesh1D(nx=8, bounds=(0.0, 1.0), bkd=bkd)
        assert isinstance(mesh, GalerkinMeshProtocol)
        assert mesh.ndim() == 1
        assert mesh.nelements() == 8

    def test_endpoints_identified(self):
        """nnodes == nx (not nx + 1); canonical fundamental-domain nodes."""
        bkd = NumpyBkd()
        nx = 8
        mesh = PeriodicStructuredMesh1D(nx=nx, bounds=(0.0, 1.0), bkd=bkd)
        assert mesh.nnodes() == nx
        nodes = bkd.to_numpy(mesh.nodes())
        assert nodes.shape == (1, nx)
        np.testing.assert_allclose(
            nodes[0], np.linspace(0.0, 1.0, nx + 1)[:-1]
        )

    def test_connectivity_wraps(self):
        """The last element connects back to logical node 0."""
        bkd = NumpyBkd()
        nx = 8
        mesh = PeriodicStructuredMesh1D(nx=nx, bounds=(0.0, 1.0), bkd=bkd)
        elements = bkd.to_numpy(mesh.elements())
        assert elements.shape == (2, nx)
        assert elements.max() == nx - 1
        assert 0 in elements[:, -1]

    def test_boundary_nodes_raises(self):
        bkd = NumpyBkd()
        mesh = PeriodicStructuredMesh1D(nx=8, bounds=(0.0, 1.0), bkd=bkd)
        with pytest.raises(ValueError, match="no boundaries"):
            mesh.boundary_nodes("left")

    @pytest.mark.parametrize("degree,expected_ndofs", [(1, 8), (2, 16)])
    def test_lagrange_basis_pairs_with_periodic_mesh(
        self, degree, expected_ndofs
    ):
        """The DG mesh-name normalization selects continuous elements."""
        bkd = NumpyBkd()
        mesh = PeriodicStructuredMesh1D(nx=8, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=degree)
        assert basis.ndofs() == expected_ndofs
