"""UnstructuredMesh2D against a real multi-block mesh from the benchmarks.

Lives in the integration tier because it reads a mesh shipped with
``pyapprox-benchmarks``: the package tier may not depend on benchmarks.

These previously read a mesh from a MrHyDE checkout under the user's
home directory, so they SKIPPED wherever that checkout was absent --
including CI, which left the loader untested there. The committed
benchmark mesh has the same structure (three material layers, four
named edges), so nothing is lost by depending on it instead.

Behaviour that needs no named regions is covered by the inline meshes in
``packages/pyapprox/tests/pde/galerkin/mesh/test_unstructured.py``.
"""

import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import os

import numpy as np
import pyapprox_benchmarks
from pyapprox.pde.galerkin.basis import VectorLagrangeBasis
from pyapprox.pde.galerkin.mesh.unstructured import UnstructuredMesh2D
from pyapprox.pde.galerkin.protocols.mesh import GalerkinMeshProtocol

_BEAM_MESH_PATH = os.path.join(
    os.path.dirname(pyapprox_benchmarks.__file__),
    "data",
    "cantilever_beam_2d_with_holes_h_2.json",
)

# Properties of that file, recorded so a silent mesh swap is visible.
_NNODES = 746
_NELEMS = 644
_TOP, _BOTTOM, _CORE = 150, 150, 344
# Extent before rescaling: x in [-50, 50], y in [-2.5, 27.5].
_WIDTH, _HEIGHT = 100.0, 30.0


class TestUnstructuredBeamMesh:
    """Named regions and index lookup on a genuine multi-block mesh."""

    def test_load_geometry(self, numpy_bkd) -> None:
        mesh = UnstructuredMesh2D(_BEAM_MESH_PATH, numpy_bkd)
        assert mesh.nnodes() == _NNODES
        assert mesh.nelements() == _NELEMS
        assert mesh.ndim() == 2

    def test_rescale_to_origin(self, numpy_bkd) -> None:
        """``rescale_origin`` shifts the minimum corner to the requested
        point without changing the extent."""
        bkd = numpy_bkd
        mesh = UnstructuredMesh2D(
            _BEAM_MESH_PATH, bkd, rescale_origin=(0.0, 0.0)
        )
        nodes = bkd.to_numpy(mesh.nodes())
        bkd.assert_allclose(
            bkd.asarray([nodes[0].min(), nodes[1].min()]),
            bkd.asarray([0.0, 0.0]),
            atol=1e-12,
        )
        bkd.assert_allclose(
            bkd.asarray([nodes[0].max(), nodes[1].max()]),
            bkd.asarray([_WIDTH, _HEIGHT]),
            atol=1e-12,
        )

    def test_boundary_nodes_left_edge(self, numpy_bkd) -> None:
        """Facet indices resolve to the nodes actually on that edge --
        the one piece of real index plumbing in the loader."""
        bkd = numpy_bkd
        mesh = UnstructuredMesh2D(
            _BEAM_MESH_PATH, bkd, rescale_origin=(0.0, 0.0)
        )
        left = bkd.to_numpy(mesh.boundary_nodes("left_edge"))
        nodes = bkd.to_numpy(mesh.nodes())
        x = nodes[0, left]
        bkd.assert_allclose(
            bkd.asarray(x), bkd.asarray(np.zeros_like(x)), atol=1e-10
        )
        y = nodes[1, left]
        bkd.assert_allclose(
            bkd.asarray([y.min(), y.max()]),
            bkd.asarray([0.0, _HEIGHT]),
            atol=1e-10,
        )

    def test_boundary_names(self, numpy_bkd) -> None:
        names = UnstructuredMesh2D(
            _BEAM_MESH_PATH, numpy_bkd
        ).boundary_names()
        for edge in ("left_edge", "right_edge", "bottom_edge", "top_edge"):
            assert edge in names

    def test_subdomain_elements_partition_the_mesh(self, numpy_bkd) -> None:
        """The three layers must tile the mesh exactly: a loader that
        dropped or duplicated elements would break this sum."""
        mesh = UnstructuredMesh2D(_BEAM_MESH_PATH, numpy_bkd)
        top = mesh.subdomain_elements("top_layer")
        bottom = mesh.subdomain_elements("bottom_layer")
        core = mesh.subdomain_elements("inner_core")
        assert (len(top), len(bottom), len(core)) == (_TOP, _BOTTOM, _CORE)
        assert len(top) + len(bottom) + len(core) == mesh.nelements()
        assert len(set(np.concatenate([top, bottom, core]))) == _NELEMS

    def test_subdomain_names(self, numpy_bkd) -> None:
        names = UnstructuredMesh2D(
            _BEAM_MESH_PATH, numpy_bkd
        ).subdomain_names()
        for layer in ("top_layer", "bottom_layer", "inner_core"):
            assert layer in names

    @pytest.mark.parametrize(
        "lookup,name",
        [("boundary_nodes", "nonexistent"),
         ("subdomain_elements", "nonexistent")],
    )
    def test_unknown_region_raises(self, numpy_bkd, lookup, name) -> None:
        mesh = UnstructuredMesh2D(_BEAM_MESH_PATH, numpy_bkd)
        with pytest.raises(ValueError):
            getattr(mesh, lookup)(name)

    def test_satisfies_galerkin_mesh_protocol(self, numpy_bkd) -> None:
        mesh = UnstructuredMesh2D(_BEAM_MESH_PATH, numpy_bkd)
        assert isinstance(mesh, GalerkinMeshProtocol)

    def test_drives_a_vector_basis(self, numpy_bkd) -> None:
        mesh = UnstructuredMesh2D(
            _BEAM_MESH_PATH, numpy_bkd, rescale_origin=(0.0, 0.0)
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        assert basis.ndofs() == 2 * mesh.nnodes()

    def test_array_shapes_follow_skfem_convention(self, numpy_bkd) -> None:
        """skfem wants (ndim, nnodes) and (nodes_per_elem, nelems); the
        JSON stores the transpose of both."""
        bkd = numpy_bkd
        mesh = UnstructuredMesh2D(_BEAM_MESH_PATH, bkd)
        assert bkd.to_numpy(mesh.elements()).shape == (4, _NELEMS)
        assert bkd.to_numpy(mesh.nodes()).shape == (2, _NNODES)
