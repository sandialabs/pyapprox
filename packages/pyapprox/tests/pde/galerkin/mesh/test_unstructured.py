"""Tests for UnstructuredMesh2D.

These build their meshes inline, so they run everywhere. Behaviour that
needs a real multi-block mesh -- named boundaries and subdomains -- is
covered in ``tests/integration/pde/galerkin/mesh/``, which may depend on
the meshes shipped with ``pyapprox-benchmarks``.
"""

import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import json

from pyapprox.pde.galerkin.basis import LagrangeBasis, VectorLagrangeBasis
from pyapprox.pde.galerkin.mesh.unstructured import UnstructuredMesh2D


class TestElementTypeDispatch:
    """The element type follows from the connectivity width."""

    @staticmethod
    def _write(tmp_path, p, t):
        path = tmp_path / "mesh.json"
        path.write_text(json.dumps({"p": p, "t": t}))
        return str(path)

    def test_three_node_elements_give_a_triangle_mesh(
        self, tmp_path, numpy_bkd
    ) -> None:
        """A curved boundary can only be meshed with triangles, so the
        loader must accept them and not assume quads."""
        path = self._write(
            tmp_path,
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            [[0, 1, 2], [0, 2, 3]],
        )
        mesh = UnstructuredMesh2D(path, numpy_bkd)
        assert type(mesh.skfem_mesh()).__name__.startswith("MeshTri")
        assert mesh.nelements() == 2
        assert numpy_bkd.to_numpy(mesh.elements()).shape == (3, 2)

    def test_four_node_elements_still_give_a_quad_mesh(
        self, tmp_path, numpy_bkd
    ) -> None:
        """The pre-existing quad path is unchanged."""
        path = self._write(
            tmp_path,
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            [[0, 1, 2, 3]],
        )
        mesh = UnstructuredMesh2D(path, numpy_bkd)
        assert type(mesh.skfem_mesh()).__name__.startswith("MeshQuad")
        assert numpy_bkd.to_numpy(mesh.elements()).shape == (4, 1)

    def test_unsupported_element_width_raises(
        self, tmp_path, numpy_bkd
    ) -> None:
        """Silently mis-typing the mesh would surface much later as a
        confusing assembly error, so refuse at load time."""
        path = self._write(
            tmp_path, [[0.0, 0.0], [1.0, 0.0]], [[0, 1]]
        )
        with pytest.raises(ValueError, match="3 .*or 4"):
            UnstructuredMesh2D(path, numpy_bkd)

    def test_triangle_mesh_supports_taylor_hood(
        self, tmp_path, numpy_bkd
    ) -> None:
        """The reason triangles were added: P2/P1 velocity-pressure on a
        mesh that conforms to a curved boundary."""
        path = self._write(
            tmp_path,
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            [[0, 1, 2], [0, 2, 3]],
        )
        mesh = UnstructuredMesh2D(path, numpy_bkd)
        vel = VectorLagrangeBasis(mesh, degree=2)
        pres = LagrangeBasis(mesh, degree=1)
        # P2 vector: 2 dofs at each vertex and edge midpoint.
        assert vel.ndofs() == 2 * (4 + 5)
        assert pres.ndofs() == 4
