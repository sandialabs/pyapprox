"""Element DOF maps of the Galerkin bases.

Each map is checked against something it must agree with rather than a
re-derivation: the mesh connectivity for P1, the sparsity of an assembled
mass matrix (dense element blocks, so no entry cancels) for the pattern,
the per-node DOF pairs for the vector and Hermite bases, and the fields'
own maps for a mixed basis.
"""

from typing import Any, Union

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np
from numpy.typing import NDArray
from skfem import asm
from skfem.models.poisson import mass

from pyapprox.pde.assembly import ElementDofMapProtocol
from pyapprox.pde.galerkin.basis import (
    CubicHermiteBasis,
    LagrangeBasis,
    MixedBasis,
    VectorLagrangeBasis,
)
from pyapprox.pde.galerkin.basis.element_dofs import skfem_element_dofs
from pyapprox.pde.galerkin.basis.mixed import MixedFieldBasisProtocol
from pyapprox.pde.galerkin.mesh import StructuredMesh1D, StructuredMesh2D
from pyapprox.util.backends.numpy import NumpyBkd


def _mesh(bkd: NumpyBkd) -> StructuredMesh2D[NDArray[Any]]:
    return StructuredMesh2D(3, 2, [(0.0, 1.0), (0.0, 1.0)], bkd)


def _pattern(element_dofs: NDArray[np.int64]) -> set[tuple[int, int]]:
    """The union of the blocks ``dofs[e] x dofs[e]``."""
    return {(int(i), int(j)) for dofs in element_dofs for i in dofs for j in dofs}


def _node_pairs(nodes: NDArray[np.int64]) -> NDArray[np.int64]:
    """Each element's interleaved two-per-node DOFs, sorted by row."""
    return np.sort(np.hstack([2 * nodes, 2 * nodes + 1]), axis=1)


class TestElementDofs:
    def test_p1_map_is_the_mesh_connectivity(self, numpy_bkd: NumpyBkd) -> None:
        basis = LagrangeBasis(_mesh(numpy_bkd), degree=1)
        dofs = basis.element_dofs()
        assert isinstance(basis, ElementDofMapProtocol)
        assert dofs.dtype == np.int64
        np.testing.assert_array_equal(dofs, basis.skfem_basis().mesh.t.T)

    @pytest.mark.parametrize("degree", [1, 2])
    def test_pattern_is_the_assembled_sparsity(
        self, numpy_bkd: NumpyBkd, degree: int
    ) -> None:
        basis = LagrangeBasis(_mesh(numpy_bkd), degree=degree)
        dofs = basis.element_dofs()
        assert dofs.shape[0] == basis.skfem_basis().mesh.t.shape[1]
        assert dofs.min() >= 0 and dofs.max() < basis.ndofs()
        assembled = asm(mass, basis.skfem_basis()).tocoo()
        structure = {(int(i), int(j)) for i, j in zip(assembled.row, assembled.col)}
        assert _pattern(dofs) == structure

    def test_vector_map_holds_every_component(self, numpy_bkd: NumpyBkd) -> None:
        mesh = _mesh(numpy_bkd)
        vector = VectorLagrangeBasis(mesh, degree=1)
        nodes = LagrangeBasis(mesh, degree=1).element_dofs()
        np.testing.assert_array_equal(
            np.sort(vector.element_dofs(), axis=1), _node_pairs(nodes)
        )

    def test_hermite_map_holds_values_and_slopes(self, numpy_bkd: NumpyBkd) -> None:
        mesh = StructuredMesh1D(nx=4, bounds=(0.0, 1.0), bkd=numpy_bkd)
        dofs = CubicHermiteBasis(mesh).element_dofs()
        assert dofs.shape == (4, 4)
        np.testing.assert_array_equal(
            np.sort(dofs, axis=1), _node_pairs(mesh.skfem_mesh().t.T)
        )

    def test_mixed_map_concatenates_offset_fields(self, numpy_bkd: NumpyBkd) -> None:
        mesh = _mesh(numpy_bkd)
        vel = VectorLagrangeBasis(mesh, degree=2)
        pres = LagrangeBasis(mesh, degree=1)
        fields: list[tuple[str, MixedFieldBasisProtocol[NDArray[Any]]]] = [
            ("velocity", vel),
            ("pressure", pres),
        ]
        mixed = MixedBasis(fields)
        dofs = mixed.element_dofs()
        nvel = vel.element_dofs().shape[1]
        np.testing.assert_array_equal(dofs[:, :nvel], vel.element_dofs())
        np.testing.assert_array_equal(dofs[:, nvel:], pres.element_dofs() + vel.ndofs())
        assert dofs.max() == mixed.ndofs() - 1


# Malformed maps: an array of the wrong dtype or rank, or not an array.
_BadMap = Union[NDArray[Any], list[list[int]]]


class _BadBasis:
    """Stands in for an skfem basis: the helper reads only this attribute."""

    def __init__(self, element_dofs: _BadMap) -> None:
        self.element_dofs = element_dofs


class TestSkfemElementDofs:
    @pytest.mark.parametrize(
        "bad",
        [np.zeros((3, 2)), np.zeros(4, dtype=np.int64), [[0, 1], [1, 2]]],
        ids=["float", "one-dimensional", "list"],
    )
    def test_refuses_a_malformed_map(self, bad: _BadMap) -> None:
        with pytest.raises(TypeError, match="2-D integer array"):
            skfem_element_dofs(_BadBasis(bad))
