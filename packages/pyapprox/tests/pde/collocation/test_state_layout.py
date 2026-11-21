"""StateLayout: where each field of a collocation state sits.

Uniform: three fields of 7 nodes, the rows ``i + k * 7``. Non-uniform, a
Stokes-like state on a 6 x 6 velocity mesh: ``ux`` and ``uy`` on its 36
nodes, ``p`` on 16 (a 4 x 4 interior grid), so ``p`` starts at row 72 and
the state has 88 entries. The row builders on that layout place a
velocity component's rows in its own block and size every column and
matrix to 88; a periodic part on a field whose node count is not the
reference points' is refused.
"""

import pickle

import pytest

from pyapprox.pde.collocation.basis import ChebyshevBasis2D
from pyapprox.pde.collocation.boundary.closure_rows import RobinPart, RobinRows
from pyapprox.pde.collocation.boundary.essential_rows import (
    DirichletPart,
    DirichletRows,
    StateRowsLayoutProtocol,
)
from pyapprox.pde.collocation.boundary.normal_operators import (
    GradientNormalOperator,
)
from pyapprox.pde.collocation.boundary.periodic_rows import (
    PeriodicPart,
    PeriodicRows,
)
from pyapprox.pde.collocation.mesh import TransformedMesh2D
from pyapprox.pde.collocation.state_layout import StateLayout
from pyapprox.util.backends.protocols import Array, Backend


def _stokes_like() -> StateLayout:
    return StateLayout([36, 36, 16], names=["ux", "uy", "p"])


class TestUniform:
    def test_rows_are_the_stacked_arithmetic(self) -> None:
        layout = StateLayout.uniform(7, 3)
        assert layout.nfields() == 3
        assert layout.nstates() == 21
        assert layout.is_uniform()
        assert layout.names() == ("0", "1", "2")
        for field in range(3):
            assert layout.npts(field) == 7
            assert layout.offset(field) == 7 * field
            assert layout.rows(field, [0, 3, 6]) == [i + 7 * field for i in (0, 3, 6)]

    def test_satisfies_the_row_builders_protocol(self) -> None:
        assert isinstance(StateLayout.uniform(4, 1), StateRowsLayoutProtocol)


class TestNonUniform:
    def test_offsets_and_rows(self) -> None:
        layout = _stokes_like()
        assert layout.nstates() == 88
        assert not layout.is_uniform()
        assert [layout.offset(k) for k in range(3)] == [0, 36, 72]
        assert [layout.npts(k) for k in range(3)] == [36, 36, 16]
        assert layout.field("p") == 2
        assert layout.rows(2, [0, 15]) == [72, 87]
        assert layout.rows(1, []) == []

    def test_equality_hash_and_pickle(self) -> None:
        layout = _stokes_like()
        assert layout == StateLayout([36, 36, 16], names=["ux", "uy", "p"])
        assert layout != StateLayout([36, 36, 16])
        assert hash(layout) == hash(pickle.loads(pickle.dumps(layout)))

    def test_refusals(self) -> None:
        with pytest.raises(ValueError, match="at least one field"):
            StateLayout([])
        with pytest.raises(ValueError, match="at least one node"):
            StateLayout([4, 0])
        with pytest.raises(ValueError, match="2 names given for 3 fields"):
            StateLayout([1, 2, 3], names=["a", "b"])
        with pytest.raises(ValueError, match="repeat"):
            StateLayout([1, 2], names=["a", "a"])
        layout = _stokes_like()
        with pytest.raises(ValueError, match="field 3 is not in"):
            layout.offset(3)
        with pytest.raises(ValueError, match=r"field 2 must lie in \[0, 16\)"):
            layout.rows(2, [16])
        with pytest.raises(ValueError, match="no field 'q'"):
            layout.field("q")


class TestRowBuilders:
    """On the Stokes-like layout, rows land in the velocity blocks."""

    def test_dirichlet_rows(self, bkd: Backend[Array]) -> None:
        mesh = TransformedMesh2D(6, 6, bkd)
        left = mesh.boundary_indices(0)
        rows = DirichletRows(
            bkd,
            _stokes_like(),
            [DirichletPart(left, 1.0, 0), DirichletPart(left, 2.0, 1)],
        )
        nodes = [bkd.to_int(i) for i in left]
        assert list(rows.rows().indices()) == sorted(
            nodes + [36 + node for node in nodes]
        )
        values = rows.values(0.0)
        assert values.shape == (88, 1)
        bkd.assert_allclose(values[nodes[0], 0], bkd.asarray(1.0))
        bkd.assert_allclose(values[36 + nodes[0], 0], bkd.asarray(2.0))
        bkd.assert_allclose(bkd.sum(values[72:]), bkd.asarray(0.0))

    def test_robin_rows(self, bkd: Backend[Array]) -> None:
        mesh = TransformedMesh2D(6, 6, bkd)
        basis = ChebyshevBasis2D(mesh, bkd)
        top = mesh.boundary_indices(3)
        # The rows' placement only: a normal term acts on the whole state,
        # and this one reads 36 entries, so it is not evaluated on the
        # 52-entry state
        normal = GradientNormalOperator(
            bkd,
            top,
            mesh.boundary_normals(3),
            [basis.derivative_matrix(1, 0), basis.derivative_matrix(1, 1)],
        )
        rows = RobinRows(
            bkd, StateLayout([36, 16]), [RobinPart(top, normal, 1.0, 0.0, 0.5)]
        )
        assert list(rows.rows().indices()) == sorted(bkd.to_int(i) for i in top)
        assert rows.rows().descriptor().size == 52

    def test_periodic_rows_refuse_another_grid(self, bkd: Backend[Array]) -> None:
        mesh = TransformedMesh2D(4, 4, bkd)
        basis = ChebyshevBasis2D(mesh, bkd)
        part = PeriodicPart(
            mesh.boundary_indices(0),
            mesh.boundary_indices(1),
            basis.derivative_matrix(1, 0),
            component=1,
        )
        with pytest.raises(ValueError, match="component 1 has 36 nodes"):
            PeriodicRows(bkd, StateLayout([16, 36]), mesh.reference_points(), [part])
        rows = PeriodicRows(bkd, StateLayout([36, 16]), mesh.reference_points(), [part])
        assert min(rows.rows().indices()) >= 36
        assert rows.jacobian(bkd.zeros((52,)), 0.0).shape == (52, 52)

    def test_a_non_layout_is_refused(self, bkd: Backend[Array]) -> None:
        mesh = TransformedMesh2D(4, 4, bkd)
        with pytest.raises(TypeError, match="StateRowsLayoutProtocol"):
            DirichletRows(
                bkd,
                16,  # type: ignore[arg-type]
                [DirichletPart(mesh.boundary_indices(0), 0.0)],
            )
