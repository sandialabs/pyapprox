"""DirichletRows: a collocation system's essential rows and their data.

2D Chebyshev mesh, 5 x 5 nodes on [-1, 1]^2, four sides. The data are
declared ``f(coords, t)`` suppliers evaluated at the boundary nodes, or
constants. The old path (``DirichletBC.apply_to_*`` in side order) is the
reference for the rows ``RowPin`` produces.
"""

import pickle
from typing import List

import numpy as np
import pytest

from pyapprox.ode.pinned_rows import RowPin
from pyapprox.ode.residual_derivatives import STATED_ZERO, backend_state_space
from pyapprox.pde.collocation.boundary.dirichlet import DirichletBC
from pyapprox.pde.collocation.boundary.essential_rows import (
    DirichletPart,
    DirichletRows,
)
from pyapprox.pde.collocation.mesh import TransformedMesh2D
from pyapprox.pde.collocation.physics.coefficients import (
    NodalField,
    NodalSupplier,
)
from pyapprox.pde.constitutive.coefficient_functions import (
    TimeDependent,
    TimeIndependent,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendDenseOperator

_N1D = 5
_NPTS = _N1D * _N1D


def _mesh(bkd: Backend[Array]) -> TransformedMesh2D[Array]:
    return TransformedMesh2D(_N1D, _N1D, bkd)


def _moving(coords: Array, time: float) -> Array:
    return (1.0 + time) * (coords[0] + 2.0 * coords[1] ** 2)


def _fixed(coords: Array) -> Array:
    return coords[0] - coords[1]


def _on_side(bkd: Backend[Array], side: int, dependent: bool) -> NodalSupplier[Array]:
    mesh = _mesh(bkd)
    coords = mesh.points()[:, mesh.boundary_indices(side)]
    if dependent:
        moving: TimeDependent[Array] = TimeDependent(_moving)
        return NodalSupplier(moving, coords)
    fixed: TimeIndependent[Array] = TimeIndependent(_fixed)
    return NodalSupplier(fixed, coords)


def _all_sides(
    bkd: Backend[Array], dependent: bool, component: int = 0
) -> List[DirichletPart[Array]]:
    mesh = _mesh(bkd)
    return [
        DirichletPart(
            mesh.boundary_indices(side), _on_side(bkd, side, dependent), component
        )
        for side in range(mesh.nboundaries())
    ]


class TestDirichletRows:
    def test_values_are_the_supplier_at_the_boundary(self, bkd: Backend[Array]) -> None:
        """On each side's rows the column holds f(x_b, t); elsewhere 0."""
        time = 0.4
        mesh = _mesh(bkd)
        rows = DirichletRows(bkd, _NPTS, 1, _all_sides(bkd, True))
        column = rows.values(time)
        assert column.shape == (_NPTS, 1)
        boundary = sorted(
            {bkd.to_int(i) for side in range(4) for i in mesh.boundary_indices(side)}
        )
        assert list(rows.rows().indices()) == boundary
        expected = _moving(mesh.points(), time)
        bkd.assert_allclose(column[boundary, 0], expected[boundary], rtol=1e-14)
        interior = [i for i in range(_NPTS) if i not in boundary]
        bkd.assert_allclose(column[interior, 0], bkd.zeros((len(interior),)))

    def test_component_offset(self, bkd: Backend[Array]) -> None:
        """A part on component 1 of a two-component state pins rows
        offset by npts and leaves component 0 alone."""
        mesh = _mesh(bkd)
        rows = DirichletRows(bkd, _NPTS, 2, _all_sides(bkd, False, component=1))
        boundary = sorted(
            {bkd.to_int(i) for side in range(4) for i in mesh.boundary_indices(side)}
        )
        assert list(rows.rows().indices()) == [i + _NPTS for i in boundary]
        column = rows.values(0.0)
        assert column.shape == (2 * _NPTS, 1)
        bkd.assert_allclose(column[:_NPTS, 0], bkd.zeros((_NPTS,)))
        offset = [i + _NPTS for i in boundary]
        bkd.assert_allclose(
            column[offset, 0], _fixed(mesh.points())[boundary], rtol=1e-14
        )

    def test_later_part_wins_a_shared_corner(self, bkd: Backend[Array]) -> None:
        """Sides 0 (x = -1) and 2 (y = -1) share a corner node; the side
        listed later supplies its value."""
        mesh = _mesh(bkd)
        left, bottom = mesh.boundary_indices(0), mesh.boundary_indices(2)
        corner = sorted({bkd.to_int(i) for i in left} & {bkd.to_int(i) for i in bottom})
        assert len(corner) == 1
        rows = DirichletRows(
            bkd, _NPTS, 1, [DirichletPart(left, 1.0), DirichletPart(bottom, 2.0)]
        )
        assert bkd.to_float(rows.values(0.0)[corner[0], 0]) == 2.0
        swapped = DirichletRows(
            bkd, _NPTS, 1, [DirichletPart(bottom, 2.0), DirichletPart(left, 1.0)]
        )
        assert bkd.to_float(swapped.values(0.0)[corner[0], 0]) == 1.0

    def test_time_invariance_from_the_data(self, bkd: Backend[Array]) -> None:
        """Invariant exactly when every part's data are declared so."""
        mesh = _mesh(bkd)
        assert DirichletRows(bkd, _NPTS, 1, _all_sides(bkd, False)).is_time_invariant()
        constant = DirichletRows(
            bkd, _NPTS, 1, [DirichletPart(mesh.boundary_indices(0), 3.0)]
        )
        assert constant.is_time_invariant()
        mixed = _all_sides(bkd, False)
        mixed[2] = DirichletPart(mesh.boundary_indices(2), _on_side(bkd, 2, True))
        assert not DirichletRows(bkd, _NPTS, 1, mixed).is_time_invariant()

    def test_no_rates_and_independent_of_parameters(self, bkd: Backend[Array]) -> None:
        rows = DirichletRows(bkd, _NPTS, 1, _all_sides(bkd, True))
        assert rows.rates() is None
        bundle = rows.parameter_derivatives()
        assert bundle.values is STATED_ZERO
        assert bundle.values_curvature is STATED_ZERO

    def test_refusals(self, bkd: Backend[Array]) -> None:
        """No parts, a component or index out of range, raw array data, and
        data of the wrong length are refused."""
        mesh = _mesh(bkd)
        side = mesh.boundary_indices(0)
        with pytest.raises(ValueError, match="at least one part"):
            DirichletRows(bkd, _NPTS, 1, [])
        with pytest.raises(ValueError, match="component 1"):
            DirichletRows(bkd, _NPTS, 1, [DirichletPart(side, 0.0, component=1)])
        outside = bkd.asarray([0, _NPTS], dtype=bkd.int64_dtype())
        with pytest.raises(ValueError, match="boundary indices"):
            DirichletRows(bkd, _NPTS, 1, [DirichletPart(outside, 0.0)])
        raw = bkd.zeros((side.shape[0],))
        with pytest.raises(TypeError, match="NodalField"):
            DirichletRows(
                bkd,
                _NPTS,
                1,
                [DirichletPart(side, raw)],  # type: ignore[arg-type]
            )
        short = DirichletRows(
            bkd, _NPTS, 1, [DirichletPart(side, NodalField(bkd.zeros((2,))))]
        )
        with pytest.raises(ValueError, match="shape"):
            short.values(0.0)

    def test_row_pin_equals_the_old_dirichlet_rows(self, bkd: Backend[Array]) -> None:
        """RowPin on the declaration gives the residual and Jacobian rows
        the old DirichletBC path gives, applied side by side in the same
        order, to 1e-14."""
        time = 0.3
        mesh = _mesh(bkd)
        rng = np.random.default_rng(1)
        residual = bkd.asarray(rng.standard_normal(_NPTS))
        state = bkd.asarray(rng.standard_normal(_NPTS))
        jacobian = bkd.asarray(rng.standard_normal((_NPTS, _NPTS)))

        old_residual, old_jacobian = residual, jacobian
        for side in range(mesh.nboundaries()):
            supplier = _on_side(bkd, side, True)
            old = DirichletBC(bkd, mesh.boundary_indices(side), supplier.values)
            old_residual = old.apply_to_residual(old_residual, state, time)
            old_jacobian = old.apply_to_jacobian(old_jacobian, state, time)

        space = backend_state_space(bkd, _NPTS)
        declaration = DirichletRows(bkd, _NPTS, 1, _all_sides(bkd, True))
        pin: RowPin[Array] = RowPin(declaration)
        new_residual = pin.residual(
            bkd.reshape(residual, (_NPTS, 1)),
            bkd.reshape(state, (_NPTS, 1)),
            time,
            space,
        )
        bkd.assert_allclose(new_residual[:, 0], old_residual, rtol=1e-14, atol=1e-14)
        operator = BackendDenseOperator(jacobian, space.descriptor(), bkd)
        new_jacobian = pin.own(operator).apply(bkd.eye(_NPTS))
        bkd.assert_allclose(new_jacobian, old_jacobian, rtol=1e-14, atol=1e-14)

    def test_pickle_round_trip(self, bkd: Backend[Array]) -> None:
        rows = DirichletRows(bkd, _NPTS, 1, _all_sides(bkd, True))
        loaded = pickle.loads(pickle.dumps(rows))
        bkd.assert_allclose(loaded.values(0.6), rows.values(0.6), rtol=1e-14)
        assert loaded.rows().indices() == rows.rows().indices()
