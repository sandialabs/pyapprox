"""DirichletRows: a collocation system's essential rows and their data.

2D Chebyshev mesh, 5 x 5 nodes on [-1, 1]^2, four sides. The data are
declared ``f(coords, t)`` suppliers evaluated at the boundary nodes, or
constants. The old path (``DirichletBC.apply_to_*`` in side order) is the
reference for the rows ``RowPin`` produces.
"""

import pickle
from typing import Callable, Generic, List, Optional, Sequence

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.ode.pinned_rows import RowPin
from pyapprox.ode.residual_derivatives import (
    STATED_ZERO,
    DerivativeOperator,
    backend_spaces,
    backend_state_space,
)
from pyapprox.pde.collocation.boundary.dirichlet import DirichletBC
from pyapprox.pde.collocation.boundary.essential_rows import (
    DirichletPart,
    DirichletRows,
    MappedBoundaryData,
)
from pyapprox.pde.collocation.mesh import TransformedMesh2D
from pyapprox.pde.collocation.physics.coefficients import (
    NodalField,
    NodalSupplier,
)
from pyapprox.pde.collocation.state_layout import StateLayout
from pyapprox.pde.constitutive.coefficient_functions import (
    TimeDependent,
    TimeIndependent,
)
from pyapprox.pde.field_maps.basis_expansion import BasisExpansion
from pyapprox.pde.field_maps.transformed import TransformedFieldMap
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
        rows = DirichletRows(bkd, StateLayout.uniform(_NPTS, 1), _all_sides(bkd, True))
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
        rows = DirichletRows(
            bkd, StateLayout.uniform(_NPTS, 2), _all_sides(bkd, False, component=1)
        )
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
        """Sides 0 (x = -1) and 2 (y = -1) share a corner node. Data
        agreeing there to round-off (a relative 1e-12 apart) are accepted,
        and the side listed later supplies its value."""
        mesh = _mesh(bkd)
        left, bottom = mesh.boundary_indices(0), mesh.boundary_indices(2)
        corner = sorted({bkd.to_int(i) for i in left} & {bkd.to_int(i) for i in bottom})
        assert len(corner) == 1
        near = 2.0 * (1.0 + 1e-12)
        rows = DirichletRows(
            bkd,
            StateLayout.uniform(_NPTS, 1),
            [DirichletPart(left, 2.0), DirichletPart(bottom, near)],
        )
        assert bkd.to_float(rows.values(0.0)[corner[0], 0]) == near
        swapped = DirichletRows(
            bkd,
            StateLayout.uniform(_NPTS, 1),
            [DirichletPart(bottom, near), DirichletPart(left, 2.0)],
        )
        assert bkd.to_float(swapped.values(0.0)[corner[0], 0]) == 2.0

    @pytest.mark.parametrize("gap", [1e-8, 1.0])
    def test_corner_mismatch_is_refused(self, bkd: Backend[Array], gap: float) -> None:
        """Data disagreeing at a shared corner by more than a relative
        1e-10 have no smooth solution: the evaluation refuses, naming the
        parts and the row."""
        mesh = _mesh(bkd)
        left, bottom = mesh.boundary_indices(0), mesh.boundary_indices(2)
        rows = DirichletRows(
            bkd,
            StateLayout.uniform(_NPTS, 1),
            [DirichletPart(left, 2.0), DirichletPart(bottom, 2.0 + gap)],
        )
        with pytest.raises(ValueError, match=r"parts 0 and 1 disagree"):
            rows.values(0.0)

    def test_trimmed_side_shares_nothing(self, bkd: Backend[Array]) -> None:
        """Leaving the corner out of one side states which condition
        holds there; different data are then accepted."""
        mesh = _mesh(bkd)
        left, bottom = mesh.boundary_indices(0), mesh.boundary_indices(2)
        left_nodes = {bkd.to_int(i) for i in left}
        trimmed = bkd.asarray(
            [bkd.to_int(i) for i in bottom if bkd.to_int(i) not in left_nodes],
            dtype=bkd.int64_dtype(),
        )
        rows = DirichletRows(
            bkd,
            StateLayout.uniform(_NPTS, 1),
            [DirichletPart(left, 1.0), DirichletPart(trimmed, 2.0)],
        )
        column = rows.values(0.0)
        bkd.assert_allclose(column[left, 0], bkd.ones((left.shape[0],)))

    def test_time_invariance_from_the_data(self, bkd: Backend[Array]) -> None:
        """Invariant exactly when every part's data are declared so."""
        mesh = _mesh(bkd)
        assert DirichletRows(
            bkd, StateLayout.uniform(_NPTS, 1), _all_sides(bkd, False)
        ).is_time_invariant()
        constant = DirichletRows(
            bkd,
            StateLayout.uniform(_NPTS, 1),
            [DirichletPart(mesh.boundary_indices(0), 3.0)],
        )
        assert constant.is_time_invariant()
        mixed = _all_sides(bkd, False)
        mixed[2] = DirichletPart(mesh.boundary_indices(2), _on_side(bkd, 2, True))
        assert not DirichletRows(
            bkd, StateLayout.uniform(_NPTS, 1), mixed
        ).is_time_invariant()

    def test_no_rates_and_independent_of_parameters(self, bkd: Backend[Array]) -> None:
        rows = DirichletRows(bkd, StateLayout.uniform(_NPTS, 1), _all_sides(bkd, True))
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
            DirichletRows(bkd, StateLayout.uniform(_NPTS, 1), [])
        with pytest.raises(ValueError, match="component 1"):
            DirichletRows(
                bkd,
                StateLayout.uniform(_NPTS, 1),
                [DirichletPart(side, 0.0, component=1)],
            )
        outside = bkd.asarray([0, _NPTS], dtype=bkd.int64_dtype())
        with pytest.raises(ValueError, match="boundary indices"):
            DirichletRows(
                bkd, StateLayout.uniform(_NPTS, 1), [DirichletPart(outside, 0.0)]
            )
        raw = bkd.zeros((side.shape[0],))
        with pytest.raises(TypeError, match="NodalField"):
            DirichletRows(
                bkd,
                StateLayout.uniform(_NPTS, 1),
                [DirichletPart(side, raw)],  # type: ignore[arg-type]
            )
        short = DirichletRows(
            bkd,
            StateLayout.uniform(_NPTS, 1),
            [DirichletPart(side, NodalField(bkd.zeros((2,))))],
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
        declaration = DirichletRows(
            bkd, StateLayout.uniform(_NPTS, 1), _all_sides(bkd, True)
        )
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
        rows = DirichletRows(bkd, StateLayout.uniform(_NPTS, 1), _all_sides(bkd, True))
        loaded = pickle.loads(pickle.dumps(rows))
        bkd.assert_allclose(loaded.values(0.6), rows.values(0.6), rtol=1e-14)
        assert loaded.rows().indices() == rows.rows().indices()


# -- data depending on the parameters ---------------------------------------
#
# Sides 0 (x = -1) and 2 (y = -1) carry one field
# g(x, y; p) = exp(0.2 + p0 y / 2 + p1 (0.3 y^2 - 0.1 + 0.2 x)), each side
# its own field map over its nodes, so they agree at their shared corner
# for every p. Sides 1 and 3 are constants on their nodes off sides 0 and
# 2, so they share no node with the parameterized sides. The exp map has a
# second derivative, so the curvature is supplied.

_NPARAMS = 2
_MAPPED_SIDES = (0, 2)
_CONSTANT_SIDES = (1, 3)


class _Exp(Generic[Array]):
    """exp as transform and both its derivatives (picklable)."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def __call__(self, x: Array) -> Array:
        return self._bkd.exp(x)


def _side_map(
    bkd: Backend[Array], side: int, second_derivative: bool = True
) -> TransformedFieldMap[Array]:
    mesh = _mesh(bkd)
    coords = mesh.points()[:, mesh.boundary_indices(side)]
    x, y = coords[0], coords[1]
    inner = BasisExpansion(bkd, 0.2, [0.5 * y, 0.3 * y**2 - 0.1 + 0.2 * x])
    exp = _Exp(bkd)
    return TransformedFieldMap(
        inner, exp, exp, bkd, transform_deriv2=exp if second_derivative else None
    )


def _params(bkd: Backend[Array]) -> Array:
    return bkd.asarray([0.4, -0.7])


def _mapped(
    bkd: Backend[Array], side: int, second_derivative: bool = True
) -> MappedBoundaryData[Array]:
    return MappedBoundaryData(
        _side_map(bkd, side, second_derivative), _params(bkd), bkd
    )


def _off_mapped_sides(bkd: Backend[Array], side: int) -> Array:
    """A side's node indices without the nodes it shares with the mapped
    sides."""
    mesh = _mesh(bkd)
    shared = {bkd.to_int(i) for s in _MAPPED_SIDES for i in mesh.boundary_indices(s)}
    kept = [
        bkd.to_int(i)
        for i in mesh.boundary_indices(side)
        if bkd.to_int(i) not in shared
    ]
    return bkd.asarray(kept, dtype=bkd.int64_dtype())


def _corner(bkd: Backend[Array]) -> int:
    """The node shared by the mapped sides."""
    mesh = _mesh(bkd)
    first, second = (
        {bkd.to_int(i) for i in mesh.boundary_indices(s)} for s in _MAPPED_SIDES
    )
    (corner,) = first & second
    return corner


def _with_mapped_sides(
    bkd: Backend[Array],
    data: Sequence[MappedBoundaryData[Array]],
    reverse: bool,
) -> DirichletRows[Array]:
    """The mapped sides (in reverse order if asked) after the constants."""
    mesh = _mesh(bkd)
    mapped = [
        DirichletPart(mesh.boundary_indices(side), side_data)
        for side, side_data in zip(_MAPPED_SIDES, data)
    ]
    constants = [
        DirichletPart(_off_mapped_sides(bkd, side), 1.0) for side in _CONSTANT_SIDES
    ]
    return DirichletRows(
        bkd,
        StateLayout.uniform(_NPTS, 1),
        constants + (mapped[::-1] if reverse else mapped),
    )


def _set(data: Sequence[MappedBoundaryData[Array]], params: Array) -> None:
    for side_data in data:
        side_data.set_parameters(params)


def _dense(operator: DerivativeOperator[Array], bkd: Backend[Array]) -> Array:
    return operator.apply(bkd.eye(operator.domain().size))


class _VectorFunction(Generic[Array]):
    """A function of the parameters with its Jacobian, for
    DerivativeChecker."""

    def __init__(
        self,
        bkd: Backend[Array],
        nqoi: int,
        fun: Callable[[Array], Array],
        jac: Callable[[Array], Array],
    ) -> None:
        self._bkd, self._nqoi, self._fun, self._jac = bkd, nqoi, fun, jac

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return _NPARAMS

    def nqoi(self) -> int:
        return self._nqoi

    def __call__(self, samples: Array) -> Array:
        return self._bkd.stack(
            [self._fun(samples[:, k]) for k in range(samples.shape[1])], axis=1
        )

    def jacobian(self, sample: Array) -> Array:
        return self._jac(sample[:, 0])

    def derivatives(self) -> Derivatives[Array]:
        return Derivatives.first_order(jacobian=self.jacobian)


def _check(function: _VectorFunction[Array], bkd: Backend[Array]) -> float:
    checker: DerivativeChecker[Array] = DerivativeChecker(function)
    errors = checker.check_derivatives(
        bkd.reshape(_params(bkd), (_NPARAMS, 1)), verbosity=0
    )
    return bkd.to_float(checker.error_ratio(errors[0]))


class TestParameterizedDirichletRows:
    @pytest.mark.parametrize("reverse", [True, False])
    def test_data_jacobian(self, bkd: Backend[Array], reverse: bool) -> None:
        """dg/dp of the whole declaration passes DerivativeChecker, in
        either order of the mapped sides; the perturbed parameters keep
        the shared corner consistent."""
        data = [_mapped(bkd, side) for side in _MAPPED_SIDES]
        rows = _with_mapped_sides(bkd, data, reverse)
        jacobian_fn = rows.parameter_derivatives().values
        assert callable(jacobian_fn)

        def fun(params: Array) -> Array:
            _set(data, params)
            return rows.values(0.0)[:, 0]

        def jac(params: Array) -> Array:
            _set(data, params)
            return _dense(jacobian_fn(0.0), bkd)

        assert _check(_VectorFunction(bkd, _NPTS, fun, jac), bkd) <= 1e-6

    @pytest.mark.parametrize("reverse", [True, False])
    def test_data_curvature(self, bkd: Backend[Array], reverse: bool) -> None:
        """sum_i w_i d2g_i/dp2 is the Jacobian of w^T dg/dp, by
        DerivativeChecker, with weights on every pinned row. The error
        ratio bottoms out near 1e-6 on both backends, so the bound is
        1e-5; a curvature 10% off stays above 1e-2."""
        data = [_mapped(bkd, side) for side in _MAPPED_SIDES]
        rows = _with_mapped_sides(bkd, data, reverse)
        bundle = rows.parameter_derivatives()
        jacobian_fn, curvature_fn = bundle.values, bundle.values_curvature
        assert callable(jacobian_fn) and callable(curvature_fn)
        rng = np.random.default_rng(4)
        weights = bkd.asarray(rng.standard_normal((_NPTS, 1)))
        weights = rows.rows().replace_rows(bkd.zeros((_NPTS, 1)), weights)

        def fun(params: Array) -> Array:
            _set(data, params)
            return (_dense(jacobian_fn(0.0), bkd).T @ weights)[:, 0]

        def jac(params: Array) -> Array:
            _set(data, params)
            return _dense(curvature_fn(0.0, weights), bkd)

        def wrong(params: Array) -> Array:
            return 1.1 * jac(params)

        assert _check(_VectorFunction(bkd, _NPARAMS, fun, jac), bkd) <= 1e-5
        assert _check(_VectorFunction(bkd, _NPARAMS, fun, wrong), bkd) > 1e-2

    def test_shared_corner_weighted_once(self, bkd: Backend[Array]) -> None:
        """A weight on the corner both mapped sides supply enters the
        curvature once, as the side owning it alone would give."""
        data = [_mapped(bkd, side) for side in _MAPPED_SIDES]
        rows = _with_mapped_sides(bkd, data, False)
        curvature_fn = rows.parameter_derivatives().values_curvature
        assert callable(curvature_fn)
        corner = _corner(bkd)
        weights = bkd.zeros((_NPTS, 1))
        weights = bkd.index_update(
            weights,
            bkd.asarray([corner], dtype=bkd.int64_dtype()),
            bkd.ones((1, 1)),
        )
        side = _mesh(bkd).boundary_indices(_MAPPED_SIDES[0])
        position = [bkd.to_int(i) for i in side].index(corner)
        side_weights = bkd.zeros((side.shape[0],))
        side_weights = bkd.index_update(
            side_weights,
            bkd.asarray([position], dtype=bkd.int64_dtype()),
            bkd.ones((1,)),
        )
        single = data[0].param_derivatives().curvature
        assert single is not None
        bkd.assert_allclose(
            _dense(curvature_fn(0.0, weights), bkd),
            single(0.0, side_weights),
            rtol=1e-14,
        )

    def test_corner_mismatch_after_a_parameter_write(self, bkd: Backend[Array]) -> None:
        """A constant side matching the mapped side's corner at the
        initial parameters is accepted; writing other parameters breaks
        the match, and the next evaluation refuses."""
        mesh = _mesh(bkd)
        data = _mapped(bkd, 0)
        corner = _corner(bkd)
        position = [bkd.to_int(i) for i in mesh.boundary_indices(0)].index(corner)
        matching = bkd.to_float(data.values(0.0)[position])
        rows = DirichletRows(
            bkd,
            StateLayout.uniform(_NPTS, 1),
            [
                DirichletPart(mesh.boundary_indices(0), data),
                DirichletPart(mesh.boundary_indices(2), matching),
            ],
        )
        rows.values(0.0)
        data.set_parameters(bkd.asarray([0.0, 0.0]))
        with pytest.raises(ValueError, match="disagree on their shared rows"):
            rows.values(0.0)

    def test_row_pin_reads_the_data_jacobian(self, bkd: Backend[Array]) -> None:
        """The shared pin's data term is dg/dp on the pinned rows and zero
        elsewhere."""
        data = [_mapped(bkd, side) for side in _MAPPED_SIDES]
        rows = _with_mapped_sides(bkd, data, False)
        states, params = backend_spaces(bkd, _NPTS, _NPARAMS)
        pin: RowPin[Array] = RowPin(rows)
        term = pin.data_param_jacobian(0.0, states, params)
        assert term is not None
        fn = rows.parameter_derivatives().values
        assert callable(fn)
        expected = rows.rows().replace_rows(
            bkd.zeros((_NPTS, _NPARAMS)), _dense(fn(0.0), bkd)
        )
        bkd.assert_allclose(_dense(term, bkd), expected, rtol=1e-14, atol=1e-15)

    def test_declared_fields(self, bkd: Backend[Array]) -> None:
        """Rates' derivatives are unavailable (no rate is supplied); the
        curvature is absent when a part's map has no second derivative."""
        data = [_mapped(bkd, side) for side in _MAPPED_SIDES]
        bundle = _with_mapped_sides(bkd, data, False).parameter_derivatives()
        assert bundle.rates is None and bundle.rates_curvature is None
        assert callable(bundle.values_curvature)
        first_order = [_mapped(bkd, 0), _mapped(bkd, 2, False)]
        assert (
            _with_mapped_sides(bkd, first_order, False)
            .parameter_derivatives()
            .values_curvature
            is None
        )

    def test_parameter_counts_must_agree(self, bkd: Backend[Array]) -> None:
        mesh = _mesh(bkd)
        y = mesh.points()[1, mesh.boundary_indices(1)]
        three = MappedBoundaryData(
            BasisExpansion(bkd, 0.0, [y, y**2, y**3]), bkd.zeros((3,)), bkd
        )
        with pytest.raises(ValueError, match="number of parameters"):
            DirichletRows(
                bkd,
                StateLayout.uniform(_NPTS, 1),
                [
                    DirichletPart(mesh.boundary_indices(0), _mapped(bkd, 0)),
                    DirichletPart(mesh.boundary_indices(1), three),
                ],
            )

    def test_mapped_data(self, bkd: Backend[Array]) -> None:
        """Writing the parameters moves the version and the values; a wrong
        parameter shape is refused; the data pickle."""
        data = _mapped(bkd, 0)
        before = data.values(0.0)
        version = data.version()
        data.set_parameters(bkd.asarray([0.1, 0.2]))
        assert data.version() == version + 1
        assert bkd.to_float(bkd.norm(data.values(0.0) - before)) > 1e-3
        with pytest.raises(ValueError, match="shape"):
            data.set_parameters(bkd.zeros((3,)))
        loaded: Optional[MappedBoundaryData[Array]] = pickle.loads(pickle.dumps(data))
        assert loaded is not None
        bkd.assert_allclose(loaded.values(0.0), data.values(0.0), rtol=1e-14)
