"""The ODE residual derivatives bundle and its declaration helpers.

The named constructors fill exactly their tier; the invariants refuse
partial tiers and operators or matrices given where a function belongs;
``STATED_ZERO`` becomes a zero operator between the right spaces; the
dense declarations reproduce the Lotka-Volterra model's derivatives, and
the transpose of ``state_param`` is the model's separate parameter-state
term; a bundle pickles; and it is covariant in the Jacobian's format.

The reference matrices are built from the model's vector methods applied
to identity columns, so they share no code with the declarations.
"""

import pickle
from typing import Callable, Generic

import pytest

from pyapprox.ode.residual_derivatives import (
    STATED_ZERO,
    ODEResidualDerivatives,
    backend_spaces,
    dense_jacobian,
    dense_operator,
    matrix_free_jacobian,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import DenseFormat
from pyapprox.util.linalg.operators import FormatProtocol, LinearizationPoint
from tests._helpers.ode_models import LotkaVolterra

_POINT = LinearizationPoint(0)
_Bundle = ODEResidualDerivatives[Array, DenseFormat[Array]]


def _columns(
    bkd: Backend[Array], action: Callable[[Array], Array], ncols: int
) -> Array:
    """The matrix of a vector action, column by column."""
    eye = bkd.eye(ncols)
    return bkd.stack([action(eye[:, k]) for k in range(ncols)], axis=1)


class _Declarations(Generic[Array]):
    """The model's derivatives as the dense helpers need them."""

    def __init__(self, model: LotkaVolterra[Array]) -> None:
        self._model = model
        self._bkd = model.bkd()

    def state_state(self, state: Array, time: float, adj: Array) -> Array:
        return _columns(
            self._bkd, lambda w: self._model.state_state_hvp(state, time, adj, w), 2
        )

    def state_param(self, state: Array, time: float, adj: Array) -> Array:
        return _columns(
            self._bkd, lambda v: self._model.state_param_hvp(state, time, adj, v), 4
        )


def _first_order(model: LotkaVolterra[Array]) -> _Bundle[Array]:
    bkd = model.bkd()
    states, params = backend_spaces(bkd, 2, 4)
    return ODEResidualDerivatives.first_order(
        states,
        params,
        jacobian=dense_jacobian(model.jacobian, states, bkd),
        param=dense_operator(model.param_jacobian, params, states, bkd),
        initial_state=STATED_ZERO,
    )


def _second_order(model: LotkaVolterra[Array]) -> _Bundle[Array]:
    bkd = model.bkd()
    states, params = backend_spaces(bkd, 2, 4)
    declared = _Declarations(model)
    return ODEResidualDerivatives.second_order(
        states,
        params,
        jacobian=dense_jacobian(model.jacobian, states, bkd),
        param=dense_operator(model.param_jacobian, params, states, bkd),
        initial_state=STATED_ZERO,
        initial_state_curvature=STATED_ZERO,
        state_state=dense_operator(
            declared.state_state, states, states, bkd, symmetric=True
        ),
        state_param=dense_operator(declared.state_param, params, states, bkd),
        param_param=STATED_ZERO,
    )


_STATE = [10.0, 5.0]
_ADJ = [0.4, -1.1]


class TestNamedConstructors:
    def test_each_tier_fills_exactly_its_fields(self, bkd: Backend[Array]) -> None:
        model = LotkaVolterra(bkd)
        empty: _Bundle[Array] = ODEResidualDerivatives.none()
        assert empty.jacobian is None and empty.param is None
        states, _ = backend_spaces(bkd, 2, 4)
        state = ODEResidualDerivatives.state_only(
            dense_jacobian(model.jacobian, states, bkd)
        )
        assert state.jacobian is not None and state.param_space is None
        first = _first_order(model)
        assert first.param is not None and first.initial_state is not None
        assert first.state_state is None and first.initial_state_curvature is None
        assert first.param_space is not None
        assert first.param_space.descriptor().size == 4
        second = _second_order(model)
        assert second.state_param is not None and second.param_param is not None

    def test_stated_zeros_are_zero_operators_between_the_right_spaces(
        self, bkd: Backend[Array]
    ) -> None:
        bundle = _second_order(LotkaVolterra(bkd))
        state, adj = bkd.asarray(_STATE), bkd.asarray(_ADJ)
        assert bundle.initial_state is not None
        assert bundle.initial_state_curvature is not None
        assert bundle.param_param is not None
        zeros = [
            (bundle.initial_state(), 4, 2),
            (bundle.initial_state_curvature(bkd.asarray(_ADJ)), 4, 4),
            (bundle.param_param(state, 0.0, adj), 4, 4),
        ]
        for op, domain, range_ in zeros:
            assert op.properties().zero
            assert (op.domain().size, op.range().size) == (domain, range_)
            bkd.assert_allclose(op.apply(bkd.ones((domain, 3))), bkd.zeros((range_, 3)))
            bkd.assert_allclose(
                op.apply_transpose(bkd.ones((range_, 2))), bkd.zeros((domain, 2))
            )
        # A given term is not a zero.
        assert bundle.state_param is not None
        assert not bundle.state_param(state, 0.0, adj).properties().zero


class TestDeclarations:
    def test_jacobians_match_the_model(self, bkd: Backend[Array]) -> None:
        model = LotkaVolterra(bkd)
        states, _ = backend_spaces(bkd, 2, 4)
        state = bkd.asarray(_STATE)
        block = bkd.asarray([[1.0, 0.3], [-0.5, 2.0]])
        expected = model.jacobian(state, 0.0)
        dense = dense_jacobian(model.jacobian, states, bkd)(state, 0.0, _POINT)
        free = matrix_free_jacobian(
            model.apply_jacobian, model.apply_jacobian_transpose, states
        )(state, 0.0, _POINT)
        for op in (dense, free):
            assert op.linearization() == _POINT
            bkd.assert_allclose(op.apply(block), bkd.dot(expected, block))
            bkd.assert_allclose(op.apply_transpose(block), bkd.dot(expected.T, block))

    def test_param_operator_maps_parameters_to_states(
        self, bkd: Backend[Array]
    ) -> None:
        model = LotkaVolterra(bkd)
        bundle = _first_order(model)
        assert bundle.param is not None
        state = bkd.asarray(_STATE)
        op = bundle.param(state, 0.0)
        assert (op.domain().size, op.range().size) == (4, 2)
        reference = _columns(
            bkd, lambda v: bkd.dot(model.param_jacobian(state, 0.0), v), 4
        )
        v = bkd.asarray([[0.5], [-0.3], [0.8], [0.1]])
        w = bkd.asarray([[0.6], [0.2]])
        bkd.assert_allclose(op.apply(v), bkd.dot(reference, v))
        bkd.assert_allclose(op.apply_transpose(w), bkd.dot(reference.T, w))

    def test_state_param_transposed_is_the_param_state_term(
        self, bkd: Backend[Array]
    ) -> None:
        """One field gives both mixed terms, so they cannot disagree."""
        model = LotkaVolterra(bkd)
        bundle = _second_order(model)
        assert bundle.state_param is not None and bundle.state_state is not None
        state, adj = bkd.asarray(_STATE), bkd.asarray(_ADJ)
        mixed = bundle.state_param(state, 0.0, adj)
        v, w = bkd.asarray([0.5, -0.3, 0.8, 0.1]), bkd.asarray([0.6, 0.2])
        bkd.assert_allclose(
            mixed.apply(bkd.reshape(v, (4, 1)))[:, 0],
            model.state_param_hvp(state, 0.0, adj, v),
            rtol=1e-12,
        )
        bkd.assert_allclose(
            mixed.apply_transpose(bkd.reshape(w, (2, 1)))[:, 0],
            model.param_state_hvp(state, 0.0, adj, w),
            rtol=1e-12,
        )
        curvature = bundle.state_state(state, 0.0, adj)
        assert curvature.properties().symmetric
        bkd.assert_allclose(
            curvature.apply(bkd.reshape(w, (2, 1)))[:, 0],
            model.state_state_hvp(state, 0.0, adj, w),
            rtol=1e-12,
        )


class TestInvariants:
    def test_partial_tiers_are_refused(self, bkd: Backend[Array]) -> None:
        model = LotkaVolterra(bkd)
        first = _first_order(model)
        second = _second_order(model)
        with pytest.raises(ValueError, match="come together"):
            ODEResidualDerivatives(
                jacobian=first.jacobian,
                param=first.param,
                param_space=first.param_space,
            )
        with pytest.raises(ValueError, match="require jacobian"):
            ODEResidualDerivatives(
                param=first.param,
                initial_state=first.initial_state,
                param_space=first.param_space,
            )
        with pytest.raises(ValueError, match="param_space"):
            ODEResidualDerivatives(
                jacobian=first.jacobian,
                param=first.param,
                initial_state=first.initial_state,
            )
        with pytest.raises(ValueError, match="four second-order"):
            ODEResidualDerivatives(
                jacobian=first.jacobian,
                param=first.param,
                initial_state=first.initial_state,
                param_space=first.param_space,
                state_state=second.state_state,
            )
        with pytest.raises(ValueError, match="requires param"):
            ODEResidualDerivatives(
                jacobian=first.jacobian,
                initial_state_curvature=second.initial_state_curvature,
                state_state=second.state_state,
                state_param=second.state_param,
                param_param=second.param_param,
            )

    def test_an_operator_or_matrix_is_refused_where_a_function_belongs(
        self, bkd: Backend[Array]
    ) -> None:
        model = LotkaVolterra(bkd)
        states, _ = backend_spaces(bkd, 2, 4)
        snapshot = dense_jacobian(model.jacobian, states, bkd)(
            bkd.asarray(_STATE), 0.0, _POINT
        )
        with pytest.raises(TypeError, match="function returning it"):
            ODEResidualDerivatives(jacobian=snapshot)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="function returning it"):
            ODEResidualDerivatives(jacobian=bkd.eye(2))  # type: ignore[arg-type]


class TestPicklingAndVariance:
    def test_a_bundle_pickles(self, numpy_bkd: Backend[Array]) -> None:
        bundle = _second_order(LotkaVolterra(numpy_bkd))
        restored = pickle.loads(pickle.dumps(bundle))
        assert restored.param is not None
        op = restored.param(numpy_bkd.asarray(_STATE), 0.0)
        numpy_bkd.assert_allclose(
            op.apply(numpy_bkd.asarray([[1.0], [0.0], [0.0], [0.0]])),
            numpy_bkd.asarray([[10.0], [0.0]]),
        )
        assert restored.param_param is not None
        zero = restored.param_param(
            numpy_bkd.asarray(_STATE), 0.0, numpy_bkd.asarray(_ADJ)
        )
        assert zero.properties().zero

    def test_covariant_in_the_jacobians_format(self, bkd: Backend[Array]) -> None:
        """A consumer that ignores the format accepts a dense bundle;
        checked by mypy, run for completeness."""

        def reads_any(bundle: ODEResidualDerivatives[Array, FormatProtocol]) -> bool:
            return bundle.jacobian is not None

        assert reads_any(_second_order(LotkaVolterra(bkd)))
