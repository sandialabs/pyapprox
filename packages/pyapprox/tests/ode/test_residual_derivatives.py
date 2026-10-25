"""The ODE residual derivatives bundle and its declaration helpers.

The named constructors fill exactly their tier; the invariants refuse
partial tiers and computed values; ``STATED_ZERO`` becomes an exact zero
of the right size and is recorded; the declaration helpers reproduce the
Lotka-Volterra model's Jacobian and parameter Jacobian; a bundle of
bound methods and stated zeros pickles; and the bundle is covariant in
the Jacobian's format.
"""

import pickle

import pytest

from pyapprox.ode.dense_jacobian_operator import DenseJacobianOperator
from pyapprox.ode.matrix_free_jacobian_operator import MatrixFreeJacobianOperator
from pyapprox.ode.residual_derivatives import (
    STATED_ZERO,
    InitialStateDerivatives,
    ODEResidualDerivatives,
    ParamActions,
    dense_jacobian,
    matrix_free_jacobian,
    param_actions,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import DenseFormat
from pyapprox.util.linalg.operators import FormatProtocol, LinearizationPoint
from tests._helpers.ode_models import LotkaVolterra

_POINT = LinearizationPoint(0)


def _echo(vec: Array) -> Array:
    """A stand-in initial-state action, for the invariant checks."""
    return vec


def _model(bkd: Backend[Array]) -> LotkaVolterra[Array]:
    return LotkaVolterra(bkd)


def _jacobian(model: LotkaVolterra[Array]) -> DenseJacobianOperator[Array]:
    return dense_jacobian(model.jacobian, 2, model.bkd())


def _param(model: LotkaVolterra[Array]) -> ParamActions[Array]:
    return param_actions(model.param_jacobian, model.bkd())


def _second_order(
    model: LotkaVolterra[Array],
) -> ODEResidualDerivatives[Array, DenseFormat[Array]]:
    bkd = model.bkd()
    return ODEResidualDerivatives.second_order(
        bkd,
        nstates=2,
        nparams=4,
        jacobian=_jacobian(model),
        param=_param(model),
        initial_state=STATED_ZERO,
        state_state=model.state_state_hvp,
        state_param=model.state_param_hvp,
        param_state=model.param_state_hvp,
        param_param=STATED_ZERO,
    )


class TestNamedConstructors:
    def test_each_tier_fills_exactly_its_fields(self, bkd: Backend[Array]) -> None:
        model = _model(bkd)
        empty: ODEResidualDerivatives[Array, DenseFormat[Array]] = (
            ODEResidualDerivatives.none()
        )
        assert empty.jacobian is None and empty.param is None
        state = ODEResidualDerivatives.state_only(_jacobian(model))
        assert state.jacobian is not None and state.param is None
        assert state.nparams == 0
        first = ODEResidualDerivatives.first_order(
            bkd, 2, 4, _jacobian(model), _param(model), STATED_ZERO
        )
        assert first.param is not None and first.initial_state is not None
        assert first.initial_state.hvp is None and first.state_state is None
        second = _second_order(model)
        assert second.state_state is not None and second.param_param is not None
        assert second.initial_state is not None
        assert second.initial_state.hvp is not None

    def test_stated_zeros_are_exact_sized_and_recorded(
        self, bkd: Backend[Array]
    ) -> None:
        bundle = _second_order(_model(bkd))
        assert bundle.zeros == frozenset({"initial_state", "param_param"})
        initial = bundle.initial_state
        assert initial is not None and initial.hvp is not None
        state, adj = bkd.asarray([10.0, 5.0]), bkd.asarray([0.4, -1.1])
        v, w = bkd.asarray([0.5, -0.3, 0.8, 0.1]), bkd.asarray([0.6, 0.2])
        bkd.assert_allclose(initial.jvp(v), bkd.zeros((2,)))
        bkd.assert_allclose(initial.vjp(w), bkd.zeros((4,)))
        bkd.assert_allclose(initial.hvp(w, v), bkd.zeros((4,)))
        param_param = bundle.param_param
        assert param_param is not None
        bkd.assert_allclose(param_param(state, 0.0, adj, v), bkd.zeros((4,)))

    def test_given_terms_are_kept(self, bkd: Backend[Array]) -> None:
        model = _model(bkd)
        bundle = _second_order(model)
        state, adj = bkd.asarray([10.0, 5.0]), bkd.asarray([0.4, -1.1])
        w = bkd.asarray([0.6, 0.2])
        state_state = bundle.state_state
        assert state_state is not None
        bkd.assert_allclose(
            state_state(state, 0.0, adj, w),
            model.state_state_hvp(state, 0.0, adj, w),
        )


class TestInvariants:
    def test_partial_tiers_are_refused(self, bkd: Backend[Array]) -> None:
        model = _model(bkd)
        jacobian, param = _jacobian(model), _param(model)
        initial: InitialStateDerivatives[Array] = InitialStateDerivatives(
            jvp=_echo, vjp=_echo
        )
        with pytest.raises(ValueError, match="come together"):
            ODEResidualDerivatives(jacobian=jacobian, param=param, nparams=4)
        with pytest.raises(ValueError, match="require jacobian"):
            ODEResidualDerivatives(param=param, initial_state=initial, nparams=4)
        with pytest.raises(ValueError, match="nparams"):
            ODEResidualDerivatives(
                jacobian=jacobian, param=param, initial_state=initial
            )
        with pytest.raises(ValueError, match="four second-order"):
            ODEResidualDerivatives(
                jacobian=jacobian,
                param=param,
                initial_state=initial,
                state_state=model.state_state_hvp,
                nparams=4,
            )
        with pytest.raises(ValueError, match="initial_state with hvp"):
            ODEResidualDerivatives.second_order(
                bkd,
                2,
                4,
                jacobian,
                param,
                initial,
                model.state_state_hvp,
                model.state_param_hvp,
                model.param_state_hvp,
                model.param_param_hvp,
            )
        with_hvp = InitialStateDerivatives(
            jvp=initial.jvp, vjp=initial.vjp, hvp=model.initial_param_hvp
        )
        with pytest.raises(ValueError, match="second order only"):
            ODEResidualDerivatives.first_order(bkd, 2, 4, jacobian, param, with_hvp)

    def test_computed_values_and_matrices_are_refused(
        self, bkd: Backend[Array]
    ) -> None:
        model = _model(bkd)
        with pytest.raises(TypeError, match="must be callable"):
            ParamActions(jvp=bkd.eye(2), vjp=model.param_jacobian)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="must be callable"):
            InitialStateDerivatives(jvp=bkd.zeros((2,)), vjp=bkd.zeros((4,)))  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="JacobianOperatorProtocol"):
            ODEResidualDerivatives(jacobian=bkd.eye(2))  # type: ignore[arg-type]


class TestDeclarationHelpers:
    def test_dense_and_matrix_free_jacobians_match_the_model(
        self, bkd: Backend[Array]
    ) -> None:
        model = _model(bkd)
        state = bkd.asarray([10.0, 5.0])
        block = bkd.asarray([[1.0, 0.3], [-0.5, 2.0]])
        expected = model.jacobian(state, 0.0)
        dense = _jacobian(model).jacobian_operator(state, 0.0, _POINT)
        free = matrix_free_jacobian(
            model.apply_jacobian, model.apply_jacobian_transpose, 2, bkd
        )
        assert isinstance(free, MatrixFreeJacobianOperator)
        action = free.jacobian_operator(state, 0.0, _POINT)
        for op in (dense, action):
            assert op.linearization() == _POINT
            bkd.assert_allclose(op.apply(block), bkd.dot(expected, block))
            bkd.assert_allclose(op.apply_transpose(block), bkd.dot(expected.T, block))

    def test_param_actions_are_the_matrix_and_its_transpose(
        self, bkd: Backend[Array]
    ) -> None:
        model = _model(bkd)
        state = bkd.asarray([10.0, 5.0])
        v, w = bkd.asarray([0.5, -0.3, 0.8, 0.1]), bkd.asarray([0.6, 0.2])
        matrix = model.param_jacobian(state, 0.0)
        actions = _param(model)
        bkd.assert_allclose(actions.jvp(state, 0.0, v), bkd.dot(matrix, v))
        bkd.assert_allclose(actions.vjp(state, 0.0, w), bkd.dot(matrix.T, w))


class TestPicklingAndVariance:
    def test_a_bundle_of_bound_methods_and_zeros_pickles(
        self, numpy_bkd: Backend[Array]
    ) -> None:
        bundle = _second_order(_model(numpy_bkd))
        restored = pickle.loads(pickle.dumps(bundle))
        assert restored.zeros == bundle.zeros
        param = restored.param
        assert param is not None
        state, v = numpy_bkd.asarray([10.0, 5.0]), numpy_bkd.asarray([1.0, 0, 0, 0])
        numpy_bkd.assert_allclose(
            param.jvp(state, 0.0, v), numpy_bkd.asarray([10.0, 0])
        )

    def test_covariant_in_the_jacobians_format(self, bkd: Backend[Array]) -> None:
        """A consumer that ignores the format accepts a dense bundle;
        checked by mypy, run for completeness."""

        def reads_any(bundle: ODEResidualDerivatives[Array, FormatProtocol]) -> bool:
            return bundle.jacobian is not None

        assert reads_any(_second_order(_model(bkd)))
