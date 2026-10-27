"""The state-constraint derivatives bundle and the identity constraint.

The named constructors fill exactly their tier; the invariants refuse
partial tiers and operators given where a function belongs;
``STATED_ZERO`` becomes a zero operator between the right spaces; the
identity constraint returns its input, declares an identity tangent and
stated zeros elsewhere, satisfies the protocol and pickles.
"""

import pickle
from typing import Generic

import pytest

from pyapprox.ode.protocols import StateConstraintProtocol
from pyapprox.ode.residual_derivatives import STATED_ZERO, backend_spaces
from pyapprox.ode.state_constraint import (
    IdentityConstraint,
    StateConstraintDerivatives,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.operators import (
    IdentityOperator,
    OperatorProperties,
    SpaceDescriptor,
    ZeroOperator,
)

_NSTATES, _NPARAMS = 3, 2


class _Tangent(Generic[Array]):
    """A tangent returning a fixed operator."""

    def __init__(self, op: IdentityOperator[Array]) -> None:
        self._op = op

    def __call__(self, state: Array, time: float) -> IdentityOperator[Array]:
        return self._op


def _tangent(bkd: Backend[Array]) -> _Tangent[Array]:
    states, _ = backend_spaces(bkd, _NSTATES, _NPARAMS)
    return _Tangent(IdentityOperator(states))


def _descriptors(op: ZeroOperator[Array]) -> tuple[SpaceDescriptor, SpaceDescriptor]:
    return op.domain(), op.range()


class TestNamedConstructors:
    def test_each_tier_fills_exactly_its_fields(self, bkd: Backend[Array]) -> None:
        states, params = backend_spaces(bkd, _NSTATES, _NPARAMS)
        empty: StateConstraintDerivatives[Array] = StateConstraintDerivatives.none()
        assert empty.tangent is None and empty.param is None
        state = StateConstraintDerivatives.state_only(_tangent(bkd))
        assert state.tangent is not None
        assert state.param is None and state.param_space is None
        first = StateConstraintDerivatives.first_order(
            states, params, _tangent(bkd), param=STATED_ZERO
        )
        assert first.param is not None and first.param_space is params
        assert first.state_state is None and first.param_param is None
        second = StateConstraintDerivatives.second_order(
            states,
            params,
            _tangent(bkd),
            param=STATED_ZERO,
            state_state=STATED_ZERO,
            state_param=STATED_ZERO,
            param_param=STATED_ZERO,
        )
        assert second.state_param is not None and second.param_param is not None

    def test_stated_zeros_are_zero_operators_between_the_right_spaces(
        self, bkd: Backend[Array]
    ) -> None:
        states, params = backend_spaces(bkd, _NSTATES, _NPARAMS)
        bundle = StateConstraintDerivatives.second_order(
            states,
            params,
            _tangent(bkd),
            param=STATED_ZERO,
            state_state=STATED_ZERO,
            state_param=STATED_ZERO,
            param_param=STATED_ZERO,
        )
        state, adj = bkd.zeros((_NSTATES,)), bkd.ones((_NSTATES,))
        s, p = states.descriptor(), params.descriptor()
        param, state_state = bundle.param, bundle.state_state
        state_param, param_param = bundle.state_param, bundle.param_param
        assert param is not None and state_state is not None
        assert state_param is not None and param_param is not None
        expected = (
            (param(state, 0.0), (p, s)),
            (state_state(state, 0.0, adj), (s, s)),
            (state_param(state, 0.0, adj), (p, s)),
            (param_param(state, 0.0, adj), (p, p)),
        )
        for op, spaces in expected:
            assert isinstance(op, ZeroOperator) and op.properties().zero
            assert _descriptors(op) == spaces


class TestInvariants:
    def test_refuses_an_operator_where_a_function_belongs(
        self, bkd: Backend[Array]
    ) -> None:
        states, _ = backend_spaces(bkd, _NSTATES, _NPARAMS)
        with pytest.raises(TypeError, match="'tangent' must be callable"):
            StateConstraintDerivatives(tangent=IdentityOperator(states))  # type: ignore[arg-type]

    def test_param_requires_tangent_and_its_space(self, bkd: Backend[Array]) -> None:
        states, params = backend_spaces(bkd, _NSTATES, _NPARAMS)
        param = StateConstraintDerivatives.first_order(
            states, params, _tangent(bkd), param=STATED_ZERO
        ).param
        with pytest.raises(ValueError, match="param requires tangent"):
            StateConstraintDerivatives(param=param, param_space=params)
        with pytest.raises(ValueError, match="param_space comes exactly"):
            StateConstraintDerivatives(tangent=_tangent(bkd), param=param)

    def test_second_order_comes_whole_and_requires_param(
        self, bkd: Backend[Array]
    ) -> None:
        states, params = backend_spaces(bkd, _NSTATES, _NPARAMS)
        full = StateConstraintDerivatives.second_order(
            states,
            params,
            _tangent(bkd),
            param=STATED_ZERO,
            state_state=STATED_ZERO,
            state_param=STATED_ZERO,
            param_param=STATED_ZERO,
        )
        with pytest.raises(ValueError, match="come together"):
            StateConstraintDerivatives(
                tangent=full.tangent,
                param=full.param,
                param_space=params,
                state_state=full.state_state,
            )
        with pytest.raises(ValueError, match="second order requires param"):
            StateConstraintDerivatives(
                tangent=full.tangent,
                state_state=full.state_state,
                state_param=full.state_param,
                param_param=full.param_param,
            )


class TestIdentityConstraint:
    def test_returns_its_input_unchanged(self, bkd: Backend[Array]) -> None:
        states, _ = backend_spaces(bkd, _NSTATES, _NPARAMS)
        constraint = IdentityConstraint(states)
        assert isinstance(constraint, StateConstraintProtocol)
        state = bkd.asarray([1.0, -2.0, 3.0])
        assert constraint.constrain(state, 0.5) is state

    def test_declares_an_identity_tangent(self, bkd: Backend[Array]) -> None:
        states, _ = backend_spaces(bkd, _NSTATES, _NPARAMS)
        bundle = IdentityConstraint(states).derivatives()
        tangent = bundle.tangent
        assert tangent is not None
        op = tangent(bkd.zeros((_NSTATES,)), 0.0)
        assert op.properties() == OperatorProperties(
            symmetric=True, spd=True, diagonal=True, identity=True
        )
        assert op.domain() == states.descriptor() == op.range()

    def test_without_a_parameter_space_is_state_only(self, bkd: Backend[Array]) -> None:
        states, _ = backend_spaces(bkd, _NSTATES, _NPARAMS)
        bundle = IdentityConstraint(states).derivatives()
        assert bundle.param is None and bundle.state_state is None

    def test_with_a_parameter_space_states_every_other_field_zero(
        self, bkd: Backend[Array]
    ) -> None:
        states, params = backend_spaces(bkd, _NSTATES, _NPARAMS)
        bundle = IdentityConstraint(states, params).derivatives()
        state, adj = bkd.zeros((_NSTATES,)), bkd.ones((_NSTATES,))
        param, state_state = bundle.param, bundle.state_state
        state_param, param_param = bundle.state_param, bundle.param_param
        assert param is not None and state_state is not None
        assert state_param is not None and param_param is not None
        assert param(state, 0.0).properties().zero
        for second in (state_state, state_param, param_param):
            assert second(state, 0.0, adj).properties().zero

    def test_pickles(self, bkd: Backend[Array]) -> None:
        states, params = backend_spaces(bkd, _NSTATES, _NPARAMS)
        restored = pickle.loads(pickle.dumps(IdentityConstraint(states, params)))
        state = bkd.asarray([1.0, 2.0, 3.0])
        bkd.assert_allclose(restored.constrain(state, 0.0), state)
        tangent = restored.derivatives().tangent
        assert tangent is not None
        assert tangent(state, 0.0).properties().identity
