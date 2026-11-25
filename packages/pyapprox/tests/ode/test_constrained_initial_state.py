r"""The constrained initial state :math:`y_0 = \Pi(\hat y_0(p);\, p)`.

A dense fake with every term nonzero: the base
:math:`\hat y_0 = u + \Phi p + r\, p_0 p_1` and the constraint
:math:`\Pi(x; p) = x + e\, s(x, p)`,
:math:`s = x_0 p_0 + x_1^2/2 + p_1^2/2`, so :math:`\Pi'`, :math:`\Pi_p`
and all three second derivatives of :math:`w^\top \Pi` are nonzero. The
Jacobian and curvature are compared with hand-composed dense matrices and
checked by finite differences; the derivative order is the weaker of the
base's and the constraint's; stated zeros stay declared.
"""

from typing import Generic, List

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.hessian import (
    FunctionWithJacobianAndHVPFromCallable,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.ode.constrained_initial_state import (
    ConstrainedInitialState,
    FixedInitialValue,
    InitialValueProtocol,
)
from pyapprox.ode.residual_derivatives import DerivativeOperator, backend_spaces
from pyapprox.ode.state_constraint import (
    IdentityConstraint,
    StateConstraintDerivatives,
)
from pyapprox.ode.trajectory_derivatives import InitialStateDerivatives
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import (
    BackendDenseOperator,
    BackendVectorSpace,
)

_NSTATES, _NPARAMS = 3, 2
_TIME = 0.3
_PARAM = [0.7, -0.4]


class _Params(Generic[Array]):
    """The parameters currently set, shared by the fakes."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._value = bkd.asarray(_PARAM)

    def set(self, value: Array) -> None:
        self._value = value

    def get(self) -> Array:
        return self._value


def _dense(
    bkd: Backend[Array],
    matrix: Array,
    domain: BackendVectorSpace[Array],
    range_: BackendVectorSpace[Array],
) -> DerivativeOperator[Array]:
    return BackendDenseOperator(
        matrix, domain.descriptor(), bkd, range=range_.descriptor()
    )


class _Base(Generic[Array]):
    r""":math:`\hat y_0 = u + \Phi p + r\, p_0 p_1`."""

    def __init__(self, bkd: Backend[Array], params: _Params[Array], order: int):
        self._bkd, self._params = bkd, params
        self._states, self._pspace = backend_spaces(bkd, _NSTATES, _NPARAMS)
        self.u = bkd.asarray([[0.5], [-1.0], [2.0]])
        self.phi = bkd.asarray([[1.0, 0.2], [-0.3, 0.8], [0.4, 0.1]])
        self.r = bkd.asarray([[0.6], [1.1], [-0.7]])
        self._derivatives: InitialStateDerivatives[Array] = (
            InitialStateDerivatives.none()
            if order == 0
            else InitialStateDerivatives.first_order(self._param_derivative)
            if order == 1
            else InitialStateDerivatives.second_order(
                self._param_derivative, self._curvature
            )
        )

    def value(self) -> Array:
        p = self._params.get()
        return self.u + self.phi @ p[:, None] + self.r * (p[0] * p[1])

    def jacobian(self) -> Array:
        p = self._params.get()
        return self.phi + self.r @ self._bkd.stack([p[1], p[0]])[None, :]

    def curvature_matrix(self, weight: Array) -> Array:
        c = self._bkd.sum(weight * self.r)
        return c * self._bkd.asarray([[0.0, 1.0], [1.0, 0.0]])

    def derivatives(self) -> InitialStateDerivatives[Array]:
        return self._derivatives

    def _param_derivative(self) -> DerivativeOperator[Array]:
        return _dense(self._bkd, self.jacobian(), self._pspace, self._states)

    def _curvature(self, weight: Array) -> DerivativeOperator[Array]:
        return _dense(
            self._bkd, self.curvature_matrix(weight), self._pspace, self._pspace
        )


class _Constraint(Generic[Array]):
    r""":math:`\Pi(x; p) = x + e\, s(x, p)`,
    :math:`s = x_0 p_0 + x_1^2/2 + p_1^2/2`."""

    def __init__(self, bkd: Backend[Array], params: _Params[Array], order: int):
        self._bkd, self._params = bkd, params
        self._states, self._pspace = backend_spaces(bkd, _NSTATES, _NPARAMS)
        self.e = bkd.asarray([[0.3], [-0.5], [0.9]])
        states, pspace = self._states, self._pspace
        self._derivs: StateConstraintDerivatives[Array] = (
            StateConstraintDerivatives.state_only(self._tangent)
            if order == 0
            else StateConstraintDerivatives.first_order(
                states, pspace, self._tangent, self._param
            )
            if order == 1
            else StateConstraintDerivatives.second_order(
                states,
                pspace,
                self._tangent,
                self._param,
                self._state_state,
                self._state_param,
                self._param_param,
            )
        )

    def constrain(self, state: Array, time: float) -> Array:
        p = self._params.get()
        s = state[0, 0] * p[0] + 0.5 * state[1, 0] ** 2 + 0.5 * p[1] ** 2
        return state + self.e * s

    def derivatives(self) -> StateConstraintDerivatives[Array]:
        return self._derivs

    def tangent_matrix(self, state: Array) -> Array:
        bkd, p = self._bkd, self._params.get()
        g = bkd.stack([p[0], state[1, 0], bkd.zeros(())])
        return bkd.eye(_NSTATES) + self.e @ g[None, :]

    def param_matrix(self, state: Array) -> Array:
        g = self._bkd.stack([state[0, 0], self._params.get()[1]])
        return self.e @ g[None, :]

    def _weighted(
        self, adj: Array, row: int, col: int, shape: tuple[int, int]
    ) -> Array:
        bkd = self._bkd
        unit = bkd.zeros(shape)
        unit = bkd.index_update(unit, (row, col), bkd.ones(()))
        return bkd.sum(adj * self.e) * unit

    def state_state_matrix(self, adj: Array) -> Array:
        return self._weighted(adj, 1, 1, (_NSTATES, _NSTATES))

    def state_param_matrix(self, adj: Array) -> Array:
        return self._weighted(adj, 0, 0, (_NSTATES, _NPARAMS))

    def param_param_matrix(self, adj: Array) -> Array:
        return self._weighted(adj, 1, 1, (_NPARAMS, _NPARAMS))

    def _tangent(self, state: Array, time: float) -> DerivativeOperator[Array]:
        return _dense(self._bkd, self.tangent_matrix(state), self._states, self._states)

    def _param(self, state: Array, time: float) -> DerivativeOperator[Array]:
        return _dense(self._bkd, self.param_matrix(state), self._pspace, self._states)

    def _state_state(
        self, state: Array, time: float, adj: Array
    ) -> DerivativeOperator[Array]:
        return _dense(
            self._bkd, self.state_state_matrix(adj), self._states, self._states
        )

    def _state_param(
        self, state: Array, time: float, adj: Array
    ) -> DerivativeOperator[Array]:
        return _dense(
            self._bkd, self.state_param_matrix(adj), self._pspace, self._states
        )

    def _param_param(
        self, state: Array, time: float, adj: Array
    ) -> DerivativeOperator[Array]:
        return _dense(
            self._bkd, self.param_param_matrix(adj), self._pspace, self._pspace
        )


class _Case(Generic[Array]):
    """The fakes over shared parameters, and the state under test."""

    def __init__(
        self, bkd: Backend[Array], base_order: int = 2, constraint_order: int = 2
    ) -> None:
        self.bkd = bkd
        self.params = _Params(bkd)
        self.base = _Base(bkd, self.params, base_order)
        self.constraint = _Constraint(bkd, self.params, constraint_order)
        self.states, self.pspace = backend_spaces(bkd, _NSTATES, _NPARAMS)
        self.initial = ConstrainedInitialState(
            self.base, self.constraint, _TIME, self.states
        )

    def jacobian(self) -> Array:
        r""":math:`\Pi' D + \Pi_p`, by hand."""
        point = self.base.value()
        return self.constraint.tangent_matrix(
            point
        ) @ self.base.jacobian() + self.constraint.param_matrix(point)

    def curvature(self, weight: Array) -> Array:
        """The curvature of ``weight^T y0(p)``, by hand."""
        constraint, base = self.constraint, self.base
        point, d = base.value(), base.jacobian()
        pulled = constraint.tangent_matrix(point).T @ weight
        xx = constraint.state_state_matrix(weight)
        xp = constraint.state_param_matrix(weight)
        pp = constraint.param_param_matrix(weight)
        return base.curvature_matrix(pulled) + d.T @ xx @ d + d.T @ xp + xp.T @ d + pp

    def operator_matrix(self, op: DerivativeOperator[Array], ncols: int) -> Array:
        """``op`` applied to each unit column."""
        bkd = self.bkd
        eye = bkd.eye(ncols)
        return bkd.hstack([op.apply(eye[:, k : k + 1]) for k in range(ncols)])


def _weight(bkd: Backend[Array]) -> Array:
    return bkd.asarray([[0.4], [-1.3], [0.8]])


def _sample(bkd: Backend[Array]) -> Array:
    return bkd.asarray(np.reshape(_PARAM, (-1, 1)))


def _param_derivative(case: _Case[Array]) -> DerivativeOperator[Array]:
    field = case.initial.derivatives().param_derivative
    assert field is not None
    return field()


def _curvature(case: _Case[Array], weight: Array) -> DerivativeOperator[Array]:
    field = case.initial.derivatives().curvature
    assert field is not None
    return field(weight)


class TestValue:
    def test_is_the_constrained_base(self, bkd: Backend[Array]) -> None:
        case = _Case(bkd)
        expected = case.constraint.constrain(case.base.value(), _TIME)
        bkd.assert_allclose(case.initial.value(), expected, rtol=1e-14)

    def test_follows_the_parameters(self, bkd: Backend[Array]) -> None:
        case = _Case(bkd)
        before = case.initial.value()
        case.params.set(bkd.asarray([1.2, 0.3]))
        after = case.initial.value()
        assert bkd.to_float(bkd.max(bkd.abs(after - before))) > 1e-2


class TestFirstOrder:
    def test_matches_the_composed_jacobian(self, bkd: Backend[Array]) -> None:
        case = _Case(bkd)
        op = _param_derivative(case)
        bkd.assert_allclose(
            case.operator_matrix(op, _NPARAMS), case.jacobian(), rtol=1e-13
        )

    def test_transpose_is_the_adjoint(self, bkd: Backend[Array]) -> None:
        case = _Case(bkd)
        op = _param_derivative(case)
        v, w = bkd.asarray([[0.9], [-0.2]]), _weight(bkd)
        lhs = bkd.sum(w * op.apply(v))
        rhs = bkd.sum(v * op.apply_transpose(w))
        bkd.assert_allclose(lhs, rhs, rtol=1e-13)

    def test_matches_finite_differences(self, bkd: Backend[Array]) -> None:
        case = _Case(bkd)

        def values(samples: Array) -> Array:
            columns: List[Array] = []
            for k in range(samples.shape[1]):
                case.params.set(samples[:, k])
                columns.append(case.initial.value()[:, 0])
            return bkd.stack(columns, axis=1)

        def jacobian(sample: Array) -> Array:
            case.params.set(sample[:, 0])
            return case.operator_matrix(_param_derivative(case), _NPARAMS)

        checker = DerivativeChecker(
            FunctionWithJacobianFromCallable(
                nqoi=_NSTATES, nvars=_NPARAMS, fun=values, jacobian=jacobian, bkd=bkd
            )
        )
        errors = checker.check_derivatives(_sample(bkd), relative=True, central=True)[0]
        assert bkd.to_float(checker.error_ratio(errors)) <= 1e-6


class TestSecondOrder:
    def test_matches_the_composed_curvature(self, bkd: Backend[Array]) -> None:
        case = _Case(bkd)
        weight = _weight(bkd)
        op = _curvature(case, weight)
        bkd.assert_allclose(
            case.operator_matrix(op, _NPARAMS), case.curvature(weight), rtol=1e-13
        )
        assert op.properties().symmetric

    def test_matches_finite_differences(self, bkd: Backend[Array]) -> None:
        case = _Case(bkd)
        weight = _weight(bkd)

        def values(samples: Array) -> Array:
            columns: List[Array] = []
            for k in range(samples.shape[1]):
                case.params.set(samples[:, k])
                columns.append(bkd.sum(weight * case.initial.value())[None])
            return bkd.stack(columns, axis=1)

        def gradient(sample: Array) -> Array:
            case.params.set(sample[:, 0])
            return _param_derivative(case).apply_transpose(weight).T

        def hvp(sample: Array, vec: Array) -> Array:
            case.params.set(sample[:, 0])
            return _curvature(case, weight).apply(vec)

        checker = DerivativeChecker(
            FunctionWithJacobianAndHVPFromCallable(
                nvars=_NPARAMS, fun=values, jacobian=gradient, hvp=hvp, bkd=bkd
            )
        )
        errors = checker.check_derivatives(_sample(bkd), relative=True, central=True)
        for error in errors:
            assert bkd.to_float(checker.error_ratio(error)) <= 1e-6


class TestOrder:
    @pytest.mark.parametrize(
        "base_order, constraint_order, expected",
        [(0, 2, 0), (2, 0, 0), (1, 2, 1), (2, 1, 1), (2, 2, 2)],
    )
    def test_is_the_weaker_of_the_two(
        self,
        bkd: Backend[Array],
        base_order: int,
        constraint_order: int,
        expected: int,
    ) -> None:
        derivs = _Case(bkd, base_order, constraint_order).initial.derivatives()
        order = (
            0
            if derivs.param_derivative is None
            else 1
            if derivs.curvature is None
            else 2
        )
        assert order == expected


class TestFixedInitialValue:
    def test_value_is_a_copy(self, bkd: Backend[Array]) -> None:
        states, pspace = backend_spaces(bkd, _NSTATES, _NPARAMS)
        state = bkd.asarray([[1.0], [2.0], [3.0]])
        fixed = FixedInitialValue(state, states, pspace)
        first = fixed.value()
        bkd.assert_allclose(first, state)
        assert first is not fixed.value()
        assert isinstance(fixed, InitialValueProtocol)

    def test_without_parameters_has_no_derivatives(self, bkd: Backend[Array]) -> None:
        states, _ = backend_spaces(bkd, _NSTATES, _NPARAMS)
        fixed = FixedInitialValue(bkd.zeros((_NSTATES, 1)), states, None)
        assert fixed.derivatives().param_derivative is None

    def test_under_the_identity_states_zero(self, bkd: Backend[Array]) -> None:
        states, pspace = backend_spaces(bkd, _NSTATES, _NPARAMS)
        state = bkd.asarray([[1.0], [2.0], [3.0]])
        initial = ConstrainedInitialState(
            FixedInitialValue(state, states, pspace),
            IdentityConstraint(states, pspace),
            _TIME,
            states,
        )
        derivs = initial.derivatives()
        assert derivs.param_derivative is not None
        assert derivs.curvature is not None
        jacobian = derivs.param_derivative()
        assert jacobian.properties().zero
        bkd.assert_allclose(
            jacobian.apply_transpose(_weight(bkd)), bkd.zeros((_NPARAMS, 1))
        )
        bkd.assert_allclose(initial.value(), state)


class TestRefusals:
    def test_base_must_have_a_value(self, bkd: Backend[Array]) -> None:
        states, pspace = backend_spaces(bkd, _NSTATES, _NPARAMS)
        with pytest.raises(TypeError, match="InitialValueProtocol"):
            ConstrainedInitialState(
                object(),  # type: ignore[arg-type]
                IdentityConstraint(states, pspace),
                _TIME,
                states,
            )

    def test_constraint_must_be_differentiable(self, bkd: Backend[Array]) -> None:
        states, pspace = backend_spaces(bkd, _NSTATES, _NPARAMS)
        with pytest.raises(TypeError, match="DifferentiableStackConstraintProtocol"):
            ConstrainedInitialState(
                FixedInitialValue(bkd.zeros((_NSTATES, 1)), states, pspace),
                object(),  # type: ignore[arg-type]
                _TIME,
                states,
            )
