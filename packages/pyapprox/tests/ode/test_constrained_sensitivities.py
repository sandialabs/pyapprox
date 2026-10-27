"""Sensitivities of a trajectory whose stored states are constrained.

Lotka-Volterra under backward Euler, with a curved projection whose
tangent depends on the state (``_PinAndRescale``). The tangent-linear and
adjoint Jacobians of the final state are DerivativeChecker-validated
against finite differences of constrained solves, and agree with each
other to near round-off. Controls: a dropped tangent, and the tangent
evaluated at the projected state, are caught by both methods. The
Hessian-vector product, which does not apply a constraint yet, refuses
one.
"""

from typing import Generic

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
from pyapprox.ode.functionals.all_states_endpoint import (
    AllStatesEndpointFunctional,
)
from pyapprox.ode.implicit_steppers.backward_euler import BackwardEulerHVP
from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
from pyapprox.ode.operator.forward_sensitivity import (
    forward_sensitivity_jacobian,
)
from pyapprox.ode.operator.qoi_jacobian import (
    TransientQoIJacobianMethod,
    adjoint_jacobian,
)
from pyapprox.ode.operator.time_adjoint_hvp import TimeAdjointOperatorWithHVP
from pyapprox.ode.protocols import StateConstraintProtocol
from pyapprox.ode.residual_derivatives import STATED_ZERO, backend_spaces
from pyapprox.ode.state_constraint import StateConstraintDerivatives
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendDenseOperator
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    FormatProtocol,
    IdentityOperator,
)
from pyapprox.util.rootfinding.newton import NewtonSolver
from tests._helpers.ode_models import LotkaVolterra

_NSTATES, _NPARAMS = 2, 4
_DELTAT, _FINAL_TIME = 0.1, 0.5
_PARAM = [1.0, 0.1, 0.075, 1.5]


_CURVATURE = 2.0


class _PinAndRescale(Generic[Array]):
    r""":math:`\Pi_t(y) = (g,\; y_1 e^{c (y_0 - g)})`, :math:`g = 1 + t`.

    Pins the first entry and rescales the second by how far the first
    was from the pin. A projection (:math:`\Pi \circ \Pi = \Pi`, since
    :math:`\Pi` sets :math:`y_0 = g`) whose tangent
    :math:`[[0, 0], [c y_1 e^{c(y_0 - g)}, e^{c(y_0 - g)}]]` depends on
    the state, so evaluating it at the projected state instead of the
    Newton solution is detectable. The controls subclass it with a wrong
    tangent.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        self._states, params = backend_spaces(bkd, _NSTATES, _NPARAMS)
        self._derivs = StateConstraintDerivatives.first_order(
            self._states, params, self._tangent, param=STATED_ZERO
        )

    def _scale(self, state: Array, time: float) -> Array:
        return self._bkd.exp(_CURVATURE * (state[0:1] - (1.0 + time)))

    def constrain(self, state: Array, time: float) -> Array:
        pinned = self._bkd.asarray([1.0 + time])
        return self._bkd.concatenate([pinned, state[1:2] * self._scale(state, time)])

    def _tangent(
        self, state: Array, time: float
    ) -> ActionOperatorProtocol[Array, FormatProtocol]:
        bkd = self._bkd
        scale = self._scale(state, time)
        matrix = bkd.stack(
            [
                bkd.zeros((2,)),
                bkd.concatenate([_CURVATURE * state[1:2] * scale, scale]),
            ],
            axis=0,
        )
        return BackendDenseOperator(matrix, self._states.descriptor(), bkd)

    def derivatives(self) -> StateConstraintDerivatives[Array]:
        return self._derivs


class _DroppedTangent(_PinAndRescale[Array]):
    """Control: the same projection declaring the identity tangent."""

    def _tangent(
        self, state: Array, time: float
    ) -> ActionOperatorProtocol[Array, FormatProtocol]:
        return IdentityOperator(self._states)


class _TangentAtProjection(_PinAndRescale[Array]):
    """Control: the exact tangent evaluated at the projected state, as a
    sweep linearizing at the stored state would."""

    def _tangent(
        self, state: Array, time: float
    ) -> ActionOperatorProtocol[Array, FormatProtocol]:
        return super()._tangent(self.constrain(state, time), time)


class _Pin(Generic[Array]):
    r""":math:`\Pi_t(y) = (1 + t,\; y_1)`: affine, so its tangent
    :math:`\mathrm{diag}(0, 1)` is constant and its second derivatives are
    stated zeros, as the Hessian-vector product requires."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        self._states, params = backend_spaces(bkd, _NSTATES, _NPARAMS)
        self._derivs = StateConstraintDerivatives.second_order(
            self._states,
            params,
            self._tangent,
            param=STATED_ZERO,
            state_state=STATED_ZERO,
            state_param=STATED_ZERO,
            param_param=STATED_ZERO,
        )

    def constrain(self, state: Array, time: float) -> Array:
        return self._bkd.concatenate([self._bkd.asarray([1.0 + time]), state[1:2]])

    def _tangent(
        self, state: Array, time: float
    ) -> ActionOperatorProtocol[Array, FormatProtocol]:
        return BackendDenseOperator(
            self._bkd.asarray([[0.0, 0.0], [0.0, 1.0]]),
            self._states.descriptor(),
            self._bkd,
        )

    def derivatives(self) -> StateConstraintDerivatives[Array]:
        return self._derivs


class _PinDroppedTangent(_Pin[Array]):
    """Control: the pin declaring the identity tangent."""

    def _tangent(
        self, state: Array, time: float
    ) -> ActionOperatorProtocol[Array, FormatProtocol]:
        return IdentityOperator(self._states)


def _integrator(
    model: LotkaVolterra[Array], constraint: StateConstraintProtocol[Array]
) -> TimeIntegrator[Array]:
    newton = NewtonSolver(BackwardEulerHVP(model))
    newton.set_options(maxiters=20, atol=1e-13, rtol=0.0)
    return TimeIntegrator(0.0, _FINAL_TIME, _DELTAT, newton, constraint=constraint)


def _init_state(bkd: Backend[Array]) -> Array:
    # Consistent at t = 0 (first entry at the pin), so the initial
    # projection does not rescale the second entry.
    return bkd.asarray([1.0, 5.0])


def _jacobian(
    bkd: Backend[Array],
    constraint: _PinAndRescale[Array],
    method: TransientQoIJacobianMethod[Array],
    sample: Array,
) -> Array:
    """The final state's parameter Jacobian by ``method``, after a
    constrained solve at ``sample``."""
    model = LotkaVolterra(bkd)
    model.set_param(sample[:, 0])
    integrator = _integrator(model, constraint)
    states, times = integrator.solve(_init_state(bkd))
    functional = AllStatesEndpointFunctional(_NSTATES, _NPARAMS, bkd)
    return method(integrator, functional, states, times, sample)


def _checker(
    bkd: Backend[Array],
    constraint: _PinAndRescale[Array],
    method: TransientQoIJacobianMethod[Array],
) -> DerivativeChecker[Array]:
    """Final state against parameters: constrained solves, and their
    Jacobian by ``method``."""
    model = LotkaVolterra(bkd)

    def final_states(samples: Array) -> Array:
        columns = []
        for ii in range(samples.shape[1]):
            model.set_param(samples[:, ii])
            states, _ = _integrator(model, constraint).solve(_init_state(bkd))
            columns.append(states[:, -1])
        return bkd.stack(columns, axis=1)

    def jacobian(sample: Array) -> Array:
        return _jacobian(bkd, constraint, method, sample)

    return DerivativeChecker(
        FunctionWithJacobianFromCallable(
            nqoi=_NSTATES,
            nvars=_NPARAMS,
            fun=final_states,
            jacobian=jacobian,
            bkd=bkd,
        )
    )


_METHODS = pytest.mark.parametrize(
    "method",
    [forward_sensitivity_jacobian, adjoint_jacobian],
    ids=["tangent_linear", "adjoint"],
)


def _sample(bkd: Backend[Array]) -> Array:
    return bkd.asarray(np.reshape(_PARAM, (-1, 1)))


class TestFirstOrder:
    @_METHODS
    def test_matches_finite_differences(
        self, bkd: Backend[Array], method: TransientQoIJacobianMethod[Array]
    ) -> None:
        checker = _checker(bkd, _PinAndRescale(bkd), method)
        errors = checker.check_derivatives(_sample(bkd), relative=True)[0]
        # One-sided differences of the Newton solves bottom out near 1e-8;
        # a wrong tangent plateaus far above (the controls below).
        assert bkd.to_float(bkd.min(errors)) <= 1e-6

    def test_adjoint_agrees_with_the_tangent_linear_model(
        self, bkd: Backend[Array]
    ) -> None:
        """Two independent sweeps, one forward, one backward, through the
        same discrete equations: they agree to near round-off."""
        constraint, sample = _PinAndRescale(bkd), _sample(bkd)
        bkd.assert_allclose(
            _jacobian(bkd, constraint, adjoint_jacobian, sample),
            _jacobian(bkd, constraint, forward_sensitivity_jacobian, sample),
            rtol=1e-10,
        )

    @_METHODS
    def test_dropping_the_tangent_is_detected(
        self, bkd: Backend[Array], method: TransientQoIJacobianMethod[Array]
    ) -> None:
        """Control: an error no finite-difference step size removes."""
        _assert_detected(bkd, _DroppedTangent(bkd), method)

    @_METHODS
    def test_the_tangent_at_the_projected_state_is_detected(
        self, bkd: Backend[Array], method: TransientQoIJacobianMethod[Array]
    ) -> None:
        """Control: linearizing at the stored state rather than the
        Newton solution is caught too."""
        _assert_detected(bkd, _TangentAtProjection(bkd), method)


def _assert_detected(
    bkd: Backend[Array],
    constraint: _PinAndRescale[Array],
    method: TransientQoIJacobianMethod[Array],
) -> None:
    checker = _checker(bkd, constraint, method)
    errors = checker.check_derivatives(_sample(bkd), relative=True)[0]
    assert bkd.to_float(bkd.min(errors)) > 1e-3


def _hvp_operator(
    bkd: Backend[Array], constraint: StateConstraintProtocol[Array]
) -> TimeAdjointOperatorWithHVP[Array]:
    """The second state at the final time, as a scalar QoI of the
    parameters, on a fresh constrained integrator."""
    functional = AllStatesEndpointFunctional(_NSTATES, _NPARAMS, bkd).row_functional(1)
    return TimeAdjointOperatorWithHVP(
        _integrator(LotkaVolterra(bkd), constraint), functional
    )


def _hvp_errors(
    bkd: Backend[Array], constraint: StateConstraintProtocol[Array]
) -> tuple[Array, Array]:
    """DerivativeChecker's gradient and HVP error sweeps: finite
    differences of Q against the adjoint gradient, and of the gradient
    against the Hessian-vector product."""

    def fun(samples: Array) -> Array:
        return bkd.hstack(
            [
                _hvp_operator(bkd, constraint)(
                    _init_state(bkd), samples[:, ii : ii + 1]
                )
                for ii in range(samples.shape[1])
            ]
        )

    def jacobian(sample: Array) -> Array:
        return _hvp_operator(bkd, constraint).jacobian(_init_state(bkd), sample)

    def hvp(sample: Array, vec: Array) -> Array:
        return _hvp_operator(bkd, constraint).hvp(_init_state(bkd), sample, vec)

    checker = DerivativeChecker(
        FunctionWithJacobianAndHVPFromCallable(
            nvars=_NPARAMS, fun=fun, jacobian=jacobian, hvp=hvp, bkd=bkd
        )
    )
    grad_errors, hvp_errors = checker.check_derivatives(_sample(bkd), relative=True)
    return grad_errors, hvp_errors


class TestSecondOrder:
    def test_hvp_matches_finite_differences_of_the_gradient(
        self, bkd: Backend[Array]
    ) -> None:
        for errors in _hvp_errors(bkd, _Pin(bkd)):
            assert bkd.to_float(bkd.min(errors)) <= 1e-6

    def test_dropping_the_tangent_is_detected(self, bkd: Backend[Array]) -> None:
        """Control: the HVP of a sweep that drops the tangent is off by
        more than any finite-difference step size explains."""
        _, hvp_errors = _hvp_errors(bkd, _PinDroppedTangent(bkd))
        assert bkd.to_float(bkd.min(hvp_errors)) > 1e-3


class TestRefusals:
    def _solved(
        self, bkd: Backend[Array], constraint: _PinAndRescale[Array]
    ) -> tuple[TimeIntegrator[Array], Array, Array]:
        model = LotkaVolterra(bkd)
        model.set_param(bkd.asarray(_PARAM))
        integrator = _integrator(model, constraint)
        states, times = integrator.solve(_init_state(bkd))
        return integrator, states, times

    def test_hvp_refuses_a_constraint_without_second_derivatives(
        self, bkd: Backend[Array]
    ) -> None:
        operator = _hvp_operator(bkd, _PinAndRescale(bkd))
        with pytest.raises(TypeError, match="second derivatives"):
            operator.hvp(_init_state(bkd), _sample(bkd), bkd.ones((_NPARAMS, 1)))

    def test_refuses_another_trajectory(self, bkd: Backend[Array]) -> None:
        integrator, states, times = self._solved(bkd, _PinAndRescale(bkd))
        functional = AllStatesEndpointFunctional(_NSTATES, _NPARAMS, bkd)
        with pytest.raises(ValueError, match="last solve"):
            forward_sensitivity_jacobian(
                integrator, functional, 2.0 * states, times, _sample(bkd)
            )
