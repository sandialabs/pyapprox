"""Sensitivities of a trajectory whose stored states are constrained.

Lotka-Volterra under backward Euler, with a constraint pinning the first
state entry to :math:`1 + t` (affine, so :math:`\\Pi' = \\mathrm{diag}(0,
1)` everywhere). The tangent-linear Jacobian of the final state is
DerivativeChecker-validated against finite differences of constrained
solves. Control: a constraint that pins identically but declares an
identity tangent, so the sweep drops :math:`\\Pi'`, is off at O(1); the
sweeps that do not apply a constraint yet refuse one.
"""

from typing import Generic

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.ode.functionals.all_states_endpoint import (
    AllStatesEndpointFunctional,
)
from pyapprox.ode.implicit_steppers.backward_euler import BackwardEulerAdjoint
from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
from pyapprox.ode.operator.forward_sensitivity import (
    forward_sensitivity_jacobian,
)
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


def _integrator(
    model: LotkaVolterra[Array], constraint: _PinAndRescale[Array]
) -> TimeIntegrator[Array]:
    newton = NewtonSolver(BackwardEulerAdjoint(model))
    newton.set_options(maxiters=20, atol=1e-13, rtol=0.0)
    return TimeIntegrator(0.0, _FINAL_TIME, _DELTAT, newton, constraint=constraint)


def _init_state(bkd: Backend[Array]) -> Array:
    # Consistent at t = 0 (first entry at the pin), so the initial
    # projection does not rescale the second entry.
    return bkd.asarray([1.0, 5.0])


def _checker(
    bkd: Backend[Array], constraint: _PinAndRescale[Array]
) -> DerivativeChecker[Array]:
    """Final state against parameters: constrained solves, and their
    tangent-linear Jacobian."""
    model = LotkaVolterra(bkd)
    functional = AllStatesEndpointFunctional(_NSTATES, _NPARAMS, bkd)

    def final_states(samples: Array) -> Array:
        columns = []
        for ii in range(samples.shape[1]):
            model.set_param(samples[:, ii])
            states, _ = _integrator(model, constraint).solve(_init_state(bkd))
            columns.append(states[:, -1])
        return bkd.stack(columns, axis=1)

    def tangent_jacobian(sample: Array) -> Array:
        model.set_param(sample[:, 0])
        integrator = _integrator(model, constraint)
        states, times = integrator.solve(_init_state(bkd))
        return forward_sensitivity_jacobian(
            integrator, functional, states, times, sample
        )

    return DerivativeChecker(
        FunctionWithJacobianFromCallable(
            nqoi=_NSTATES,
            nvars=_NPARAMS,
            fun=final_states,
            jacobian=tangent_jacobian,
            bkd=bkd,
        )
    )


def _sample(bkd: Backend[Array]) -> Array:
    return bkd.asarray(np.reshape(_PARAM, (-1, 1)))


class TestTangentLinear:
    def test_matches_finite_differences(self, bkd: Backend[Array]) -> None:
        checker = _checker(bkd, _PinAndRescale(bkd))
        errors = checker.check_derivatives(_sample(bkd), relative=True)[0]
        # One-sided differences of the Newton solves bottom out near 1e-8;
        # a wrong tangent plateaus far above (the controls below).
        assert bkd.to_float(bkd.min(errors)) <= 1e-6

    def test_dropping_the_tangent_is_detected(self, bkd: Backend[Array]) -> None:
        """Control: an error no finite-difference step size removes."""
        _assert_detected(bkd, _DroppedTangent(bkd))

    def test_the_tangent_at_the_projected_state_is_detected(
        self, bkd: Backend[Array]
    ) -> None:
        """Control: linearizing at the stored state rather than the
        Newton solution is caught too."""
        _assert_detected(bkd, _TangentAtProjection(bkd))


def _assert_detected(bkd: Backend[Array], constraint: _PinAndRescale[Array]) -> None:
    errors = _checker(bkd, constraint).check_derivatives(_sample(bkd), relative=True)[0]
    assert bkd.to_float(bkd.min(errors)) > 1e-3


class TestRefusals:
    def _solved(
        self, bkd: Backend[Array], constraint: _PinAndRescale[Array]
    ) -> tuple[TimeIntegrator[Array], Array, Array]:
        model = LotkaVolterra(bkd)
        model.set_param(bkd.asarray(_PARAM))
        integrator = _integrator(model, constraint)
        states, times = integrator.solve(_init_state(bkd))
        return integrator, states, times

    def test_adjoint_gradient_refuses_a_constraint(self, bkd: Backend[Array]) -> None:
        integrator, states, times = self._solved(bkd, _PinAndRescale(bkd))
        integrator.set_functional(AllStatesEndpointFunctional(1, _NPARAMS, bkd))
        with pytest.raises(NotImplementedError, match="adjoint solve"):
            integrator.gradient(states, times, _sample(bkd))

    def test_refuses_another_trajectory(self, bkd: Backend[Array]) -> None:
        integrator, states, times = self._solved(bkd, _PinAndRescale(bkd))
        functional = AllStatesEndpointFunctional(_NSTATES, _NPARAMS, bkd)
        with pytest.raises(ValueError, match="last solve"):
            forward_sensitivity_jacobian(
                integrator, functional, 2.0 * states, times, _sample(bkd)
            )
