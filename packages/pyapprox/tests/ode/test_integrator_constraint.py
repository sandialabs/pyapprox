"""The time integrator applies a state constraint at stored states.

With no constraint (the identity) the stored states are exactly what an
unconstrained integrator produces. With one, every stored state, the
initial one included, satisfies it; each step starts from the stored
state; and the unprojected states keep the given initial state and each
step's Newton solution. The reference for a step is an independent
unconstrained integrator started from the stored state.
"""

from typing import Generic

import pytest

from pyapprox.ode.implicit_steppers.backward_euler import BackwardEulerStepper
from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
from pyapprox.ode.protocols import StateConstraintProtocol
from pyapprox.ode.residual_derivatives import backend_state_space
from pyapprox.ode.state_constraint import (
    IdentityConstraint,
    StateConstraintDerivatives,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.rootfinding.newton import NewtonSolver
from tests._helpers.ode_models import LotkaVolterra

_DELTAT = 0.1
_FINAL_TIME = 0.5


class _PinFirst(Generic[Array]):
    r"""Fixes the first entry to :math:`1 + t`."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def constrain(self, state: Array, time: float) -> Array:
        pinned = self._bkd.asarray([1.0 + time])
        return self._bkd.concatenate([pinned, state[1:]])

    def derivatives(self) -> StateConstraintDerivatives[Array]:
        return StateConstraintDerivatives.none()


def _integrator(
    bkd: Backend[Array],
    init_time: float,
    final_time: float,
    constraint: StateConstraintProtocol[Array] | None = None,
) -> TimeIntegrator[Array]:
    newton = NewtonSolver(BackwardEulerStepper(LotkaVolterra(bkd)))
    newton.set_options(maxiters=20, atol=1e-13, rtol=0.0)
    return TimeIntegrator(init_time, final_time, _DELTAT, newton, constraint=constraint)


def _init_state(bkd: Backend[Array]) -> Array:
    return bkd.asarray([10.0, 5.0])


class TestIdentity:
    def test_default_and_identity_store_the_unconstrained_states(
        self, bkd: Backend[Array]
    ) -> None:
        default = _integrator(bkd, 0.0, _FINAL_TIME)
        states, _ = default.solve(_init_state(bkd))
        identity = _integrator(
            bkd, 0.0, _FINAL_TIME, IdentityConstraint(backend_state_space(bkd, 2))
        )
        identity_states, _ = identity.solve(_init_state(bkd))
        bkd.assert_allclose(identity_states, states, rtol=0.0, atol=0.0)
        bkd.assert_allclose(default.unprojected_states(), states, rtol=0.0, atol=0.0)


class TestConstraint:
    def test_every_stored_state_satisfies_it(self, bkd: Backend[Array]) -> None:
        integrator = _integrator(bkd, 0.0, _FINAL_TIME, _PinFirst(bkd))
        states, times = integrator.solve(_init_state(bkd))
        bkd.assert_allclose(states[0], 1.0 + times, rtol=1e-14)

    def test_each_step_starts_from_the_stored_state(self, bkd: Backend[Array]) -> None:
        r"""Step :math:`n`'s Newton solution :math:`z_{n+1}` is a raw
        (unconstrained) step from the stored :math:`y_n`, and the stored
        :math:`y_{n+1}` is its projection, in every entry. Control: a raw
        step from :math:`z_n`, which differs from :math:`y_n` in the
        pinned entry, gives something else, so the check tells the two
        starts apart."""
        constraint = _PinFirst(bkd)
        integrator = _integrator(bkd, 0.0, _FINAL_TIME, constraint)
        states, times = integrator.solve(_init_state(bkd))
        unprojected = integrator.unprojected_states()
        for n in range(times.shape[0] - 1):
            t_n = bkd.to_float(times[n])
            from_stored, _ = _integrator(bkd, t_n, t_n + _DELTAT).solve(states[:, n])
            bkd.assert_allclose(unprojected[:, n + 1], from_stored[:, 1], rtol=1e-12)
            bkd.assert_allclose(
                states[:, n + 1],
                constraint.constrain(from_stored[:, 1], t_n + _DELTAT),
                rtol=1e-12,
            )
            from_unprojected, _ = _integrator(bkd, t_n, t_n + _DELTAT).solve(
                unprojected[:, n]
            )
            gap = bkd.max(bkd.abs(from_unprojected[:, 1] - unprojected[:, n + 1]))
            assert bkd.to_float(gap) > 1e-3

    def test_unprojected_states_keep_the_given_initial_state(
        self, bkd: Backend[Array]
    ) -> None:
        integrator = _integrator(bkd, 0.0, _FINAL_TIME, _PinFirst(bkd))
        states, _ = integrator.solve(_init_state(bkd))
        bkd.assert_allclose(
            integrator.unprojected_states()[:, 0], _init_state(bkd), rtol=0.0
        )
        assert bkd.to_float(states[0, 0]) == 1.0


class TestErrors:
    def test_unprojected_states_before_a_solve(self, bkd: Backend[Array]) -> None:
        with pytest.raises(RuntimeError, match="after solve"):
            _integrator(bkd, 0.0, _FINAL_TIME).unprojected_states()

    def test_refuses_a_non_constraint(self, bkd: Backend[Array]) -> None:
        newton = NewtonSolver(BackwardEulerStepper(LotkaVolterra(bkd)))
        with pytest.raises(TypeError, match="StateConstraintProtocol"):
            TimeIntegrator(0.0, 1.0, 0.1, newton, constraint=object())  # type: ignore[arg-type]
