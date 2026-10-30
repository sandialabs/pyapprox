r"""The integrator: marching a stepper and recording the trajectory.

On Lotka-Volterra with a mass, from (10, 5), dt = 0.05 to T = 0.5: for
backward and forward Euler, Crank-Nicolson, implicit midpoint and Heun,
the new integrator's trajectory equals the old path's (the tiered
stepper of the same method through the old ``TimeIntegrator``) to 1e-10
relative, both stepped to Newton tolerance 1e-13; each recorded window
reproduces its step (the step residual is below 1e-10 there) and holds
the step's unknown; the times are recorded as given.

A constraint is applied at the stored states only: a pin of the second
component to 3 holds at every stored state, initial state included,
while the recorded unknowns, where the steps are linearized, are left
as solved. The identity constraint through the residual form changes
nothing. Malformed times and a non-empty store are refused.
"""

from typing import Callable, Generic, List, Optional

import pytest

from pyapprox.ode.accumulating_sink import PerStateSink
from pyapprox.ode.explicit_steppers import ForwardEulerHVP, HeunHVP
from pyapprox.ode.implicit_steppers import (
    BackwardEulerHVP,
    CrankNicolsonHVP,
    ImplicitMidpointHVP,
)
from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
from pyapprox.ode.integration import (
    Integrator,
    ResidualFormConstraint,
    StackConstraintProtocol,
)
from pyapprox.ode.mass_form_system import dense_mass_form_system
from pyapprox.ode.protocols import TimeSteppingResidualProtocol
from pyapprox.ode.residual_derivatives import backend_spaces
from pyapprox.ode.residual_form import BackendColumnForm
from pyapprox.ode.state_constraint import IdentityConstraint
from pyapprox.ode.tableau import (
    ButcherTableau,
    backward_euler,
    crank_nicolson,
    forward_euler,
    heun,
    implicit_midpoint,
)
from pyapprox.ode.trajectory_store import (
    InMemoryTrajectoryStore,
    TrajectorySinkProtocol,
    TrajectoryStoreProtocol,
    trajectory_array,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.rootfinding.newton import NewtonSolver
from tests._helpers.ode.runge_kutta_cases import (
    lotka_volterra_with_mass,
    rk_stepper,
)

_TIMES = [0.05 * n for n in range(11)]
_START = [[10.0], [5.0]]

# The old path's stepper class for each tableau, for the same method.
_OLD_STEPPERS: list[tuple[Callable[[], ButcherTableau], Callable[..., object]]] = [
    (backward_euler, BackwardEulerHVP),
    (forward_euler, ForwardEulerHVP),
    (crank_nicolson, CrankNicolsonHVP),
    (implicit_midpoint, ImplicitMidpointHVP),
    (heun, HeunHVP),
]
_OLD_IDS = [make().name for make, _ in _OLD_STEPPERS]


def _new_trajectory(
    bkd: Backend[Array], tableau: ButcherTableau
) -> InMemoryTrajectoryStore[Array]:
    stepper = rk_stepper(tableau, dense_mass_form_system(lotka_volterra_with_mass(bkd)))
    store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
    Integrator(stepper).solve(bkd.asarray(_START), _TIMES, store)
    return store


class TestAgainstTheOldPath:
    @pytest.mark.parametrize("make, old", _OLD_STEPPERS, ids=_OLD_IDS)
    def test_same_trajectory(
        self,
        bkd: Backend[Array],
        make: Callable[[], ButcherTableau],
        old: Callable[..., object],
    ) -> None:
        store = _new_trajectory(bkd, make())
        old_stepper = old(lotka_volterra_with_mass(bkd))
        assert isinstance(old_stepper, TimeSteppingResidualProtocol)
        newton: NewtonSolver[Array] = NewtonSolver(old_stepper)
        newton.set_options(atol=1e-13, rtol=0.0, maxiters=30)
        old_states, old_times = TimeIntegrator(0.0, 0.5, 0.05, newton).solve(
            bkd.asarray([10.0, 5.0])
        )
        new_states = bkd.hstack([store.state(n) for n in range(store.ntimes())])
        bkd.assert_allclose(new_states, old_states, rtol=1e-10)
        bkd.assert_allclose(bkd.asarray(store.times()), old_times, rtol=1e-12)


class TestRecordedWindows:
    @pytest.mark.parametrize(
        "times",
        [_TIMES, [0.0, 0.05, 0.15, 0.2, 0.32, 0.4, 0.43]],
        ids=["uniform", "uneven_short_last"],
    )
    @pytest.mark.parametrize("make", [backward_euler, heun, implicit_midpoint])
    def test_each_window_reproduces_its_step(
        self,
        bkd: Backend[Array],
        make: Callable[[], ButcherTableau],
        times: List[float],
    ) -> None:
        tableau = make()
        stepper = rk_stepper(
            tableau, dense_mass_form_system(lotka_volterra_with_mass(bkd))
        )
        store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
        Integrator(stepper).solve(bkd.asarray(_START), times, store)
        assert store.nsteps() == len(times) - 1
        assert store.times() == tuple(times)
        for n in range(store.nsteps()):
            window = store.window(n)
            assert window.times == (times[n + 1], times[n])
            residual = stepper.residual_space().norm(stepper.step_residual(window))
            assert bkd.to_float(residual[0, 0]) < 1e-10
            stepped = stepper.readout().apply(window.unknown).blocks[0]
            bkd.assert_allclose(stepped, store.state(n + 1), rtol=0.0)


class _PinSecond(Generic[Array]):
    """Pins the second component of a two-state column to 3."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def constrain(self, state: Array, time: float) -> Array:
        return self._bkd.vstack([state[0:1], self._bkd.asarray([[3.0]])])


class TestConstraint:
    def test_applied_at_stored_states_only(self, bkd: Backend[Array]) -> None:
        pin: _PinSecond[Array] = _PinSecond(bkd)
        assert isinstance(pin, StackConstraintProtocol)
        stepper = rk_stepper(
            backward_euler(), dense_mass_form_system(lotka_volterra_with_mass(bkd))
        )
        store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
        Integrator(stepper, pin).solve(bkd.asarray(_START), _TIMES, store)
        for n in range(store.ntimes()):
            assert bkd.to_float(store.state(n)[1, 0]) == 3.0
        unknowns = [store.window(n).unknown.blocks[0] for n in range(store.nsteps())]
        assert any(abs(bkd.to_float(u[1, 0]) - 3.0) > 1e-3 for u in unknowns)

    def test_identity_through_the_residual_form_changes_nothing(
        self, bkd: Backend[Array]
    ) -> None:
        stepper = rk_stepper(
            heun(), dense_mass_form_system(lotka_volterra_with_mass(bkd))
        )
        states, params = backend_spaces(bkd, 2, 4)
        identity = ResidualFormConstraint(
            IdentityConstraint(states, params), BackendColumnForm(bkd)
        )
        constrained: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
        Integrator(stepper, identity).solve(bkd.asarray(_START), _TIMES, constrained)
        plain = _new_trajectory(bkd, heun())
        for n in range(plain.ntimes()):
            bkd.assert_allclose(constrained.state(n), plain.state(n), rtol=0.0)


class _WeightedSum(Generic[Array]):
    r"""Accumulates :math:`\sum_m c^\top y_m` as states arrive and keeps
    none of them."""

    def __init__(self, weight: Array) -> None:
        self._weight = weight
        self._total: Optional[Array] = None

    def add(self, index: int, time: float, state: Array) -> None:
        term = self._weight.T @ state
        self._total = term if self._total is None else self._total + term

    def value(self) -> Array:
        if self._total is None:
            raise ValueError("no state received")
        return self._total


class TestSinks:
    def test_a_sink_keeping_nothing(self, bkd: Backend[Array]) -> None:
        """A sink accumulating a sum of the states as they arrive gives
        the sum over the stored trajectory, with no states kept."""
        weight = bkd.asarray([[0.3], [-1.2]])
        sink: PerStateSink[Array, Array] = PerStateSink(_WeightedSum(weight))
        assert isinstance(sink, TrajectorySinkProtocol)
        assert not isinstance(sink, TrajectoryStoreProtocol)
        stepper = rk_stepper(
            heun(), dense_mass_form_system(lotka_volterra_with_mass(bkd))
        )
        Integrator(stepper).solve(bkd.asarray(_START), _TIMES, sink)
        store = _new_trajectory(bkd, heun())
        expected = bkd.reshape(
            bkd.sum(weight.T @ trajectory_array(store, bkd)), (1, 1)
        )
        assert sink.ntimes() == store.ntimes()
        bkd.assert_allclose(sink.value(), expected, rtol=1e-12)


class TestRefusals:
    def test_times_and_store(self, bkd: Backend[Array]) -> None:
        stepper = rk_stepper(
            forward_euler(), dense_mass_form_system(lotka_volterra_with_mass(bkd))
        )
        integrator = Integrator(stepper)
        start = bkd.asarray(_START)
        with pytest.raises(ValueError, match="at least two"):
            integrator.solve(start, [0.0], InMemoryTrajectoryStore())
        with pytest.raises(ValueError, match="strictly increase"):
            integrator.solve(start, [0.0, 0.1, 0.1], InMemoryTrajectoryStore())
        used: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
        used.append(0.0, start)
        with pytest.raises(ValueError, match="must be empty"):
            integrator.solve(start, [0.0, 0.1], used)

    def test_a_non_stepper_is_refused(self) -> None:
        with pytest.raises(TypeError, match="ForwardStepperProtocol"):
            Integrator(object())  # type: ignore[arg-type]
