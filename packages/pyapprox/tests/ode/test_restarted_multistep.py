r"""A multistep run that restarts its history mid-run, differentiated.

Lotka-Volterra with the mass ``[[2, 0.5], [0.5, 1]]`` from (10, 5) at
``LV_PARAM``, ten steps of 0.05 to 0.5, under the Tikhonov-augmented MSE
of ``native_tikhonov_mse``; BDF2 (implicit) and Adams-Bashforth 2
(explicit). The step plan restarts the history at step 5 (t = 0.25):
that step reads one state, taking the method's startup formula, though a
full step reads two. The store records how many states each step read:

- The restarted step's window holds one state and a one-size span; the
  steps either side hold two.
- The restart changes the trajectory (the last state differs from the
  unrestarted run's by more than 1e-6), so it is not a no-op.
- The adjoint gradient passes central differences of the whole solve
  (smallest relative error below 1e-7), and the HVP passes central
  differences of the gradient.
- Wrong-formula control: the same recorded run read back as if every
  step had read the method's depth (the full formula at the restarted
  step) gives a gradient off by more than 1e-5 relative, a hundred times
  the finite-difference tolerance the right gradient meets (Adams-Bashforth
  2: 8.5e-4; BDF2: above 1e-3).
"""

from typing import Callable, List, Optional, Sequence, Tuple, TypeVar

import pytest

from pyapprox.ode.integration import Integrator
from pyapprox.ode.multistep_methods import (
    MultistepFormulaProtocol,
    adams_bashforth2,
    bdf2,
)
from pyapprox.ode.protocols import StepTakerProtocol
from pyapprox.ode.step_plan import TakenStep
from pyapprox.ode.step_span import StepSpan
from pyapprox.ode.stepper_derivatives import StepWindow
from pyapprox.ode.trajectory_derivatives import TrajectoryDerivatives
from pyapprox.ode.trajectory_store import InMemoryTrajectoryStore
from pyapprox.util.backends.protocols import Array, Backend
from tests._helpers.ode.runge_kutta_cases import LV_PARAM
from tests._helpers.ode.trajectory_cases import (
    START,
    TIMES,
    TrajectoryProblem,
    min_gradient_fd_error,
    min_hvp_fd_error,
)

W = TypeVar("W")
_RESTART = 5
_SIZES = tuple(later - earlier for earlier, later in zip(TIMES, TIMES[1:]))


class _RestartingSequence:
    """Steps of the given sizes, each reading the states there are (at
    most one more than the step before read), except the restart step,
    which reads the latest state alone."""

    def __init__(self, origin: float, sizes: Tuple[float, ...], restart: int) -> None:
        self._origin, self._sizes, self._restart = origin, sizes, restart
        self._taken = 0
        self._last: Optional[StepSpan] = None

    def next_step(
        self, stepper: StepTakerProtocol[W], past_states: Sequence[W]
    ) -> Optional[TakenStep[W]]:
        if self._taken == len(self._sizes):
            return None
        size = self._sizes[self._taken]
        if self._last is None or self._taken == self._restart:
            start = self._origin if self._last is None else self._last.end()
            span = StepSpan.first(start, size)
        else:
            nread = min(len(past_states), len(self._last.sizes) + 1)
            span = self._last.advanced(size, nread)
        unknown = stepper.step(past_states[: len(span.sizes)], span)
        self._last = span
        self._taken += 1
        return TakenStep(span, unknown)


class _RestartingPlan:
    """:data:`TIMES`' steps, the history restarted at ``restart``."""

    def __init__(self, restart: int) -> None:
        self._restart = restart

    def origin(self) -> float:
        return TIMES[0]

    def start(self) -> _RestartingSequence:
        return _RestartingSequence(TIMES[0], _SIZES, self._restart)


class _FullDepthStore(InMemoryTrajectoryStore[Array]):
    """Reads every window back as if its step had read the method's
    depth of states (or all there are at startup): the wrong formula at a
    restarted step. Only windows are read so; a gradient reads nothing
    else of a step."""

    def __init__(self, depth: int) -> None:
        super().__init__()
        self._depth = depth

    def window(self, step: int) -> StepWindow[Array]:
        recorded = super().window(step)
        step %= self.nsteps()
        nread = min(self._depth, step + 1)
        span = StepSpan.from_times(self.times()[step + 1 - nread : step + 2])
        past = tuple(self.state(step - lag) for lag in range(nread))
        return StepWindow(
            span,
            past,
            recorded.unknown,
            recorded.parameters_version,
            recorded.origin,
        )


_METHODS: List[Callable[[], MultistepFormulaProtocol]] = [bdf2, adams_bashforth2]
_IDS = ["bdf2", "adams_bashforth2"]


def _problem(
    bkd: Backend[Array], make: Callable[[], MultistepFormulaProtocol]
) -> TrajectoryProblem[Array]:
    return TrajectoryProblem(bkd, make(), plan=_RestartingPlan(_RESTART))


class TestRestartedMultistep:
    @pytest.mark.parametrize("make", _METHODS, ids=_IDS)
    def test_the_store_records_the_restart(
        self, bkd: Backend[Array], make: Callable[[], MultistepFormulaProtocol]
    ) -> None:
        param = bkd.asarray(LV_PARAM)
        store = _problem(bkd, make).solve(param)
        assert store.times() == tuple(TIMES)
        reads = [len(store.window(n).past_states) for n in range(store.nsteps())]
        assert reads == [1, 2, 2, 2, 2, 1, 2, 2, 2, 2]
        assert store.span(_RESTART).times() == (TIMES[5], TIMES[6])
        plain = TrajectoryProblem(bkd, make()).solve(param)
        difference = bkd.to_float(bkd.max(bkd.abs(store.state(-1) - plain.state(-1))))
        assert difference > 1e-6

    @pytest.mark.parametrize("make", _METHODS, ids=_IDS)
    def test_gradient_matches_finite_differences(
        self, bkd: Backend[Array], make: Callable[[], MultistepFormulaProtocol]
    ) -> None:
        assert min_gradient_fd_error(bkd, _problem(bkd, make)) < 1e-7

    @pytest.mark.parametrize("make", _METHODS, ids=_IDS)
    def test_hvp_matches_finite_differences(
        self, bkd: Backend[Array], make: Callable[[], MultistepFormulaProtocol]
    ) -> None:
        assert min_hvp_fd_error(bkd, _problem(bkd, make)) < 1e-7

    @pytest.mark.parametrize("make", _METHODS, ids=_IDS)
    def test_the_full_depth_formula_is_wrong(
        self, bkd: Backend[Array], make: Callable[[], MultistepFormulaProtocol]
    ) -> None:
        param = bkd.asarray(LV_PARAM)
        problem = _problem(bkd, make)
        gradient = problem.gradient(param)
        # The same run, recorded into a store that reads it back at the
        # method's depth; the model is at param from the solve above.
        stepper = problem.stepper()
        full: _FullDepthStore[Array] = _FullDepthStore(stepper.depth())
        Integrator(stepper).solve(bkd.asarray(START), _RestartingPlan(_RESTART), full)
        sweeps: TrajectoryDerivatives[Array, Array] = TrajectoryDerivatives(
            stepper, None, None
        )
        wrong = sweeps.gradient(full, problem.coupling(full, param)).T
        error = bkd.to_float(bkd.norm(wrong - gradient) / bkd.norm(gradient))
        assert error > 1e-5
