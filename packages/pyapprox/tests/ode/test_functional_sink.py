r"""The value engine: a functional's terms evaluated as the trajectory
arrives.

:math:`M \dot y = K y` (3 states, a nontrivial mass) from (1, -0.5, 2)
over unequal dyadic steps, with a time integral of :math:`q = \cos(t)\,
y^\top y` and an endpoint term :math:`\mathbf 1^\top y_N`.

- Injected combinations: the recursion run forward
  (:class:`RunningRecursion`) and kept and weighed at the end by its
  transpose (:class:`StoredContributions`) give the same integral to
  1e-13, for Crank-Nicolson and for BDF2 (whose weights are not 1); a
  plain sum is the sum of the contributions.
- The sink's value equals the same terms replayed over the recorded
  trajectory (each state term at the states it reads, each step term at
  each step's window, combined alike), so the value and the derivatives
  read the same states.
- Streaming: an endpoint term reads the newest state as the last, so the
  value after any number of steps is that state's, though the length of
  the trajectory is not known; a parameter term's value is added once.
- Refusals: a value before any state, a step before the initial state,
  a second initial state, a step not starting where the last ended.
"""

import math
from typing import Any, Callable, Generic

import pytest

from pyapprox.ode.accumulating_sink import MultiSink
from pyapprox.ode.functional_sink import (
    FunctionalSink,
    PlainSum,
    RunningRecursion,
    StepCombinationProtocol,
    StoredContributions,
)
from pyapprox.ode.integration import Integrator
from pyapprox.ode.mass_form_system import dense_mass_form_system
from pyapprox.ode.multistep_methods import bdf2
from pyapprox.ode.step_plan import GivenSizes
from pyapprox.ode.step_quadrature import StepQuadratureProtocol
from pyapprox.ode.step_span import StepSpan
from pyapprox.ode.stepper_derivatives import StepWindow
from pyapprox.ode.tableau import crank_nicolson
from pyapprox.ode.time_integral import IntegrandDerivatives, TimeIntegral
from pyapprox.ode.trajectory_store import InMemoryTrajectoryStore
from pyapprox.ode.trajectory_terms import (
    ParameterTermDerivatives,
    StateTermDerivatives,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.operators import BlockVector
from tests._helpers.ode.multistep_cases import lm_stepper
from tests._helpers.ode.runge_kutta_cases import LinearMassModel, rk_stepper

_Y0 = [[1.0], [-0.5], [2.0]]
_SIZES = (0.125, 0.25, 0.0625, 0.5, 0.125, 0.25)


class _Energy(Generic[Array]):
    r""":math:`q(y, t) = \cos(t)\, y^\top y`, value only."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def nqoi(self) -> int:
        return 1

    def value(self, state: Array, time: float) -> Array:
        return math.cos(time) * self._bkd.dot(state.T, state)

    def derivatives(self) -> IntegrandDerivatives[Array, Array]:
        return IntegrandDerivatives.none()


class _Endpoint(Generic[Array]):
    r"""The state term :math:`\mathbf 1^\top y_N`: the last state only."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def nqoi(self) -> int:
        return 1

    def reads(self, index: int, last: bool) -> bool:
        return last

    def value(self, index: int, time: float, state: Array) -> Array:
        return self._bkd.sum(state, axis=0, keepdims=True)

    def derivatives(self) -> StateTermDerivatives[Array, Array]:
        return StateTermDerivatives.none()


class _Constant(Generic[Array]):
    """A parameter term of value 0.5."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def nqoi(self) -> int:
        return 1

    def value(self) -> Array:
        return self._bkd.full((1, 1), 0.5)

    def derivatives(self) -> ParameterTermDerivatives[Array, Array]:
        return ParameterTermDerivatives.none()


def _stepper(bkd: Backend[Array], method: str) -> Any:
    system = dense_mass_form_system(LinearMassModel(bkd))
    if method == "BDF2":
        return lm_stepper(bdf2(), system)
    return rk_stepper(crank_nicolson(), system)


Combination = Callable[[StepQuadratureProtocol, int], StepCombinationProtocol[Any]]


def _integrate(
    bkd: Backend[Array], method: str, combination: Combination
) -> tuple[Array, InMemoryTrajectoryStore[Array], TimeIntegral[Array, Array]]:
    stepper = _stepper(bkd, method)
    integral: TimeIntegral[Array, Array] = TimeIntegral(
        _Energy(bkd), stepper, bkd, combination
    )
    sink = integral.sink()
    store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
    Integrator(stepper).solve(
        bkd.asarray(_Y0), GivenSizes(0.0, _SIZES), MultiSink([sink, store])
    )
    return sink.value(), store, integral


class TestCombinations:
    @pytest.mark.parametrize("method", ["Crank-Nicolson", "BDF2"])
    def test_running_and_stored_agree(self, bkd: Backend[Array], method: str) -> None:
        running, _, _ = _integrate(bkd, method, RunningRecursion)
        stored, _, _ = _integrate(bkd, method, StoredContributions)
        bkd.assert_allclose(running, stored, rtol=1e-13)

    def test_plain_sum(self, bkd: Backend[Array]) -> None:
        combination: PlainSum[Array] = PlainSum()
        assert combination.value() is None
        span = StepSpan.first(0.0, 1.0)
        for value in (1.0, 2.0, 4.0):
            combination.add(span, bkd.full((1, 1), value))
        total = combination.value()
        assert total is not None
        bkd.assert_allclose(total, bkd.full((1, 1), 7.0))


class TestReplay:
    @pytest.mark.parametrize("method", ["Crank-Nicolson", "BDF2"])
    def test_the_sink_is_the_terms_over_the_store(
        self, bkd: Backend[Array], method: str
    ) -> None:
        """The integral and the endpoint through one sink equal the same
        terms evaluated over the recorded trajectory."""
        stepper = _stepper(bkd, method)
        integral: TimeIntegral[Array, Array] = TimeIntegral(
            _Energy(bkd), stepper, bkd
        )
        endpoint = _Endpoint(bkd)
        sink: FunctionalSink[Array, Array] = FunctionalSink(
            bkd.zeros((1, 1)),
            state_terms=(endpoint,),
            step_terms=(
                (
                    integral.step_term(),
                    RunningRecursion(integral.quadrature(), stepper.depth()),
                ),
            ),
        )
        store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
        Integrator(stepper).solve(
            bkd.asarray(_Y0), GivenSizes(0.0, _SIZES), MultiSink([sink, store])
        )
        ntimes = store.ntimes()
        replay = bkd.zeros((1, 1))
        for index in range(ntimes):
            if endpoint.reads(index, index == ntimes - 1):
                replay = replay + endpoint.value(
                    index, store.time(index), store.state(index)
                )
        combination = RunningRecursion[Array](integral.quadrature(), stepper.depth())
        for step in range(store.nsteps()):
            window = store.window(step, min(stepper.depth(), step + 1))
            combination.add(window.span, integral.step_term().value(step, window))
        combined = combination.value()
        assert combined is not None
        bkd.assert_allclose(sink.value(), replay + combined, rtol=1e-14)


class TestStreaming:
    def test_the_newest_state_is_read_as_the_last(
        self, bkd: Backend[Array]
    ) -> None:
        sink: FunctionalSink[Array, Array] = FunctionalSink(
            bkd.zeros((1, 1)),
            parameter_terms=(_Constant(bkd),),
            state_terms=(_Endpoint(bkd),),
        )
        states = [bkd.full((3, 1), float(n)) for n in range(4)]
        sink.append(0.0, states[0])
        bkd.assert_allclose(sink.value(), bkd.full((1, 1), 0.5))
        span = StepSpan.first(0.0, 0.25)
        unknown = BlockVector((states[0],))
        for n in range(1, 4):
            sink.append_step(StepWindow(span, (states[n - 1],), unknown), states[n])
            # The endpoint is always the newest state, never an earlier one.
            bkd.assert_allclose(sink.value(), bkd.full((1, 1), 3.0 * n + 0.5))
            span = span.advanced(0.25, 1)
        assert sink.ntimes() == 4


class TestRefusals:
    def test_out_of_order(self, bkd: Backend[Array]) -> None:
        sink: FunctionalSink[Array, Array] = FunctionalSink(bkd.zeros((1, 1)))
        y = bkd.ones((3, 1))
        window = StepWindow(StepSpan.first(0.0, 0.5), (y,), BlockVector((y,)))
        with pytest.raises(ValueError, match="no state has been received"):
            sink.value()
        with pytest.raises(ValueError, match="append one first"):
            sink.append_step(window, y)
        sink.append(0.0, y)
        with pytest.raises(ValueError, match="only the initial state"):
            sink.append(0.0, y)
        late = StepWindow(StepSpan.first(0.25, 0.5), (y,), BlockVector((y,)))
        with pytest.raises(ValueError, match="not where the last one ended"):
            sink.append_step(late, y)
