"""Tests for the trajectory store."""

from typing import Sequence, TypeVar

import numpy as np
import pytest

from pyapprox.ode.step_span import StepSpan
from pyapprox.ode.stepper_derivatives import StepWindow
from pyapprox.ode.trajectory_store import (
    InMemoryTrajectoryStore,
    TrajectoryStoreProtocol,
    trajectory_array,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.operators import BlockVector, SpaceDescriptor
from tests._helpers.linalg.mock_stack import MockVector, peek, wrap

_TIMES = (0.0, 0.25, 0.5, 1.0)
V = TypeVar("V")


def _states(bkd: Backend[Array], shape: tuple[int, ...]) -> list[Array]:
    rng = np.random.default_rng(0)
    return [bkd.asarray(rng.normal(size=shape)) for _ in _TIMES]


def _filled(store: InMemoryTrajectoryStore[V], states: Sequence[V]) -> None:
    """``states`` at ``_TIMES``: the first appended, each later one as a
    step whose unknown is the state itself."""
    store.append(_TIMES[0], states[0])
    for n, state in enumerate(states[1:]):
        span = StepSpan.from_times(_TIMES[n : n + 2])
        window = StepWindow(span, (states[n],), BlockVector((state,)), 0)
        store.append_step(window, state)


class TestInMemoryTrajectoryStore:
    def test_stores_states_and_times_in_order(self, bkd: Backend[Array]) -> None:
        store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
        assert isinstance(store, TrajectoryStoreProtocol)
        states = _states(bkd, (4,))
        _filled(store, states)
        assert store.ntimes() == 4
        assert store.times() == _TIMES
        assert store.time(-2) == 0.5
        assert store.state(1) is states[1]
        assert store.state(-1) is states[-1]

    def test_times_are_derived_from_the_sizes(self, bkd: Backend[Array]) -> None:
        """Steps of 0.1 from 0, each reading up to three states: the
        stored times are the running sums, not 0.1 n, and every window's
        span gives them back bit for bit."""
        store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
        state = bkd.zeros((2, 1))
        store.append(0.0, state)
        span = StepSpan.first(0.0, 0.1)
        for n in range(10):
            read = (state,) * len(span.sizes)
            store.append_step(StepWindow(span, read, BlockVector((state,))), state)
            span = span.advanced(0.1, min(3, n + 2))
        sums = [0.0]
        for _ in range(10):
            sums.append(sums[-1] + 0.1)
        assert store.times() == tuple(sums)
        assert store.time(-1) != 1.0
        for step in range(2, 10):
            expected = tuple(sums[step - 2 : step + 2])
            assert store.window(step).span.times() == expected
            assert store.window(step).deltat() == 0.1

    def test_refuses_a_second_initial_state_and_a_misplaced_step(
        self, bkd: Backend[Array]
    ) -> None:
        store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
        state = bkd.zeros((3, 1))
        store.append(0.0, state)
        with pytest.raises(ValueError, match="only the initial state"):
            store.append(1.0, state)
        with pytest.raises(ValueError, match="not at the last stored time"):
            store.append_step(
                StepWindow(StepSpan.first(0.5, 0.5), (state,), BlockVector((state,))),
                state,
            )

    def test_index_out_of_range(self, bkd: Backend[Array]) -> None:
        store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
        store.append(0.0, bkd.zeros((3,)))
        with pytest.raises(IndexError):
            store.state(1)
        with pytest.raises(IndexError):
            store.time(-2)

    def test_holds_opaque_vectors(self) -> None:
        """Nothing in the store looks inside a state."""
        space = SpaceDescriptor("u", 3)
        store: InMemoryTrajectoryStore[MockVector] = InMemoryTrajectoryStore()
        _filled(store, [wrap(np.full((3, 1), time), space) for time in _TIMES])
        np.testing.assert_array_equal(peek(store.state(2)), np.full((3, 1), 0.5))


def _stepped_store(
    bkd: Backend[Array], reads: Sequence[int] = (1, 2, 3)
) -> tuple[InMemoryTrajectoryStore[Array], list[Array], list[BlockVector[Array]]]:
    """A store of the four states, each later one appended as a step
    reading ``reads[n]`` states, whose unknown is two blocks and whose
    parameter version is its index."""
    states = _states(bkd, (2, 1))
    unknowns = [BlockVector((state, 2.0 * state)) for state in states[1:]]
    store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
    store.append(_TIMES[0], states[0])
    for n, state in enumerate(states[1:]):
        nread = reads[n]
        span = StepSpan.from_times(_TIMES[n + 1 - nread : n + 2])
        past = tuple(states[n - lag] for lag in range(nread))
        store.append_step(StepWindow(span, past, unknowns[n], n), state)
    return store, states, unknowns


class TestStepWindows:
    def test_window_holds_the_step(self, bkd: Backend[Array]) -> None:
        store, states, unknowns = _stepped_store(bkd)
        assert store.nsteps() == 3
        window = store.window(1)
        assert window.time() == 0.25
        assert window.deltat() == 0.25
        assert window.past_states == (states[1], states[0])
        assert window.unknown is unknowns[1]
        assert window.parameters_version == 1
        assert store.window(-1).unknown is unknowns[2]

    def test_deeper_windows_read_past_states_most_recent_first(
        self, bkd: Backend[Array]
    ) -> None:
        store, states, _ = _stepped_store(bkd)
        assert store.window(2).past_states == (states[2], states[1], states[0])

    def test_a_restarted_step_reads_what_it_read(self, bkd: Backend[Array]) -> None:
        """A step reading one state after a step that read two (a
        multistep history restarted mid-run) is read back with one state
        and one size, not the method's depth."""
        store, states, _ = _stepped_store(bkd, reads=(1, 2, 1))
        window = store.window(2)
        assert window.past_states == (states[2],)
        assert window.span.times() == (0.5, 1.0)
        assert store.span(2) == window.span
        assert store.window(1).span.times() == (0.0, 0.25, 0.5)

    def test_windows_carry_their_origin(self, bkd: Backend[Array]) -> None:
        """Every read of a step's window names the same step of this
        store's trajectory, by ``-1`` too; other steps and other stores
        name others; a window built by hand names none."""
        store, states, unknowns = _stepped_store(bkd)
        origin = store.window(2).origin
        assert origin is not None and origin.step() == 2
        assert store.window(2).origin == origin
        assert store.window(-1).origin == origin
        assert store.window(1).origin != origin
        other, _, _ = _stepped_store(bkd)
        assert other.window(2).origin != origin
        span = StepSpan.from_times(_TIMES[:2])
        assert StepWindow(span, (states[0],), unknowns[0]).origin is None

    def test_span_is_the_windows_span(self, bkd: Backend[Array]) -> None:
        """The span read alone equals the window's at every step, the last
        step by ``-1`` too."""
        store, _, _ = _stepped_store(bkd)
        for step in range(store.nsteps()):
            assert store.span(step) == store.window(step).span
        assert store.span(-1) == store.span(2)

    def test_refusals(self, bkd: Backend[Array]) -> None:
        empty: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
        state = bkd.zeros((2, 1))
        with pytest.raises(ValueError, match="append one first"):
            empty.append_step(
                StepWindow(StepSpan.first(0.0, 1.0), (state,), BlockVector((state,))),
                state,
            )
        store, _, _ = _stepped_store(bkd)
        with pytest.raises(IndexError):
            store.window(3)
        with pytest.raises(IndexError):
            store.window(-4)
        with pytest.raises(IndexError):
            store.span(3)
        short: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
        short.append(0.25, state)
        with pytest.raises(ValueError, match="reads 2 states, but 1 are stored"):
            short.append_step(
                StepWindow(
                    StepSpan.from_times((0.0, 0.25, 0.5)),
                    (state, state),
                    BlockVector((state,)),
                ),
                state,
            )


class TestTrajectoryArray:
    @pytest.mark.parametrize("shape", [(4,), (4, 1)], ids=["1d", "column"])
    def test_stacks_states_as_columns(
        self, bkd: Backend[Array], shape: tuple[int, ...]
    ) -> None:
        """Today's (n,) states and (n, 1) blocks give the same array."""
        store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
        states = _states(bkd, shape)
        _filled(store, states)
        array = trajectory_array(store, bkd)
        assert tuple(array.shape) == (4, 4)
        bkd.assert_allclose(
            array, bkd.stack([bkd.reshape(s, (-1,)) for s in states], axis=1)
        )

    def test_refuses_an_empty_store(self, bkd: Backend[Array]) -> None:
        empty: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
        with pytest.raises(ValueError, match="empty"):
            trajectory_array(empty, bkd)
