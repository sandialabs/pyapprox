"""Tests for the trajectory store."""

import numpy as np
import pytest

from pyapprox.ode.trajectory_store import (
    InMemoryTrajectoryStore,
    TrajectoryStoreProtocol,
    trajectory_array,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.operators import BlockVector, SpaceDescriptor
from tests._helpers.linalg.mock_stack import MockVector, peek, wrap

_TIMES = (0.0, 0.25, 0.5, 1.0)


def _states(bkd: Backend[Array], shape: tuple[int, ...]) -> list[Array]:
    rng = np.random.default_rng(0)
    return [bkd.asarray(rng.normal(size=shape)) for _ in _TIMES]


class TestInMemoryTrajectoryStore:
    def test_stores_states_and_times_in_order(self, bkd: Backend[Array]) -> None:
        store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
        assert isinstance(store, TrajectoryStoreProtocol)
        states = _states(bkd, (4,))
        for time, state in zip(_TIMES, states):
            store.append(time, state)
        assert store.ntimes() == 4
        assert store.times() == _TIMES
        assert store.time(-2) == 0.5
        assert store.state(1) is states[1]
        assert store.state(-1) is states[-1]

    def test_refuses_times_that_do_not_increase(self, bkd: Backend[Array]) -> None:
        store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
        store.append(0.0, bkd.zeros((3,)))
        with pytest.raises(ValueError, match="strictly increase"):
            store.append(0.0, bkd.zeros((3,)))
        with pytest.raises(ValueError, match="strictly increase"):
            store.append(-1.0, bkd.zeros((3,)))

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
        for time in _TIMES:
            store.append(time, wrap(np.full((3, 1), time), space))
        np.testing.assert_array_equal(peek(store.state(2)), np.full((3, 1), 0.5))


def _stepped_store(
    bkd: Backend[Array],
) -> tuple[InMemoryTrajectoryStore[Array], list[Array], list[BlockVector[Array]]]:
    """A store of the four states, each later one appended as a step whose
    unknown is two blocks and whose parameter version is its index."""
    states = _states(bkd, (2, 1))
    unknowns = [BlockVector((state, 2.0 * state)) for state in states[1:]]
    store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
    store.append(_TIMES[0], states[0])
    for n, (time, state) in enumerate(zip(_TIMES[1:], states[1:])):
        store.append_step(time, state, unknowns[n], n)
    return store, states, unknowns


class TestStepWindows:
    def test_window_holds_the_step(self, bkd: Backend[Array]) -> None:
        store, states, unknowns = _stepped_store(bkd)
        assert store.nsteps() == 3
        window = store.window(1)
        assert window.time() == 0.25
        assert window.deltat() == 0.25
        assert window.past_states == (states[1],)
        assert window.unknown is unknowns[1]
        assert window.parameters_version == 1
        assert store.window(-1).unknown is unknowns[2]

    def test_deeper_windows_read_past_states_most_recent_first(
        self, bkd: Backend[Array]
    ) -> None:
        store, states, _ = _stepped_store(bkd)
        assert store.window(2, depth=3).past_states == (
            states[2],
            states[1],
            states[0],
        )
        with pytest.raises(ValueError, match="preceded by 2 states"):
            store.window(1, depth=3)

    def test_refusals(self, bkd: Backend[Array]) -> None:
        empty: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
        state = bkd.zeros((2, 1))
        with pytest.raises(ValueError, match="append one first"):
            empty.append_step(0.0, state, BlockVector((state,)), 0)
        store, _, _ = _stepped_store(bkd)
        with pytest.raises(IndexError):
            store.window(3)
        with pytest.raises(IndexError):
            store.window(-4)
        store.append(2.0, state)
        assert store.nsteps() == 4
        with pytest.raises(ValueError, match="without its unknown"):
            store.window(3)


class TestTrajectoryArray:
    @pytest.mark.parametrize("shape", [(4,), (4, 1)], ids=["1d", "column"])
    def test_stacks_states_as_columns(
        self, bkd: Backend[Array], shape: tuple[int, ...]
    ) -> None:
        """Today's (n,) states and (n, 1) blocks give the same array."""
        store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
        states = _states(bkd, shape)
        for time, state in zip(_TIMES, states):
            store.append(time, state)
        array = trajectory_array(store, bkd)
        assert tuple(array.shape) == (4, 4)
        bkd.assert_allclose(
            array, bkd.stack([bkd.reshape(s, (-1,)) for s in states], axis=1)
        )

    def test_refuses_an_empty_store(self, bkd: Backend[Array]) -> None:
        empty: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
        with pytest.raises(ValueError, match="empty"):
            trajectory_array(empty, bkd)
