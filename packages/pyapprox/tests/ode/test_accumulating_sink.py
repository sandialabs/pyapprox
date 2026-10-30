"""A sink passing each stored state to an accumulator.

- States reach the accumulator in order with their index and time, from
  ``append`` and ``append_step`` alike (a step's unknown is not passed);
  the sink keeps none of them and reports the accumulator's value.
- It satisfies the sink and accumulating-sink protocols, not the store's.
- Refusals: a time that does not increase, and an accumulator that does
  not satisfy its protocol.

A sink passing everything to several sinks: each receives every state in
order; it refuses no sinks, a sink already written to, and an object
that is not a sink.
"""

from typing import Generic, List, Tuple

import pytest

from pyapprox.ode.accumulating_sink import (
    AccumulatingSinkProtocol,
    MultiSink,
    PerStateSink,
    StateAccumulatorProtocol,
)
from pyapprox.ode.trajectory_store import (
    TrajectorySinkProtocol,
    TrajectoryStoreProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.operators import BlockVector


class _Recording(Generic[Array]):
    """Records what it is given; its value is the number of states."""

    def __init__(self) -> None:
        self.received: List[Tuple[int, float, Array]] = []

    def add(self, index: int, time: float, state: Array) -> None:
        self.received.append((index, time, state))

    def value(self) -> int:
        return len(self.received)


class TestPerStateSink:
    def test_states_reach_the_accumulator_in_order(self, bkd: Backend[Array]) -> None:
        recording: _Recording[Array] = _Recording()
        assert isinstance(recording, StateAccumulatorProtocol)
        sink: PerStateSink[Array, int] = PerStateSink(recording)
        states = [bkd.asarray([[float(n)], [2.0 * n]]) for n in range(3)]
        sink.append(0.0, states[0])
        unknown = BlockVector((bkd.asarray([[9.0], [9.0]]),))
        sink.append_step(0.1, states[1], unknown, 0)
        sink.append_step(0.3, states[2], unknown, 0)
        assert sink.ntimes() == 3
        assert sink.value() == 3
        for n, (index, time, state) in enumerate(recording.received):
            assert (index, time) == (n, (0.0, 0.1, 0.3)[n])
            assert state is states[n]

    def test_protocols(self, bkd: Backend[Array]) -> None:
        sink: PerStateSink[Array, int] = PerStateSink(_Recording())
        sink.append(0.0, bkd.asarray([[1.0]]))
        assert isinstance(sink, TrajectorySinkProtocol)
        assert isinstance(sink, AccumulatingSinkProtocol)
        assert not isinstance(sink, TrajectoryStoreProtocol)

    def test_refusals(self, bkd: Backend[Array]) -> None:
        sink: PerStateSink[Array, int] = PerStateSink(_Recording())
        sink.append(0.5, bkd.asarray([[1.0]]))
        with pytest.raises(ValueError, match="strictly increase"):
            sink.append(0.5, bkd.asarray([[1.0]]))
        with pytest.raises(TypeError, match="StateAccumulatorProtocol"):
            PerStateSink(object())  # type: ignore[arg-type]


class TestMultiSink:
    def test_every_sink_receives_everything(self, bkd: Backend[Array]) -> None:
        first: _Recording[Array] = _Recording()
        second: _Recording[Array] = _Recording()
        sinks: List[PerStateSink[Array, int]] = [
            PerStateSink(first),
            PerStateSink(second),
        ]
        multi: MultiSink[Array] = MultiSink(sinks)
        assert isinstance(multi, TrajectorySinkProtocol)
        states = [bkd.asarray([[float(n)]]) for n in range(3)]
        multi.append(0.0, states[0])
        unknown = BlockVector((bkd.asarray([[9.0]]),))
        multi.append_step(0.1, states[1], unknown, 0)
        multi.append_step(0.2, states[2], unknown, 0)
        assert multi.ntimes() == 3
        for recording in (first, second):
            assert [state for *_, state in recording.received] == states

    def test_refusals(self, bkd: Backend[Array]) -> None:
        with pytest.raises(ValueError, match="at least one sink"):
            MultiSink[Array]([])
        used: PerStateSink[Array, int] = PerStateSink(_Recording())
        used.append(0.0, bkd.asarray([[1.0]]))
        with pytest.raises(ValueError, match="must be empty"):
            MultiSink([used])
        with pytest.raises(TypeError, match="TrajectorySinkProtocol"):
            MultiSink([object()])  # type: ignore[arg-type]
