r"""Functionals accumulated while a trajectory is computed.

A time integrator writes into a :class:`TrajectorySinkProtocol`. A sink
that also reports a :meth:`~AccumulatingSinkProtocol.value` computes a
functional of the trajectory as the states arrive, so a solve that needs
only the functional keeps no trajectory and never reads one back.

A functional reused across solves cannot itself be the sink, since an
integrator takes an empty sink each time: it makes a fresh sink per solve.
A functional of the stored states one at a time (an endpoint, a running
maximum, states picked at given steps) is a :class:`StateAccumulatorProtocol`
wrapped in a :class:`PerStateSink`, which supplies the sink's side; a
functional that reads more of each step, such as its stage values, is a
sink of its own. A :class:`MultiSink` lets one solve feed several sinks:
several functionals, or a functional and a store for a later sweep.
"""

from typing import Generic, Optional, Protocol, Sequence, TypeVar, runtime_checkable

from pyapprox.ode.step_span import StepSpan
from pyapprox.ode.trajectory_store import TrajectorySinkProtocol
from pyapprox.util.linalg.operators import BlockVector

V = TypeVar("V")
V_contra = TypeVar("V_contra", contravariant=True)
R = TypeVar("R")
R_co = TypeVar("R_co", covariant=True)


@runtime_checkable
class AccumulatingSinkProtocol(TrajectorySinkProtocol[V], Protocol[V, R_co]):
    """A sink computing a functional of what it receives."""

    def value(self) -> R_co:
        """The functional of the trajectory received so far."""
        ...


@runtime_checkable
class StateAccumulatorProtocol(Protocol[V_contra, R_co]):
    """A functional built up from the stored states one at a time."""

    def add(self, index: int, time: float, state: V_contra) -> None:
        """Take stored state ``index`` (0 the initial state), at ``time``.
        The state must not be modified, and is not modified here."""
        ...

    def value(self) -> R_co:
        """The functional of the states taken so far."""
        ...


class PerStateSink(Generic[V, R]):
    """A sink passing each stored state to an accumulator and keeping
    nothing else.

    Parameters
    ----------
    accumulator : StateAccumulatorProtocol
        Receives the states in order; its value is the sink's.

    Raises
    ------
    TypeError
        If ``accumulator`` does not satisfy its protocol.
    ValueError
        From :meth:`append`, if a state was already received; from
        :meth:`append_step`, if none was, or the step does not start where
        the last one ended.
    """

    def __init__(self, accumulator: StateAccumulatorProtocol[V, R]) -> None:
        if not isinstance(accumulator, StateAccumulatorProtocol):
            raise TypeError(
                "accumulator must satisfy StateAccumulatorProtocol, got "
                f"{type(accumulator).__name__}"
            )
        self._accumulator = accumulator
        self._ntimes = 0
        self._last_time: Optional[float] = None

    def _take(self, time: float, state: V) -> None:
        self._accumulator.add(self._ntimes, time, state)
        self._ntimes += 1
        self._last_time = time

    def append(self, time: float, state: V, unconstrained: Optional[V] = None) -> None:
        """Pass the initial state on; the unconstrained value is not
        needed."""
        if self._last_time is not None:
            raise ValueError(
                "only the initial state is appended at a time; every later state "
                "comes with its step (append_step)"
            )
        self._take(time, state)

    def append_step(
        self,
        span: StepSpan,
        state: V,
        unknown: BlockVector[V],
        parameters_version: int,
        unconstrained: Optional[V] = None,
    ) -> None:
        """Pass the step's state on, at the span's end; the step's unknown
        is not needed."""
        if self._last_time is None:
            raise ValueError("a step starts from a received state; append one first")
        if span.time() != self._last_time:
            raise ValueError(
                f"the step starts at {span.time()}, not where the last one ended, "
                f"{self._last_time}"
            )
        self._take(span.end(), state)

    def ntimes(self) -> int:
        """The number of states received."""
        return self._ntimes

    def value(self) -> R:
        """The accumulator's value."""
        return self._accumulator.value()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._accumulator!r})"


class MultiSink(Generic[V]):
    """A sink passing everything it receives to each of several sinks, so
    one solve feeds them all (several functionals, or a functional and a
    store).

    Parameters
    ----------
    sinks : sequence of TrajectorySinkProtocol
        Receive every state and step, in order; each must be empty.

    Raises
    ------
    TypeError
        If a sink does not satisfy its protocol.
    ValueError
        If there is no sink or one is not empty.
    """

    def __init__(self, sinks: Sequence[TrajectorySinkProtocol[V]]) -> None:
        if not sinks:
            raise ValueError("a MultiSink needs at least one sink")
        for sink in sinks:
            if not isinstance(sink, TrajectorySinkProtocol):
                raise TypeError(
                    "each sink must satisfy TrajectorySinkProtocol, got "
                    f"{type(sink).__name__}"
                )
            if sink.ntimes() != 0:
                raise ValueError("each sink must be empty")
        self._sinks = list(sinks)

    def append(self, time: float, state: V, unconstrained: Optional[V] = None) -> None:
        for sink in self._sinks:
            sink.append(time, state, unconstrained)

    def append_step(
        self,
        span: StepSpan,
        state: V,
        unknown: BlockVector[V],
        parameters_version: int,
        unconstrained: Optional[V] = None,
    ) -> None:
        for sink in self._sinks:
            sink.append_step(span, state, unknown, parameters_version, unconstrained)

    def ntimes(self) -> int:
        """The number of states received, the same for every sink."""
        return self._sinks[0].ntimes()
