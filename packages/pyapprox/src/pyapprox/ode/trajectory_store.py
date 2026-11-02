r"""Storage of a time integrator's trajectory, independent of the stack.

A trajectory is the sequence of states :math:`y_0, y_1, \dots, y_N` at
increasing times :math:`t_0 < t_1 < \dots < t_N`. Sweeps over it (the
adjoint, tangent and second-order sweeps) read states by step index and
never assume the states are arrays, so a store holds the stack's opaque
vectors. Where the states live is the store's business: in memory today;
on disk, recomputed from checkpoints, or distributed later, without
changing a sweep.

A store keeps the states it is given, without copying: a state, once
appended, must not be modified, so a buffer reused through an ``out``
argument must never be appended.

A store also keeps each step's unknown :math:`u_n` (the stage values the
step solved for) with the state it produced, so a sweep can rebuild step
:math:`n`'s :class:`~pyapprox.ode.stepper_derivatives.StepWindow`: the
step's span, the stored states it read, its unknown, and the parameter
version it was taken at. Sweeps ask for windows in reverse step order,
which lets a checkpointing store recompute instead of holding every step.

A store keeps the initial time and each step's size, and derives the
times from them (see :mod:`pyapprox.ode.step_span`): the initial state is
appended at its time, every later state with the step that produced it,
so a window's span is rebuilt from the recorded sizes and never from a
difference of times.

A time integrator only writes, so it takes a
:class:`TrajectorySinkProtocol`; a store is a sink that also reads back.
A sink that keeps nothing (accumulating a functional as states arrive)
serves a solve that needs no sweep.
"""

from typing import Generic, Optional, Protocol, TypeVar, runtime_checkable

from pyapprox.ode.step_span import StepSpan
from pyapprox.ode.stepper_derivatives import StepWindow
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.operators import BlockVector

V = TypeVar("V")


@runtime_checkable
class TrajectorySinkProtocol(Protocol, Generic[V]):
    """Receives a trajectory as it is computed: the write side of a
    store, and all a time integrator needs.

    A sink need not keep what it receives: one accumulating a functional
    of the states as they arrive keeps no states at all. It receives the
    initial state at its time, then each step, the step's span starting
    where the last one ended.
    """

    def append(self, time: float, state: V, unconstrained: Optional[V] = None) -> None:
        """Add the initial state at ``time``, with the value the state
        constraint was applied to, when there is one."""
        ...

    def append_step(
        self,
        span: StepSpan,
        state: V,
        unknown: BlockVector[V],
        parameters_version: int,
        unconstrained: Optional[V] = None,
    ) -> None:
        """Add the state a step over ``span`` produced, at ``span.end()``,
        with the step's unknown, the parameter version it was taken at and
        the value the state constraint was applied to, when there is
        one."""
        ...

    def ntimes(self) -> int:
        """The number of states received."""
        ...


@runtime_checkable
class TrajectoryStoreProtocol(TrajectorySinkProtocol[V], Protocol[V]):
    """A sink that keeps the trajectory: states at strictly increasing
    times, read by step index, as the sweeps read them.

    Indices follow Python's convention: ``-1`` is the last state.
    """

    def times(self) -> tuple[float, ...]:
        """Every stored time, in order, derived from the initial time and
        the step sizes."""
        ...

    def time(self, index: int) -> float:
        """The time of state ``index``."""
        ...

    def state(self, index: int) -> V:
        """State ``index``. Not to be modified."""
        ...

    def unconstrained(self, index: int) -> V:
        r"""The value the state constraint mapped to state ``index``,
        where the sweeps linearize :math:`\Pi`; the state itself when it
        was appended without one. Not to be modified."""
        ...

    def nsteps(self) -> int:
        """The number of recorded steps."""
        ...

    def span(self, step: int, depth: int = 1) -> StepSpan:
        """Step ``step``'s span, reading the ``depth`` states up to state
        ``step``: the span of :meth:`window` without reading any state, so
        a store that recomputes states need not, for what depends on the
        step sizes alone."""
        ...

    def window(self, step: int, depth: int = 1) -> StepWindow[V]:
        """Step ``step``'s window: from state ``step`` to state
        ``step + 1``, reading the ``depth`` states up to state ``step``,
        over :meth:`span`."""
        ...


class InMemoryTrajectoryStore(Generic[V]):
    """A trajectory held in memory: the initial time, the step sizes, the
    states and each step's unknown.

    Satisfies :class:`TrajectoryStoreProtocol` for any vector type.

    Raises
    ------
    ValueError
        From :meth:`append`, if the store is not empty; from
        :meth:`append_step`, if it is, or the step does not start at the
        last stored time.
    IndexError
        From :meth:`time`, :meth:`state` and :meth:`window`, for an index
        out of range.
    """

    def __init__(self) -> None:
        # The times are derived: the initial time, then each step's end
        # (its start plus its size), kept so they are computed once.
        self._times: list[float] = []
        self._sizes: list[float] = []
        self._states: list[V] = []
        # Step n's unknown and parameter version, aligned with the steps.
        self._steps: list[tuple[BlockVector[V], int]] = []
        # The value the constraint was applied to, aligned with the
        # states; None where the state was appended without one.
        self._unconstrained: list[Optional[V]] = []

    def append(self, time: float, state: V, unconstrained: Optional[V] = None) -> None:
        if self._times:
            raise ValueError(
                "only the initial state is appended at a time; every later state "
                "comes with its step (append_step)"
            )
        self._times.append(float(time))
        self._states.append(state)
        self._unconstrained.append(unconstrained)

    def append_step(
        self,
        span: StepSpan,
        state: V,
        unknown: BlockVector[V],
        parameters_version: int,
        unconstrained: Optional[V] = None,
    ) -> None:
        """Add the state a step over ``span`` produced, with the step's
        unknown.

        Raises
        ------
        ValueError
            If the store is empty (a step starts from a stored state) or
            the span does not start at the last stored time.
        """
        if not self._times:
            raise ValueError("a step starts from a stored state; append one first")
        if span.time() != self._times[-1]:
            raise ValueError(
                f"the step starts at {span.time()}, not at the last stored time "
                f"{self._times[-1]}"
            )
        self._sizes.append(span.deltat())
        self._times.append(span.end())
        self._states.append(state)
        self._unconstrained.append(unconstrained)
        self._steps.append((unknown, parameters_version))

    def nsteps(self) -> int:
        return len(self._steps)

    def _step_index(self, step: int, depth: int) -> int:
        """``step`` as a nonnegative index, checked against ``depth``.

        Raises
        ------
        IndexError
            If ``step`` is out of range.
        ValueError
            If fewer than ``depth`` states precede it.
        """
        if not -len(self._steps) <= step < len(self._steps):
            raise IndexError(f"step {step} is out of range for {self.nsteps()} steps")
        step = step % len(self._steps)
        if depth > step + 1:
            raise ValueError(
                f"step {step} is preceded by {step + 1} states, not {depth}"
            )
        return step

    def span(self, step: int, depth: int = 1) -> StepSpan:
        """Step ``step``'s span from the recorded sizes alone.

        Parameters
        ----------
        step : int
            The step, from state ``step`` to state ``step + 1``; ``-1`` is
            the last step.
        depth : int, default 1
            How many stored states the step reads.

        Raises
        ------
        IndexError
            If ``step`` is out of range.
        ValueError
            If fewer than ``depth`` states precede it.
        """
        step = self._step_index(step, depth)
        # From the oldest state read, by the recorded sizes: the same
        # additions that made the stored times.
        oldest = step + 1 - depth
        return StepSpan(self._times[oldest], tuple(self._sizes[oldest : step + 1]))

    def window(self, step: int, depth: int = 1) -> StepWindow[V]:
        """Step ``step``'s window, over :meth:`span`.

        Parameters
        ----------
        step : int
            The step, from state ``step`` to state ``step + 1``; ``-1`` is
            the last step.
        depth : int, default 1
            How many stored states the step reads, most recent first.

        Raises
        ------
        IndexError
            If ``step`` is out of range.
        ValueError
            If fewer than ``depth`` states precede it.
        """
        span = self.span(step, depth)
        step = self._step_index(step, depth)
        unknown, version = self._steps[step]
        past = tuple(self._states[step - lag] for lag in range(depth))
        return StepWindow(span, past, unknown, version)

    def ntimes(self) -> int:
        return len(self._times)

    def times(self) -> tuple[float, ...]:
        return tuple(self._times)

    def time(self, index: int) -> float:
        return self._times[index]

    def state(self, index: int) -> V:
        return self._states[index]

    def unconstrained(self, index: int) -> V:
        given = self._unconstrained[index]
        return self._states[index] if given is None else given

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(ntimes={self.ntimes()})"


def trajectory_array(
    store: TrajectoryStoreProtocol[Array], bkd: Backend[Array]
) -> Array:
    """The stored states as one backend array, shape ``(nstates, ntimes)``.

    For consumers that still take a whole trajectory as an array; each
    state is flattened first, so states of shape ``(nstates,)`` and
    ``(nstates, 1)`` give the same result. Only for stores of backend
    arrays.

    Raises
    ------
    ValueError
        If the store is empty.
    """
    if store.ntimes() == 0:
        raise ValueError("an empty trajectory has no array form")
    states: list[Array] = [
        bkd.reshape(store.state(index), (-1,)) for index in range(store.ntimes())
    ]
    return bkd.stack(states, axis=1)
