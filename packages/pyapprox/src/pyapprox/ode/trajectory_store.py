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
step's time and size, the stored states it read, its unknown, and the
parameter version it was taken at. Sweeps ask for windows in reverse
step order, which lets a checkpointing store recompute instead of
holding every step.
"""

from typing import Generic, Optional, Protocol, TypeVar, runtime_checkable

from pyapprox.ode.stepper_derivatives import StepWindow
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.operators import BlockVector

V = TypeVar("V")


@runtime_checkable
class TrajectoryStoreProtocol(Protocol, Generic[V]):
    """States at strictly increasing times, read by step index.

    Indices follow Python's convention: ``-1`` is the last state.
    """

    def append(self, time: float, state: V, unconstrained: Optional[V] = None) -> None:
        """Add ``state`` at ``time``, later than every stored time, with
        the value the state constraint was applied to, when there is
        one."""
        ...

    def ntimes(self) -> int:
        """The number of stored states."""
        ...

    def times(self) -> tuple[float, ...]:
        """Every stored time, in order."""
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

    def append_step(
        self,
        time: float,
        state: V,
        unknown: BlockVector[V],
        parameters_version: int,
        unconstrained: Optional[V] = None,
    ) -> None:
        """Add the state a step produced at ``time``, with the step's
        unknown, the parameter version it was taken at and the value the
        state constraint was applied to, when there is one."""
        ...

    def nsteps(self) -> int:
        """The number of recorded steps."""
        ...

    def window(self, step: int, depth: int = 1) -> StepWindow[V]:
        """Step ``step``'s window: from state ``step`` to state
        ``step + 1``, reading the ``depth`` states up to state ``step``."""
        ...


class InMemoryTrajectoryStore(Generic[V]):
    """A trajectory held in memory as lists of times and states.

    Satisfies :class:`TrajectoryStoreProtocol` for any vector type.

    Raises
    ------
    ValueError
        From :meth:`append`, if the time does not exceed the last stored
        time.
    IndexError
        From :meth:`time`, :meth:`state` and :meth:`window`, for an index
        out of range.
    """

    def __init__(self) -> None:
        self._times: list[float] = []
        self._states: list[V] = []
        # Step n's unknown and parameter version, aligned with the steps
        # (state n + 1); None for a state appended without a step.
        self._steps: list[Optional[tuple[BlockVector[V], int]]] = []
        # The value the constraint was applied to, aligned with the
        # states; None where the state was appended without one.
        self._unconstrained: list[Optional[V]] = []

    def append(self, time: float, state: V, unconstrained: Optional[V] = None) -> None:
        if self._times and not time > self._times[-1]:
            raise ValueError(
                f"times must strictly increase: {time} after {self._times[-1]}"
            )
        if self._times:
            self._steps.append(None)
        self._times.append(float(time))
        self._states.append(state)
        self._unconstrained.append(unconstrained)

    def append_step(
        self,
        time: float,
        state: V,
        unknown: BlockVector[V],
        parameters_version: int,
        unconstrained: Optional[V] = None,
    ) -> None:
        """Add the state a step produced, with the step's unknown.

        Raises
        ------
        ValueError
            If the store is empty (a step starts from a stored state) or
            ``time`` does not exceed the last stored time.
        """
        if not self._times:
            raise ValueError("a step starts from a stored state; append one first")
        self.append(time, state, unconstrained)
        self._steps[-1] = (unknown, parameters_version)

    def nsteps(self) -> int:
        return len(self._steps)

    def window(self, step: int, depth: int = 1) -> StepWindow[V]:
        """Step ``step``'s window.

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
            If the step was recorded without its unknown, or fewer than
            ``depth`` states precede it.
        """
        if not -len(self._steps) <= step < len(self._steps):
            raise IndexError(f"step {step} is out of range for {self.nsteps()} steps")
        step = step % len(self._steps)
        recorded = self._steps[step]
        if recorded is None:
            raise ValueError(f"step {step} was recorded without its unknown")
        if depth > step + 1:
            raise ValueError(
                f"step {step} is preceded by {step + 1} states, not {depth}"
            )
        unknown, version = recorded
        past = tuple(self._states[step - lag] for lag in range(depth))
        return StepWindow(
            self._times[step],
            self._times[step + 1] - self._times[step],
            past,
            unknown,
            version,
        )

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
