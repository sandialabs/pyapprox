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
"""

from typing import Generic, Protocol, TypeVar, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend

V = TypeVar("V")


@runtime_checkable
class TrajectoryStoreProtocol(Protocol, Generic[V]):
    """States at strictly increasing times, read by step index.

    Indices follow Python's convention: ``-1`` is the last state.
    """

    def append(self, time: float, state: V) -> None:
        """Add ``state`` at ``time``, later than every stored time."""
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


class InMemoryTrajectoryStore(Generic[V]):
    """A trajectory held in memory as lists of times and states.

    Satisfies :class:`TrajectoryStoreProtocol` for any vector type.

    Raises
    ------
    ValueError
        From :meth:`append`, if the time does not exceed the last stored
        time.
    IndexError
        From :meth:`time` and :meth:`state`, for an index out of range.
    """

    def __init__(self) -> None:
        self._times: list[float] = []
        self._states: list[V] = []

    def append(self, time: float, state: V) -> None:
        if self._times and not time > self._times[-1]:
            raise ValueError(
                f"times must strictly increase: {time} after {self._times[-1]}"
            )
        self._times.append(float(time))
        self._states.append(state)

    def ntimes(self) -> int:
        return len(self._times)

    def times(self) -> tuple[float, ...]:
        return tuple(self._times)

    def time(self, index: int) -> float:
        return self._times[index]

    def state(self, index: int) -> V:
        return self._states[index]

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
