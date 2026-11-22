r"""A trajectory kept on disk, one NumPy ``.npz`` file per stored state.

:class:`NpzTrajectoryStore` satisfies
:class:`~pyapprox.ode.trajectory_store.TrajectoryStoreProtocol` with the
states, the values the constraint was applied to and each step's unknown
written to files in a directory, and only the times, step sizes,
parameter versions and read counts held in memory. A sweep reading a
window loads the files it needs and nothing else, so a long trajectory
does not have to fit in memory, and each subdomain of a distributed run
can keep its own trajectory in its own directory: the caller chooses the
directory.

File ``state_<index>.npz`` holds state ``index`` (key ``state``), the
value the constraint was applied to when one was given
(``unconstrained``), and, for every state after the first, the step that
produced it: its unknown's blocks (``unknown_0``, ``unknown_1``, ...)
and the step's size, parameter version and number of states read
(``size``, ``version``, ``nread``). The files describe the trajectory
completely.

States are backend arrays, converted to NumPy at one boundary when
written and back with ``bkd.asarray`` when read (on the backend's
configured dtype and device): the store is not differentiable through.
"""

import os
from typing import Generic, Optional

import numpy as np
from numpy.typing import NDArray

from pyapprox.ode.step_span import StepSpan
from pyapprox.ode.stepper_derivatives import StepOrigin, StepWindow, TrajectoryId
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.operators import BlockVector


class NpzTrajectoryStore(Generic[Array]):
    """A trajectory of backend arrays written to ``directory``, one file
    per state (see the module docstring).

    Satisfies :class:`~pyapprox.ode.trajectory_store.
    TrajectoryStoreProtocol` for backend arrays.

    Parameters
    ----------
    directory : str
        Where the files go; created if missing, and must hold no files.
    bkd : Backend
        The backend of the states.

    Raises
    ------
    ValueError
        If ``directory`` holds files; from :meth:`append`, if the store is
        not empty; from :meth:`append_step`, if it is, the step does not
        start at the last stored time, or it reads more states than are
        stored.
    IndexError
        From :meth:`time`, :meth:`state`, :meth:`span` and :meth:`window`,
        for an index out of range.
    """

    def __init__(self, directory: str, bkd: Backend[Array]) -> None:
        os.makedirs(directory, exist_ok=True)
        if os.listdir(directory):
            raise ValueError(f"the directory {directory!r} must hold no files")
        self._directory, self._bkd = directory, bkd
        self._times: list[float] = []
        self._sizes: list[float] = []
        # Step n's parameter version, number of states read and number of
        # unknown blocks; the rest of the step is on disk.
        self._steps: list[tuple[int, int, int]] = []
        self._trajectory = TrajectoryId()

    def directory(self) -> str:
        """The directory holding the files."""
        return self._directory

    def _path(self, index: int) -> str:
        return os.path.join(self._directory, f"state_{index:06d}.npz")

    def _index(self, index: int, count: int, what: str) -> int:
        if not -count <= index < count:
            raise IndexError(f"{what} {index} is out of range for {count}")
        return index % count

    def _write(
        self,
        index: int,
        state: Array,
        unconstrained: Optional[Array],
        extra: dict[str, NDArray[np.generic]],
    ) -> None:
        to_numpy = self._bkd.to_numpy
        arrays: dict[str, NDArray[np.generic]] = {"state": to_numpy(state)}
        if unconstrained is not None:
            arrays["unconstrained"] = to_numpy(unconstrained)
        arrays.update(extra)
        # Plain arrays only: nothing pickled is written, or loaded back.
        np.savez(self._path(index), allow_pickle=False, **arrays)

    def _read(self, index: int, key: str) -> Optional[Array]:
        with np.load(self._path(index)) as data:
            if key not in data:
                return None
            return self._bkd.asarray(data[key])

    def append(
        self, time: float, state: Array, unconstrained: Optional[Array] = None
    ) -> None:
        if self._times:
            raise ValueError(
                "only the initial state is appended at a time; every later state "
                "comes with its step (append_step)"
            )
        self._write(0, state, unconstrained, {})
        self._times.append(float(time))

    def append_step(
        self,
        window: StepWindow[Array],
        state: Array,
        unconstrained: Optional[Array] = None,
    ) -> None:
        """Write the state a step produced with the step's unknown, size,
        parameter version and number of states read; the window's past
        states are the store's own and are not written twice.

        Raises
        ------
        ValueError
            If the store is empty, the span does not start at the last
            stored time, or the step reads more states than are stored.
        """
        span = window.span
        if not self._times:
            raise ValueError("a step starts from a stored state; append one first")
        if span.time() != self._times[-1]:
            raise ValueError(
                f"the step starts at {span.time()}, not at the last stored time "
                f"{self._times[-1]}"
            )
        nread = len(span.sizes)
        if nread > len(self._times):
            raise ValueError(
                f"the step reads {nread} states, but {len(self._times)} are stored"
            )
        blocks = window.unknown.blocks
        to_numpy = self._bkd.to_numpy
        extra: dict[str, NDArray[np.generic]] = {
            f"unknown_{k}": to_numpy(block) for k, block in enumerate(blocks)
        }
        extra["size"] = np.asarray(span.deltat())
        extra["version"] = np.asarray(window.parameters_version)
        extra["nread"] = np.asarray(nread)
        self._write(len(self._times), state, unconstrained, extra)
        self._sizes.append(span.deltat())
        self._times.append(span.end())
        self._steps.append((window.parameters_version, nread, len(blocks)))

    def ntimes(self) -> int:
        return len(self._times)

    def nsteps(self) -> int:
        return len(self._steps)

    def times(self) -> tuple[float, ...]:
        return tuple(self._times)

    def time(self, index: int) -> float:
        return self._times[index]

    def state(self, index: int) -> Array:
        """State ``index``, read from its file."""
        state = self._read(self._index(index, self.ntimes(), "state"), "state")
        if state is None:
            raise ValueError(f"the file of state {index} holds no state")
        return state

    def unconstrained(self, index: int) -> Array:
        index = self._index(index, self.ntimes(), "state")
        given = self._read(index, "unconstrained")
        return self.state(index) if given is None else given

    def span(self, step: int) -> StepSpan:
        """Step ``step``'s span from the recorded sizes, one per state the
        step read, without reading a file.

        Raises
        ------
        IndexError
            If ``step`` is out of range.
        """
        step = self._index(step, self.nsteps(), "step")
        oldest = step + 1 - self._steps[step][1]
        return StepSpan(self._times[oldest], tuple(self._sizes[oldest : step + 1]))

    def window(self, step: int) -> StepWindow[Array]:
        """Step ``step``'s window, reading its unknown and the states it
        read, most recent first.

        Raises
        ------
        IndexError
            If ``step`` is out of range.
        """
        span = self.span(step)
        step = self._index(step, self.nsteps(), "step")
        version, nread, nblocks = self._steps[step]
        blocks = []
        for k in range(nblocks):
            block = self._read(step + 1, f"unknown_{k}")
            if block is None:
                raise ValueError(f"the file of step {step} lacks unknown block {k}")
            blocks.append(block)
        past = tuple(self.state(step - lag) for lag in range(nread))
        origin = StepOrigin(self._trajectory, step)
        return StepWindow(span, past, BlockVector(tuple(blocks)), version, origin)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._directory!r}, ntimes={self.ntimes()})"
