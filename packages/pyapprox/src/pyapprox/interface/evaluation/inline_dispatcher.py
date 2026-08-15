"""Serial dispatch: one task at a time, in the calling thread.

The trivial dispatcher, and the one that makes the evaluator's semantics
testable without concurrency in the way. Failure handling, streaming
collection and cost accounting are all exercised here first; a bug found
through a thread pool is a bug debugged through two independent sources
of nondeterminism at once.

**Tasks run lazily, on first poll, rather than eagerly in ``submit``.**
Running them in ``submit`` would be simpler, but it makes ``progress()``
report either nothing or everything -- the work is already over by the
time anyone can ask. Running each task when it is first polled means a
serial batch genuinely advances one task at a time, so partial results
are actionable mid-run and a caller can stop early. It also keeps
``submit`` non-blocking, which is what every other dispatcher promises;
a serial dispatcher that blocked there would be the one implementation
whose ``submit`` behaved differently from the protocol's contract.

Nothing here is concurrent, so nothing here needs a lock.
"""

import time
from typing import Callable, Generic, Optional, Sequence, TypeVar

from pyapprox.interface.evaluation.protocols import TaskProtocol
from pyapprox.interface.evaluation.records import (
    ComputeProvenance,
    JobStatus,
    Outcome,
)

Task = TypeVar("Task", bound=TaskProtocol)
Payload = TypeVar("Payload")


class InlineJobHandle(Generic[Task, Payload]):
    """A task that runs when first asked about.

    Idempotent: the task runs at most once, and the resulting
    :class:`Outcome` is returned by every subsequent call.
    """

    def __init__(
        self,
        task: Task,
        indices: Sequence[int],
        run: Callable[[Task], Payload],
        ncores: int = 1,
    ) -> None:
        self._task = task
        self._indices = indices
        self._run = run
        self._ncores = ncores
        self._outcome: Optional[Outcome[Task, Payload]] = None
        self._cancelled = False

    def done(self) -> bool:
        """Whether this job has finished.

        Deliberately does **not** run the task. ``done`` is what a
        progress report calls, and a progress report must not cause the
        work it is reporting on -- otherwise asking about a batch would
        run it, and ``progress()`` could never show anything
        outstanding.
        """
        return self._outcome is not None

    def _execute(self) -> Outcome[Task, Payload]:
        """Run the task once, recording however it turned out."""
        if self._outcome is not None:
            return self._outcome
        if self._cancelled:
            self._outcome = Outcome(
                task=self._task,
                indices=self._indices,
                status=JobStatus.CANCELLED,
                detail="cancelled before it ran",
            )
            return self._outcome

        start = time.perf_counter()
        try:
            payload = self._run(self._task)
        except Exception as exc:
            # A failure is a return value here, not an exception: the
            # evaluator records these indices as failed and carries on
            # with the rest of the batch. Raising would abandon every
            # other task, which is the legacy behavior this replaces.
            self._outcome = Outcome(
                task=self._task,
                indices=self._indices,
                status=JobStatus.FAILED,
                wall_time=time.perf_counter() - start,
                ncores=self._ncores,
                detail=f"{type(exc).__name__}: {exc}",
            )
        else:
            self._outcome = Outcome(
                task=self._task,
                indices=self._indices,
                status=JobStatus.SUCCEEDED,
                payload=payload,
                wall_time=time.perf_counter() - start,
                ncores=self._ncores,
            )
        return self._outcome

    def outcome(
        self, timeout: Optional[float] = None
    ) -> Outcome[Task, Payload]:
        """Run the task if it has not run, and return what happened.

        ``timeout`` is accepted for protocol conformance and ignored:
        the work happens in this thread, so there is nothing to wait
        for and no way to abandon it partway. A serial dispatcher
        cannot honor a deadline without threads, and pretending
        otherwise would be worse than saying so.
        """
        return self._execute()

    def cancel(self) -> bool:
        """Prevent a task that has not run yet from running.

        Returns ``True`` if this call stopped the task, ``False`` if it
        had already run. Matches the protocol's weaker guarantee --
        pending work will not start -- which is the only one available
        without a separate thread of control.
        """
        if self._outcome is not None:
            return False
        self._cancelled = True
        return True


class InlineDispatcher(Generic[Task, Payload]):
    """Runs tasks serially in the calling thread.

    Parameters
    ----------
    run : Callable[[Task], Payload]
        How to execute one task. Injected rather than inherited, so this
        dispatcher is not tied to any particular task family.
    """

    def __init__(self, run: Callable[[Task], Payload]) -> None:
        if not callable(run):
            raise TypeError(
                f"run must be callable, got {type(run).__name__}"
            )
        self._run = run
        self._closed = False

    def concurrency(self) -> int:
        """One: this dispatcher runs a single task at a time."""
        return 1

    def compute_provenance(self) -> ComputeProvenance:
        """Measured -- with one worker the two clocks coincide exactly."""
        return ComputeProvenance.MEASURED

    def submit(
        self, tasks: Sequence[Task]
    ) -> Sequence[InlineJobHandle[Task, Payload]]:
        """Accept tasks and return handles, without running anything yet.

        Returns immediately, as the protocol requires. The work happens
        when each handle is first polled.
        """
        if self._closed:
            raise RuntimeError("cannot submit to a closed dispatcher")
        return [
            InlineJobHandle(
                task=task,
                indices=task.indices,
                run=self._run,
            )
            for task in tasks
        ]

    def close(self) -> None:
        """Refuse further submissions. Idempotent; holds no resources."""
        self._closed = True

    def __enter__(self) -> "InlineDispatcher[Task, Payload]":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
