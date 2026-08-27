"""Concurrent dispatch over any ``concurrent.futures`` executor.

The dispatcher that makes the framework's central claim true rather than
argued: ``submit`` hands tasks to workers and returns while they run, so
a caller can ask about progress, act on partial results, and cancel what
has not started.

**One class covers threads and processes**, because both satisfy the
``Executor`` interface. So the pool a user actually wants in production
and the pool that makes concurrency testable in CI are the same code
path, rather than a test-only implementation shadowing a real one.

**A handle here is nearly a future already.** ``done()`` is
``Future.done()``, ``outcome(timeout)`` wraps ``Future.result(timeout)``,
``cancel()`` is ``Future.cancel()``. That the protocol turned out to fit
so closely is not a coincidence -- it was shaped after futures precisely
because their semantics are well understood.

**Timing is stamped in the worker and returned**, never measured in the
parent. Measuring from submit to completion counts queue wait, which for
any batch larger than the pool is most of the elapsed time: the plan's
measurement put 8 jobs of 0.2s on 2 workers at 4.0s of "compute" against
a true 1.6s, a 2.5x overstatement that would make a measured cost a lie.
The shim below is a module-level class rather than a closure so it
survives pickling to a worker process.
"""

import time
from concurrent.futures import (
    CancelledError,
    Executor,
    Future,
    ThreadPoolExecutor,
    TimeoutError,
)
from dataclasses import dataclass
from typing import Callable, Generic, List, Optional, Sequence, TypeVar

from pyapprox.interface.evaluation.protocols import TaskProtocol
from pyapprox.interface.evaluation.records import (
    ComputeProvenance,
    JobStatus,
    Outcome,
    Resources,
)

Task = TypeVar("Task", bound=TaskProtocol)
Payload = TypeVar("Payload")


@dataclass(frozen=True)
class TimedResult(Generic[Payload]):
    """What a worker produced, and how long it took *in the worker*.

    A duration rather than a pair of timestamps, because that is what
    survives the trip out of a worker process: ``perf_counter`` origins
    are per-process and cannot be compared across one, while an
    interval means the same thing wherever it was measured.
    """

    payload: Payload
    wall_time: float


class _TimedCall(Generic[Task, Payload]):
    """Runs a task and returns its duration alongside its result.

    A class rather than a closure so it pickles into a worker process,
    which a process pool requires. Strictly this is only *needed* for
    processes -- a thread could mutate shared state -- but one mechanism
    serving both is worth more than saving a small record on the thread
    path.
    """

    def __init__(self, run: Callable[[Task], Payload]) -> None:
        self._run = run

    def __call__(self, task: Task) -> TimedResult[Payload]:
        picked_up = time.perf_counter()
        payload = self._run(task)
        return TimedResult(
            payload=payload,
            wall_time=time.perf_counter() - picked_up,
        )


class ExecutorJobHandle(Generic[Task, Payload]):
    """One submitted job, backed by a future."""

    def __init__(
        self,
        task: Task,
        future: "Future[TimedResult[Payload]]",
        resources: Optional[Resources] = None,
    ) -> None:
        self._task = task
        self._future = future
        self._resources = Resources() if resources is None else resources
        self._finished: Optional[float] = None
        # Stamped as each future completes rather than when the caller
        # gets round to reading it. Reading the clock in `outcome()`
        # instead dates every job in a batch to the moment collection
        # reached it: with four jobs two at a time, all four then appear
        # to have finished together, their derived starts collapse onto
        # one origin, and the ledger reports a single job's duration for
        # the whole batch. The callback runs in the driver process --
        # for a process pool, on the thread that reaps results -- so the
        # clock it reads is the one the ledger unions on.
        future.add_done_callback(self._stamp_finish)

    def _stamp_finish(self, _future: "Future[TimedResult[Payload]]") -> None:
        """Record when this job finished, on the driver's clock."""
        if self._finished is None:
            self._finished = time.perf_counter()

    def done(self) -> bool:
        """Whether the job has finished, without consuming anything."""
        return self._future.done()

    def outcome(
        self, timeout: Optional[float] = None
    ) -> Outcome[Task, Payload]:
        """What the job produced, waiting up to ``timeout`` seconds.

        A timeout yields an ``OUTSTANDING`` outcome rather than raising:
        it is a fact about how long the caller chose to wait, not about
        the work, and raising would wrap an ordinary polling result in
        ``try``/``except``.
        """
        try:
            timed = self._future.result(timeout)
        except TimeoutError:
            return Outcome(
                task=self._task,
                indices=self._task.indices,
                status=JobStatus.OUTSTANDING,
            )
        except CancelledError:
            return Outcome(
                task=self._task,
                indices=self._task.indices,
                status=JobStatus.CANCELLED,
                detail="cancelled before it ran",
            )
        except Exception as exc:
            # A raising task is a failed sample, not a failed batch.
            # Wall time is unknown -- the shim never returned -- so it is
            # reported as zero rather than guessed at.
            return Outcome(
                task=self._task,
                indices=self._task.indices,
                status=JobStatus.FAILED,
                resources=self._resources,
                detail=f"{type(exc).__name__}: {exc}",
            )
        return Outcome(
            task=self._task,
            indices=self._task.indices,
            status=JobStatus.SUCCEEDED,
            payload=timed.payload,
            wall_time=timed.wall_time,
            # The worker's own start cannot be used: perf_counter is
            # comparable only within a process, so origins stamped in
            # separate workers cannot be ordered against each other. A
            # duration crosses the boundary intact, so subtracting it
            # from the driver-side finish lands the start on the
            # driver's clock -- the one the ledger unions on.
            started=(
                None
                if self._finished is None
                else self._finished - timed.wall_time
            ),
            resources=self._resources,
        )

    def cancel(self) -> bool:
        """Prevent a job that has not started from starting.

        Returns ``False`` for a job already running or finished. An
        executor cannot interrupt a running future, so this is the
        weaker of the two guarantees the protocol permits -- pending
        work will not start. A dispatcher holding child processes can do
        better; this one cannot, and says so rather than appearing to.
        """
        return self._future.cancel()


class ExecutorDispatcher(Generic[Task, Payload]):
    """Runs tasks on a ``concurrent.futures`` executor.

    Parameters
    ----------
    run : Callable[[Task], Payload]
        How to execute one task, typically a marshaller's bound method.
        Must be picklable if ``executor`` runs processes.
    executor : Executor
        Where the work happens. **Not closed by this dispatcher unless
        ``owns_executor``**, since a caller may be sharing one pool
        across several models -- which is how a throttle spans an
        ensemble.
    concurrency : int
        How many tasks run at once. A constructor argument because
        ``Executor`` exposes no public worker count, and both grouping
        and estimated costs depend on it.
    resources : Resources, optional
        What one task needs from the machine. A property of the wrapped
        code rather than of the pool, which is what lets one shared
        executor serve a serial model and a 32-rank model at once.
        Defaults to one core and nothing else specified.
    owns_executor : bool
        Whether :meth:`close` shuts the executor down. ``False`` by
        default, so sharing is safe and the caller who built the pool
        keeps the right to close it.
    """

    def __init__(
        self,
        run: Callable[[Task], Payload],
        executor: Executor,
        concurrency: int,
        resources: Optional[Resources] = None,
        owns_executor: bool = False,
    ) -> None:
        if not callable(run):
            raise TypeError(f"run must be callable, got {type(run).__name__}")
        if not isinstance(executor, Executor):
            raise TypeError(
                "executor must be a concurrent.futures.Executor, got "
                f"{type(executor).__name__}"
            )
        if concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, got {concurrency}")
        self._timed = _TimedCall(run)
        self._executor = executor
        self._concurrency = concurrency
        self._resources = Resources() if resources is None else resources
        self._owns_executor = owns_executor
        self._closed = False

    def concurrency(self) -> int:
        """How many tasks this dispatcher runs at once."""
        return self._concurrency

    def compute_provenance(self) -> ComputeProvenance:
        """Measured: every duration is stamped inside its own worker."""
        return ComputeProvenance.MEASURED

    def submit(
        self, tasks: Sequence[Task]
    ) -> Sequence[ExecutorJobHandle[Task, Payload]]:
        """Hand tasks to the executor and return at once.

        Returns before the work finishes, which is what makes waiting
        after submitting worthwhile: the workers proceed while the
        caller does something else.
        """
        if self._closed:
            raise RuntimeError("cannot submit to a closed dispatcher")
        handles: List[ExecutorJobHandle[Task, Payload]] = []
        for task in tasks:
            future = self._executor.submit(self._timed, task)
            handles.append(
                ExecutorJobHandle(
                    task=task, future=future, resources=self._resources
                )
            )
        return handles

    def close(self) -> None:
        """Refuse further submissions, and shut down a pool we own.

        Idempotent. A shared executor is left running, because closing
        one out from under another model would be worse than leaking it.
        """
        self._closed = True
        if self._owns_executor:
            self._executor.shutdown(wait=True)

    def __enter__(self) -> "ExecutorDispatcher[Task, Payload]":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def thread_dispatcher(
    run: Callable[[Task], Payload],
    concurrency: int,
    resources: Optional[Resources] = None,
) -> ExecutorDispatcher[Task, Payload]:
    """A dispatcher over a thread pool it owns.

    The right choice for IO-bound work -- an HTTP model, a scheduler
    client, anything that spends its time waiting -- and for making
    concurrency testable without process-spawn overhead.
    """
    return ExecutorDispatcher(
        run=run,
        executor=ThreadPoolExecutor(max_workers=concurrency),
        concurrency=concurrency,
        resources=resources,
        owns_executor=True,
    )


def process_dispatcher(
    run: Callable[[Task], Payload],
    concurrency: int,
    resources: Optional[Resources] = None,
) -> ExecutorDispatcher[Task, Payload]:
    """A dispatcher over a process pool it owns.

    Uses loky's executor rather than the standard library's, matching
    what this repository already depends on elsewhere. The difference
    matters and the type system cannot warn about it: loky cloudpickles,
    so closures, lambdas and locally-defined functions survive the trip
    to a worker, while the stdlib pool rejects them with an unhelpful
    error. A caller who wants the stdlib pool constructs it and passes
    it to :class:`ExecutorDispatcher` directly.

    The import is deferred because joblib is third-party and not free,
    matching how the parallel backends reach it. Everything else this
    module needs is standard library and imported normally.
    """
    from joblib.externals.loky import ProcessPoolExecutor

    return ExecutorDispatcher(
        run=run,
        executor=ProcessPoolExecutor(max_workers=concurrency),
        concurrency=concurrency,
        resources=resources,
        owns_executor=True,
    )
