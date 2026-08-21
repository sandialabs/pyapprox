"""Concurrency, and the claims that only a real pool can settle.

Everything the framework says about non-blocking evaluation was, until
this dispatcher existed, either argued or demonstrated on a serial
dispatcher that satisfies the contract trivially. These tests measure
the things that distinguish a working design from one that merely
typechecks: that submit returns while workers run, that progress
advances mid-flight, that the throttle binds, and that measured compute
is not inflated by queue wait.

Thread pools throughout. They are the right tool for the IO-bound work
this dispatcher mostly serves, they avoid process-spawn overhead
distorting the timing assertions, and they sidestep the shared-memory
tensor transfer that can hang a process pool in CI. A dispatcher is
array-free by construction, so these use ``numpy_bkd`` where a backend
is needed at all.

Delay units are 20-50ms: long enough that the ratios below are not
measuring scheduler noise, short enough that the module stays well under
a second.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Sequence

import pytest
from pyapprox.interface.evaluation.executor_dispatcher import (
    ExecutorDispatcher,
    thread_dispatcher,
)
from pyapprox.interface.evaluation.protocols import (
    DispatcherProtocol,
    JobHandle,
)
from pyapprox.interface.evaluation.records import (
    ComputeProvenance,
    JobStatus,
)

#: Simulated work per task, sized so the signal dominates the noise.
#:
#: Jitter is a millisecond or two on an idle developer machine, which
#: made 0.05 look generous. A shared CI runner is a different regime: a
#: 0.05 sleep was measured at 0.10 and 0.19 there, so an assertion
#: written as "under two units" failed while the quantity it exists to
#: exclude -- a four-unit queue wait -- was never in play. Scheduling
#: delay is roughly constant rather than proportional, so the fix is a
#: longer unit rather than looser ratios: at 0.25 the same 50ms of
#: overhead is a fifth of a unit instead of a whole one.
UNIT = 0.25


@dataclass(frozen=True)
class _Task:
    indices: Sequence[int]
    value: int = 0
    delay: float = 0.0
    explode: bool = False


def _run(task: _Task) -> int:
    if task.delay:
        time.sleep(task.delay)
    if task.explode:
        raise RuntimeError(f"task {task.value} diverged")
    return task.value * 10


def _tasks(n: int, delay: float = 0.0) -> list:
    return [
        _Task(indices=[i], value=i, delay=delay) for i in range(n)
    ]


class TestConformance:
    def test_satisfies_dispatcher_protocol(self):
        with thread_dispatcher(_run, concurrency=2) as dispatcher:
            assert isinstance(dispatcher, DispatcherProtocol)

    def test_handles_satisfy_job_handle(self):
        with thread_dispatcher(_run, concurrency=2) as dispatcher:
            (handle,) = dispatcher.submit([_Task(indices=[0])])
            assert isinstance(handle, JobHandle)

    def test_reports_its_concurrency(self):
        with thread_dispatcher(_run, concurrency=4) as dispatcher:
            assert dispatcher.concurrency() == 4

    def test_provenance_is_measured(self):
        """Durations are stamped in the worker, so they are observed."""
        with thread_dispatcher(_run, concurrency=2) as dispatcher:
            assert (
                dispatcher.compute_provenance()
                is ComputeProvenance.MEASURED
            )

    def test_rejects_bad_arguments(self):
        executor = ThreadPoolExecutor(max_workers=1)
        with pytest.raises(TypeError, match="callable"):
            ExecutorDispatcher("not callable", executor, concurrency=1)
        with pytest.raises(TypeError, match="Executor"):
            ExecutorDispatcher(_run, "not an executor", concurrency=1)
        with pytest.raises(ValueError, match="concurrency"):
            ExecutorDispatcher(_run, executor, concurrency=0)
        executor.shutdown()


class TestSubmitDoesNotBlock:
    """The claim the whole design rests on."""

    def test_submit_returns_before_the_work_finishes(self):
        """Elapsed at return must be far below the known total."""
        with thread_dispatcher(_run, concurrency=2) as dispatcher:
            start = time.perf_counter()
            dispatcher.submit(_tasks(4, delay=UNIT))
            submit_seconds = time.perf_counter() - start
        assert submit_seconds < UNIT

    def test_waiting_after_submit_is_worth_something(self):
        """Unlike a serial dispatcher, the work proceeds during a wait.

        This is precisely what a lazy or blocking submit cannot do, and
        the reason the inline dispatcher is honest about running its
        work up front rather than pretending otherwise.
        """
        with thread_dispatcher(_run, concurrency=4) as dispatcher:
            handles = dispatcher.submit(_tasks(4, delay=UNIT))
            time.sleep(UNIT * 3)
            assert all(h.done() for h in handles)

    def test_progress_can_be_observed_mid_flight(self):
        """Some done, some not -- the state a blocking call cannot show."""
        with thread_dispatcher(_run, concurrency=2) as dispatcher:
            handles = dispatcher.submit(
                [_Task(indices=[i], value=i, delay=UNIT * (i + 1))
                 for i in range(4)]
            )
            time.sleep(UNIT * 1.5)
            ndone = sum(1 for h in handles if h.done())
            assert 0 < ndone < 4


class TestThrottle:
    def test_never_more_than_concurrency_running(self):
        """The throttle binds, and is not merely declared."""
        running = []
        peak = []

        def counting(task: _Task) -> int:
            running.append(1)
            peak.append(len(running))
            time.sleep(task.delay)
            running.pop()
            return task.value

        with thread_dispatcher(counting, concurrency=2) as dispatcher:
            handles = dispatcher.submit(_tasks(8, delay=UNIT))
            for handle in handles:
                handle.outcome()
        assert max(peak) <= 2

    def test_a_wider_pool_runs_more_at_once(self):
        """Distinguishes a real throttle from one that ignores its limit."""
        running = []
        peak = []

        def counting(task: _Task) -> int:
            running.append(1)
            peak.append(len(running))
            time.sleep(task.delay)
            running.pop()
            return task.value

        with thread_dispatcher(counting, concurrency=4) as dispatcher:
            handles = dispatcher.submit(_tasks(8, delay=UNIT))
            for handle in handles:
                handle.outcome()
        assert max(peak) > 2


class TestWorkerSideTiming:
    """Durations come from the worker, never from the parent.

    Measuring submit-to-completion in the parent counts queue wait,
    which for any batch larger than the pool is most of the elapsed
    time. The error is large and one-directional: it inflates compute,
    which is what a budget spends against.
    """

    def test_wall_time_excludes_queue_wait(self):
        """A queued task must report its own duration, not its latency.

        With 4 tasks on 1 worker the last finishes about 4 units after
        submission, but ran for one. A parent-side stamp would report
        the former.
        """
        with thread_dispatcher(_run, concurrency=1) as dispatcher:
            handles = dispatcher.submit(_tasks(4, delay=UNIT))
            outcomes = [h.outcome() for h in handles]
        for outcome in outcomes:
            # Generous against a one-unit task, and still far below the
            # four-unit queue wait this exists to prove is excluded.
            assert outcome.wall_time < UNIT * 2.5

    def test_summed_compute_approximates_the_true_total(self):
        """8 tasks of one unit is 8 units of compute, however queued."""
        ntasks = 8
        with thread_dispatcher(_run, concurrency=2) as dispatcher:
            handles = dispatcher.submit(_tasks(ntasks, delay=UNIT))
            total = sum(h.outcome().wall_time for h in handles)
        expected = UNIT * ntasks
        # The lower bound is the one that matters: a parent-side stamp
        # would report roughly 2.5x this, so an upper bound loose enough
        # to survive per-task overhead still catches it.
        assert 0.7 * expected <= total <= 2.0 * expected

    def test_compute_over_wall_clock_reflects_the_worker_count(self):
        """The ratio a parent-side stamp gets wrong by roughly 2.5x."""
        ntasks, nworkers = 8, 2
        with thread_dispatcher(_run, concurrency=nworkers) as dispatcher:
            start = time.perf_counter()
            handles = dispatcher.submit(_tasks(ntasks, delay=UNIT))
            compute = sum(h.outcome().wall_time for h in handles)
            wall = time.perf_counter() - start
        assert 0.7 * nworkers <= compute / wall <= 1.4 * nworkers


class TestFailureIsAReturnValue:
    def test_raising_task_becomes_a_failed_outcome(self):
        with thread_dispatcher(_run, concurrency=2) as dispatcher:
            (handle,) = dispatcher.submit(
                [_Task(indices=[0], explode=True)]
            )
            outcome = handle.outcome()
        assert outcome.status is JobStatus.FAILED
        assert outcome.payload is None
        assert "diverged" in (outcome.detail or "")

    def test_one_failure_does_not_stop_the_others(self):
        with thread_dispatcher(_run, concurrency=2) as dispatcher:
            handles = dispatcher.submit(
                [
                    _Task(indices=[0], value=1),
                    _Task(indices=[1], value=2, explode=True),
                    _Task(indices=[2], value=3),
                ]
            )
            statuses = [h.outcome().status for h in handles]
        assert statuses == [
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.SUCCEEDED,
        ]

    def test_a_failure_still_carries_its_indices(self):
        with thread_dispatcher(_run, concurrency=2) as dispatcher:
            (handle,) = dispatcher.submit(
                [_Task(indices=[5, 6], explode=True)]
            )
            assert list(handle.outcome().indices) == [5, 6]


class TestTimeout:
    def test_expiry_returns_outstanding_rather_than_raising(self):
        """A timeout is a fact about the wait, not about the work."""
        with thread_dispatcher(_run, concurrency=1) as dispatcher:
            (handle,) = dispatcher.submit(
                [_Task(indices=[0], delay=UNIT * 4)]
            )
            outcome = handle.outcome(timeout=UNIT * 0.2)
            assert outcome.status is JobStatus.OUTSTANDING
            handle.outcome()

    def test_the_same_handle_resolves_once_the_work_lands(self):
        """A timed-out poll must not consume or invalidate anything."""
        with thread_dispatcher(_run, concurrency=1) as dispatcher:
            (handle,) = dispatcher.submit(
                [_Task(indices=[0], value=7, delay=UNIT)]
            )
            assert (
                handle.outcome(timeout=0.0).status is JobStatus.OUTSTANDING
            )
            assert handle.outcome().payload == 70


class TestCancel:
    def test_pending_work_does_not_start(self):
        """The guarantee an executor can actually make."""
        ran = []

        def recording(task: _Task) -> int:
            ran.append(task.value)
            time.sleep(task.delay)
            return task.value

        with thread_dispatcher(recording, concurrency=1) as dispatcher:
            handles = dispatcher.submit(_tasks(6, delay=UNIT))
            cancelled = [h for h in handles[2:] if h.cancel()]
            for handle in handles:
                handle.outcome()
        assert cancelled
        assert len(ran) < 6

    def test_a_running_job_cannot_be_cancelled(self):
        """Stated plainly rather than papered over.

        An executor cannot interrupt a running future, so cancel reports
        False and the work completes. A dispatcher holding child
        processes can do better; this one says what it can do.
        """
        with thread_dispatcher(_run, concurrency=2) as dispatcher:
            (handle,) = dispatcher.submit(
                [_Task(indices=[0], delay=UNIT)]
            )
            time.sleep(UNIT * 0.4)
            assert handle.cancel() is False
            assert handle.outcome().status is JobStatus.SUCCEEDED

    def test_a_cancelled_job_reports_cancelled(self):
        with thread_dispatcher(_run, concurrency=1) as dispatcher:
            handles = dispatcher.submit(_tasks(6, delay=UNIT))
            tail = handles[-1]
            if tail.cancel():
                assert tail.outcome().status is JobStatus.CANCELLED
            for handle in handles[:-1]:
                handle.outcome()

    def test_cancelled_status_is_retryable(self):
        """It says nothing about the parameter point."""
        assert JobStatus.CANCELLED.is_retryable()


class TestLifecycle:
    def test_close_refuses_further_submission(self):
        dispatcher = thread_dispatcher(_run, concurrency=1)
        dispatcher.close()
        with pytest.raises(RuntimeError, match="closed"):
            dispatcher.submit([_Task(indices=[0])])

    def test_close_is_idempotent(self):
        dispatcher = thread_dispatcher(_run, concurrency=1)
        dispatcher.close()
        dispatcher.close()

    def test_a_shared_executor_survives_close(self):
        """Closing one model's dispatcher must not break another's.

        A shared pool is how a throttle spans an ensemble, so a
        dispatcher that did not own its executor must leave it running.
        """
        executor = ThreadPoolExecutor(max_workers=2)
        first = ExecutorDispatcher(_run, executor, concurrency=2)
        second = ExecutorDispatcher(_run, executor, concurrency=2)
        first.close()
        (handle,) = second.submit([_Task(indices=[0], value=3)])
        assert handle.outcome().payload == 30
        second.close()
        executor.shutdown()

    def test_empty_submission_returns_no_handles(self):
        with thread_dispatcher(_run, concurrency=2) as dispatcher:
            assert list(dispatcher.submit([])) == []
