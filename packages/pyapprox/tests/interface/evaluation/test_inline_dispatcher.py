"""Serial dispatch, tested directly against the protocol.

No evaluator here on purpose. The dispatcher contract -- submit returns
without running, handles are idempotent, a raising task becomes a FAILED
outcome rather than an exception -- is what everything else is built on,
so it is worth pinning before anything composes it.

A dispatcher is array-free by construction, so these run on
``numpy_bkd`` only; running them twice would exercise nothing new.
"""

from dataclasses import dataclass
from typing import Sequence

import pytest
from pyapprox.interface.evaluation.inline_dispatcher import InlineDispatcher
from pyapprox.interface.evaluation.protocols import (
    DispatcherProtocol,
    JobHandle,
    TaskProtocol,
)
from pyapprox.interface.evaluation.records import (
    ComputeProvenance,
    JobStatus,
)


@dataclass(frozen=True)
class _Task:
    """A task carrying just enough to be dispatched and identified."""

    indices: Sequence[int]
    value: int = 0
    explode: bool = False


def _run(task: _Task) -> int:
    if task.explode:
        raise RuntimeError(f"task {task.value} diverged")
    return task.value * 10


class TestProtocolConformance:
    def test_satisfies_dispatcher_protocol(self):
        assert isinstance(InlineDispatcher(_run), DispatcherProtocol)

    def test_handles_satisfy_job_handle(self):
        (handle,) = InlineDispatcher(_run).submit([_Task(indices=[0])])
        assert isinstance(handle, JobHandle)

    def test_task_satisfies_task_protocol(self):
        assert isinstance(_Task(indices=[0]), TaskProtocol)

    def test_concurrency_is_one(self):
        assert InlineDispatcher(_run).concurrency() == 1

    def test_provenance_is_measured(self):
        """One worker means the two clocks genuinely coincide."""
        assert (
            InlineDispatcher(_run).compute_provenance()
            is ComputeProvenance.MEASURED
        )

    def test_rejects_non_callable(self):
        with pytest.raises(TypeError, match="callable"):
            InlineDispatcher("not callable")


class TestLazyExecution:
    """Tasks run on first poll, not in submit.

    This is what lets a serial batch report genuine partial progress. An
    eager dispatcher would finish everything inside ``submit``, leaving
    ``progress()`` able to say only "nothing" or "everything".
    """

    def test_submit_does_not_run_tasks(self):
        ran = []

        def record(task: _Task) -> int:
            ran.append(task.value)
            return task.value

        dispatcher = InlineDispatcher(record)
        dispatcher.submit([_Task(indices=[i], value=i) for i in range(3)])
        assert ran == []

    def test_done_does_not_run_the_task(self):
        """Asking about work must not cause it.

        Otherwise ``progress()`` could never report anything
        outstanding, because reporting would complete the batch.
        """
        ran = []

        def record(task: _Task) -> int:
            ran.append(task.value)
            return task.value

        (handle,) = InlineDispatcher(record).submit([_Task(indices=[0])])
        assert not handle.done()
        assert ran == []

    def test_outcome_runs_the_task(self):
        (handle,) = InlineDispatcher(_run).submit(
            [_Task(indices=[0], value=4)]
        )
        assert handle.outcome().payload == 40
        assert handle.done()

    def test_tasks_run_one_at_a_time_as_polled(self):
        """Polling the second handle must not run the third."""
        ran = []

        def record(task: _Task) -> int:
            ran.append(task.value)
            return task.value

        dispatcher = InlineDispatcher(record)
        handles = dispatcher.submit(
            [_Task(indices=[i], value=i) for i in range(3)]
        )
        handles[0].outcome()
        assert ran == [0]
        handles[1].outcome()
        assert ran == [0, 1]


class TestIdempotence:
    """A handle has one outcome, readable as often as you like."""

    def test_task_runs_at_most_once(self):
        ran = []

        def record(task: _Task) -> int:
            ran.append(task.value)
            return task.value

        (handle,) = InlineDispatcher(record).submit([_Task(indices=[0])])
        handle.outcome()
        handle.outcome()
        handle.outcome()
        assert len(ran) == 1

    def test_repeated_outcomes_are_identical(self):
        (handle,) = InlineDispatcher(_run).submit(
            [_Task(indices=[0], value=7)]
        )
        first = handle.outcome()
        assert handle.outcome() is first

    def test_done_is_repeatable_and_free(self):
        (handle,) = InlineDispatcher(_run).submit([_Task(indices=[0])])
        handle.outcome()
        assert handle.done()
        assert handle.done()


class TestFailureIsAReturnValue:
    def test_raising_task_becomes_failed_outcome(self):
        """The exception must not escape and abandon the batch."""
        (handle,) = InlineDispatcher(_run).submit(
            [_Task(indices=[0], explode=True)]
        )
        outcome = handle.outcome()
        assert outcome.status is JobStatus.FAILED
        assert outcome.payload is None

    def test_failure_detail_records_the_cause(self):
        (handle,) = InlineDispatcher(_run).submit(
            [_Task(indices=[0], value=3, explode=True)]
        )
        assert "diverged" in outcome_detail(handle)

    def test_one_failure_does_not_stop_the_others(self):
        dispatcher = InlineDispatcher(_run)
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

    def test_failed_outcome_still_carries_indices(self):
        """A failure must still say which samples it was for."""
        (handle,) = InlineDispatcher(_run).submit(
            [_Task(indices=[5, 6], explode=True)]
        )
        assert list(handle.outcome().indices) == [5, 6]

    def test_failed_outcome_is_charged(self):
        """Failures consumed real compute and must reach the ledger."""
        (handle,) = InlineDispatcher(_run).submit(
            [_Task(indices=[0], explode=True)]
        )
        assert handle.outcome().wall_time >= 0.0
        assert handle.outcome().status.is_retryable() is False


def outcome_detail(handle) -> str:
    detail = handle.outcome().detail
    return "" if detail is None else detail


class TestIndicesAreExplicit:
    def test_outcome_carries_the_task_indices(self):
        """Never inferred from completion order."""
        dispatcher = InlineDispatcher(_run)
        handles = dispatcher.submit(
            [_Task(indices=[9]), _Task(indices=[2]), _Task(indices=[5])]
        )
        # Poll out of order; each outcome still knows its own columns.
        assert list(handles[2].outcome().indices) == [5]
        assert list(handles[0].outcome().indices) == [9]
        assert list(handles[1].outcome().indices) == [2]

    def test_multi_sample_task_keeps_all_indices(self):
        (handle,) = InlineDispatcher(_run).submit(
            [_Task(indices=[3, 4, 5])]
        )
        assert list(handle.outcome().indices) == [3, 4, 5]


class TestCancel:
    def test_cancel_prevents_a_pending_task_from_running(self):
        ran = []

        def record(task: _Task) -> int:
            ran.append(task.value)
            return task.value

        (handle,) = InlineDispatcher(record).submit([_Task(indices=[0])])
        assert handle.cancel() is True
        assert handle.outcome().status is JobStatus.CANCELLED
        assert ran == []

    def test_cancel_after_running_reports_false(self):
        (handle,) = InlineDispatcher(_run).submit([_Task(indices=[0])])
        handle.outcome()
        assert handle.cancel() is False

    def test_cancelled_outcome_is_still_readable(self):
        """Cancelling must not discard the record of what happened."""
        (handle,) = InlineDispatcher(_run).submit([_Task(indices=[1, 2])])
        handle.cancel()
        outcome = handle.outcome()
        assert list(outcome.indices) == [1, 2]
        assert outcome.status.is_retryable() is True


class TestLifecycle:
    def test_close_refuses_further_submission(self):
        dispatcher = InlineDispatcher(_run)
        dispatcher.close()
        with pytest.raises(RuntimeError, match="closed"):
            dispatcher.submit([_Task(indices=[0])])

    def test_close_is_idempotent(self):
        dispatcher = InlineDispatcher(_run)
        dispatcher.close()
        dispatcher.close()

    def test_context_manager_closes(self):
        with InlineDispatcher(_run) as dispatcher:
            dispatcher.submit([_Task(indices=[0])])
        with pytest.raises(RuntimeError, match="closed"):
            dispatcher.submit([_Task(indices=[0])])

    def test_empty_submission_returns_no_handles(self):
        assert list(InlineDispatcher(_run).submit([])) == []
