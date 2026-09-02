"""The composition, proven serially before concurrency is added.

Every test here runs on the inline dispatcher. That is deliberate: the
evaluator's job is the failure and streaming paths, and debugging those
through a thread pool means mixing two independent sources of
nondeterminism.

The ``bkd`` fixture throughout, because these assert on
``EvalResult.values``, ``succeeded`` and ``failed`` -- empty ``(nqoi, 0)``
construction, integer index arrays and column concatenation are exactly
where NumPy and Torch diverge, and they occur on the failure and partial
paths.
"""

import logging
import time

import pytest
from pyapprox.interface.evaluation.callable_marshaller import (
    CallableMarshaller,
)
from pyapprox.interface.evaluation.evaluator import Evaluator
from pyapprox.interface.evaluation.inline_dispatcher import InlineDispatcher
from pyapprox.interface.evaluation.protocols import (
    BatchProtocol,
    EvaluatorProtocol,
)
from pyapprox.interface.evaluation.records import (
    CostLedger,
    Decoded,
    JobStatus,
    Request,
)

#: Seconds of simulated work per sample, for the timing tests. Small
#: enough that the whole module stays well under a second, large enough
#: that the ratios asserted below are not measuring scheduler noise.
DELAY_PER_SAMPLE = 0.02


def _build(bkd, fn=None, **kwargs):
    """An evaluator over a vectorized sum-of-squares model."""
    if fn is None:

        def fn(samples):
            return bkd.sum(samples * samples, axis=0)[None, :]

    marshaller = CallableMarshaller(fn, bkd, nvars=2, nqoi=1, **kwargs)
    return Evaluator(marshaller, InlineDispatcher(marshaller.run))


def _columns(bkd, n):
    """(2, n) whose column j sums to 2*j^2, so values are distinguishable."""
    return bkd.array([[float(j) for j in range(n)] for _ in range(2)])


class TestConformance:
    def test_evaluator_satisfies_protocol(self, bkd):
        assert isinstance(_build(bkd), EvaluatorProtocol)

    def test_batch_satisfies_protocol(self, bkd):
        assert isinstance(_build(bkd).submit(bkd.zeros((2, 1))), BatchProtocol)

    def test_dimensions_come_from_the_marshaller(self, bkd):
        ev = _build(bkd)
        assert ev.nvars() == 2
        assert ev.nqoi() == 1

    def test_rejects_bad_marshaller(self):
        with pytest.raises(TypeError, match="MarshallerProtocol"):
            Evaluator("not a marshaller", InlineDispatcher(abs))


class TestSubmitValidation:
    """Validation precedes task construction, so a rejection costs nothing."""

    def test_rejects_1d_samples(self, bkd):
        with pytest.raises(ValueError, match="2D"):
            _build(bkd).submit(bkd.zeros((2,)))

    def test_rejects_wrong_nvars(self, bkd):
        with pytest.raises(ValueError, match="rows"):
            _build(bkd).submit(bkd.zeros((3, 4)))

    def test_rejects_request_for_absent_capability(self, bkd):
        """At submit, where the caller can still act on it."""
        with pytest.raises(ValueError, match="jacobians"):
            _build(bkd).submit(bkd.zeros((2, 1)), Request(jacobians=True))

    def test_validation_happens_before_any_work(self, bkd):
        """A rejected submission must not have called the model."""
        calls = []

        def counting(samples):
            calls.append(samples.shape[1])
            return bkd.sum(samples, axis=0)[None, :]

        with pytest.raises(ValueError):
            _build(bkd, fn=counting).submit(
                bkd.zeros((2, 4)), Request(jacobians=True)
            )
        assert calls == []


class TestValues:
    def test_round_trip(self, bkd):
        X = bkd.array([[1.0, 2.0], [3.0, 4.0]])
        result = _build(bkd).submit(X).collect()
        bkd.assert_allclose(result.values, bkd.array([[10.0, 20.0]]))

    def test_succeeded_indexes_the_submitted_batch(self, bkd):
        result = _build(bkd).submit(_columns(bkd, 5)).collect()
        bkd.assert_allclose(result.succeeded, bkd.asarray([0, 1, 2, 3, 4]))

    def test_values_align_with_succeeded_when_split(self, bkd):
        """Columns must follow index order, not arrival order.

        With several tasks the pieces are decoded separately, so an
        implementation that sorted the indices while concatenating
        columns in arrival order would pair each value with the wrong
        sample -- silently, and only for split marshallers.
        """
        X = _columns(bkd, 4)
        result = _build(bkd, samples_per_task=1).submit(X).collect()
        bkd.assert_allclose(result.values, bkd.sum(X * X, axis=0)[None, :])

    def test_empty_batch(self, bkd):
        result = _build(bkd).submit(bkd.zeros((2, 0))).collect()
        assert result.values.shape == (1, 0)
        assert result.nsucceeded() == 0

    def test_single_sample_is_a_column(self, bkd):
        result = _build(bkd).submit(bkd.array([[3.0], [4.0]])).collect()
        bkd.assert_allclose(result.values, bkd.array([[25.0]]))


class TestGrouping:
    def test_vectorized_model_called_once(self, bkd):
        """The regression the default grouping exists to prevent."""
        widths = []

        def counting(samples):
            widths.append(samples.shape[1])
            return bkd.sum(samples * samples, axis=0)[None, :]

        _build(bkd, fn=counting).submit(bkd.ones((2, 20))).collect()
        assert widths == [20]

    def test_split_marshaller_calls_per_sample(self, bkd):
        widths = []

        def counting(samples):
            widths.append(samples.shape[1])
            return bkd.sum(samples * samples, axis=0)[None, :]

        ev = _build(bkd, fn=counting, samples_per_task=1)
        ev.submit(bkd.ones((2, 4))).collect()
        assert widths == [1, 1, 1, 1]


class TestEagerTiming:
    """Where the time goes, and what a caller may conclude from it.

    A serial dispatcher has nowhere to run work in the background, so
    ``submit`` does it and ``collect`` is free. The alternative -- return
    from submit having done nothing, run on first poll -- looks
    non-blocking and is not: a caller who submits, waits out the expected
    duration, then collects would pay the full cost at collection having
    gained nothing from the wait.
    """

    def test_submit_carries_the_cost_and_collect_is_cheap(self, bkd):
        def slow(samples):
            time.sleep(DELAY_PER_SAMPLE * samples.shape[1])
            return bkd.sum(samples, axis=0)[None, :]

        ev = _build(bkd, fn=slow)
        X = bkd.ones((2, 4))
        expected = DELAY_PER_SAMPLE * 4

        start = time.perf_counter()
        batch = ev.submit(X)
        submit_seconds = time.perf_counter() - start

        start = time.perf_counter()
        batch.collect()
        collect_seconds = time.perf_counter() - start

        assert submit_seconds >= expected * 0.5
        assert collect_seconds < expected * 0.5

    def test_progress_is_truthful_immediately_after_submit(self, bkd):
        """The work is done, so progress must not report it outstanding.

        Counting only what has been *collected* would report ``0/4`` here
        -- describing the collection state while claiming to describe the
        work.
        """
        batch = _build(bkd).submit(bkd.ones((2, 4)))
        progress = batch.progress()
        assert progress.nsucceeded == 4
        assert progress.noutstanding == 0
        assert progress.is_complete()

    @pytest.mark.parametrize("samples_per_task", [None, 1])
    def test_grouping_does_not_change_what_a_caller_sees(
        self, bkd, samples_per_task
    ):
        """One task of four, or four of one -- same observable result."""
        kwargs = (
            {} if samples_per_task is None
            else {"samples_per_task": samples_per_task}
        )
        batch = _build(bkd, **kwargs).submit(_columns(bkd, 4))
        assert batch.progress().is_complete()
        assert batch.collect().nsucceeded() == 4


class TestSeparateBatches:
    """Two submissions are independent, and the ledger spans them.

    This is what makes a retry a new submit rather than an extension of
    a live batch: each batch tracks its own work, and cost accumulates
    across both.
    """

    def test_each_batch_reports_only_its_own_work(self, bkd):
        ev = _build(bkd)
        first = ev.submit(bkd.ones((2, 2)))
        second = ev.submit(bkd.ones((2, 5)))

        assert first.nsubmitted() == 2
        assert second.nsubmitted() == 5
        assert first.progress().nsucceeded == 2
        assert second.progress().nsucceeded == 5

    def test_results_go_to_the_batch_that_asked(self, bkd):
        ev = _build(bkd)
        first = ev.submit(_columns(bkd, 2))
        second = ev.submit(_columns(bkd, 5))

        assert first.collect().nsucceeded() == 2
        assert second.collect().nsucceeded() == 5

    def test_shared_ledger_accumulates_across_batches(self, bkd):
        """What a budget spanning submissions reads."""

        def slow(samples):
            time.sleep(DELAY_PER_SAMPLE * samples.shape[1])
            return bkd.sum(samples, axis=0)[None, :]

        ev = _build(bkd, fn=slow)
        ev.submit(bkd.ones((2, 2))).collect()
        after_first = ev.ledger().total().compute
        ev.submit(bkd.ones((2, 3))).collect()
        after_second = ev.ledger().total().compute

        assert after_first >= DELAY_PER_SAMPLE * 2 * 0.5
        assert after_second > after_first

    def test_one_batch_does_not_complete_another(self, bkd):
        """Collecting the first must not consume the second's outcomes."""
        ev = _build(bkd)
        first = ev.submit(_columns(bkd, 3))
        second = ev.submit(_columns(bkd, 4))
        first.collect()
        assert second.progress().nsucceeded == 4
        assert second.collect().nsucceeded() == 4


class TestFailureIsAReturnValue:
    def test_failing_task_reported_not_raised(self, bkd):
        def explodes(samples):
            raise RuntimeError("solver diverged")

        result = _build(bkd, fn=explodes).submit(bkd.ones((2, 3))).collect()
        assert result.nsucceeded() == 0
        assert result.nfailed() == 3

    def test_one_failure_does_not_lose_the_batch(self, bkd):
        """The sharpest claim: a bad sample costs that sample only."""

        def picky(samples):
            if bkd.to_float(bkd.max(samples)) > 2.5:
                raise RuntimeError("diverged")
            return bkd.sum(samples * samples, axis=0)[None, :]

        result = (
            _build(bkd, fn=picky, samples_per_task=1)
            .submit(_columns(bkd, 5))
            .collect()
        )
        assert result.nsucceeded() == 3
        assert result.nfailed() == 2
        bkd.assert_allclose(result.succeeded, bkd.asarray([0, 1, 2]))

    def test_values_narrow_to_the_successes(self, bkd):
        def picky(samples):
            if bkd.to_float(bkd.max(samples)) > 2.5:
                raise RuntimeError("diverged")
            return bkd.sum(samples * samples, axis=0)[None, :]

        result = (
            _build(bkd, fn=picky, samples_per_task=1)
            .submit(_columns(bkd, 5))
            .collect()
        )
        bkd.assert_allclose(result.values, bkd.array([[0.0, 2.0, 8.0]]))

    def test_failure_is_evidence_about_the_point(self, bkd):
        """FAILED is not retryable; the solver diverged *there*."""

        def explodes(samples):
            raise RuntimeError("diverged")

        result = _build(bkd, fn=explodes).submit(bkd.ones((2, 1))).collect()
        assert result.statuses[0] is JobStatus.FAILED
        assert not result.statuses[0].is_retryable()

    def test_marshal_error_fails_only_that_task(self, bkd):
        """A wrong-shaped return is a decode failure, not a crash."""

        def wrong_shape(samples):
            return bkd.zeros((1, samples.shape[1] + 1))

        result = (
            _build(bkd, fn=wrong_shape).submit(bkd.ones((2, 2))).collect()
        )
        assert result.nfailed() == 2
        assert result.nsucceeded() == 0


class TestStreamingContract:
    def test_collect_ready_then_collect_covers_the_batch_once(self, bkd):
        """The check a consuming-cursor design would fail."""
        batch = _build(bkd, samples_per_task=1).submit(_columns(bkd, 4))
        first = batch.collect_ready()
        second = batch.collect()
        seen = sorted(
            [int(i) for i in first.succeeded]
            + [int(i) for i in second.succeeded]
        )
        assert seen == [0, 1, 2, 3]

    def test_no_index_is_returned_twice(self, bkd):
        batch = _build(bkd, samples_per_task=1).submit(_columns(bkd, 3))
        batch.collect()
        assert batch.collect().nsucceeded() == 0

    def test_union_over_calls_is_the_whole_batch(self, bkd):
        batch = _build(bkd, samples_per_task=1).submit(_columns(bkd, 6))
        total = 0
        for _ in range(3):
            total += batch.collect_ready().nsucceeded()
        total += batch.collect().nsucceeded()
        assert total == 6


class TestCumulativeStatuses:
    """The whole-batch view that a per-harvest result cannot give.

    ``EvalResult.statuses`` covers one harvest, so a caller streaming
    with ``collect_ready`` holds several disjoint partial maps and no
    complete one.
    """

    def test_successes_are_recorded_not_only_failures(self, bkd) -> None:
        batch = _build(bkd, samples_per_task=1).submit(_columns(bkd, 3))
        batch.collect()
        assert batch.statuses() == {
            0: JobStatus.SUCCEEDED,
            1: JobStatus.SUCCEEDED,
            2: JobStatus.SUCCEEDED,
        }

    def test_the_result_also_carries_its_own_successes(self, bkd) -> None:
        """The per-harvest map is no longer failure-only either."""
        result = _build(bkd, samples_per_task=1).submit(
            _columns(bkd, 2)
        ).collect()
        assert result.statuses == {
            0: JobStatus.SUCCEEDED,
            1: JobStatus.SUCCEEDED,
        }

    def test_it_accumulates_across_streaming_calls(self, bkd) -> None:
        batch = _build(bkd, samples_per_task=1).submit(_columns(bkd, 6))
        for _ in range(3):
            batch.collect_ready()
        batch.collect()
        assert len(batch.statuses()) == 6
        assert set(batch.statuses()) == {0, 1, 2, 3, 4, 5}

    def test_outstanding_indices_are_absent(self, bkd) -> None:
        batch = _build(bkd, samples_per_task=1).submit(_columns(bkd, 4))
        assert batch.statuses() == {}

    def test_a_failure_keeps_its_own_status(self, bkd) -> None:
        def explodes(samples):
            raise RuntimeError("diverged")

        batch = _build(bkd, fn=explodes, samples_per_task=1).submit(
            _columns(bkd, 2)
        )
        batch.collect()
        assert batch.statuses() == {
            0: JobStatus.FAILED,
            1: JobStatus.FAILED,
        }

    def test_the_map_is_a_copy(self, bkd) -> None:
        """A caller must not be able to rewrite the batch's record."""
        batch = _build(bkd, samples_per_task=1).submit(_columns(bkd, 2))
        batch.collect()
        batch.statuses()[0] = JobStatus.CANCELLED
        assert batch.statuses()[0] is JobStatus.SUCCEEDED


class TestProgress:
    def test_progress_completes_after_collect(self, bkd):
        batch = _build(bkd).submit(bkd.ones((2, 4)))
        batch.collect()
        assert batch.progress().is_complete()

    def test_repeated_polling_does_not_multiply_cost(self, bkd):
        """A running total must not be re-accumulated."""
        batch = _build(bkd).submit(bkd.ones((2, 2)))
        batch.collect()
        first = batch.progress().cost.compute
        batch.progress()
        assert batch.progress().cost.compute == pytest.approx(first)

    def test_empty_batch_is_complete(self, bkd):
        batch = _build(bkd).submit(bkd.zeros((2, 0)))
        assert batch.progress().is_complete()
        assert batch.progress().fraction_returned() == pytest.approx(1.0)


class TestLedger:
    def test_shared_ledger_is_the_same_object(self, bkd):
        ledger = CostLedger()
        m1 = CallableMarshaller(abs, bkd, nvars=2, nqoi=2)
        m2 = CallableMarshaller(abs, bkd, nvars=2, nqoi=2)
        ev1 = Evaluator(m1, InlineDispatcher(m1.run), ledger=ledger)
        ev2 = Evaluator(m2, InlineDispatcher(m2.run), ledger=ledger)
        assert ev1.ledger() is ev2.ledger()

    def test_private_ledger_by_default(self, bkd):
        assert _build(bkd).ledger() is not _build(bkd).ledger()


class TestJobReportedTime:
    """A duration the job reports beats one the wrapper measured.

    A dispatcher can only time what it waited on. For in-process work
    that is the job, but anything external also covers process startup,
    input staging and filesystem sync -- and on a scheduler, the entire
    queue wait. A wrapper can report hours for a solve that took
    minutes, which would put queue time into a compute budget.
    """

    def test_a_decoded_runtime_replaces_the_measured_one(self, bkd):
        reported = 0.001

        class SelfReporting(CallableMarshaller):
            def values(self, outcome):
                decoded = super().values(outcome)
                return Decoded(
                    values=decoded.values,
                    indices=decoded.indices,
                    wall_time=reported,
                )

        def slow(samples):
            time.sleep(DELAY_PER_SAMPLE * 4)
            return bkd.sum(samples, axis=0)[None, :]

        marshaller = SelfReporting(slow, bkd, nvars=2, nqoi=1)
        ev = Evaluator(marshaller, InlineDispatcher(marshaller.run))
        ev.submit(bkd.ones((2, 2))).collect()

        # The wrapper waited four delay units; the job says it took one
        # millisecond. The ledger must believe the job.
        assert ev.ledger().total().wall_clock == pytest.approx(
            reported, abs=1e-6
        )

    def test_the_wrapper_measurement_stands_when_none_is_reported(
        self, bkd
    ):
        """The common case: no runtime in the output, so time the call."""

        def slow(samples):
            time.sleep(DELAY_PER_SAMPLE)
            return bkd.sum(samples, axis=0)[None, :]

        ev = _build(bkd, fn=slow)
        ev.submit(bkd.ones((2, 1))).collect()
        assert ev.ledger().total().wall_clock >= DELAY_PER_SAMPLE * 0.5


class TestCompletionHook:
    def test_called_once_per_task(self, bkd):
        seen = []
        marshaller = CallableMarshaller(
            lambda X: bkd.sum(X * X, axis=0)[None, :],
            bkd,
            nvars=2,
            nqoi=1,
            samples_per_task=1,
        )
        ev = Evaluator(
            marshaller,
            InlineDispatcher(marshaller.run),
            on_complete=lambda outcome, decoded, cost: seen.append(outcome),
        )
        ev.submit(_columns(bkd, 3)).collect()
        assert len(seen) == 3

    def test_decoded_is_none_for_a_failure(self, bkd):
        """A store recording only successes would recompute them forever."""
        seen = []

        def explodes(samples):
            raise RuntimeError("diverged")

        marshaller = CallableMarshaller(explodes, bkd, nvars=2, nqoi=1)
        ev = Evaluator(
            marshaller,
            InlineDispatcher(marshaller.run),
            on_complete=lambda outcome, decoded, cost: seen.append(decoded),
        )
        ev.submit(bkd.ones((2, 1))).collect()
        assert seen == [None]

    def test_hook_receives_the_outcome_and_a_cost(self, bkd):
        records = []
        marshaller = CallableMarshaller(
            lambda X: bkd.sum(X, axis=0)[None, :], bkd, nvars=2, nqoi=1
        )
        ev = Evaluator(
            marshaller,
            InlineDispatcher(marshaller.run),
            on_complete=lambda o, d, c: records.append((o, d, c)),
        )
        ev.submit(bkd.ones((2, 2))).collect()
        outcome, decoded, cost = records[0]
        assert outcome.status is JobStatus.SUCCEEDED
        assert decoded is not None
        assert cost.compute >= 0.0


class TestCollaboratorFailuresDoNotEndTheBatch:
    """A hook or a release that raises costs its own task, not the run.

    Both are called per task inside the harvest loop, which is also
    where handles leave ``_pending``. Propagating would abandon every
    handle after the failure: never released, never charged, and absent
    from the returned result.
    """

    def _marshaller(self, bkd, **kwargs):
        return CallableMarshaller(
            lambda X: bkd.sum(X * X, axis=0)[None, :],
            bkd,
            nvars=2,
            nqoi=1,
            samples_per_task=1,
            **kwargs,
        )

    def test_a_raising_hook_still_yields_every_sample(self, bkd) -> None:
        marshaller = self._marshaller(bkd)

        def explodes(outcome, decoded, cost):
            raise RuntimeError("the store is full")

        ev = Evaluator(
            marshaller,
            InlineDispatcher(marshaller.run),
            on_complete=explodes,
        )
        result = ev.submit(_columns(bkd, 3)).collect()
        assert result.nsucceeded() == 3
        assert result.values.shape == (1, 3)

    def test_a_raising_hook_does_not_skip_release(self, bkd) -> None:
        """The failure must not cascade into the next collaborator."""
        released = []
        marshaller = self._marshaller(bkd)
        original = marshaller.release

        def watch(outcome):
            released.append(outcome)
            original(outcome)

        marshaller.release = watch

        def explodes(outcome, decoded, cost):
            raise RuntimeError("the store is full")

        ev = Evaluator(
            marshaller,
            InlineDispatcher(marshaller.run),
            on_complete=explodes,
        )
        ev.submit(_columns(bkd, 3)).collect()
        assert len(released) == 3

    def test_a_raising_release_still_yields_every_sample(
        self, bkd
    ) -> None:
        marshaller = self._marshaller(bkd)

        def explodes(outcome):
            raise OSError("scratch is read-only")

        marshaller.release = explodes
        ev = Evaluator(marshaller, InlineDispatcher(marshaller.run))
        result = ev.submit(_columns(bkd, 3)).collect()
        assert result.nsucceeded() == 3

    def test_the_failure_is_logged_rather_than_swallowed(
        self, bkd, caplog
    ) -> None:
        """Silence would leave an unwritten store undiscoverable."""
        marshaller = self._marshaller(bkd)

        def explodes(outcome, decoded, cost):
            raise RuntimeError("the store is full")

        ev = Evaluator(
            marshaller,
            InlineDispatcher(marshaller.run),
            on_complete=explodes,
        )
        with caplog.at_level(
            logging.ERROR, logger="pyapprox.interface.evaluation.evaluator"
        ):
            ev.submit(_columns(bkd, 2)).collect()
        assert "completion hook failed" in caplog.text
        assert "the store is full" in caplog.text
