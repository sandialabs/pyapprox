"""Boundary records: cost arithmetic, provenance, statuses and results.

Most of these are backend-independent value objects, so they use
``numpy_bkd``. The exception is ``EvalResult``, whose index arrays and
narrowed values are built by the evaluator from backend arrays -- empty
``(nqoi, 0)`` construction and integer-dtype index arrays are exactly
where NumPy and Torch diverge, so those run on both.

The properties under test here are the ones that propagate into
statistics rather than control flow, and are correspondingly expensive
to retrofit: that wall-clock does not sum across concurrent work, that a
NaN duration cannot enter a ledger, and that a retryable stop is
distinguishable from evidence about a parameter point.
"""

import pytest
from pyapprox.interface.evaluation.records import (
    ComputeProvenance,
    Cost,
    CostLedger,
    EvalProgress,
    EvalResult,
    JobStatus,
    Outcome,
    Resources,
    TimeSource,
)


class TestCost:
    def test_serial_clocks_coincide(self, numpy_bkd):
        cost = Cost.serial(2.5)
        numpy_bkd.assert_allclose(
            numpy_bkd.asarray([cost.wall_clock]),
            numpy_bkd.asarray([cost.compute]),
        )
        assert cost.provenance is ComputeProvenance.MEASURED

    def test_estimated_caps_busy_workers_at_nsamples(self, numpy_bkd):
        """A 4-sample batch on a 64-wide pool occupies four workers.

        Charging it 64x would corrupt the very cost measurement a pilot
        exists to make.
        """
        cost = Cost.estimated(wall_clock=1.0, concurrency=64, nsamples=4)
        numpy_bkd.assert_allclose(
            numpy_bkd.asarray([cost.compute]), numpy_bkd.asarray([4.0])
        )
        assert cost.provenance is ComputeProvenance.ESTIMATED

    def test_estimated_uses_concurrency_when_it_binds(self, numpy_bkd):
        cost = Cost.estimated(wall_clock=2.0, concurrency=4, nsamples=64)
        numpy_bkd.assert_allclose(
            numpy_bkd.asarray([cost.compute]), numpy_bkd.asarray([8.0])
        )

    def test_unaccounted_reports_zero_compute(self, numpy_bkd):
        """A socket wait times a core count is not a bound on anything."""
        cost = Cost.unaccounted(3.0)
        numpy_bkd.assert_allclose(
            numpy_bkd.asarray([cost.compute]), numpy_bkd.asarray([0.0])
        )
        assert cost.provenance is ComputeProvenance.NOT_APPLICABLE

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_rejects_non_finite(self, bad):
        """``nan < 0.0`` is False, so non-negativity alone admits NaN.

        One NaN would silently turn an entire accumulated ledger into
        NaN, and a failed job is exactly where a NaN duration arises.
        """
        with pytest.raises(ValueError, match="finite"):
            Cost(wall_clock=bad, compute=1.0, provenance=ComputeProvenance.MEASURED)
        with pytest.raises(ValueError, match="finite"):
            Cost(wall_clock=1.0, compute=bad, provenance=ComputeProvenance.MEASURED)

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match="non-negative"):
            Cost(
                wall_clock=-1.0,
                compute=0.0,
                provenance=ComputeProvenance.MEASURED,
            )

    def test_estimated_rejects_bad_arguments(self):
        with pytest.raises(ValueError, match="concurrency"):
            Cost.estimated(1.0, concurrency=0, nsamples=1)
        with pytest.raises(ValueError, match="nsamples"):
            Cost.estimated(1.0, concurrency=1, nsamples=-1)

    def test_has_no_add(self):
        """Wall-clock is not additive; the ledger exists for this reason."""
        assert not hasattr(Cost.zero(), "__add__") or not isinstance(
            getattr(Cost, "__add__", None), type(lambda: None)
        )


class TestCostLedger:
    def test_empty_ledger_is_zero(self, numpy_bkd):
        total = CostLedger().total()
        numpy_bkd.assert_allclose(
            numpy_bkd.asarray([total.wall_clock, total.compute]),
            numpy_bkd.asarray([0.0, 0.0]),
        )

    def test_compute_sums(self, numpy_bkd):
        ledger = CostLedger()
        ledger.add(Cost.serial(1.0), start=0.0)
        ledger.add(Cost.serial(2.0), start=10.0)
        numpy_bkd.assert_allclose(
            numpy_bkd.asarray([ledger.total().compute]),
            numpy_bkd.asarray([3.0]),
        )

    def test_concurrent_wall_clock_is_union_not_sum(self, numpy_bkd):
        """Two jobs of one hour, run side by side, took one hour.

        This is the property that makes overlapping batches expressible
        at all, and the reason there is no ``Cost.__add__``.
        """
        ledger = CostLedger()
        ledger.add(Cost.serial(1.0), start=0.0)
        ledger.add(Cost.serial(1.0), start=0.0)
        total = ledger.total()
        numpy_bkd.assert_allclose(
            numpy_bkd.asarray([total.wall_clock]), numpy_bkd.asarray([1.0])
        )
        numpy_bkd.assert_allclose(
            numpy_bkd.asarray([total.compute]), numpy_bkd.asarray([2.0])
        )

    def test_sequential_wall_clock_sums(self, numpy_bkd):
        ledger = CostLedger()
        ledger.add(Cost.serial(1.0), start=0.0)
        ledger.add(Cost.serial(1.0), start=5.0)
        numpy_bkd.assert_allclose(
            numpy_bkd.asarray([ledger.total().wall_clock]),
            numpy_bkd.asarray([2.0]),
        )

    def test_partially_overlapping_spans(self, numpy_bkd):
        """(0,2) and (1,4) cover 4 seconds, not 5."""
        ledger = CostLedger()
        ledger.add(Cost.serial(2.0), start=0.0)
        ledger.add(Cost.serial(3.0), start=1.0)
        numpy_bkd.assert_allclose(
            numpy_bkd.asarray([ledger.total().wall_clock]),
            numpy_bkd.asarray([4.0]),
        )

    def test_measured_plus_estimated_is_estimated(self):
        """A total containing an estimate is an estimate."""
        ledger = CostLedger()
        ledger.add(Cost.serial(1.0))
        ledger.add(Cost.estimated(1.0, concurrency=2, nsamples=2))
        assert ledger.total().provenance is ComputeProvenance.ESTIMATED

    def test_not_applicable_dominates(self):
        ledger = CostLedger()
        ledger.add(Cost.serial(1.0))
        ledger.add(Cost.unaccounted(1.0))
        assert ledger.total().provenance is ComputeProvenance.NOT_APPLICABLE

    def test_all_measured_stays_measured(self):
        ledger = CostLedger()
        ledger.add(Cost.serial(1.0))
        ledger.add(Cost.measured(1.0, 4.0))
        assert ledger.total().provenance is ComputeProvenance.MEASURED

    def test_omitted_start_reduces_to_longest_span(self, numpy_bkd):
        ledger = CostLedger()
        ledger.add(Cost.serial(1.0))
        ledger.add(Cost.serial(3.0))
        numpy_bkd.assert_allclose(
            numpy_bkd.asarray([ledger.total().wall_clock]),
            numpy_bkd.asarray([3.0]),
        )


class TestJobStatus:
    @pytest.mark.parametrize(
        "status,retryable",
        [
            (JobStatus.FAILED, False),
            (JobStatus.TIMED_OUT, True),
            (JobStatus.CANCELLED, True),
            (JobStatus.SUCCEEDED, False),
            (JobStatus.OUTSTANDING, False),
        ],
    )
    def test_retryable_partition(self, status, retryable):
        """FAILED is evidence about the point; the other two are not.

        Collapsing these would make "resubmit what failed" unsafe.
        """
        assert status.is_retryable() is retryable

    def test_only_outstanding_is_unfinished(self):
        assert not JobStatus.OUTSTANDING.is_finished()
        for status in (
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.TIMED_OUT,
            JobStatus.CANCELLED,
        ):
            assert status.is_finished()


class TestOutcome:
    def test_cost_scales_with_ncores(self, numpy_bkd):
        """The 32-rank MPI case: compute is wall_time * ncores.

        The core count rides on the outcome's ``Resources`` because it
        is a property of the wrapped code rather than of the machine --
        which is what lets a serial model and a 32-rank one share one
        dispatcher, and one throttle.
        """
        outcome = Outcome(
            task="t",
            indices=[0],
            status=JobStatus.SUCCEEDED,
            payload="p",
            wall_time=2.0,
            resources=Resources(ncores=32),
        )
        numpy_bkd.assert_allclose(
            numpy_bkd.asarray([outcome.cost().compute]),
            numpy_bkd.asarray([64.0]),
        )
        numpy_bkd.assert_allclose(
            numpy_bkd.asarray([outcome.cost().wall_clock]),
            numpy_bkd.asarray([2.0]),
        )

    def test_succeeded_requires_payload(self):
        with pytest.raises(ValueError, match="payload"):
            Outcome(task="t", indices=[0], status=JobStatus.SUCCEEDED)

    def test_failed_needs_no_payload(self):
        outcome = Outcome(task="t", indices=[0], status=JobStatus.FAILED)
        assert outcome.payload is None

    def test_rejects_non_finite_wall_time(self):
        with pytest.raises(ValueError, match="finite"):
            Outcome(
                task="t",
                indices=[0],
                status=JobStatus.FAILED,
                wall_time=float("nan"),
            )

    def test_rejects_zero_ncores(self):
        """Validated on the record that owns the field."""
        with pytest.raises(ValueError, match="ncores"):
            Resources(ncores=0)

    def test_nsamples_counts_indices(self):
        outcome = Outcome(
            task="t", indices=[3, 7, 11], status=JobStatus.FAILED
        )
        assert outcome.nsamples() == 3


class TestResources:
    """What a task needs from the machine, declared by the code.

    Lives on the task rather than the dispatcher so that one shared
    dispatcher -- and therefore one throttle -- can serve models with
    different requirements. A per-dispatcher constant would force an
    ensemble holding a serial model and a 32-rank model into two
    dispatchers, which cannot share a throttle.
    """

    def test_defaults_to_one_serial_core(self):
        resources = Resources()
        assert resources.ncores == 1
        assert resources.walltime_seconds is None
        assert resources.memory_mb is None
        assert resources.queue is None
        assert dict(resources.extra) == {}

    def test_serial_constructor(self):
        assert Resources.serial() == Resources()

    def test_heterogeneous_models_differ(self):
        """The case the record exists for."""
        serial = Resources(ncores=1)
        parallel = Resources(ncores=32, queue="compute")
        assert serial.ncores != parallel.ncores
        assert parallel.queue == "compute"

    @pytest.mark.parametrize("bad", [0, -1])
    def test_rejects_bad_ncores(self, bad):
        with pytest.raises(ValueError, match="ncores"):
            Resources(ncores=bad)

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_rejects_nonpositive_walltime(self, bad):
        with pytest.raises(ValueError, match="walltime"):
            Resources(walltime_seconds=bad)

    def test_rejects_non_finite_walltime(self):
        with pytest.raises(ValueError, match="finite"):
            Resources(walltime_seconds=float("inf"))

    def test_rejects_zero_memory(self):
        with pytest.raises(ValueError, match="memory_mb"):
            Resources(memory_mb=0)

    def test_rejects_empty_queue_name(self):
        """None means unspecified; empty string means nothing."""
        with pytest.raises(ValueError, match="queue"):
            Resources(queue="")

    def test_extra_carries_site_specific_arguments(self):
        """The escape hatch: what no record could enumerate."""
        resources = Resources(
            ncores=8, extra={"account": "m1234", "constraint": "gpu"}
        )
        assert resources.extra["account"] == "m1234"

    def test_adding_a_field_would_not_break_callers(self):
        """Defaults everywhere, so an older construction still works."""
        assert Resources(ncores=4) == Resources(
            ncores=4,
            walltime_seconds=None,
            memory_mb=None,
            queue=None,
        )


class TestTimeSource:
    """Which clock produced a duration, since they differ hugely.

    A wrapper timing a scheduler submission counts queue wait, staging
    and startup; a job reporting its own runtime counts none of them.
    Mixing the two in one batch without saying which is which compares
    two different quantities.
    """

    def test_outcome_defaults_to_wrapper(self):
        """What a dispatcher timing its own call can honestly claim."""
        outcome = Outcome(
            task="t", indices=[0], status=JobStatus.FAILED
        )
        assert outcome.time_source is TimeSource.WRAPPER

    def test_a_job_reported_time_can_be_recorded(self):
        outcome = Outcome(
            task="t",
            indices=[0],
            status=JobStatus.SUCCEEDED,
            payload="p",
            wall_time=12.0,
            time_source=TimeSource.JOB,
        )
        assert outcome.time_source is TimeSource.JOB


class TestEvalProgress:
    def test_counts_and_completion(self):
        progress = EvalProgress(
            nsucceeded=40,
            nfailed=3,
            noutstanding=157,
            cost=Cost.serial(6.0),
            elapsed_seconds=6.0,
        )
        assert progress.nsubmitted() == 200
        assert not progress.is_complete()
        assert progress.fraction_returned() == pytest.approx(43 / 200)

    def test_complete_when_nothing_outstanding(self):
        progress = EvalProgress(
            nsucceeded=2,
            nfailed=1,
            noutstanding=0,
            cost=Cost.zero(),
            elapsed_seconds=1.0,
        )
        assert progress.is_complete()
        assert progress.fraction_returned() == pytest.approx(1.0)

    def test_empty_batch_is_complete_and_fully_returned(self):
        """Nothing outstanding means nothing is being waited on."""
        progress = EvalProgress(
            nsucceeded=0,
            nfailed=0,
            noutstanding=0,
            cost=Cost.zero(),
            elapsed_seconds=0.0,
        )
        assert progress.is_complete()
        assert progress.fraction_returned() == pytest.approx(1.0)

    def test_elapsed_is_distinct_from_cost_wall_clock(self):
        """Queue wait is elapsed but covered by no job's span."""
        progress = EvalProgress(
            nsucceeded=1,
            nfailed=0,
            noutstanding=1,
            cost=Cost.serial(0.5),
            elapsed_seconds=10.0,
        )
        assert progress.elapsed() == pytest.approx(10.0)
        assert progress.cost.wall_clock == pytest.approx(0.5)

    def test_rejects_negative_counts(self):
        with pytest.raises(ValueError, match="non-negative"):
            EvalProgress(
                nsucceeded=-1,
                nfailed=0,
                noutstanding=0,
                cost=Cost.zero(),
                elapsed_seconds=0.0,
            )


class TestEvalResult:
    def test_counts_across_three_index_arrays(self, bkd):
        result = EvalResult(
            values=bkd.zeros((2, 3)),
            succeeded=bkd.arange(3),
            failed=bkd.arange(2),
            cancelled=bkd.arange(1),
            cost=Cost.serial(1.0),
        )
        assert result.nsucceeded() == 3
        assert result.nfailed() == 2
        assert result.ncancelled() == 1
        assert result.nreturned() == 6

    def test_empty_batch_shape(self, bkd):
        """``(nqoi, 0)`` values with no indices, on both backends."""
        result = EvalResult(
            values=bkd.zeros((2, 0)),
            succeeded=bkd.zeros((0,), dtype=int),
            failed=bkd.zeros((0,), dtype=int),
            cancelled=bkd.zeros((0,), dtype=int),
            cost=Cost.zero(),
        )
        assert result.values.shape == (2, 0)
        assert result.nreturned() == 0

    def test_failed_and_cancelled_are_separate(self, bkd):
        """Only one of the two is safe to resubmit."""
        result = EvalResult(
            values=bkd.zeros((1, 0)),
            succeeded=bkd.zeros((0,), dtype=int),
            failed=bkd.asarray([0]),
            cancelled=bkd.asarray([1]),
            cost=Cost.zero(),
            statuses={0: JobStatus.FAILED, 1: JobStatus.CANCELLED},
        )
        assert result.statuses[0].is_retryable() is False
        assert result.statuses[1].is_retryable() is True
