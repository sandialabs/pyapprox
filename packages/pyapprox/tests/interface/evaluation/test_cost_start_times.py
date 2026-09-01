"""The start time a dispatcher records must reach the cost ledger.

CostLedger measures wall clock as the union of ``(start, start + wall)``
spans, which is the only way a batch that overlaps and a batch that does
not can report different totals. Its own unit tests pass ``start``
directly and prove the union arithmetic; what they cannot show is
whether a real dispatcher supplies one.

It did not. ``Outcome`` carried a duration and no origin, so every span
anchored at zero, every span overlapped, and the total collapsed to the
longest single job -- understating a concurrent batch, and equally
refusing to sum a serial one. These tests run real dispatchers over real
sleeps so the reported figure has to match the clock on the wall.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from pyapprox.interface.evaluation.callable_marshaller import (
    CallableMarshaller,
)
from pyapprox.interface.evaluation.evaluator import Evaluator
from pyapprox.interface.evaluation.executor_dispatcher import (
    ExecutorDispatcher,
)
from pyapprox.interface.evaluation.inline_dispatcher import InlineDispatcher
from pyapprox.interface.evaluation.records import CostLedger

# Long enough that scheduling jitter cannot reorder the result, short
# enough to keep the suite quick. The assertions compare against the
# measured elapsed time rather than against this constant, so a slow
# machine cannot fail them.
NAP = 0.05


def _sleeper(nap=NAP):
    def fn(sample):
        time.sleep(nap)
        return sample * 2.0

    return fn


class TestOutcomeCarriesStart:
    """The record itself must have somewhere to put the origin."""

    def test_inline_dispatcher_records_a_start(self, bkd) -> None:
        marshaller = CallableMarshaller(
            _sleeper(), bkd, nvars=1, nqoi=1, samples_per_task=1
        )
        dispatcher = InlineDispatcher(marshaller.run)
        before = time.perf_counter()
        handles = dispatcher.submit(
            list(marshaller.tasks(bkd.ones((1, 1)), [0], _values_request()))
        )
        outcome = handles[0].outcome()
        after = time.perf_counter()
        assert outcome.started is not None
        # On the driver's clock, so it must lie inside the window the
        # driver itself observed.
        assert before <= outcome.started <= after

    def test_executor_start_is_on_the_drivers_clock(self, bkd) -> None:
        """The subtlety that decides whether the union means anything.

        perf_counter is comparable only within a process, so a start
        stamped inside a worker cannot be ordered against one from
        another worker. The dispatcher recovers a driver-side origin by
        subtracting the reported duration from the moment it observed
        completion -- and this pins that the result lands on the
        driver's clock rather than a worker's.
        """
        marshaller = CallableMarshaller(
            _sleeper(), bkd, nvars=1, nqoi=1, samples_per_task=1
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            dispatcher = ExecutorDispatcher(
                run=marshaller.run, executor=pool, concurrency=2
            )
            before = time.perf_counter()
            handles = dispatcher.submit(
                list(
                    marshaller.tasks(
                        bkd.ones((1, 1)), [0], _values_request()
                    )
                )
            )
            outcome = handles[0].outcome()
            after = time.perf_counter()
        assert outcome.started is not None
        assert before <= outcome.started <= after


class TestLedgerSeesConcurrency:
    """The end-to-end property, measured against the wall clock."""

    def test_serial_batch_sums(self, bkd) -> None:
        """Consecutive jobs occupy the sum of their durations.

        The bug hid here too, not only in the concurrent case: with
        every span anchored at zero, three sequential naps reported one
        nap. Nothing about that is a conservative approximation.
        """
        ledger = CostLedger()
        marshaller = CallableMarshaller(
            _sleeper(), bkd, nvars=1, nqoi=1, samples_per_task=1
        )
        evaluator = Evaluator(
            marshaller, InlineDispatcher(marshaller.run), ledger=ledger
        )
        started = time.perf_counter()
        evaluator.submit(bkd.ones((1, 3))).collect()
        elapsed = time.perf_counter() - started
        wall = ledger.total().wall_clock
        # Three naps ran back to back, so the union spans all three.
        assert wall == pytest.approx(elapsed, rel=0.5)
        assert wall > 2 * NAP

    def test_concurrent_batch_reports_union_not_sum(self, bkd) -> None:
        """Four jobs, two at a time: two naps of wall clock, not four.

        Compute still sums -- core-hours are additive -- so this pins
        the two measures diverging, which is the whole reason the ledger
        keeps spans rather than a running total.
        """
        ledger = CostLedger()
        marshaller = CallableMarshaller(
            _sleeper(), bkd, nvars=1, nqoi=1, samples_per_task=1
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            evaluator = Evaluator(
                marshaller,
                ExecutorDispatcher(
                    run=marshaller.run, executor=pool, concurrency=2
                ),
                ledger=ledger,
            )
            started = time.perf_counter()
            evaluator.submit(bkd.ones((1, 4))).collect()
            elapsed = time.perf_counter() - started
        total = ledger.total()
        # The union tracks the elapsed time however much the pool
        # actually overlapped, so this holds whether the runner granted
        # two workers or serialized them.
        assert total.wall_clock == pytest.approx(elapsed, rel=0.5)
        # Four naps of compute happened regardless of the overlap:
        # compute sums per-job durations and never consults the
        # driver's clock. Bounding it against the jobs rather than
        # against ``elapsed`` is what keeps this honest. Whether the
        # ratio of compute to elapsed reaches two is a fact about the
        # runner, not about the ledger, so the earlier
        # ``compute > 1.5 * elapsed`` form failed on contended macOS
        # runners while the ledger under test was correct.
        #
        # The bounds stay lopsided. Billing too little is the defect
        # this pins and contention cannot cause it, so the lower bound
        # is tight enough to catch a batch that dropped a job.
        # Oversleeping is what a busy runner produces, so the upper
        # bound is loose.
        assert total.compute > 3.5 * NAP
        # The upper bound counts jobs rather than seconds. A contended
        # runner oversleeps -- one CI failure recorded four 0.05s naps
        # taking 0.72s between them -- so any ceiling expressed in
        # multiples of NAP eventually trips on a slow machine while the
        # ledger is correct. What cannot exceed four jobs' worth of
        # time, however slow each job is, is the sum of four spans that
        # each fit inside the batch: double counting is the defect this
        # guards, and it would report more than the whole batch took.
        assert total.compute <= 4.0 * elapsed
        # The two measures diverge whenever the jobs really did overlap,
        # which is what the defect destroyed by collapsing wall clock
        # onto the longest job. Overlap is the runner's to grant, so
        # this is asserted where it occurred rather than demanded: a
        # serialized batch legitimately reports the two as equal, and
        # failing then would test the machine instead of the ledger.
        if total.wall_clock < 3.5 * NAP:
            assert total.compute > total.wall_clock

    def test_concurrent_wall_clock_exceeds_one_job(self, bkd) -> None:
        """The specific number the defect produced.

        With spans anchored at zero the union was max(wall_clock) -- one
        nap, however many jobs ran. Asserting strictly more than that
        fails on the old behaviour and passes on the new, whatever the
        machine's speed.
        """
        ledger = CostLedger()
        marshaller = CallableMarshaller(
            _sleeper(), bkd, nvars=1, nqoi=1, samples_per_task=1
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            evaluator = Evaluator(
                marshaller,
                ExecutorDispatcher(
                    run=marshaller.run, executor=pool, concurrency=2
                ),
                ledger=ledger,
            )
            evaluator.submit(bkd.ones((1, 4))).collect()
        longest_single_job = 1.5 * NAP
        assert ledger.total().wall_clock > longest_single_job


def _values_request():
    from pyapprox.interface.evaluation.records import Request

    return Request(values=True)
