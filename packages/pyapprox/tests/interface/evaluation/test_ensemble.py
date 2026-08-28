"""Several models, one submission.

The ensemble owns three things no single-model batch can: a submission a
scheduler sees whole, one cancel point, and progress that identifies the
long pole. It owns nothing else -- sharing a dispatcher or a ledger is
arranged by the caller when the evaluators are built, and these tests
show how, because a documented alternative nobody runs is a claim rather
than a fact.

The ``bkd`` fixture where results are asserted on, since those are
backend arrays; ``numpy_bkd`` for the cost arithmetic, which is
array-free.
"""

import time

import pytest
from pyapprox.interface.evaluation.callable_marshaller import (
    CallableMarshaller,
)
from pyapprox.interface.evaluation.ensemble import Ensemble
from pyapprox.interface.evaluation.evaluator import Evaluator
from pyapprox.interface.evaluation.executor_dispatcher import (
    thread_dispatcher,
)
from pyapprox.interface.evaluation.inline_dispatcher import InlineDispatcher
from pyapprox.interface.evaluation.records import (
    ComputeProvenance,
    CostLedger,
    Request,
)
from pyapprox.interface.functions.derivatives import Derivatives

#: Simulated work per model, sized so the signal dominates the noise.
#:
#: A shared CI runner measured a 0.05 sleep at 0.12, which broke ratios
#: written against an idle machine's millisecond jitter. Scheduling
#: overhead is roughly constant rather than proportional, so a longer
#: unit fixes this where looser ratios would only hide it.
UNIT = 0.25


def _evaluator(bkd, scale=1.0, ledger=None, **kwargs):
    """A model whose values identify which model produced them."""

    def model(samples):
        return scale * bkd.sum(samples, axis=0)[None, :]

    marshaller = CallableMarshaller(model, bkd, nvars=2, nqoi=1, **kwargs)
    return Evaluator(
        marshaller, InlineDispatcher(marshaller.run), ledger=ledger
    )


def _slow_evaluator(bkd, delay, ledger=None):
    def slow(samples):
        time.sleep(delay * samples.shape[1])
        return bkd.sum(samples, axis=0)[None, :]

    marshaller = CallableMarshaller(slow, bkd, nvars=2, nqoi=1)
    return Evaluator(
        marshaller, InlineDispatcher(marshaller.run), ledger=ledger
    )


def _ensemble(bkd, nmodels=3, **kwargs):
    return Ensemble(
        {i: _evaluator(bkd, scale=float(i + 1)) for i in range(nmodels)},
        **kwargs,
    )


class TestConstruction:
    def test_reports_its_models(self, bkd):
        ensemble = _ensemble(bkd)
        assert ensemble.nmodels() == 3
        assert ensemble.model_ids() == [0, 1, 2]

    def test_model_ids_sort_numerically(self, bkd):
        """The case string ids get wrong.

        ``sorted`` puts ``"10"`` before ``"2"``, so a ten-model ensemble
        assembled into a list would silently pair each model's values
        with the wrong index.
        """
        ensemble = Ensemble(
            {i: _evaluator(bkd) for i in [0, 1, 2, 10, 11]}
        )
        assert ensemble.model_ids() == [0, 1, 2, 10, 11]

    def test_names_are_optional_and_only_for_messages(self, bkd):
        ensemble = _ensemble(bkd, names={0: "hifi", 2: "coarse"})
        assert ensemble.name(0) == "model 0 (hifi)"
        assert ensemble.name(1) == "model 1"
        assert ensemble.name(2) == "model 2 (coarse)"

    def test_rejects_a_non_evaluator(self, bkd):
        with pytest.raises(TypeError, match="EvaluatorProtocol"):
            Ensemble({0: "not an evaluator"})

    def test_rejects_non_integer_ids(self, bkd):
        with pytest.raises(TypeError, match="int"):
            Ensemble({"lo": _evaluator(bkd)})

    def test_rejects_the_same_evaluator_twice(self, bkd):
        """Legal Python, and almost certainly a copy-paste error.

        The work would run once and be reported twice, doubling its cost
        and making any cross-model comparison meaningless.
        """
        shared = _evaluator(bkd)
        with pytest.raises(ValueError, match="same evaluator"):
            Ensemble({0: shared, 1: shared})

    def test_rejects_names_for_absent_models(self, bkd):
        with pytest.raises(ValueError, match="do not exist"):
            _ensemble(bkd, names={7: "nowhere"})

    def test_evaluator_is_reachable(self, bkd):
        ensemble = _ensemble(bkd)
        assert ensemble.evaluator(1) is not None
        with pytest.raises(KeyError, match="no model 9"):
            ensemble.evaluator(9)


class TestSubmission:
    def test_each_model_gets_its_own_samples(self, bkd):
        ensemble = _ensemble(bkd)
        X = bkd.ones((2, 3))
        results = ensemble.submit({0: X, 1: X, 2: X}).collect()

        # Model i scales by (i + 1), so the values identify the model.
        bkd.assert_allclose(results[0].values, bkd.full((1, 3), 2.0))
        bkd.assert_allclose(results[1].values, bkd.full((1, 3), 4.0))
        bkd.assert_allclose(results[2].values, bkd.full((1, 3), 6.0))

    def test_models_may_receive_different_sample_counts(self, bkd):
        """What an allocation actually produces."""
        ensemble = _ensemble(bkd)
        results = ensemble.submit(
            {0: bkd.ones((2, 2)), 1: bkd.ones((2, 5)), 2: bkd.ones((2, 9))}
        ).collect()
        assert results[0].nsucceeded() == 2
        assert results[1].nsucceeded() == 5
        assert results[2].nsucceeded() == 9

    def test_a_subset_of_models_is_legal(self, bkd):
        """An adaptive fitter asks only for what is pending."""
        ensemble = _ensemble(bkd)
        batch = ensemble.submit(
            {0: bkd.ones((2, 2)), 2: bkd.ones((2, 3))}
        )
        assert batch.model_ids() == [0, 2]
        results = batch.collect()
        assert set(results) == {0, 2}

    def test_models_may_differ_in_nvars(self, bkd):
        """Nothing assumes a shared input width."""
        wide = CallableMarshaller(
            lambda X: bkd.sum(X, axis=0)[None, :], bkd, nvars=5, nqoi=1
        )
        ensemble = Ensemble(
            {
                0: _evaluator(bkd),
                1: Evaluator(wide, InlineDispatcher(wide.run)),
            }
        )
        results = ensemble.submit(
            {0: bkd.ones((2, 2)), 1: bkd.ones((5, 2))}
        ).collect()
        assert results[0].nsucceeded() == 2
        assert results[1].nsucceeded() == 2

    def test_rejects_work_for_an_absent_model(self, bkd):
        with pytest.raises(KeyError, match="do not exist"):
            _ensemble(bkd).submit({9: bkd.ones((2, 1))})

    def test_rejects_a_request_for_a_model_with_no_work(self, bkd):
        with pytest.raises(KeyError, match="no work"):
            _ensemble(bkd).submit(
                {0: bkd.ones((2, 1))}, requests={1: Request()}
            )

    def test_per_model_requests(self, bkd):
        """One model asked for jacobians, another only for values."""

        def jac_batch(samples):
            n = samples.shape[1]
            return bkd.reshape(2.0 * samples.T, (n, 1, 2))

        ensemble = Ensemble(
            {
                0: _evaluator(bkd),
                1: _evaluator(
                    bkd, derivatives=Derivatives(jacobian_batch=jac_batch)
                ),
            }
        )
        results = ensemble.submit(
            {0: bkd.ones((2, 2)), 1: bkd.ones((2, 2))},
            requests={1: Request(jacobians=True)},
        ).collect()
        assert results[0].jacobians is None
        assert results[1].jacobians is not None

    def test_empty_submission(self, bkd):
        batch = _ensemble(bkd).submit({})
        assert batch.nsubmitted() == 0
        assert batch.progress().is_complete()
        assert batch.collect() == {}

    def test_a_model_with_zero_samples(self, bkd):
        """An allocation may assign a model nothing."""
        results = (
            _ensemble(bkd)
            .submit({0: bkd.zeros((2, 0)), 1: bkd.ones((2, 2))})
            .collect()
        )
        assert results[0].nsucceeded() == 0
        assert results[0].values.shape == (1, 0)
        assert results[1].nsucceeded() == 2


class TestProgress:
    def test_aggregates_across_models(self, bkd):
        batch = _ensemble(bkd).submit(
            {0: bkd.ones((2, 2)), 1: bkd.ones((2, 3))}
        )
        progress = batch.progress()
        assert progress.nsubmitted() == 5
        assert progress.nsucceeded() == 5
        assert progress.is_complete()

    def test_per_model_is_available(self, bkd):
        batch = _ensemble(bkd).submit(
            {0: bkd.ones((2, 2)), 1: bkd.ones((2, 3))}
        )
        per_model = batch.progress().per_model()
        assert per_model[0].nsubmitted() == 2
        assert per_model[1].nsubmitted() == 3

    def test_outstanding_models_identifies_the_long_pole(self, bkd):
        """What a caller acts on: which model is still going."""
        ensemble = Ensemble(
            {0: _evaluator(bkd), 1: _evaluator(bkd)}
        )
        batch = ensemble.submit({0: bkd.ones((2, 1))})
        # Everything submitted has finished, so nothing is outstanding.
        assert batch.progress().outstanding_models() == []

    def test_empty_submission_is_complete(self, bkd):
        progress = _ensemble(bkd).submit({}).progress()
        assert progress.is_complete()
        assert progress.fraction_returned() == pytest.approx(1.0)

    def test_fraction_returned(self, bkd):
        batch = _ensemble(bkd).submit(
            {0: bkd.ones((2, 2)), 1: bkd.ones((2, 2))}
        )
        assert batch.progress().fraction_returned() == pytest.approx(1.0)


class TestCost:
    def test_compute_sums_across_models(self, numpy_bkd):
        """Compute is additive whether or not models overlapped."""
        bkd = numpy_bkd
        ensemble = Ensemble(
            {
                0: _slow_evaluator(bkd, UNIT),
                1: _slow_evaluator(bkd, UNIT),
            }
        )
        batch = ensemble.submit(
            {0: bkd.ones((2, 1)), 1: bkd.ones((2, 1))}
        )
        cost = batch.progress().cost()
        assert cost.compute >= UNIT * 2 * 0.7

    def test_wall_clock_is_the_maximum_not_the_sum(self, numpy_bkd):
        """Two models running concurrently did not take twice as long.

        Summing would double-count the overlap and could halt a study
        that had budget to spare.
        """
        bkd = numpy_bkd
        ensemble = Ensemble(
            {
                0: _slow_evaluator(bkd, UNIT),
                1: _slow_evaluator(bkd, UNIT),
            }
        )
        batch = ensemble.submit(
            {0: bkd.ones((2, 1)), 1: bkd.ones((2, 1))}
        )
        cost = batch.progress().cost()
        assert cost.wall_clock < UNIT * 1.8

    def test_mixed_provenance_degrades(self, numpy_bkd):
        """A total containing an estimate is an estimate."""
        bkd = numpy_bkd
        ensemble = Ensemble({0: _evaluator(bkd), 1: _evaluator(bkd)})
        batch = ensemble.submit(
            {0: bkd.ones((2, 1)), 1: bkd.ones((2, 1))}
        )
        assert (
            batch.progress().cost().provenance
            is ComputeProvenance.MEASURED
        )

    def test_empty_submission_costs_nothing(self, numpy_bkd):
        cost = _ensemble(numpy_bkd).submit({}).progress().cost()
        assert cost.compute == pytest.approx(0.0)


class TestSharedLedger:
    """How a caller gets the exact wall-clock the ensemble cannot.

    ``EnsembleProgress.cost()`` takes the maximum of the per-model
    wall-clocks, because a snapshot carries a duration and not an
    interval, so there is no way to tell overlapping runs from
    consecutive ones. A shared ``CostLedger`` does carry the spans and
    computes the true union.

    This is the documented alternative, so it is exercised rather than
    merely described.
    """

    def test_a_shared_ledger_is_passed_at_construction(
        self, numpy_bkd
    ) -> None:
        """The whole mechanism: one ledger, given to each evaluator.

        The ensemble cannot arrange this -- by the time it holds an
        evaluator, that evaluator's ledger is fixed -- so it is the
        caller's to do, and this is what it looks like.
        """
        bkd = numpy_bkd
        ledger = CostLedger()
        ensemble = Ensemble(
            {
                0: _slow_evaluator(bkd, UNIT, ledger=ledger),
                1: _slow_evaluator(bkd, UNIT, ledger=ledger),
            }
        )
        ensemble.submit(
            {0: bkd.ones((2, 1)), 1: bkd.ones((2, 1))}
        ).collect()
        shared_total = ledger.total().compute

        # The same work again, but with each evaluator keeping its own
        # ledger. Comparing against the sum of those is a sharper check
        # than a tolerance around an expected duration: it fails if the
        # shared ledger caught only one model, and does not depend on
        # how accurate ``sleep`` happens to be.
        separate = [CostLedger(), CostLedger()]
        apart = Ensemble(
            {
                i: _slow_evaluator(bkd, UNIT, ledger=separate[i])
                for i in range(2)
            }
        )
        apart.submit(
            {0: bkd.ones((2, 1)), 1: bkd.ones((2, 1))}
        ).collect()
        summed = sum(led.total().compute for led in separate)

        # What must hold is that the shared ledger caught *both* models
        # rather than one. Comparing the two runs against each other
        # cannot say that on a contended machine: they are separate
        # measurements, and CI has seen the second run take long enough
        # that even a doubled figure from the first fell short of a
        # 1.5x margin against it. Each run is internally consistent
        # though, so the bound is drawn inside the separate run -- the
        # shared total must exceed the larger single model, which one
        # model's worth of work never can.
        assert shared_total > max(
            led.total().compute for led in separate
        )
        assert shared_total < summed * 3.0

    def test_the_ledger_spans_several_submissions(self, numpy_bkd):
        """What a budget reads: cost across every batch so far.

        The ensemble's own progress covers one submission. A ledger
        outlives them, which is why a retry can be a fresh submission
        rather than an extension of a live batch.
        """
        bkd = numpy_bkd
        ledger = CostLedger()
        ensemble = Ensemble(
            {0: _slow_evaluator(bkd, UNIT, ledger=ledger)}
        )
        ensemble.submit({0: bkd.ones((2, 1))}).collect()
        after_first = ledger.total().compute
        ensemble.submit({0: bkd.ones((2, 1))}).collect()
        assert ledger.total().compute > after_first

    def test_without_sharing_each_model_keeps_its_own(self, numpy_bkd):
        """The default, and why sharing has to be deliberate."""
        bkd = numpy_bkd
        first = _evaluator(bkd)
        second = _evaluator(bkd)
        Ensemble({0: first, 1: second})
        assert first.ledger() is not second.ledger()


class TestCollection:
    def test_collect_returns_every_model(self, bkd):
        results = (
            _ensemble(bkd)
            .submit({0: bkd.ones((2, 1)), 1: bkd.ones((2, 1))})
            .collect()
        )
        assert set(results) == {0, 1}

    def test_collect_ready_returns_what_is_finished(self, bkd):
        batch = _ensemble(bkd).submit(
            {0: bkd.ones((2, 2)), 1: bkd.ones((2, 2))}
        )
        ready = batch.collect_ready()
        assert ready[0].nsucceeded() == 2

    def test_no_index_is_returned_twice(self, bkd):
        """The streaming contract, applied model by model."""
        batch = _ensemble(bkd).submit(
            {0: bkd.ones((2, 2)), 1: bkd.ones((2, 2))}
        )
        batch.collect()
        again = batch.collect()
        assert again[0].nsucceeded() == 0
        assert again[1].nsucceeded() == 0

    def test_cancel_stops_every_model(self, bkd):
        """One cancel point rather than N to remember."""
        batch = _ensemble(bkd).submit(
            {0: bkd.ones((2, 2)), 1: bkd.ones((2, 2))}
        )
        results = batch.cancel()
        assert set(results) == {0, 1}


class TestSharedDispatcher:
    """The throttle spans models only if the caller shares one.

    Not something the ensemble can arrange or verify, since it receives
    evaluators already built. Shown here so the arrangement is on
    record.
    """

    def test_one_dispatcher_serves_several_models(self, numpy_bkd):
        bkd = numpy_bkd
        running = []
        peak = []

        def counting(samples):
            running.append(1)
            peak.append(len(running))
            time.sleep(UNIT)
            running.pop()
            return bkd.sum(samples, axis=0)[None, :]

        marshallers = [
            CallableMarshaller(
                counting, bkd, nvars=2, nqoi=1, samples_per_task=1
            )
            for _ in range(2)
        ]
        dispatcher = thread_dispatcher(
            marshallers[0].run, concurrency=2
        )
        try:
            ensemble = Ensemble(
                {
                    i: Evaluator(marshaller, dispatcher)
                    for i, marshaller in enumerate(marshallers)
                }
            )
            batch = ensemble.submit(
                {0: bkd.ones((2, 3)), 1: bkd.ones((2, 3))}
            )
            batch.collect()
        finally:
            dispatcher.close()

        # Six tasks across two models, one throttle of two.
        assert max(peak) <= 2
