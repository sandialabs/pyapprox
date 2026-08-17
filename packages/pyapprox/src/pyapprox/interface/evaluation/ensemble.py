"""Several models, one submission, one place to ask and to stop.

Multifidelity work evaluates a family of models whose allocation says
how many samples each one gets. Submitting those as N independent
batches loses three things, and this class exists to hold them:

- **the whole allocation is visible at once**, so a scheduler sees all
  of it rather than discovering the second model after the first has
  queued;
- **one cancel point**, rather than N to remember;
- **progress across models**, including which one is the long pole.

**What it does not own, and cannot.** The ensemble receives evaluators
already built, so it can neither supply nor verify that they share a
dispatcher or a ledger. Sharing a dispatcher is what makes the throttle
global, and sharing a ledger is what makes a budget span models -- but
both are established when the caller constructs the evaluators, and by
the time this class holds one its dispatcher is fixed. Those are
documented preconditions, not enforced invariants, and pretending
otherwise would mean a setter on an object the design treats as
configured at construction.

**Models are keyed by integer id.** Strings sort wrong: ``sorted`` puts
``"10"`` before ``"2"``, which at a ``List[Array]`` boundary is a
silent-wrong-answer path, and a ten-model ensemble is not exotic.
Integers also match the vocabulary the estimators already use, where a
model subset is a list of indices. Display names are separate and
carried only for messages, so a label is never load-bearing.

A ``Mapping`` rather than a ``Sequence``, because a submission may cover
a subset: an adaptive fitter asks for whichever configurations are
pending, and a sequence would need placeholders for the rest.
"""

import time
from typing import Dict, Generic, List, Mapping, Optional

from pyapprox.interface.evaluation.protocols import (
    BatchProtocol,
    EvaluatorProtocol,
)
from pyapprox.interface.evaluation.records import (
    ComputeProvenance,
    Cost,
    EvalProgress,
    EvalResult,
    Request,
)
from pyapprox.util.backends.protocols import Array

#: How a model is identified. An integer, for the reasons in the module
#: docstring; aliased so the intent reads at every use.
ModelId = int


class EnsembleProgress:
    """How a whole ensemble submission is getting on.

    Three things here are not derivable from the per-model records, and
    are the reason this is a record rather than a bare mapping.

    ``cost`` sums compute across models and takes the **maximum** of
    their wall-clocks. Compute is additive -- two models each burning
    ten core-seconds burnt twenty, whether or not they overlapped --
    while wall-clock is not: two models each running ten seconds, one
    starting four seconds after the other, occupied fourteen seconds of
    a user's life, not twenty. The correct figure is the measure of the
    union of their spans.

    That union cannot be computed here. Each model reports a *duration*
    and not an interval, so there is no way to tell whether two ten-
    second runs overlapped completely, partially, or were separated by
    an hour. Of the two available answers, the maximum under-reports and
    the sum over-reports, and they are not equally bad: a sum is wrong
    by a factor approaching the number of models and looks authoritative
    while doing it, which could halt a study with budget to spare. The
    maximum errs toward letting work proceed, and says so.

    **It is a loose bound when models do not overlap.** Two ten-second
    models running back to back have a true union of twenty seconds and
    are reported as ten. The bound is tight exactly when one model's
    span contains the others', which is the common case under a shared
    throttle and not a guarantee.

    A caller wanting the exact figure shares one ``CostLedger`` across
    its evaluators: that record keeps each job's ``(start, end)`` and
    merges overlapping intervals, so it answers fourteen seconds where
    this reports ten.

    ``elapsed_seconds`` measures from the ensemble's submission, which
    is not any single model's elapsed time.

    ``per_model`` is what identifies the long pole -- the model still
    outstanding when the others have finished, which is what a caller
    acts on.
    """

    def __init__(
        self,
        per_model: Mapping[ModelId, EvalProgress],
        elapsed_seconds: float,
    ) -> None:
        self._per_model = dict(per_model)
        self._elapsed_seconds = elapsed_seconds

    def per_model(self) -> Mapping[ModelId, EvalProgress]:
        """Each model's own progress, keyed by id."""
        return self._per_model

    def nsucceeded(self) -> int:
        """Samples returned across every model."""
        return sum(p.nsucceeded for p in self._per_model.values())

    def nfailed(self) -> int:
        """Samples that failed across every model."""
        return sum(p.nfailed for p in self._per_model.values())

    def noutstanding(self) -> int:
        """Samples still running across every model."""
        return sum(p.noutstanding for p in self._per_model.values())

    def nsubmitted(self) -> int:
        """Samples the whole submission started with."""
        return sum(p.nsubmitted() for p in self._per_model.values())

    def is_complete(self) -> bool:
        """Whether every model has finished."""
        return self.noutstanding() == 0

    def fraction_returned(self) -> float:
        """Share of the submission that has come back, one way or another.

        One for an empty submission: nothing is outstanding, so nothing
        is being waited on.
        """
        submitted = self.nsubmitted()
        if submitted == 0:
            return 1.0
        return (self.nsucceeded() + self.nfailed()) / submitted

    def elapsed(self) -> float:
        """Seconds since the ensemble was submitted."""
        return self._elapsed_seconds

    def cost(self) -> Cost:
        """What the submission has spent, across models.

        Compute sums; wall-clock takes the maximum, since models under
        one throttle overlap. Provenance degrades pessimistically, so a
        total containing an estimate is an estimate.
        """
        return _combine_costs(
            [p.cost for p in self._per_model.values()]
        )

    def outstanding_models(self) -> List[ModelId]:
        """Which models still have work, in id order.

        What a caller reads to find the long pole.
        """
        return sorted(
            model_id
            for model_id, progress in self._per_model.items()
            if not progress.is_complete()
        )


class EnsembleBatch(Generic[Array]):
    """Work in flight across several models.

    **Not thread-safe**, like the single-model batch it composes. A
    caller driving one from several threads serializes its own calls.

    The streaming contract is the per-model one, applied model by model:
    each of ``collect_ready``, ``collect`` and ``cancel`` returns what a
    previous call did not, and the union over all calls is the whole
    submission with each index returned exactly once.
    """

    def __init__(
        self,
        batches: Mapping[ModelId, BatchProtocol[Array]],
        started: float,
    ) -> None:
        self._batches = dict(batches)
        self._started = started

    def model_ids(self) -> List[ModelId]:
        """Which models this submission covers, in id order."""
        return sorted(self._batches)

    def nsubmitted(self) -> int:
        """Samples submitted across every model."""
        return sum(b.nsubmitted() for b in self._batches.values())

    def progress(self) -> EnsembleProgress:
        """Counts and cost so far, without consuming or waiting."""
        return EnsembleProgress(
            per_model={
                model_id: batch.progress()
                for model_id, batch in self._batches.items()
            },
            elapsed_seconds=time.perf_counter() - self._started,
        )

    def collect_ready(self) -> Mapping[ModelId, EvalResult[Array]]:
        """Return what has finished across every model, without waiting."""
        return {
            model_id: batch.collect_ready()
            for model_id, batch in self._batches.items()
        }

    def collect(
        self, timeout: Optional[float] = None
    ) -> Mapping[ModelId, EvalResult[Array]]:
        """Wait for the rest of the submission and return it.

        ``timeout`` applies to the whole submission rather than to each
        model, so a caller asking for five seconds waits five seconds in
        total. Models are polled in id order, and one that exhausts the
        budget leaves the rest to a later call.
        """
        deadline = (
            None if timeout is None else time.perf_counter() + timeout
        )
        results: Dict[ModelId, EvalResult[Array]] = {}
        for model_id in sorted(self._batches):
            remaining = (
                None
                if deadline is None
                else max(0.0, deadline - time.perf_counter())
            )
            results[model_id] = self._batches[model_id].collect(remaining)
        return results

    def cancel(self) -> Mapping[ModelId, EvalResult[Array]]:
        """Stop everything outstanding, across every model.

        One cancel point rather than N, which is a large part of why an
        ensemble is an object at all. Returns results rather than
        ``None`` so compute already spent reaches the ledger.
        """
        return {
            model_id: batch.cancel()
            for model_id, batch in self._batches.items()
        }


class Ensemble(Generic[Array]):
    """Submits to several models at once, and reports on all of them.

    Parameters
    ----------
    evaluators : Mapping[ModelId, EvaluatorProtocol[Array]]
        The models, keyed by integer id. Built by the caller, because
        pairing a marshaller with a dispatcher is checked at that call
        site and constructing them here would erase the check.

        **Sharing one dispatcher across these is what makes a throttle
        global**, and sharing one ``CostLedger`` is what makes a budget
        span models. Both are the caller's to arrange when constructing
        the evaluators; this class cannot verify either, and says so
        rather than implying a guarantee.
    names : Mapping[ModelId, str], optional
        Display names, used only in messages. Never looked up, never
        sorted, never required -- so "model 2 (coarse)" reads well
        without the label carrying meaning.
    """

    def __init__(
        self,
        evaluators: Mapping[ModelId, EvaluatorProtocol[Array]],
        names: Optional[Mapping[ModelId, str]] = None,
    ) -> None:
        for model_id, evaluator in evaluators.items():
            if not isinstance(model_id, int) or isinstance(model_id, bool):
                raise TypeError(
                    f"model ids must be int, got {model_id!r}"
                )
            if not isinstance(evaluator, EvaluatorProtocol):
                raise TypeError(
                    f"evaluator for model {model_id} must satisfy "
                    f"EvaluatorProtocol, got {type(evaluator).__name__}"
                )
        _reject_duplicate_evaluators(evaluators)
        self._evaluators = dict(evaluators)
        self._names = dict(names) if names is not None else {}
        unknown = set(self._names) - set(self._evaluators)
        if unknown:
            raise ValueError(
                f"names given for models that do not exist: "
                f"{sorted(unknown)}"
            )

    def nmodels(self) -> int:
        """How many models this ensemble holds."""
        return len(self._evaluators)

    def model_ids(self) -> List[ModelId]:
        """Every model id, in order."""
        return sorted(self._evaluators)

    def name(self, model_id: ModelId) -> str:
        """A readable label for one model, for messages only."""
        label = self._names.get(model_id)
        return (
            f"model {model_id}"
            if label is None
            else f"model {model_id} ({label})"
        )

    def evaluator(self, model_id: ModelId) -> EvaluatorProtocol[Array]:
        """One model's evaluator, for a caller that wants it directly."""
        if model_id not in self._evaluators:
            raise KeyError(f"no model {model_id} in this ensemble")
        return self._evaluators[model_id]

    def submit(
        self,
        work: Mapping[ModelId, Array],
        requests: Optional[Mapping[ModelId, Request[Array]]] = None,
    ) -> EnsembleBatch[Array]:
        """Submit samples to several models at once.

        ``work`` may cover a **subset** of the ensemble: an adaptive
        fitter asks only for the configurations currently pending, and a
        model absent from the mapping simply receives nothing.

        Returns without waiting, as each evaluator does.

        Samples are handed over **model by model**, in id order. That
        ordering is worth stating because a shared dispatcher queues
        FIFO, so a model submitting many cheap tasks ahead of one
        submitting a few expensive ones delays the latter for the whole
        run. Where that matters -- a numerous cheap model alongside a
        scarce expensive one -- submit the expensive model first, or use
        separate submissions.
        """
        unknown = set(work) - set(self._evaluators)
        if unknown:
            raise KeyError(
                f"work given for models that do not exist: "
                f"{sorted(unknown)}"
            )
        asks = dict(requests) if requests is not None else {}
        unknown_requests = set(asks) - set(work)
        if unknown_requests:
            raise KeyError(
                "requests given for models with no work: "
                f"{sorted(unknown_requests)}"
            )

        started = time.perf_counter()
        batches: Dict[ModelId, BatchProtocol[Array]] = {}
        for model_id in sorted(work):
            batches[model_id] = self._evaluators[model_id].submit(
                work[model_id], asks.get(model_id)
            )
        return EnsembleBatch(batches=batches, started=started)

    def __repr__(self) -> str:
        listed = ", ".join(self.name(i) for i in self.model_ids())
        return f"Ensemble({listed})"


def _reject_duplicate_evaluators(
    evaluators: Mapping[ModelId, EvaluatorProtocol[Array]],
) -> None:
    """Refuse the same evaluator under two ids.

    Legal Python, and almost certainly a copy-paste error: the work
    would run once but be reported twice, doubling its cost and making
    an intersection across "two" models meaningless. Identity rather
    than equality, since evaluators do not define equality and two
    genuinely distinct ones may be configured identically.
    """
    seen: Dict[int, ModelId] = {}
    for model_id in sorted(evaluators):
        key = id(evaluators[model_id])
        if key in seen:
            raise ValueError(
                f"models {seen[key]} and {model_id} are the same "
                "evaluator; work would run once and be counted twice"
            )
        seen[key] = model_id


def _combine_costs(costs: List[Cost]) -> Cost:
    """Sum compute, take the longest wall-clock, degrade provenance.

    Wall-clock is a maximum rather than a sum because models under one
    throttle overlap, and adding their elapsed times would count the
    overlap once per model. It is a lower bound on the true union, which
    cannot be computed from snapshots that do not carry their spans.
    """
    if not costs:
        return Cost.zero()
    provenance = costs[0].provenance
    for cost in costs[1:]:
        if cost.provenance is provenance:
            continue
        if ComputeProvenance.NOT_APPLICABLE in (
            provenance,
            cost.provenance,
        ):
            provenance = ComputeProvenance.NOT_APPLICABLE
        else:
            provenance = ComputeProvenance.ESTIMATED
    return Cost(
        wall_clock=max(cost.wall_clock for cost in costs),
        compute=sum(cost.compute for cost in costs),
        provenance=provenance,
    )
