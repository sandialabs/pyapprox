"""The composition: samples in, work out, results back.

An evaluator pairs one marshaller with one dispatcher and owns the four
things neither of them can:

- **grouping** samples into tasks, by the rule below;
- **catching** :class:`MarshalError` and recording the affected samples
  as failed rather than losing the batch;
- **buffering** which indices have already been returned, so the
  streaming contract holds;
- **accumulating** cost, including the cost of failures.

**An evaluator is a model without caching.** Samples in, values out. It
has no memory between submissions, no notion of what a sample *is*
beyond a column, and no policy about what to do when one fails. Indices
are local to the submitted batch, and mapping them onto anything else --
a latent draw, an allocation partition, a design row -- is the caller's,
because the caller is the only party that knows the mapping. Caching,
budget and retry are wrappers around this, not features of it.

The completion hook is the one seam for durable writing: crash-safe
recording needs task-completion timing, and only this class sees it.
"""

import logging
import time
from math import ceil
from typing import (
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

from pyapprox.interface.evaluation.protocols import (
    DispatcherProtocol,
    JobHandle,
    MarshalError,
    MarshallerProtocol,
    SubmissionAware,
    TaskProtocol,
)
from pyapprox.interface.evaluation.records import (
    ComputeProvenance,
    Cost,
    CostLedger,
    Decoded,
    EvalProgress,
    EvalResult,
    JobStatus,
    Outcome,
    Request,
)
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.util.backends.protocols import Array, Backend

Task = TypeVar("Task", bound=TaskProtocol)
Payload = TypeVar("Payload")

#: Where cleanup failures go. A library announcing its own progress is
#: logging configuration's business rather than an argument, and a
#: failure that must not end the batch still has to be visible somewhere.
_LOGGER = logging.getLogger(__name__)

#: Called once per finished task, before ``release``. ``Decoded`` is
#: ``None`` when the task did not produce usable output.
CompletionHook = Callable[
    [Outcome[Task, Payload], Optional[Decoded[Array]], Cost], None
]


class Batch(Generic[Array, Task, Payload]):
    """Work in flight for one submission.

    **The streaming contract.** Each of :meth:`collect_ready`,
    :meth:`collect` and :meth:`cancel` returns the outcomes not returned
    by a previous call. The union over all calls is the submitted batch,
    each index exactly once.

    **Not thread-safe.** A caller driving one batch from several threads
    serializes its own calls; the returned-index set is ordinary mutable
    state. What is guaranteed regardless of call order is that no index
    is returned twice, because that set is the arbiter.

    **Collection outlives its collaborators' failures.** A raising
    completion hook or ``release`` is logged and the batch carries on.
    Both run per task inside the harvest loop, so propagating would
    strand every handle behind the failure -- unreleased, uncharged and
    missing from the result -- which loses far more than the one task
    that actually broke.
    """

    def __init__(
        self,
        handles: Sequence[JobHandle[Task, Payload]],
        marshaller: MarshallerProtocol[Array, Task, Payload],
        nsubmitted: int,
        ledger: CostLedger,
        started: float,
        hvp_weights: Optional[Array] = None,
        on_complete: Optional[CompletionHook[Task, Payload, Array]] = None,
    ) -> None:
        self._handles = list(handles)
        self._marshaller = marshaller
        self._bkd = marshaller.bkd()
        self._nsubmitted = nsubmitted
        # A property of the submission, not of any sample: one weight
        # vector covers the whole request. Held here rather than read
        # back from decoded pieces, so a result that collected only
        # failures still reports what its hvps would have been weighted
        # by, and so no single task's view can become the batch's.
        self._hvp_weights = hvp_weights
        self._ledger = ledger
        self._started = started
        self._on_complete = on_complete
        self._pending: List[JobHandle[Task, Payload]] = list(handles)
        self._returned: set[int] = set()
        # Every index this batch has returned, successes included, and
        # cumulative across calls. Each ``EvalResult`` covers one harvest
        # of a stream, so a caller looping on ``collect_ready`` ends up
        # holding several disjoint partial maps and no whole one. This is
        # the whole one.
        self._statuses: Dict[int, JobStatus] = {}
        self._nsucceeded = 0
        self._nfailed = 0

    def nsubmitted(self) -> int:
        """How many samples this batch was submitted with."""
        return self._nsubmitted

    def statuses(self) -> Dict[int, JobStatus]:
        """How every returned index ended, cumulative across calls.

        A copy, so a caller cannot rewrite the batch's own record of
        what happened. Indices still outstanding are simply absent:
        their absence is the question ``progress`` answers, and
        inventing an ``OUTSTANDING`` entry for them would make "was
        this index returned" a comparison against a sentinel rather
        than a membership test.
        """
        return dict(self._statuses)

    def progress(self) -> EvalProgress:
        """Counts and cost so far, without consuming or waiting.

        Counts what the **handles** report, not what has been
        collected. A finished job is finished whether or not anyone has
        asked for its result yet, and reporting it as outstanding until
        collection would make ``progress`` useless for the case it
        exists for -- deciding whether to wait.

        ``done()`` is specified as free and repeatable, so this costs a
        poll per outstanding handle and consumes nothing.
        """
        nsucceeded = self._nsucceeded
        nfailed = self._nfailed
        # Cost, like the counts, must include work that has finished but
        # not been collected. The ledger holds only what harvesting has
        # already charged, so reading it alone would report zero for a
        # batch whose jobs are all done and uncollected -- accurate
        # counts beside a cost of nothing, which is worse than either
        # being wrong on its own. Cost-so-far exists to be read *while*
        # work runs, which is exactly when nothing has been collected.
        pending_costs: List[Cost] = []
        for handle in self._pending:
            if not handle.done():
                continue
            outcome = handle.outcome()
            covered = len(outcome.indices)
            if outcome.status is JobStatus.SUCCEEDED:
                nsucceeded += covered
            else:
                nfailed += covered
            pending_costs.append(outcome.cost())
        return EvalProgress(
            nsucceeded=nsucceeded,
            nfailed=nfailed,
            noutstanding=self._nsubmitted - nsucceeded - nfailed,
            cost=_with_uncollected(self._ledger.total(), pending_costs),
            elapsed_seconds=time.perf_counter() - self._started,
        )

    def collect_ready(self) -> EvalResult[Array]:
        """Return outcomes finished now, without waiting."""
        ready = [h for h in self._pending if h.done()]
        return self._harvest(ready)

    def collect(self, timeout: Optional[float] = None) -> EvalResult[Array]:
        """Wait for the rest of the batch and return it.

        On timeout, returns what finished; the remainder stays
        outstanding and collectable.
        """
        deadline = None if timeout is None else time.perf_counter() + timeout
        finished = []
        # Two-pass: gather first, mutate after, so the pending list is
        # never modified while being iterated.
        for handle in list(self._pending):
            remaining = (
                None if deadline is None
                else max(0.0, deadline - time.perf_counter())
            )
            outcome = handle.outcome(remaining)
            if outcome.status is JobStatus.OUTSTANDING:
                continue
            finished.append(handle)
        return self._harvest(finished)

    def cancel(self) -> EvalResult[Array]:
        """Stop what has not finished and return everything outstanding.

        Returns a result rather than ``None`` so compute already burnt
        reaches the ledger and ``release`` runs for every task.
        """
        for handle in self._pending:
            handle.cancel()
        return self._harvest(list(self._pending))

    def _harvest(
        self, handles: Sequence[JobHandle[Task, Payload]]
    ) -> EvalResult[Array]:
        """Decode finished handles into one result, once each."""
        succeeded: List[int] = []
        failed: List[int] = []
        cancelled: List[int] = []
        statuses: Dict[int, JobStatus] = {}
        columns: List[Tuple[int, Array]] = []
        # Derivative pieces, gathered per quantity so a batch split
        # across tasks reassembles each one in index order.
        derivative_pieces: Dict[str, List[Tuple[int, Array]]] = {
            name: [] for name in _DERIVATIVE_FIELDS
        }

        for handle in handles:
            if handle not in self._pending:
                continue
            self._pending.remove(handle)
            outcome = handle.outcome()

            decoded: Optional[Decoded[Array]] = None
            if outcome.status is JobStatus.SUCCEEDED:
                try:
                    decoded = self._marshaller.values(outcome)
                except MarshalError:
                    # A statement about one task: record its samples as
                    # failed and keep the rest of the batch running.
                    decoded = None

            # Decoding first, because it may carry a better duration than
            # the dispatcher measured. A wrapper times whatever it waited
            # through -- process startup, input staging, and on a
            # scheduler the whole queue wait -- while a solver reporting
            # its own runtime is timing the work itself.
            cost = _outcome_cost(outcome, decoded)
            # The start is what lets the ledger tell overlapping jobs
            # from consecutive ones. Without it every span anchors at
            # zero, they all overlap, and the wall-clock total collapses
            # to the longest single job -- understating a concurrent
            # batch and, equally, refusing to sum a serial one.
            self._ledger.add(cost, start=outcome.started)

            if decoded is None:
                target = (
                    cancelled
                    if outcome.status.is_retryable()
                    else failed
                )
                for idx in outcome.indices:
                    if idx in self._returned:
                        continue
                    self._returned.add(idx)
                    target.append(idx)
                    statuses[idx] = (
                        outcome.status
                        if outcome.status is not JobStatus.SUCCEEDED
                        else JobStatus.FAILED
                    )
                    self._nfailed += 1
            else:
                fresh = [
                    (pos, idx)
                    for pos, idx in enumerate(decoded.indices)
                    if idx not in self._returned
                ]
                for _, idx in fresh:
                    self._returned.add(idx)
                    succeeded.append(idx)
                    statuses[idx] = JobStatus.SUCCEEDED
                    self._nsucceeded += 1
                if fresh:
                    # Tag each piece with the batch index of its first
                    # column, so assembly can restore index order.
                    #
                    # Handles arrive in whatever order the dispatcher
                    # finished them, which for anything concurrent is
                    # unrelated to submission order. Sorting the indices
                    # while concatenating columns as they arrive pairs
                    # every value with the wrong sample -- silently, and
                    # only when completion is reordered.
                    #
                    # One integer per task, not per sample: a task's own
                    # columns are already contiguous and ordered, so
                    # only the tasks need reordering. The tag comes from
                    # the *fresh* columns rather than the task, since an
                    # earlier partial collection may already have taken
                    # some of them.
                    take = [pos for pos, _ in fresh]
                    first = fresh[0][1]
                    if decoded.values.shape[1] > 0:
                        columns.append((first, decoded.values[:, take]))
                # Derivatives are gathered whether or not these indices
                # are newly succeeded. Where one request became several
                # tasks -- a solver computing values and a jacobian by
                # separate invocations -- the second task covers indices
                # the first already reported, and skipping it would
                # discard exactly the quantity it was run to produce.
                # Success is per quantity, and this is where that holds.
                if decoded.indices:
                    at = list(range(len(decoded.indices)))
                    origin = decoded.indices[0]
                    for name in _DERIVATIVE_FIELDS:
                        piece = getattr(decoded, name)
                        if piece is not None:
                            derivative_pieces[name].append(
                                (origin, _select(piece, name, at))
                            )

                # A task that decoded fewer samples than it covered
                # fails the remainder -- per quantity, not per sample.
                undecoded = set(outcome.indices) - set(decoded.indices)
                for idx in sorted(undecoded):
                    if idx in self._returned:
                        continue
                    self._returned.add(idx)
                    failed.append(idx)
                    statuses[idx] = JobStatus.FAILED
                    self._nfailed += 1

            # Neither of these may take the batch down with it. Both run
            # once per finished task, inside the loop that also removes
            # handles from ``_pending`` -- so an exception escaping here
            # abandons every handle after this one: never released,
            # never charged, and absent from the result the caller gets
            # back. A hook writing to a full disk, or a marshaller whose
            # cleanup hits a read-only scratch, would turn one failed
            # sample into a lost batch. Logged rather than swallowed
            # silently, because a store that never wrote is a fact its
            # owner needs.
            if self._on_complete is not None:
                try:
                    self._on_complete(outcome, decoded, cost)
                except Exception:
                    _LOGGER.exception(
                        "completion hook failed for indices %s; "
                        "continuing with the rest of the batch",
                        list(outcome.indices),
                    )
            try:
                self._marshaller.release(outcome)
            except Exception:
                _LOGGER.exception(
                    "release failed for indices %s; its working "
                    "directory may survive",
                    list(outcome.indices),
                )

        # Accumulated once here rather than at each of the three sites
        # that write ``statuses``, so a branch added later cannot forget
        # to. The streaming contract makes these disjoint: an index is
        # returned once, so no key is ever overwritten.
        self._statuses.update(statuses)

        return self._assemble(
            succeeded,
            failed,
            cancelled,
            statuses,
            columns,
            derivative_pieces,
        )

    def _assemble(
        self,
        succeeded: Sequence[int],
        failed: Sequence[int],
        cancelled: Sequence[int],
        statuses: Dict[int, JobStatus],
        columns: Sequence[Tuple[int, Array]],
        derivative_pieces: Dict[str, List[Tuple[int, Array]]],
    ) -> EvalResult[Array]:
        """Build the result record, empty-safe on both backends."""
        bkd = self._bkd
        nqoi = self._marshaller.nqoi()
        # Sort by batch index so columns line up with ``succeeded``,
        # which is also sorted. Without this, arrival order wins and
        # every value pairs with the wrong sample.
        ordered = [piece for _, piece in sorted(columns, key=_by_index)]
        values = (
            bkd.hstack(ordered) if ordered else bkd.zeros((nqoi, 0))
        )
        joined = {
            name: _join(bkd, name, pieces)
            for name, pieces in derivative_pieces.items()
        }
        return EvalResult(
            values=values,
            succeeded=_index_array(bkd, succeeded),
            failed=_index_array(bkd, failed),
            cancelled=_index_array(bkd, cancelled),
            cost=self._ledger.total(),
            statuses=statuses,
            jacobians=joined["jacobians"],
            hessians=joined["hessians"],
            jvps=joined["jvps"],
            hvps=joined["hvps"],
            hvp_weights=self._hvp_weights,
        )

    def __enter__(self) -> "Batch[Array, Task, Payload]":
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._pending:
            self.cancel()


class Evaluator(Generic[Array, Task, Payload]):
    """One marshaller, one dispatcher, and the grouping between them.

    Parameters
    ----------
    marshaller : MarshallerProtocol[Array, Task, Payload]
        Builds tasks and decodes their output. Owns the format and the
        mathematics.
    dispatcher : DispatcherProtocol[Task, Payload]
        Launches tasks and reports on them. Owns the machine.
    ledger : CostLedger, optional
        Where cost accumulates. Pass one shared instance to several
        evaluators to get a budget spanning models; omit it for a
        private ledger.
    on_complete : callable, optional
        Called once per finished task, before ``release``, with the
        outcome, its decoded output (``None`` if the task failed, timed
        out, was cancelled, or could not be decoded) and its cost. This
        is the seam for crash-safe recording: a store that saw only
        successes would make a resumed run recompute every failure
        forever.

        Typed on this class rather than on the protocol, because the
        protocol erases ``Task`` -- reaching a task's own fields through
        it would need a cast.
    """

    def __init__(
        self,
        marshaller: MarshallerProtocol[Array, Task, Payload],
        dispatcher: DispatcherProtocol[Task, Payload],
        ledger: Optional[CostLedger] = None,
        on_complete: Optional[CompletionHook[Task, Payload, Array]] = None,
    ) -> None:
        if not isinstance(marshaller, MarshallerProtocol):
            raise TypeError(
                "marshaller must satisfy MarshallerProtocol, got "
                f"{type(marshaller).__name__}"
            )
        if not isinstance(dispatcher, DispatcherProtocol):
            raise TypeError(
                "dispatcher must satisfy DispatcherProtocol, got "
                f"{type(dispatcher).__name__}"
            )
        self._marshaller = marshaller
        self._dispatcher = dispatcher
        self._ledger = CostLedger() if ledger is None else ledger
        self._on_complete = on_complete

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._marshaller.bkd()

    def nvars(self) -> int:
        """Number of input variables the model takes."""
        return self._marshaller.nvars()

    def nqoi(self) -> int:
        """Number of quantities of interest the model returns."""
        return self._marshaller.nqoi()

    def ledger(self) -> CostLedger:
        """The ledger this evaluator accumulates into."""
        return self._ledger

    def derivatives(self) -> Derivatives[Array]:
        """Which derivative capabilities this evaluator can serve.

        Mirrors the marshaller's bundle, so a marshalled model is
        inspected exactly like any other objective.
        """
        return self._marshaller.derivatives()

    def submit(
        self, samples: Array, request: Optional[Request[Array]] = None
    ) -> Batch[Array, Task, Payload]:
        """Submit samples of shape ``(nvars, nsamples)``.

        Returns without waiting for the work. Validation happens here,
        before a single task is built, so a rejected submission costs
        nothing.
        """
        if samples.ndim != 2:
            raise ValueError(
                "samples must be 2D (nvars, nsamples), got shape "
                f"{tuple(samples.shape)}; a single sample is (nvars, 1) "
                "and nothing is reshaped automatically"
            )
        if samples.shape[0] != self.nvars():
            raise ValueError(
                f"samples has {samples.shape[0]} rows, expected "
                f"{self.nvars()}"
            )
        req = Request.values_only() if request is None else request
        self._validate_request(req)

        nsamples = int(samples.shape[1])
        started = time.perf_counter()
        # After validation, so a rejected submission opens nothing, and
        # before the chunk loop, because a chunk boundary is not a
        # submission boundary -- ``tasks`` is called once per chunk and
        # cannot tell the difference from inside.
        if isinstance(self._marshaller, SubmissionAware):
            self._marshaller.begin_submission()
        tasks: List[Task] = []
        for lo, hi in _chunks(
            nsamples,
            self._marshaller.max_samples_per_task(),
            self._dispatcher.concurrency(),
        ):
            tasks.extend(
                self._marshaller.tasks(
                    samples[:, lo:hi], list(range(lo, hi)), req
                )
            )
        handles = self._dispatcher.submit(tasks)
        return Batch(
            handles=handles,
            marshaller=self._marshaller,
            nsubmitted=nsamples,
            ledger=self._ledger,
            started=started,
            hvp_weights=req.hvp_weights,
            on_complete=self._on_complete,
        )

    def _validate_request(self, request: Request[Array]) -> None:
        """Refuse a request for a capability the model does not have.

        At submit, where the caller can act on it, rather than returning
        a result with a silently absent field.
        """
        derivs = self._marshaller.derivatives()
        wanted: List[Tuple[bool, Optional[object], str]] = [
            (request.jacobians, derivs.jacobian_batch, "jacobians"),
            (request.hessians, derivs.hessian_batch, "hessians"),
            (request.wants_jvp(), derivs.jvp, "jvp"),
        ]
        if request.wants_hvp():
            field = (
                derivs.whvp_batch
                if request.is_weighted_hvp()
                else derivs.hvp_batch
            )
            wanted.append((True, field, "hvp"))
        for asked, capability, name in wanted:
            if asked and capability is None:
                raise ValueError(
                    f"request asked for {name}, which this model does not "
                    "provide; its derivatives() bundle has no such "
                    "capability"
                )


def _chunks(
    nsamples: int, max_per_task: int, concurrency: int
) -> List[Tuple[int, int]]:
    """Split a batch into ``(lo, hi)`` column ranges.

    ``min(max_samples_per_task, ceil(nsamples / concurrency))`` -- which
    reduces to one task per sample for a marshaller that inherently
    works per sample, and to one chunk per worker for a vectorized one.
    """
    if nsamples == 0:
        return []
    per_worker = max(1, ceil(nsamples / max(1, concurrency)))
    size = max(1, min(max_per_task, per_worker))
    return [(lo, min(lo + size, nsamples)) for lo in range(0, nsamples, size)]


#: Derivative fields carried from ``Decoded`` to ``EvalResult``, and
#: which axis indexes the sample.
#:
#: Mixing these up silently transposes a result rather than raising, and
#: the grouping is not the intuitive one:
#:
#: - ``jacobians`` ``(n, nqoi, nvars)`` and ``hessians``
#:   ``(n, nvars, nvars)`` are sample-**first**;
#: - ``hvps`` is **also** sample-first, ``(n, nvars)``. The batch form
#:   is scalar-implicit -- no nqoi axis at all -- which the derivative
#:   bundle itself calls out as a recurring trap when reshaping. The
#:   single-sample ``hvp`` is ``(nvars, 1)``, so the batch form is not
#:   simply the single form widened;
#: - ``jvps`` ``(nqoi, n)`` is sample-**last**, matching ``values``.
#:
#: So "directional capabilities are sample-last" is false, and only
#: ``jvps`` follows the ``values`` convention.
_DERIVATIVE_FIELDS: Dict[str, int] = {
    "jacobians": 0,
    "hessians": 0,
    "jvps": 1,
    "hvps": 0,
}


def _select(piece: Array, name: str, take: Sequence[int]) -> Array:
    """Take the given samples from one decoded derivative piece."""
    if _DERIVATIVE_FIELDS[name] == 0:
        return piece[take]
    return piece[:, take]


def _with_uncollected(
    charged: Cost, pending: Sequence[Cost]
) -> Cost:
    """Add finished-but-uncollected job costs to what is already charged.

    Compute sums. Wall-clock takes the larger of the two rather than
    their sum, since the uncollected jobs ran concurrently with the
    collected ones under the same dispatcher -- adding them would count
    the overlap twice. A lower bound, and the same trade the ensemble
    makes for the same reason: the exact figure needs the spans, which
    a total no longer carries.

    Provenance degrades pessimistically, so a measured total containing
    an estimate is reported as an estimate.
    """
    if not pending:
        return charged
    provenance = charged.provenance
    for cost in pending:
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
        wall_clock=max(
            charged.wall_clock,
            max(cost.wall_clock for cost in pending),
        ),
        compute=charged.compute
        + sum(cost.compute for cost in pending),
        provenance=provenance,
    )


def _outcome_cost(
    outcome: Outcome[Task, Payload], decoded: Optional[Decoded[Array]]
) -> Cost:
    """What a job spent, preferring a duration the job reported itself.

    A dispatcher can only time what it waited on. For in-process work
    that is the job, but for anything external it also covers process
    startup, input staging, filesystem sync, and on a scheduler the
    entire queue wait -- so the figure can exceed the real one by orders
    of magnitude. Where the marshaller decoded a runtime out of the
    job's own output, that is the better number and is used instead.
    """
    if decoded is None or decoded.wall_time is None:
        return outcome.cost()
    return Cost.measured(
        wall_clock=decoded.wall_time,
        compute=decoded.wall_time * outcome.resources.ncores,
    )


def _by_index(tagged: Tuple[int, Array]) -> int:
    """Sort key: the batch index a decoded piece starts at."""
    return tagged[0]


def _join(
    bkd: Backend[Array], name: str, pieces: Sequence[Tuple[int, Array]]
) -> Optional[Array]:
    """Concatenate tagged pieces in index order, or ``None`` if absent.

    Sorted by batch index for the same reason values are: pieces arrive
    in completion order, and a derivative paired with the wrong sample
    is as wrong as a value paired with the wrong sample, but harder to
    notice.

    ``None`` rather than an empty array when nothing was decoded,
    because absence of a capability and a capability that returned
    nothing are different, and the record's contract is that ``None``
    means the former.
    """
    if not pieces:
        return None
    ordered = [piece for _, piece in sorted(pieces, key=_by_index)]
    if len(ordered) == 1:
        return ordered[0]
    return bkd.concatenate(ordered, axis=_DERIVATIVE_FIELDS[name])


def _index_array(bkd: Backend[Array], indices: Sequence[int]) -> Array:
    """Build an integer index array, empty-safe on both backends."""
    if not indices:
        return bkd.zeros((0,), dtype=int)
    return bkd.asarray(sorted(indices))
