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
    TaskProtocol,
)
from pyapprox.interface.evaluation.records import (
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
        self._nsucceeded = 0
        self._nfailed = 0

    def nsubmitted(self) -> int:
        """How many samples this batch was submitted with."""
        return self._nsubmitted

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
        for handle in self._pending:
            if not handle.done():
                continue
            outcome = handle.outcome()
            covered = len(outcome.indices)
            if outcome.status is JobStatus.SUCCEEDED:
                nsucceeded += covered
            else:
                nfailed += covered
        return EvalProgress(
            nsucceeded=nsucceeded,
            nfailed=nfailed,
            noutstanding=self._nsubmitted - nsucceeded - nfailed,
            cost=self._ledger.total(),
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
            cost = outcome.cost()
            self._ledger.add(cost)

            decoded: Optional[Decoded[Array]] = None
            if outcome.status is JobStatus.SUCCEEDED:
                try:
                    decoded = self._marshaller.values(outcome)
                except MarshalError:
                    # A statement about one task: record its samples as
                    # failed and keep the rest of the batch running.
                    decoded = None

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
                    for name in _DERIVATIVE_FIELDS:
                        piece = getattr(decoded, name)
                        if piece is not None:
                            derivative_pieces[name].append(
                                (first, _select(piece, name, take))
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

            if self._on_complete is not None:
                self._on_complete(outcome, decoded, cost)
            self._marshaller.release(outcome)

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
