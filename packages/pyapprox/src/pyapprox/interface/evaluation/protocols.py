"""The seams: dispatch, marshalling, and the evaluator that composes them.

Three protocols and a handle, split so that a model running locally today
and on a cluster tomorrow differs by one constructed object.

**Dispatch is array-free and format-free.** A dispatcher never sees an
``Array``, never sees a ``Backend``, and never sees a file format. It
launches opaque tasks, reports on them, and hands back what they
produced, so it is not generic in ``Array`` at all. Marshalling is where
both the format and the mathematics live.

Typing note, established by running the cases rather than reasoning about
them. With the invariant ``Array`` type variable:

======================================================  =================
Where ``Array`` occurs in a protocol                    mypy verdict
======================================================  =================
argument position **only**                              error: contravariant expected
argument position, plus anywhere in return              fine -- inferred invariant
return only, nested in an **invariant** generic         fine
return only, nested in a **covariant** generic          error: covariant expected
declared covariant and used as a parameter              error: cannot use as parameter
======================================================  =================

Two consequences are load-bearing here. First, splitting ``submit`` away
from ``collect`` into a submit-only protocol fails -- the *submit* half
breaks, not the collect half, because its only ``Array`` would be in
argument position. Second, **boundary records stay invariant frozen
dataclasses**: :class:`~pyapprox.interface.evaluation.records.EvalResult`
is nominal and therefore invariant, so :class:`BatchProtocol` -- whose
only occurrence of ``Array`` is nested inside it -- typechecks. Restating
that record as a return-only ``Protocol`` would make it covariant and
break exactly that shape. This complements the note in
``functions/protocols/objective.py``, which states the other half: a
genuinely return-only protocol must declare a covariant type variable
rather than adding a dummy argument to silence the checker.

**A subscripted protocol cannot be used with ``isinstance``.**
``isinstance(d, DispatcherProtocol[T])`` raises ``TypeError``; only the
bare protocol works, and it cannot check the task pairing at all.
Runtime checks are therefore a backstop at public boundaries, for an
implementer who missed a method. The primary guard is mypy at the call
site, where pairing a marshaller and dispatcher of different task
families fails to infer ``Task``.
"""

from typing import (
    Generic,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    runtime_checkable,
)

from pyapprox.interface.evaluation.records import (
    ComputeProvenance,
    Cost,
    Decoded,
    EvalProgress,
    EvalResult,
    Outcome,
    Request,
)
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.util.backends.protocols import Array, Backend

Task = TypeVar("Task")
Payload = TypeVar("Payload")


@runtime_checkable
class TaskProtocol(Protocol):
    """The one thing every task must carry: which columns it covers.

    Dispatch is otherwise entirely opaque about tasks -- it never reads
    a format, a path or an array. But an ``Outcome`` records its indices
    **explicitly** rather than letting them be inferred from completion
    order, because inferring them silently scrambles results under
    priority queueing, retry-at-end, or any scheduler that returns out
    of order. That guarantee has to come from somewhere, and the task is
    the only thing dispatch holds.

    Structural, so task types stay plain frozen dataclasses with no base
    class to inherit.
    """

    @property
    def indices(self) -> Sequence[int]:
        """Which columns of the submitted batch this task covers."""
        ...


class MarshalError(Exception):
    """A finished job's output could not be turned into values.

    Missing, unparseable, or the wrong shape. Raised by
    :meth:`MarshallerProtocol.values` and caught by the evaluator, which
    records the affected samples as failed and carries on.

    It is an exception rather than a return value because it is a
    statement about *one task*, and the evaluator's response -- record
    those indices, keep the rest of the batch running -- is uniform.
    Raising it out of a whole batch, as a blocking wrapper must, would
    orphan every other in-flight job and leak their handles.
    """


@runtime_checkable
class JobHandle(Protocol, Generic[Task, Payload]):
    """A single submitted job, shaped like a ``concurrent.futures`` future.

    Deliberately per-job and **idempotent**, rather than a batch-level
    cursor that yields whatever has completed since it was last called.
    A cursor loses outcomes to whichever caller polls first, cannot count
    completions without consuming them, and gives two readers of one
    handle a way to silently steal each other's results. A handle has one
    outcome, readable as often as you like, and is freed with the object.
    """

    def done(self) -> bool:
        """Whether the job has finished, without consuming anything.

        Free and repeatable: this is what lets a progress report count
        completions without disturbing them.
        """
        ...

    def outcome(
        self, timeout: Optional[float] = None
    ) -> Outcome[Task, Payload]:
        """What the job produced, waiting up to ``timeout`` seconds.

        Returns an ``OUTSTANDING`` outcome if the timeout expires rather
        than raising. A timeout is only a failure relative to a policy
        the dispatcher does not know, and raising would wrap a wholly
        non-exceptional event in ``try``/``except`` inside an ordinary
        polling loop.

        ``timeout=None`` waits indefinitely.
        """
        ...

    def cancel(self) -> bool:
        """Try to stop the job; report whether it was stopped.

        **Best-effort, and the guarantee differs by dispatcher.** The
        protocol promises only the weaker form: *pending work will not
        start*. Interrupting work already running is an upgrade that
        some dispatchers provide and others cannot -- an executor cannot
        interrupt a running future, while a subprocess dispatcher can
        signal and reap the child. Returns ``True`` if the job will not
        run or was stopped, ``False`` if it was already running or
        finished.

        The outcome remains readable afterwards, so compute already
        burnt reaches the ledger.
        """
        ...


@runtime_checkable
class DispatcherProtocol(Protocol, Generic[Task, Payload]):
    """Launches opaque tasks and reports on them. Varies with the machine."""

    def concurrency(self) -> int:
        """How many tasks this dispatcher runs at once.

        A constructor argument for every real implementation rather than
        something inferred: ``Executor`` exposes no public worker count.
        Grouping and estimated costs both depend on it, so it is
        validated rather than trusted.
        """
        ...

    def compute_provenance(self) -> ComputeProvenance:
        """Whether this dispatcher's core-seconds mean anything.

        Declared by the dispatcher because only it knows how its
        ``wall_time`` figures were obtained.
        """
        ...

    def submit(self, tasks: Sequence[Task]) -> Sequence[JobHandle[Task, Payload]]:
        """Launch tasks, returning one handle each, in order.

        **Must return before the work finishes.** A dispatcher whose
        ``submit`` blocks for the whole batch satisfies this protocol
        while defeating its purpose, and does so invisibly -- no test
        that fails to measure *when submit returns* would catch it.

        Handles are returned for tasks that have not started yet, so a
        throttling dispatcher needs an internal pending queue. That is
        what a real executor does anyway.

        **A handle corresponds to a task, not to a job.** An
        implementation may map several tasks onto one unit of work --
        one scheduler job, one array element, one slot in an
        allocation it already holds -- and return handles that all
        resolve when that unit finishes. Nothing above may assume
        one-task-one-job: that assumption forecloses task packing and
        pilot-style allocations, where work for models with different
        resource requests shares a single submission.

        **A blocking backend does not need rewriting to satisfy this.**
        An interface that waits for its resources before returning is a
        slow callable, and running a slow callable without blocking is
        what a thread pool does: hand the call to the pool, return the
        handles, let them resolve later. Cancellation then degrades to
        the weaker guarantee this protocol already permits, since a
        thread blocked inside the call cannot be interrupted.
        """
        ...

    def close(self) -> None:
        """Release resources; no further ``submit`` calls are valid.

        Required on the protocol because keeping an executor alive
        across submissions deletes the ``with`` block that would
        otherwise guarantee shutdown on every exit path including
        exceptions. Concrete dispatchers are also context managers.
        Idempotent.
        """
        ...


@runtime_checkable
class MarshallerProtocol(Protocol, Generic[Array, Task, Payload]):
    """Samples to tasks, payloads back to values. Varies with the code wrapped.

    Carries ``bkd()``, ``nvars()`` and ``nqoi()`` -- the entire
    mathematical surface of a model in this repo -- because for an
    external solver "the problem" and "the code being wrapped" are the
    same axis. You do not change the input format without changing the
    solver, and you do not change the solver without changing the
    mathematics.

    **The marshaller owns working directories, end to end**, where there
    are any: it creates the directory, writes inputs into it, puts the
    path in the task, and applies the retention policy in ``release``.
    Dispatch merely passes the path as ``cwd``. Assigning directories to
    dispatch instead is unbuildable -- the marshaller would have nowhere
    to write inputs, because the directory would not exist until submit
    ran -- and would be dishonest anyway, since thread and HTTP
    dispatchers have no working directory at all.

    **A marshaller must never call ``os.chdir``.** It is process-global
    rather than per-thread, so it is unsafe the moment anything runs in
    a thread pool, and an exception between the two calls leaves the
    interpreter in the wrong directory, breaking every subsequent
    relative path. Absolute paths and ``cwd=`` remove the need for it.
    """

    def bkd(self) -> Backend[Array]:
        """Return the backend used to build values and index arrays."""
        ...

    def nvars(self) -> int:
        """Number of input variables the model takes."""
        ...

    def nqoi(self) -> int:
        """Number of quantities of interest the model returns."""
        ...

    def max_samples_per_task(self) -> int:
        """Most samples this marshaller will put in one task.

        One where a task is inherently per-sample, such as one working
        directory per evaluation. Larger -- or effectively unbounded --
        where the wrapped code is vectorized, which is the normal case
        in this repo and must not be regressed: fanning every batch into
        one task per sample turns a single ``f(samples)`` call into
        ``nsamples`` calls, four orders of magnitude more dispatch
        operations for a fast model.

        The evaluator groups by
        ``min(max_samples_per_task(), ceil(nsamples / concurrency()))``,
        which reduces to one-per-sample for subprocess work and to
        chunked behavior for in-memory work.
        """
        ...

    def tasks(
        self,
        samples: Array,
        indices: Sequence[int],
        request: Request[Array],
    ) -> Sequence[Task]:
        """Build the tasks that satisfy ``request`` for ``samples``.

        Returns **however many tasks the wrapped code needs**, which is
        the point of taking a request rather than exposing one method
        per quantity. Codes differ in how much they fuse: an adjoint
        solver has usually formed everything a gradient needs by the
        time the forward solve finishes, so value and jacobian come back
        from one invocation; other codes expose them as separate entry
        points. That is a property of the code, not of the caller's ask,
        so the caller states the ask and the marshaller decides the
        invocations -- one task for a fused code, several for a split
        one. A new capability never adds a method here.

        Parameters
        ----------
        samples : Array
            Shape ``(nvars, len(indices))`` -- the columns these tasks
            cover, already sliced out of the submitted batch.
        indices : Sequence[int]
            Which columns of the submitted batch those are. Carried into
            each ``Outcome`` so results need no positional reassembly.
        request : Request[Array]
            Which quantities are wanted, and any vectors they contract
            against. Already validated against :meth:`derivatives`, so
            an implementation may assume it advertised everything asked
            for.
        """
        ...

    def values(self, outcome: Outcome[Task, Payload]) -> Decoded[Array]:
        """Decode a finished job into arrays and the indices they cover.

        Returns the columns that decoded and the indices they came from,
        rather than one column, because a task may carry several
        samples. A task whose output is entirely unreadable raises
        :class:`MarshalError`; one where some samples decoded and others
        did not returns the subset, and the evaluator records the rest
        as failed for the quantities that task was carrying.

        A :class:`Decoded` populates only the quantities *its own* task
        computed. Where one request became several tasks, several
        ``Decoded`` records cover the same indices with different fields
        populated, and the evaluator merges them by index -- which is
        what makes success per-quantity rather than per-sample.

        **The implementer's obligation: entry ``k`` of every returned
        array must describe ``indices[k]``.**

        This sits here rather than in the evaluator because only a
        marshaller *can* honor it. The evaluator reorders whole tasks by
        the indices their outcomes carry, so out-of-order completion by
        a dispatcher is already handled -- the information it needs is
        present. Within one task it receives values and indices already
        paired, and re-deriving the pairing would mean recomputing the
        model. A permuted decode therefore yields a result of the right
        shape, the right size, and wrong, which nothing downstream can
        detect.

        Three ways this is commonly broken, all worth checking in a new
        marshaller:

        - **Reading outputs by directory listing.** ``glob("out_*")``
          sorts lexicographically, so ``out_10`` precedes ``out_2``.
          Sort by the integer the name encodes, not by the name.
        - **An unordered map inside the marshaller.**
          ``imap_unordered`` and its equivalents yield results as they
          finish, which is exactly what must not be zipped against an
          ordered index list. Send the index *with* each work item so
          each result carries its own identity back.
        - **A solver that reorders its own output** -- MPI ranks writing
          to a shared file, or a code that sorts before writing.

        The safe construction in every case is the same: derive each
        entry's index from the output itself, and build ``indices``
        alongside the arrays rather than assuming the two already agree.

        A marshaller that hands a whole slice to one call and receives
        one array back -- the in-memory case -- satisfies this by
        construction and needs no care.

        **Every marshaller must carry a test for this.** Since the
        obligation cannot be enforced at any boundary, a test is the
        only thing standing between a permuted decode and a silently
        wrong study. Submit samples whose values identify their own
        column -- column ``j`` encoding ``j`` -- decode, and assert that
        entry ``k`` is the one ``indices[k]`` names. Shape and count
        assertions do not substitute: a permutation passes both.

        Include a multi-sample task, since a per-sample marshaller
        cannot express the failure at all, and the marshallers that can
        get this wrong are exactly those grouping several samples per
        task.
        """
        ...

    def derivatives(self) -> Derivatives[Array]:
        """Which derivative capabilities this marshalled model has.

        Discovery works exactly as it does for any other bundle:
        inspect the fields, and a non-``None`` field means the
        capability is available. There is no separate capability
        predicate to learn, and consumers branch on the bundle once at
        construction rather than probing with ``hasattr``.

        A field being non-``None`` here is a statement that this
        marshaller can *dispatch* that capability -- that it knows how
        to ask the wrapped code for it and how to decode the answer.
        The evaluator mirrors this bundle into its own, so a marshalled
        model is inspected like any other objective, and validates
        requests against it: asking for a quantity whose field is
        ``None`` is an error at submit rather than a silently absent
        result.

        Nothing here is ever reconstructed from anything else. A
        directional capability takes its vector as an **input to the
        solve** -- a seeded tangent or second-order adjoint returns the
        product for the cost of one solve without forming the operator
        -- so a marshaller that cannot do a real directional solve
        leaves the field ``None`` rather than materializing a hessian to
        contract afterwards. That conversion turns a matrix-free
        algorithm into an O(nvars^2)-memory one, which is exactly what
        ``Derivatives.resolved_hvp`` refuses to do silently.

        Return ``Derivatives.none()`` when the wrapped model has no
        derivative capability -- the common case for an external forward
        solver.
        """
        ...

    def release(self, outcome: Outcome[Task, Payload]) -> None:
        """Release whatever the task held -- scratch directories, handles.

        Called exactly once per outcome, including for failed and
        cancelled jobs, so a cancelled batch cleans up after itself.
        """
        ...


@runtime_checkable
class ResultStore(Protocol, Generic[Array]):
    """Somewhere finished results survive the process that produced them.

    A handle is an in-memory object, so a crashed workflow loses its
    handles. For a model taking hours per sample, recomputing everything
    after a restart is the sharpest version of the failure this design
    exists to prevent -- so results are written down as they are
    collected, and a resumed run skips what is already known.

    Two failures need separating. If the **workflow process** dies but
    the work does not -- Slurm jobs keep running, an orphaned subprocess
    still writes its output, a remote solver finishes regardless -- only
    the bookkeeping was lost, and reattaching to live work additionally
    needs a dispatcher that can re-adopt job ids. If the **machine**
    dies, the work is gone too and only completed results can be
    salvaged. This protocol addresses the second and most of the first.

    **Keys are the caller's, and so is the decision to consult a store
    at all.** An evaluator never skips work, never refuses a duplicate,
    and never returns something it did not just compute: it has no
    notion of sample identity, because the caller that built the samples
    is the only party that has one. A caller resuming asks its store
    what it already has, submits the remainder, and maps the returned
    column indices back itself.

    That trades a visible performance bug -- a caller who forgets to
    check recomputes -- against an invisible correctness one, where a
    framework hands back stored values for a key that no longer means
    what it did.

    Deliberately **not** keyed on a hash of the sample values: that is
    bit-exact, silently dtype- and contiguity-dependent, and two
    mathematically identical samples can key differently. It would also
    put a value comparison in a control path, which nothing here does.

    Implementations are injected, so a caller may back this with an
    ``.npz`` file, a database, or anything else. Nothing here enumerates
    the possibilities.

    Decoded arrays are the currency because they are portable: one
    generic store then works for any model. A marshaller that owns a
    native on-disk format may instead supply its *own* implementation of
    this protocol -- one that records where the solver already wrote its
    output and re-decodes lazily on ``load`` -- which avoids duplicating
    data that is durable already, and keeps whatever the native format
    carries that ``values`` does not. Same protocol either way; only the
    internals differ.
    """

    def save(self, key: str, decoded: Decoded[Array], cost: Cost) -> None:
        """Record one task's decoded output under ``key``.

        Called as results stream in, not at the end -- a store written
        only on completion would be empty in exactly the case it exists
        for. Must be safe to call again with the same key (a resumed run
        may recompute a sample whose save was interrupted).
        """
        ...

    def load(self, key: str) -> Optional[Tuple[Decoded[Array], Cost]]:
        """Return what was stored under ``key``, or ``None`` if absent."""
        ...

    def keys(self) -> Sequence[str]:
        """Every key currently stored.

        Lets a resuming evaluator decide what to skip in one call rather
        than probing per sample.
        """
        ...


@runtime_checkable
class BatchProtocol(Protocol, Generic[Array]):
    """Work in flight: what can be asked, and how results come back.

    **The streaming contract.** Each of ``collect_ready``, ``collect``
    and ``cancel`` returns the outcomes *not returned by a previous
    call*. The union over all calls is the submitted batch, each index
    exactly once. So ``collect`` on a batch nobody streamed returns
    everything -- the common case -- and ``collect`` after some
    ``collect_ready`` calls returns only the remainder.

    A ``Batch`` is an object rather than an opaque token because a token
    over an integer is forgeable, and two evaluators each counting from
    zero would silently return each other's results. An object cannot be
    forged, needs no handle registry, and frees its state when it goes
    out of scope. It is also a context manager, so
    ``with evaluator.submit(samples) as batch:`` cancels and releases on
    ``KeyboardInterrupt``.
    """

    def nsubmitted(self) -> int:
        """How many samples this batch was submitted with."""
        ...

    def progress(self) -> EvalProgress:
        """Counts and cost so far, without consuming or waiting."""
        ...

    def collect_ready(self) -> EvalResult[Array]:
        """Return outcomes that are finished now, without waiting."""
        ...

    def collect(self, timeout: Optional[float] = None) -> EvalResult[Array]:
        """Wait for the rest of the batch and return it.

        ``timeout=None`` waits indefinitely. On expiry, returns what has
        finished; the remainder stays outstanding and collectable.
        """
        ...

    def cancel(self) -> EvalResult[Array]:
        """Stop what has not finished and return everything outstanding.

        Returns an :class:`EvalResult` rather than ``None`` so the
        compute already burnt by killed jobs reaches the ledger and
        ``release`` runs for their working directories. A cancel that
        discarded its own cost would be the sharpest possible version of
        the failure this design exists to prevent.
        """
        ...


@runtime_checkable
class EvaluatorProtocol(Protocol, Generic[Array]):
    """A model that can be submitted to and asked about.

    ``Task`` and ``Payload`` are erased here: the concrete evaluator is
    generic in all three, but callers only ever name ``Array``.
    """

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        ...

    def nvars(self) -> int:
        """Number of input variables the model takes."""
        ...

    def nqoi(self) -> int:
        """Number of quantities of interest the model returns."""
        ...

    def submit(
        self, samples: Array, request: Optional[Request[Array]] = None
    ) -> BatchProtocol[Array]:
        """Submit samples of shape ``(nvars, nsamples)`` for evaluation.

        Returns without waiting for the work. Raises ``ValueError`` for a
        1D array or the wrong ``nvars``; single samples are ``(nvars, 1)``
        and nothing is reshaped automatically.

        ``request`` defaults to values only. Asking for a quantity whose
        field is ``None`` in :meth:`derivatives` raises ``ValueError``
        here, where the caller can act on it, rather than returning a
        result with a silently absent field.

        This is also the only path that can exploit a fused code, since
        a request names several quantities at once and the marshaller
        decides how many invocations that is. The per-field callables in
        :meth:`derivatives` necessarily ask for one quantity each.

        **Results are indexed by column, and nothing else.** An evaluator
        holds no notion of what a sample *is* beyond its position in what
        was submitted, so mapping those indices onto anything durable --
        a latent draw, an allocation partition, a design row -- belongs
        to the caller, which is the only party that knows the mapping.
        """
        ...

    def derivatives(self) -> Derivatives[Array]:
        """Which derivative capabilities this evaluator can serve.

        Mirrors the marshaller's bundle, so a marshalled model is
        inspected exactly like any other objective -- a non-``None``
        field means the capability is available, and there is no second
        capability mechanism to learn. The callables submit and collect
        internally, so each is a blocking round trip.
        """
        ...
