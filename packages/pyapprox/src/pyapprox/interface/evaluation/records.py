"""What an evaluation costs, what came back, and what is still running.

An evaluator submits work and collects it later, rather than blocking on
a call. The reason is not that a queued model would be awkward to drive
with ``map`` -- though it would -- but that **work in flight is a state
the workflow can be asked about**. A blocking call has no such state:
everything is unknown until everything is known.

Outstanding work makes several things possible that a blocking call
forecloses:

- a decision point can see *partial* progress -- forty of two hundred
  back, three failed, six hours gone -- and act on it, rather than
  learning everything at the end;
- the response to a failure can be chosen while the rest is still
  running, which is when it is worth choosing;
- submitting and collecting need not be the same call or the same
  decision;
- a budget cap can refuse the next submission instead of reporting an
  overrun after the fact.

Two properties live in the records rather than the mechanism, and each
would be expensive to add later because both propagate into the
*statistics* rather than the control flow:

**Partial failure is a return value, not an exception.** A downstream
estimator has to be told which samples actually came back: it must be
computed on the realized set rather than the requested one, sizing rules
must use the realized count, and a budget must charge for failures
because they consumed real compute. Fabricating NaN values for a failed
sample and caching them as a result poisons that sample permanently;
returning the narrower set does not.

**Two clocks, and where the second one came from.** Wall-clock and
compute coincide for serial work and diverge the moment anything runs
concurrently -- compute being the *larger*, by roughly the number of busy
workers. Ranking arms by time needs both, and only the dispatcher knows
whether its core-seconds are real accounting, an inference, or a figure
that means nothing at all, so that distinction travels with the numbers
as :class:`ComputeProvenance`.

**Variance rule for every record here.** Boundary records stay invariant
frozen dataclasses. A ``@dataclass(frozen=True)`` is nominal and
therefore invariant, so a protocol returning one is safe even when the
protocol's *only* occurrence of ``Array`` is inside it. Restating such a
record as a return-only ``Protocol`` makes it covariant and breaks any
protocol of that shape -- ``BatchProtocol`` is exactly that shape, so
this is a live hazard rather than a theoretical one.
"""

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Generic, Mapping, Optional, Sequence, Tuple, TypeVar

from pyapprox.util.backends.protocols import Array

Task = TypeVar("Task")
Payload = TypeVar("Payload")


class ComputeProvenance(Enum):
    """Where a ``compute`` figure came from.

    Three states rather than a boolean. ``MEASURED`` and ``ESTIMATED``
    are the obvious two. The third exists because some transports have
    no meaningful core-second figure at all: for HTTP dispatch the local
    ``wall_time`` is time spent blocked on a socket, and multiplying it
    by a core count yields a number that is neither an upper nor a lower
    bound on anything. Reporting that as measured would corrupt exactly
    the accounting the distinction exists to protect.
    """

    MEASURED = "measured"
    """Real accounting -- the figure was observed, not inferred."""

    ESTIMATED = "estimated"
    """Inferred from elapsed time and a worker count. An upper bound."""

    NOT_APPLICABLE = "not_applicable"
    """No meaningful local core-second figure exists for this work."""


class TimeSource(Enum):
    """Who measured a job's duration.

    The same argument as :class:`ComputeProvenance`, applied to the
    other clock: the number travels with a statement of where it came
    from, because the alternatives differ by orders of magnitude and a
    caller ranking models on runtime cannot otherwise know what it is
    comparing.

    A wrapper timing a scheduler submission may report hours for a job
    that ran for twenty minutes, since it also counts queue wait,
    staging and filesystem sync. Mixing that with a self-reported figure
    in one batch would silently compare two different quantities.
    """

    WRAPPER = "wrapper"
    """Timed around the call, in the process that launched it.

    Exact for in-process work, where the call *is* the job. For anything
    external it is an upper bound that includes whatever the wrapper
    waited through.
    """

    JOB = "job"
    """Reported by the job itself, or by the system that ran it.

    A solver writing its own runtime, or a scheduler's accounting. The
    most accurate available, because it excludes startup and staging.
    """


@dataclass(frozen=True)
class Cost:
    """What an evaluation spent, on two clocks.

    Attributes
    ----------
    wall_clock : float
        Elapsed seconds. What a user waits.
    compute : float
        Core-seconds. What an allocation is charged. Equal to
        ``wall_clock`` for serial work, and *larger* -- by roughly the
        number of busy workers -- once anything runs concurrently. Zero
        when ``provenance`` is ``NOT_APPLICABLE``.
    provenance : ComputeProvenance
        Whether ``compute`` was measured, estimated, or is meaningless
        for this transport. Travels with the number because a branch
        table ranked by core-hours reads very differently when the
        figures are estimates, and because a run record should say which
        it was rather than leave a reader to guess.

    Notes
    -----
    There is deliberately no ``__add__``. ``compute`` is additive but
    ``wall_clock`` is not: two batches each running an hour concurrently
    did not take two hours, and this design exists to make overlapping
    batches normal. Use :class:`CostLedger`, which adds compute and takes
    the measure of the *union* of the elapsed spans.
    """

    wall_clock: float
    compute: float
    provenance: ComputeProvenance

    def __post_init__(self) -> None:
        # Non-finite is rejected as well as negative. ``nan < 0.0`` is
        # False, so a bare non-negativity check admits NaN -- and one NaN
        # anywhere silently turns an entire accumulated ledger into NaN.
        # Failed jobs are charged here, and a failure is exactly where a
        # NaN duration is most likely to be produced.
        for name, value in (
            ("wall_clock", self.wall_clock),
            ("compute", self.compute),
        ):
            if not isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}")

    @classmethod
    def measured(cls, wall_clock: float, compute: float) -> "Cost":
        """Both figures come from real accounting."""
        return cls(
            wall_clock=wall_clock,
            compute=compute,
            provenance=ComputeProvenance.MEASURED,
        )

    @classmethod
    def serial(cls, wall_clock: float) -> "Cost":
        """Serial work, where the two clocks genuinely coincide.

        Counts as measured: with one worker there is nothing to
        estimate.
        """
        return cls(
            wall_clock=wall_clock,
            compute=wall_clock,
            provenance=ComputeProvenance.MEASURED,
        )

    @classmethod
    def estimated(
        cls, wall_clock: float, concurrency: int, nsamples: int
    ) -> "Cost":
        """Infer core-seconds from elapsed time and worker count.

        ``wall_clock * min(concurrency, nsamples)``. The cap matters: a
        4-sample batch submitted to a 64-wide evaluator occupies four
        workers, and charging it 64x would corrupt the very cost
        measurement a pilot exists to make.

        This is an **upper bound**, exact only if every worker stayed
        busy for the whole batch. Ragged per-sample times leave workers
        idle and the true figure is lower. Overestimating is the safe
        direction for a budget cap and the wrong one for ranking arms by
        core-hours, which is why the result is flagged rather than
        silently mixed with measured values.
        """
        if concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, got {concurrency}")
        if nsamples < 0:
            raise ValueError(f"nsamples must be >= 0, got {nsamples}")
        busy = min(concurrency, nsamples)
        return cls(
            wall_clock=wall_clock,
            compute=wall_clock * busy,
            provenance=ComputeProvenance.ESTIMATED,
        )

    @classmethod
    def unaccounted(cls, wall_clock: float) -> "Cost":
        """Elapsed time is known; core-seconds are not a meaningful figure.

        For a remote service the local process burnt almost nothing and
        the remote server burnt an unknown amount. ``compute`` is zero
        and says so through its provenance rather than through a
        plausible-looking product.
        """
        return cls(
            wall_clock=wall_clock,
            compute=0.0,
            provenance=ComputeProvenance.NOT_APPLICABLE,
        )

    @classmethod
    def zero(cls) -> "Cost":
        """Nothing was spent. The identity for an empty batch."""
        return cls(
            wall_clock=0.0,
            compute=0.0,
            provenance=ComputeProvenance.MEASURED,
        )


class CostLedger:
    """Accumulates cost across jobs that may have run concurrently.

    ``compute`` sums; ``wall_clock`` does not. Two jobs each running an
    hour side by side occupied two core-hours but only one hour of a
    user's life, so wall-clock is the measure of the **union** of the
    ``(start, end)`` spans rather than their sum. That is right for
    concurrent and sequential use alike, and it is why this is a mutable
    accumulator rather than a ``Cost.__add__``.

    Provenance combines pessimistically: a total is ``MEASURED`` only if
    every contribution was. Mixing one estimate into a sum of measured
    figures makes the total an estimate, and a ledger that reported
    otherwise would launder the estimate.
    """

    def __init__(self) -> None:
        self._spans: list[Tuple[float, float]] = []
        self._compute = 0.0
        self._provenance: Optional[ComputeProvenance] = None

    def add(self, cost: Cost, start: Optional[float] = None) -> None:
        """Add one job's cost.

        Parameters
        ----------
        cost : Cost
            What the job spent.
        start : float, optional
            When the job started, on the same clock the batch uses. When
            given, the span ``(start, start + cost.wall_clock)`` joins
            the union; when omitted the span is treated as starting at
            zero, which makes the union reduce to the longest single
            span. Concurrent dispatchers supply it; serial ones need not.
        """
        self._compute += cost.compute
        origin = 0.0 if start is None else start
        self._spans.append((origin, origin + cost.wall_clock))
        if self._provenance is None:
            self._provenance = cost.provenance
        elif self._provenance is not cost.provenance:
            # Any disagreement degrades the total. NOT_APPLICABLE is the
            # weakest claim, so it wins outright; otherwise a measured
            # total containing an estimate is an estimate.
            if ComputeProvenance.NOT_APPLICABLE in (
                self._provenance,
                cost.provenance,
            ):
                self._provenance = ComputeProvenance.NOT_APPLICABLE
            else:
                self._provenance = ComputeProvenance.ESTIMATED

    def total(self) -> Cost:
        """The accumulated cost so far."""
        if self._provenance is None:
            return Cost.zero()
        return Cost(
            wall_clock=_union_measure(self._spans),
            compute=self._compute,
            provenance=self._provenance,
        )


def _union_measure(spans: Sequence[Tuple[float, float]]) -> float:
    """Total length covered by a set of possibly-overlapping spans.

    Sorting by start and merging is O(n log n) and exact. Summing the
    lengths instead would double-count every overlap, which is precisely
    the error this function exists to avoid.
    """
    if not spans:
        return 0.0
    ordered = sorted(spans)
    total = 0.0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start > current_end:
            total += current_end - current_start
            current_start, current_end = start, end
        elif end > current_end:
            current_end = end
    return total + (current_end - current_start)


class JobStatus(Enum):
    """How a job ended, or that it has not.

    The distinction that matters is **retryable versus not**.
    ``FAILED`` is a statement about the parameter point -- the solver
    diverged *there* -- so blindly resubmitting it is both wasteful and
    statistically wrong. ``TIMED_OUT`` and ``CANCELLED`` say nothing
    about the point and are the ones a caller may legitimately resubmit.
    Collapsing these into a single failed flag would make "resubmit what
    failed" unsafe advice.
    """

    OUTSTANDING = "outstanding"
    """Not finished. Also what a timed-out poll reports."""

    SUCCEEDED = "succeeded"
    """Produced a payload."""

    FAILED = "failed"
    """The job ran and did not produce a usable result."""

    TIMED_OUT = "timed_out"
    """Exceeded a deadline imposed on it."""

    CANCELLED = "cancelled"
    """Stopped, or never started, by request."""

    def is_retryable(self) -> bool:
        """Whether resubmitting this sample is defensible.

        False for ``FAILED``, which is evidence about the parameter
        point itself, and for the two terminal-success and not-yet-run
        states.
        """
        return self in (JobStatus.TIMED_OUT, JobStatus.CANCELLED)

    def is_finished(self) -> bool:
        """Whether this job will not change state again."""
        return self is not JobStatus.OUTSTANDING


@dataclass(frozen=True)
class Resources:
    """What one task needs from the machine that runs it.

    **A property of the wrapped code, not of the dispatcher.** A
    multifidelity ensemble may hold a serial model, a 32-rank model and
    a 128-rank one, running on the same cluster through the same queue.
    If the requirement lived on the dispatcher, that ensemble would need
    three dispatchers differing only by the solver they serve -- and
    three dispatchers cannot share one throttle, which is the whole
    reason an ensemble wants a shared one.

    So the marshaller declares what its code needs, and the dispatcher
    decides what to do about it. **A dispatcher may ignore any of
    this.** An in-process or thread dispatcher has no notion of a queue
    or a memory limit and reads only ``ncores``, for accounting; a
    scheduler dispatcher turns the rest into submission arguments. That
    asymmetry is deliberate: a declaration a dispatcher cannot act on is
    still worth making, because a different dispatcher can.

    Attributes
    ----------
    ncores : int
        Cores one task occupies. Multiplies ``wall_time`` to give the
        compute a job is charged, so it is the one field every
        dispatcher reads.
    walltime_seconds : float, optional
        How long the task may run before it should be considered lost.
        A dispatcher holding a fixed allocation also needs this to avoid
        starting work that cannot finish before the allocation ends.
    memory_mb : int, optional
        Memory the task needs. ``None`` where the caller has no figure,
        which is not the same as needing none.
    queue : str, optional
        Which partition, queue or pool the work belongs in. Meaningless
        to a local dispatcher and load-bearing to a scheduler.
    extra : Mapping[str, str]
        Site-specific submission arguments, passed through verbatim by
        whichever dispatcher understands them -- an account to charge, a
        node constraint, a GPU request.

        The typed fields above are the ones this framework reads or that
        every scheduler shares. Beyond them the space is not
        enumerable: accounts, constraints, reservations and generic
        resources differ by site, and a record that tried to name them
        all would still be wrong somewhere. Without this, a site whose
        need is missing has to edit this file or fork it, which is worse
        than an escape hatch.

        **Prefer a typed field where one fits.** Anything here is
        invisible to type checking, unvalidated, and meaningful only to
        a dispatcher that happens to recognize the key. A value that
        turns out to be common belongs above, not here.

        Strings rather than arbitrary objects, so the record stays
        comparable, hashable in principle, and safe to write into a log
        or a submission script without a serializer.
    """

    ncores: int = 1
    walltime_seconds: Optional[float] = None
    memory_mb: Optional[int] = None
    queue: Optional[str] = None
    extra: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.ncores < 1:
            raise ValueError(f"ncores must be >= 1, got {self.ncores}")
        if self.walltime_seconds is not None:
            if not isfinite(self.walltime_seconds):
                raise ValueError(
                    "walltime_seconds must be finite, got "
                    f"{self.walltime_seconds}"
                )
            if self.walltime_seconds <= 0.0:
                raise ValueError(
                    "walltime_seconds must be positive, got "
                    f"{self.walltime_seconds}"
                )
        if self.memory_mb is not None and self.memory_mb < 1:
            raise ValueError(
                f"memory_mb must be >= 1, got {self.memory_mb}"
            )
        if self.queue is not None and not self.queue:
            raise ValueError("queue must be a non-empty name or None")

    @classmethod
    def serial(cls) -> "Resources":
        """One core, nothing else specified. The common case."""
        return cls()


@dataclass(frozen=True)
class Outcome(Generic[Task, Payload]):
    """What a finished job hands back.

    Attributes
    ----------
    task : Task
        What was run. Returned so the marshaller can decode the result
        without keeping its own side table keyed on job identity.
    indices : Sequence[int]
        Which columns of the submitted batch this job covers. **Explicit
        rather than inferred.** Recovering the mapping by sorting job
        ids, on the assumption that they are assigned in strictly
        increasing dispatch order, silently scrambles results under
        priority queueing, retry-at-end, or any scheduler that returns
        out of order -- with no error at all.
    status : JobStatus
        How it ended.
    payload : Payload, optional
        What it produced, for the marshaller to decode. ``None`` unless
        ``status`` is ``SUCCEEDED``. For an in-process job this *is* the
        answer, which is why dispatch carries a ``Payload`` type at all:
        a lone ``Task`` is the argument half of a call with the return
        half missing.
    wall_time : float
        Seconds this job itself took. For a concurrent dispatcher this
        must be stamped **in the worker and returned**, not measured in
        the parent from submit to done: the latter counts queue wait,
        which for any batch larger than the pool is most of the elapsed
        time, and overstates compute severalfold.
    time_source : TimeSource
        Who measured ``wall_time``. Defaults to ``WRAPPER``, which is
        what a dispatcher timing its own call can honestly claim. A
        marshaller that reads a runtime out of the solver's own output,
        or a dispatcher reading a scheduler's accounting, replaces both
        the figure and this label.
    resources : Resources
        What the job asked of the machine. Carried on the outcome as
        well as the task because cost is computed from it, and because a
        run record that says what a job requested is more useful than
        one that says only how long it took.
    detail : str, optional
        Human-readable note: exit code, exception text, which deadline
        expired. Diagnostics only; nothing branches on it.
    """

    task: Task
    indices: Sequence[int]
    status: JobStatus
    payload: Optional[Payload] = None
    wall_time: float = 0.0
    time_source: TimeSource = TimeSource.WRAPPER
    resources: Resources = field(default_factory=Resources)
    detail: Optional[str] = None

    def __post_init__(self) -> None:
        if not isfinite(self.wall_time):
            raise ValueError(f"wall_time must be finite, got {self.wall_time}")
        if self.wall_time < 0.0:
            raise ValueError(
                f"wall_time must be non-negative, got {self.wall_time}"
            )
        if self.status is JobStatus.SUCCEEDED and self.payload is None:
            raise ValueError("a SUCCEEDED outcome must carry a payload")

    def ncores(self) -> int:
        """How many cores this job occupied."""
        return self.resources.ncores

    def cost(self) -> Cost:
        """What this job spent.

        ``compute`` is ``wall_time * ncores``, which is correct for a
        heterogeneous batch precisely because the core count is per-job
        rather than a property of the dispatcher that ran it.
        """
        return Cost.measured(
            wall_clock=self.wall_time,
            compute=self.wall_time * self.resources.ncores,
        )

    def nsamples(self) -> int:
        """How many samples of the submitted batch this job covers."""
        return len(self.indices)


#: How each ``Derivatives`` bundle field is carried through evaluation.
#:
#: The evaluation records name their capabilities in typed fields rather
#: than deriving them from the bundle at runtime, which keeps ``mypy``
#: checking call sites -- but it also means the two can drift apart
#: silently if the bundle grows a field. This mapping is the one place
#: that states the correspondence, and
#: ``test_bundle_coupling.py`` fails if any bundle field is missing from
#: it, so a new capability cannot be added upstream without this layer
#: noticing.
#:
#: A value of ``None`` means the capability is deliberately not
#: marshalled, and the comment says why.
BUNDLE_FIELD_CARRIERS: dict[str, Optional[str]] = {
    "jacobian": "jacobians",
    "jacobian_batch": "jacobians",
    "hessian": "hessians",
    "hessian_batch": "hessians",
    "jvp": "jvps",
    "hvp": "hvps",
    # The same tensor as ``hvp``, contracted with QoI weights; the
    # bundle converts between them via resolved_hvp/resolved_whvp.
    "whvp": "hvps",
    "hvp_batch": "hvps",
    "whvp_batch": "hvps",
    # Tolerance-aware evaluation. Not marshalled: ``tol`` is an input
    # the caller varies per call, and a request that carried it would
    # make every distinct tolerance its own dispatch.
    "inexact": None,
}


@dataclass(frozen=True)
class Request(Generic[Array]):
    """What a caller wants computed for a set of samples.

    A request says *which quantities*, never *how many invocations*.
    That distinction is the whole point: codes differ in how much they
    fuse. A solver computing derivatives alongside its values has
    usually formed everything a gradient needs by the time the forward
    solve finishes, so value and jacobian come back from one
    invocation; other codes expose them as separate entry points, or
    separate executables. Which one you have
    is a property of the **wrapped code**, not of the caller's ask.

    So the caller states the ask, and
    :meth:`~pyapprox.interface.evaluation.protocols.MarshallerProtocol.tasks`
    returns however many tasks satisfy it -- one for a fused code, two
    or three for a split one. Neither shape is penalized, and a new
    capability never adds a method to the protocol.

    Attributes
    ----------
    values : bool
        Whether the model's values are wanted. Usually ``True``; a
        caller that already has values and wants only a gradient sets
        it ``False``.
    jacobians : bool
        Whether jacobians are wanted, shape ``(n, nqoi, nvars)``.
    hessians : bool
        Whether hessians are wanted, shape ``(n, nvars, nvars)``. Only
        meaningful for ``nqoi == 1``.
    jvp_vecs : Array, optional
        Shape ``(nvars, n)``, one direction per sample. Requests a
        jacobian-vector product, returned in QoI space as ``(nqoi, n)``.
        The vector is an input to the solve, not something applied to a
        returned jacobian.
    hvp_vecs : Array, optional
        Shape ``(nvars, n)``, one direction per sample. Requests a
        Hessian-vector product, returned in parameter space as
        ``(nvars, n)``. With ``hvp_weights`` this is the weighted
        (adjoint-Hessian) form.
    hvp_weights : Array, optional
        Shape ``(nqoi, 1)``. Present for a weighted Hessian-vector
        product, absent for the plain one, which requires ``nqoi == 1``.
    """

    values: bool = True
    jacobians: bool = False
    hessians: bool = False
    jvp_vecs: Optional[Array] = None
    hvp_vecs: Optional[Array] = None
    hvp_weights: Optional[Array] = None

    def __post_init__(self) -> None:
        if self.hvp_weights is not None and self.hvp_vecs is None:
            raise ValueError(
                "hvp_weights given without hvp_vecs: weights only apply to "
                "a Hessian-vector product"
            )
        if not self.wants_anything():
            raise ValueError("a request must ask for at least one quantity")

    def wants_anything(self) -> bool:
        """Whether this request asks for any quantity at all."""
        return (
            self.values
            or self.jacobians
            or self.hessians
            or self.jvp_vecs is not None
            or self.hvp_vecs is not None
        )

    def wants_jvp(self) -> bool:
        """Whether a jacobian-vector product was asked for."""
        return self.jvp_vecs is not None

    def wants_hvp(self) -> bool:
        """Whether a Hessian-vector product was asked for."""
        return self.hvp_vecs is not None

    def is_weighted_hvp(self) -> bool:
        """Whether the Hessian-vector product carries QoI weights."""
        return self.hvp_vecs is not None and self.hvp_weights is not None

    @classmethod
    def values_only(cls) -> "Request[Array]":
        """The ordinary forward evaluation."""
        return cls()


@dataclass(frozen=True)
class Decoded(Generic[Array]):
    """One task's output, turned back into arrays.

    What :meth:`MarshallerProtocol.values` returns. A record rather than
    a tuple because a task may decode into values *and* derivatives, and
    a growing tuple is where positional-unpacking bugs come from.

    **Entry ``k`` of every array describes ``indices[k]``.** That
    positional correspondence is what makes this record usable, and it
    is the marshaller's to uphold: nothing downstream can detect a
    marshaller that returned its values in a different order from its
    own indices, because the result would be the right shape, the right
    size, and wrong. ``__post_init__`` checks the counts, which catches
    truncation but not permutation.

    ``indices`` are the columns of the submitted batch these arrays
    belong to, which may be a **subset** of the task's own indices: a
    task where some samples decoded and others did not returns the
    subset, and the evaluator records the remainder as failed.

    Attributes
    ----------
    values : Array
        Shape ``(nqoi, len(indices))``.
    indices : Sequence[int]
        Which columns of the submitted batch decoded.
    jacobians : Array, optional
        Shape ``(len(indices), nqoi, nvars)``, or ``None``.
    hessians : Array, optional
        Shape ``(len(indices), nvars, nvars)``, or ``None``. Only
        meaningful for ``nqoi == 1``.
    jvps : Array, optional
        Jacobian-vector products, shape ``(nqoi, len(indices))`` --
        **QoI space**.
    wall_time : float, optional
        How long the job took, **as reported by the job itself**, when
        its output says so. A solver that writes its own runtime knows
        better than anything wrapping it: the wrapper's figure includes
        process startup, input staging, filesystem sync, and on a
        scheduler the whole queue wait. Where this is present the
        evaluator prefers it and records the outcome's time source as
        ``JOB`` rather than ``WRAPPER``.

        ``None`` when the output carries no such figure, which is the
        common case and leaves the wrapper's measurement in place.
    hvps : Array, optional
        Hessian-vector products, shape ``(len(indices), nvars)`` --
        sample-**first**, unlike ``values`` and ``jvps``, because the
        batch form is scalar-implicit and carries no nqoi axis --
        **parameter space**. Carries the weighted (adjoint-Hessian)
        form too: those are the same tensor contracted differently, and
        the bundle itself converts between them with ``resolved_hvp`` /
        ``resolved_whvp`` rather than treating them as unrelated.

    Notes
    -----
    The two directional fields are split by *output space* rather than
    lumped behind a tag, so a field's shape never depends on a
    discriminator a consumer has to read first. They are also never
    derived from ``jacobians`` or ``hessians``: the vector is an input
    to the solve, so these are what the job computed.

    ``values`` is empty with shape ``(nqoi, 0)`` for a task that was
    asked only for a derivative. A task decodes only the quantities its
    request asked of it, so a marshaller that splits one request across
    several tasks returns several ``Decoded`` records covering the same
    indices with different fields populated; the evaluator merges them
    by index.
    """

    values: Array
    indices: Sequence[int]
    jacobians: Optional[Array] = None
    hessians: Optional[Array] = None
    jvps: Optional[Array] = None
    hvps: Optional[Array] = None
    hvp_weights: Optional[Array] = None
    wall_time: Optional[float] = None

    def __post_init__(self) -> None:
        # Positional correspondence is the whole contract of this
        # record, and it is invisible when broken: a marshaller whose
        # values came back in a different order from its own indices
        # would produce a result that is the right shape, the right
        # size, and wrong. Checking what can be checked cheaply is
        # worth more here than elsewhere for exactly that reason.
        n = len(self.indices)
        if len(set(self.indices)) != n:
            raise ValueError(
                f"indices must be unique, got {list(self.indices)}"
            )
        # Which axis indexes the sample. Not uniform, and not the
        # intuitive grouping: hvps is sample-first like jacobians and
        # hessians, because its batch form is scalar-implicit --
        # (n, nvars), with no nqoi axis. Only jvps follows the
        # sample-last convention that values uses.
        for name, axis in (
            ("values", 1),
            ("jacobians", 0),
            ("hessians", 0),
            ("jvps", 1),
            ("hvps", 0),
        ):
            array = getattr(self, name)
            if array is None:
                continue
            if name == "values" and int(array.shape[1]) == 0 and n > 0:
                # A decode-nothing task: succeeded, nothing decoded.
                continue
            if int(array.shape[axis]) != n:
                raise ValueError(
                    f"{name} covers {int(array.shape[axis])} samples but "
                    f"indices names {n}; entry k of each must describe "
                    "indices[k]"
                )

    def nsamples(self) -> int:
        """How many samples decoded."""
        return len(self.indices)


@dataclass(frozen=True)
class EvalProgress:
    """How a submitted batch is getting on, without waiting for it.

    What a decision point sees when it asks about work in flight.

    Carries **cost so far**, which is the whole point of asking. A
    wrapper accumulating cost around ``collect`` would match the
    existing tracked-model idiom, but it only ever sees what ``collect``
    returned -- and by then the compute is already spent. A budget cap
    needs the figure while the batch is still running, which is the
    overrun-after-the-fact failure this design exists to prevent. What
    the cap *is*, and whether to refuse or truncate, stays with the
    caller; what belongs here is the number it needs, in time to act.

    Attributes
    ----------
    nsucceeded : int
        Samples that have returned a value so far.
    nfailed : int
        Samples known to have failed, timed out, or been cancelled.
    noutstanding : int
        Samples neither returned nor failed yet.
    cost : Cost
        What the batch has spent so far, including failures.
    elapsed_seconds : float
        Wall-clock seconds since the batch was submitted. Distinct from
        ``cost.wall_clock``, which measures only spans where a job was
        actually running: a batch throttled behind a full queue has
        elapsed time that no job's span covers.
    """

    nsucceeded: int
    nfailed: int
    noutstanding: int
    cost: Cost
    elapsed_seconds: float

    def __post_init__(self) -> None:
        for name, value in (
            ("nsucceeded", self.nsucceeded),
            ("nfailed", self.nfailed),
            ("noutstanding", self.noutstanding),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if not isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0.0:
            raise ValueError(
                "elapsed_seconds must be finite and non-negative, got "
                f"{self.elapsed_seconds}"
            )

    def elapsed(self) -> float:
        """Wall-clock seconds since the batch was submitted."""
        return self.elapsed_seconds

    def nsubmitted(self) -> int:
        """How many samples the batch started with."""
        return self.nsucceeded + self.nfailed + self.noutstanding

    def is_complete(self) -> bool:
        """Whether collecting would return immediately."""
        return self.noutstanding == 0

    def fraction_returned(self) -> float:
        """Share of the batch that has come back, one way or another.

        One for an empty batch: nothing is outstanding, so nothing is
        being waited on.
        """
        submitted = self.nsubmitted()
        if submitted == 0:
            return 1.0
        return (self.nsucceeded + self.nfailed) / submitted


@dataclass(frozen=True)
class EvalResult(Generic[Array]):
    """Values that came back, and which samples produced them.

    ``values`` holds one column per *succeeded* sample, so it is
    narrower than the submitted batch whenever anything failed.
    ``succeeded``, ``failed`` and ``cancelled`` index into the submitted
    batch, which is what lets a caller line results up across models and
    keep only the samples every model returned.

    ``failed`` and ``cancelled`` are separate arrays rather than one,
    for the reason :class:`JobStatus` gives: only one of them is safe to
    resubmit.

    Attributes
    ----------
    values : Array
        Shape ``(nqoi, len(succeeded))``, ordered to match
        ``succeeded``.
    succeeded : Array
        Indices into the submitted samples that produced a value.
    failed : Array
        Indices whose job ran and did not produce one. Evidence about
        those parameter points.
    cancelled : Array
        Indices stopped or never started, including timeouts. Says
        nothing about the parameter point.
    statuses : Mapping[int, JobStatus]
        Per-index outcome, for a caller that needs to tell a timeout
        from a cancellation without re-deriving it.
    cost : Cost
        What this portion of the batch spent, including its failures.
    jacobians : Array, optional
        Shape ``(len(succeeded), nqoi, nvars)``.
    hessians : Array, optional
        Shape ``(len(succeeded), nvars, nvars)``; ``nqoi == 1`` only.
    jvps : Array, optional
        Jacobian-vector products, shape ``(nqoi, len(succeeded))`` --
        QoI space.
    hvps : Array, optional
        Hessian-vector products, shape ``(len(succeeded), nvars)`` --
        sample-**first**, unlike ``values`` and ``jvps``, because the
        batch form is scalar-implicit and carries no nqoi axis --
        parameter space.
    hvp_weights : Array, optional
        The QoI weights ``hvps`` was contracted with, or ``None`` for a
        plain (unweighted) product.

        Shape ``(nqoi, 1)`` -- **one weight vector for the whole
        submission**, not one per sample. That is inherited from the
        derivative bundle, whose weighted batch signature specifies "one
        weight vector applied to every sample", rather than a choice
        made here. Per-sample weightings are meaningful, and a caller
        who needs them submits once per distinct weight vector. Nothing
        in this record forecloses widening it later: the field is a
        single array either way.

        One field holds both forms because they are contractions of the
        same Hessian tensor, in the same shape -- the derivative bundle
        itself treats them as convertible rather than as separate
        capabilities. But they are not the same number: a plain product
        requires ``nqoi == 1``, while a weighted one is
        ``(sum_i w_i H_i) @ v`` for any ``nqoi``, and the two coincide
        only when ``nqoi == 1`` and ``w = [1]``.

        So the weights travel with the result rather than being implied
        by the request that produced it. A result outlives its request
        as soon as it is stored, reloaded, or passed on, and at that
        point "weighted, and by what?" must be answerable from the
        record alone.
    derivative_failures : Mapping[str, Array]
        Indices that produced a value but whose *derivative* did not,
        keyed by quantity name (``"jacobians"``, ``"hvps"``, ...).

        Non-empty only where a marshaller splits one request across
        several tasks. **Success is per quantity**: if the value task
        succeeds and the jacobian task fails for the same sample, that
        sample appears in ``succeeded`` with its value and is listed
        here under ``"jacobians"``. A caller doing forward UQ is
        unaffected; one doing gradient-based work checks this. Failing
        the whole sample instead would discard a perfectly good value
        because a gradient diverged.

    Notes
    -----
    Every derivative field is ``None`` unless it was asked for in the
    :class:`Request` *and* the marshaller advertised the capability.
    Absence is ``None``, never a missing attribute. Nothing here is
    reconstructed from anything else: a ``jvp`` or ``hvp`` is what the
    job computed with the vector as an input to its solve, never a
    contraction applied to a returned jacobian, and no hessian is
    materialized in order to produce one.

    These are fields from the outset rather than a later addition
    because the record is frozen and sits at a boundary: an external
    solver that computes its own derivatives is exactly the case that
    most needs non-blocking evaluation, and routing one through an
    evaluator must not silently cost it its gradient.
    """

    values: Array
    succeeded: Array
    failed: Array
    cancelled: Array
    cost: Cost
    statuses: dict[int, JobStatus] = field(default_factory=dict)
    jacobians: Optional[Array] = None
    hessians: Optional[Array] = None
    jvps: Optional[Array] = None
    hvps: Optional[Array] = None
    hvp_weights: Optional[Array] = None
    derivative_failures: dict[str, Array] = field(default_factory=dict)

    def nsucceeded(self) -> int:
        """Number of samples that produced a value."""
        return int(self.succeeded.shape[0])

    def nfailed(self) -> int:
        """Number of samples whose job ran and produced nothing."""
        return int(self.failed.shape[0])

    def ncancelled(self) -> int:
        """Number of samples stopped or never started."""
        return int(self.cancelled.shape[0])

    def nreturned(self) -> int:
        """Total samples accounted for by this result."""
        return self.nsucceeded() + self.nfailed() + self.ncancelled()
