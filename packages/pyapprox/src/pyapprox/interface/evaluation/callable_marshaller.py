"""In-memory marshalling: the model is a Python callable.

The simplest marshaller, and the on-ramp for every model that already
exists. There is no file format and no working directory -- a task
carries the samples it covers, and the payload *is* the returned array.
That is why dispatch carries a ``Payload`` type at all: for an
in-process job the return value is the answer, and a lone ``Task`` would
be the argument half of a call with the return half missing.

**Grouping is a constructor argument, and defaults to unbounded.** Most
models in this repo are vectorized -- ``f(samples)`` handles every column
at once -- so the default puts a whole batch in one task and calls the
function once. Fanning a batch into one task per sample instead would
turn a single call into ``nsamples`` calls, four orders of magnitude more
dispatch operations for a fast model, and would make this on-ramp a trap
rather than a convenience.

Some models are not vectorized: ``__call__`` takes a batch but loops
internally, one sample at a time. Grouping is useless to those, and worse,
it hides them -- a whole batch in one task means progress reports nothing
until everything is done, and a process pool can only ever occupy one
worker. Passing ``samples_per_task=1`` splits them, which is what buys
per-sample progress and real parallelism. The caller chooses, because only
the caller knows which kind of model they have.

**Indices are local to the submitted batch**, as everywhere else here.
This marshaller does not know what a sample *is* beyond a column; giving
it a notion of identity would be inventing one the caller already has.

**When not to use this.** Paired with the inline dispatcher, wrapping a
fast in-process function buys little over calling it: same thread, same
blocking, plus dispatch overhead. Call the function.

What the wrapping buys, in rough order of how much it matters:

- **failure as a return value** -- a function that raises on one sample
  of two hundred yields the other 199 values and reports the one, where
  a bare call yields a traceback and nothing. Doing that by hand means a
  per-sample ``try``/``except`` loop, which gives up vectorization to
  get it;
- **cost with provenance**, accumulating into a ledger that can span
  several models;
- **one surface over many machines** -- the same calling code drives an
  in-process model, a process pool, or a scheduler, and swapping is a
  constructor argument;
- **something to compose with.** Caching, budget caps and retry are
  wrappers, and the completion hook is where durable recording attaches.
  None of that can wrap a bare callable, because resume needs to know
  which columns are already done and a callable has no such notion.

The first three are available today. The fourth describes wrappers that
are designed but not yet written, so treat it as the direction rather
than a feature.
"""

from dataclasses import dataclass, field
from typing import Callable, Generic, Optional, Sequence

from pyapprox.interface.evaluation.protocols import MarshalError
from pyapprox.interface.evaluation.records import (
    Decoded,
    Outcome,
    Request,
)
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.util.backends.protocols import Array, Backend

#: Effectively no limit on samples per task. Large enough that the
#: evaluator's grouping rule is always decided by the other term, without
#: pretending an unbounded integer exists.
UNBOUNDED_SAMPLES_PER_TASK = 1 << 30


@dataclass(frozen=True)
class CallableTask(Generic[Array]):
    """A call to make, and which columns of the batch it covers.

    Carries the samples themselves rather than a reference into the
    submitted array, so the task is self-contained and survives crossing
    a process boundary.
    """

    indices: Sequence[int]
    samples: Array
    request: Request[Array] = field(default_factory=Request)


@dataclass(frozen=True)
class CallablePayload(Generic[Array]):
    """What one in-process call produced.

    A record rather than a bare array because one call may answer for
    several quantities at once -- which is the point of a fused code,
    where the value and the jacobian fall out of the same solve.
    """

    values: Optional[Array] = None
    jacobians: Optional[Array] = None
    hessians: Optional[Array] = None
    jvps: Optional[Array] = None
    hvps: Optional[Array] = None


class CallableMarshaller(Generic[Array]):
    """Marshals samples to and from an in-process callable.

    Parameters
    ----------
    fn : Callable[[Array], Array]
        The model. Takes ``(nvars, n)`` and returns ``(nqoi, n)``.
    bkd : Backend[Array]
        Backend used to build values and index arrays.
    nvars : int
        Number of input variables.
    nqoi : int
        Number of quantities of interest.
    samples_per_task : int, optional
        Most samples to put in one task. Defaults to unbounded, which
        calls ``fn`` once per batch and is right for a vectorized model.
        Pass ``1`` for a model whose ``__call__`` loops internally, to
        get per-sample progress and let a pool use more than one worker.
    derivatives : Derivatives[Array], optional
        Which derivative capabilities ``fn`` has. Defaults to none. The
        callables are invoked inside the task, so what they cost is
        measured and charged like any other work.
    ncores : int, optional
        How many cores one call occupies. Defaults to 1. Set it for a
        model that parallelizes internally, since that is a property of
        the wrapped code rather than of the machine.
    """

    def __init__(
        self,
        fn: Callable[[Array], Array],
        bkd: Backend[Array],
        nvars: int,
        nqoi: int,
        samples_per_task: int = UNBOUNDED_SAMPLES_PER_TASK,
        derivatives: Optional[Derivatives[Array]] = None,
        ncores: int = 1,
    ) -> None:
        if not callable(fn):
            raise TypeError(f"fn must be callable, got {type(fn).__name__}")
        if nvars < 1:
            raise ValueError(f"nvars must be >= 1, got {nvars}")
        if nqoi < 1:
            raise ValueError(f"nqoi must be >= 1, got {nqoi}")
        if samples_per_task < 1:
            raise ValueError(
                f"samples_per_task must be >= 1, got {samples_per_task}"
            )
        if ncores < 1:
            raise ValueError(f"ncores must be >= 1, got {ncores}")
        self._fn = fn
        self._bkd = bkd
        self._nvars = nvars
        self._nqoi = nqoi
        self._samples_per_task = samples_per_task
        self._ncores = ncores
        self._derivatives = (
            Derivatives.none() if derivatives is None else derivatives
        )

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def nvars(self) -> int:
        """Number of input variables."""
        return self._nvars

    def nqoi(self) -> int:
        """Number of quantities of interest."""
        return self._nqoi

    def ncores(self) -> int:
        """How many cores one call occupies."""
        return self._ncores

    def max_samples_per_task(self) -> int:
        """Most samples this marshaller puts in one task."""
        return self._samples_per_task

    def derivatives(self) -> Derivatives[Array]:
        """Which derivative capabilities the wrapped function has."""
        return self._derivatives

    def tasks(
        self,
        samples: Array,
        indices: Sequence[int],
        request: Request[Array],
    ) -> Sequence[CallableTask[Array]]:
        """Build one task covering ``samples``.

        A single task regardless of what was asked for: an in-process
        callable answers every requested quantity in one invocation, so
        this is the fused case. A marshaller wrapping a code with
        separate entry points would return several tasks here instead,
        and neither shape needs the protocol to change.
        """
        if samples.shape[1] != len(indices):
            raise ValueError(
                f"got {samples.shape[1]} columns for {len(indices)} "
                "indices; they must agree"
            )
        return [
            CallableTask(
                indices=tuple(indices),
                samples=samples,
                request=_narrow(request, indices),
            )
        ]

    def run(self, task: CallableTask[Array]) -> CallablePayload[Array]:
        """Execute one task. This is what a dispatcher is given.

        A bound method rather than a closure, so the marshaller stays
        picklable -- which a process pool requires.
        """
        request = task.request
        derivs = self._derivatives
        values = self._fn(task.samples) if request.values else None

        jacobians = None
        if request.jacobians:
            jacobians = _require(derivs.jacobian_batch, "jacobian_batch")(
                task.samples
            )

        hessians = None
        if request.hessians:
            hessians = _require(derivs.hessian_batch, "hessian_batch")(
                task.samples
            )

        jvps = None
        if request.jvp_vecs is not None:
            jvps = _require(derivs.jvp, "jvp")(
                task.samples, request.jvp_vecs
            )

        hvps = None
        if request.hvp_vecs is not None:
            if request.hvp_weights is not None:
                hvps = _require(derivs.whvp_batch, "whvp_batch")(
                    task.samples, request.hvp_vecs, request.hvp_weights
                )
            else:
                hvps = _require(derivs.hvp_batch, "hvp_batch")(
                    task.samples, request.hvp_vecs
                )

        return CallablePayload(
            values=values,
            jacobians=jacobians,
            hessians=hessians,
            jvps=jvps,
            hvps=hvps,
        )

    def values(
        self, outcome: Outcome[CallableTask[Array], CallablePayload[Array]]
    ) -> Decoded[Array]:
        """Turn a finished call back into arrays.

        There is no parsing here -- the payload is already arrays -- so
        the only way to fail is a shape the caller cannot have meant.
        Checking is still worth it: a function returning the wrong shape
        otherwise corrupts a batch silently, and the failure surfaces
        much later somewhere unrelated.
        """
        payload = outcome.payload
        if payload is None:
            raise MarshalError(
                f"job for indices {list(outcome.indices)} produced no payload"
            )
        nsamples = len(outcome.indices)
        values = payload.values
        if values is None:
            # A request that asked for no values decodes none. The
            # indices stay populated, which marks these samples
            # succeeded-with-nothing-decoded rather than failed.
            values = self._bkd.zeros((self._nqoi, 0))
        elif values.shape != (self._nqoi, nsamples):
            raise MarshalError(
                f"expected values of shape ({self._nqoi}, {nsamples}), got "
                f"{tuple(values.shape)}"
            )
        return Decoded(
            values=values,
            indices=outcome.indices,
            jacobians=payload.jacobians,
            hessians=payload.hessians,
            jvps=payload.jvps,
            hvps=payload.hvps,
            # Carried from the request so the decoded record says
            # whether its hvps are weighted, without needing the request
            # that produced them.
            hvp_weights=outcome.task.request.hvp_weights,
        )

    def release(
        self, outcome: Outcome[CallableTask[Array], CallablePayload[Array]]
    ) -> None:
        """Nothing to release: in-process work holds no external resource."""
        return None


def _narrow(
    request: Request[Array], indices: Sequence[int]
) -> Request[Array]:
    """Restrict a request's per-sample vectors to the columns of a task.

    ``jvp_vecs`` and ``hvp_vecs`` carry one direction per sample, so a
    task covering a slice of the batch must receive the matching slice
    of the vectors. Passing the whole array instead asks a one-sample
    task to contract against every sample's direction, which produces
    an answer of the wrong width -- caught by ``Decoded`` validation,
    but only after the model has run.

    ``hvp_weights`` is deliberately not narrowed: it is one vector for
    the whole submission rather than per-sample data.
    """
    if request.jvp_vecs is None and request.hvp_vecs is None:
        return request
    columns = list(indices)
    return Request(
        values=request.values,
        jacobians=request.jacobians,
        hessians=request.hessians,
        jvp_vecs=(
            None
            if request.jvp_vecs is None
            else request.jvp_vecs[:, columns]
        ),
        hvp_vecs=(
            None
            if request.hvp_vecs is None
            else request.hvp_vecs[:, columns]
        ),
        hvp_weights=request.hvp_weights,
    )


def _require(
    capability: Optional[Callable[..., Array]], name: str
) -> Callable[..., Array]:
    """Fail loudly if a request got past validation without capability.

    The evaluator validates a request against the marshaller's bundle
    before dispatching, so reaching this is a bug rather than a user
    error -- but a clear message beats ``None is not callable`` raised
    inside a worker process.
    """
    if capability is None:
        raise MarshalError(
            f"request asked for {name}, which this model does not provide"
        )
    return capability
