"""Meeting the blocking world.

Most of this library calls a model and waits. An evaluator submits and
collects, which is the point of it -- but a great deal of existing code
takes an ``ObjectiveProtocol`` and calls it, and that code should be able
to drive a marshalled model without knowing what a dispatcher is.

``BlockingModel`` is that adapter. It submits, waits, and returns values,
raising if anything failed -- because a blocking return has nowhere to
report a partial result, and silently handing back a narrower array
would be worse than raising.

It satisfies ``ObjectiveProtocol`` rather than merely ``FunctionProtocol``
because the wrappers it exists to compose with -- parallel wrappers,
timing wrappers, derivative checkers -- all check for the derivatives
accessor at construction. A plain function shape would be rejected by
all of them.
"""

from typing import Generic, Optional

from pyapprox.interface.evaluation.protocols import EvaluatorProtocol
from pyapprox.interface.evaluation.records import (
    EvalResult,
    JobStatus,
    Request,
)
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.util.backends.protocols import Array, Backend


class EvaluationFailure(RuntimeError):
    """A blocking call could not return every value it was asked for.

    Raised rather than returning a narrower array, because a caller
    expecting ``(nqoi, nsamples)`` has nowhere to learn that it received
    fewer columns, and would silently misalign them against its own
    samples. Callers who can act on partial results should use the
    evaluator directly, where failure is a return value.
    """


class BlockingModel(Generic[Array]):
    """Presents an evaluator as an ordinary callable model.

    Parameters
    ----------
    evaluator : EvaluatorProtocol[Array]
        The model to wrap. Its dispatcher, marshaller and cost ledger
        are unchanged; this only changes how results are asked for.
    """

    def __init__(self, evaluator: EvaluatorProtocol[Array]) -> None:
        if not isinstance(evaluator, EvaluatorProtocol):
            raise TypeError(
                "evaluator must satisfy EvaluatorProtocol, got "
                f"{type(evaluator).__name__}"
            )
        self._evaluator = evaluator
        self._derivatives = _blocking_bundle(evaluator)

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._evaluator.bkd()

    def nvars(self) -> int:
        """Number of input variables the model takes."""
        return self._evaluator.nvars()

    def nqoi(self) -> int:
        """Number of quantities of interest the model returns."""
        return self._evaluator.nqoi()

    def evaluator(self) -> EvaluatorProtocol[Array]:
        """The wrapped evaluator, for a caller that wants the async path."""
        return self._evaluator

    def __call__(self, samples: Array, /) -> Array:
        """Evaluate ``samples``, waiting for every result.

        Raises :class:`EvaluationFailure` if any sample did not return,
        naming the columns and what happened to them.
        """
        result = self._evaluator.submit(samples).collect()
        _require_complete(result, int(samples.shape[1]))
        return result.values

    def derivatives(self) -> Derivatives[Array]:
        """Which derivative capabilities this model has.

        Mirrors the evaluator's bundle, with each capability wrapped so
        it submits and waits. So a marshalled model is inspected exactly
        like any other objective, and a caller that never intends to
        touch a dispatcher can still get its gradients.

        Every capability the evaluator advertises is mirrored, including
        the directional forms: a wrapped model that can serve a
        Hessian-vector product exposes one here rather than appearing to
        have lost it on the way through.
        """
        return self._derivatives


class _BlockingJacobianBatch(Generic[Array]):
    """Submits for jacobians and waits.

    A class rather than a closure so the bundle stays picklable, which
    matters as soon as this model is handed to a process pool.
    """

    def __init__(self, evaluator: EvaluatorProtocol[Array]) -> None:
        self._evaluator = evaluator

    def __call__(self, samples: Array) -> Array:
        result = self._evaluator.submit(
            samples, Request(values=False, jacobians=True)
        ).collect()
        _require_complete(result, int(samples.shape[1]))
        if result.jacobians is None:
            raise EvaluationFailure(
                "jacobians were requested but none were returned"
            )
        return result.jacobians


class _BlockingHessianBatch(Generic[Array]):
    """Submits for hessians and waits."""

    def __init__(self, evaluator: EvaluatorProtocol[Array]) -> None:
        self._evaluator = evaluator

    def __call__(self, samples: Array) -> Array:
        result = self._evaluator.submit(
            samples, Request(values=False, hessians=True)
        ).collect()
        _require_complete(result, int(samples.shape[1]))
        if result.hessians is None:
            raise EvaluationFailure(
                "hessians were requested but none were returned"
            )
        return result.hessians


class _BlockingJacobian(Generic[Array]):
    """Single-sample jacobian, served by submitting one column.

    A model that can produce a batch of jacobians can produce one, and
    optimizers and derivative checkers ask for the single-sample form.
    Deriving it here rather than requiring the marshaller to declare
    both keeps a marshaller from having to implement the same capability
    twice.

    The shape convention differs between the two, which is the whole
    reason this is not a pass-through: ``jacobian_batch`` returns
    ``(n, nqoi, nvars)``, while ``jacobian`` returns ``(nqoi, nvars)``
    for its single sample.
    """

    def __init__(self, evaluator: EvaluatorProtocol[Array]) -> None:
        self._batch = _BlockingJacobianBatch(evaluator)

    def __call__(self, sample: Array) -> Array:
        if sample.ndim != 2 or sample.shape[1] != 1:
            raise ValueError(
                "jacobian takes a single sample of shape (nvars, 1), got "
                f"{tuple(sample.shape)}"
            )
        return self._batch(sample)[0]


class _BlockingHessian(Generic[Array]):
    """Single-sample hessian, served by submitting one column."""

    def __init__(self, evaluator: EvaluatorProtocol[Array]) -> None:
        self._batch = _BlockingHessianBatch(evaluator)

    def __call__(self, sample: Array) -> Array:
        if sample.ndim != 2 or sample.shape[1] != 1:
            raise ValueError(
                "hessian takes a single sample of shape (nvars, 1), got "
                f"{tuple(sample.shape)}"
            )
        return self._batch(sample)[0]


class _BlockingJVP(Generic[Array]):
    """Submits for a jacobian-vector product and waits.

    The direction rides along in the request rather than being applied
    to a returned jacobian, because a tangent-linear solve takes the
    seed as an input. Contracting a materialized jacobian would compute
    nvars columns to use one.
    """

    def __init__(self, evaluator: EvaluatorProtocol[Array]) -> None:
        self._evaluator = evaluator

    def __call__(self, sample: Array, vec: Array) -> Array:
        _require_single(sample, "jvp", "sample")
        _require_single(vec, "jvp", "vec")
        result = self._evaluator.submit(
            sample, Request(values=False, jvp_vecs=vec)
        ).collect()
        _require_complete(result, 1)
        if result.jvps is None:
            raise EvaluationFailure(
                "a jacobian-vector product was requested but none was "
                "returned"
            )
        # jvps are (nqoi, n) -- sample-last -- and one sample was sent,
        # so this is already the (nqoi, 1) the bundle promises.
        return result.jvps


class _BlockingHVPBatch(Generic[Array]):
    """Submits for Hessian-vector products and waits.

    Serves both the plain and weighted forms: the two differ only by
    whether the request carries weights, and the evaluator returns the
    same ``hvps`` array either way. Splitting them into two classes
    would duplicate the submit-and-check body to vary one argument.
    """

    def __init__(
        self,
        evaluator: EvaluatorProtocol[Array],
        weighted: bool,
    ) -> None:
        self._evaluator = evaluator
        self._weighted = weighted

    def _submit(
        self, samples: Array, vecs: Array, weights: Optional[Array]
    ) -> Array:
        result = self._evaluator.submit(
            samples,
            Request(values=False, hvp_vecs=vecs, hvp_weights=weights),
        ).collect()
        _require_complete(result, int(samples.shape[1]))
        if result.hvps is None:
            raise EvaluationFailure(
                "a Hessian-vector product was requested but none was "
                "returned"
            )
        # hvps are (n, nvars) -- sample-first, unlike values and jvps.
        return result.hvps

    def __call__(
        self,
        samples: Array,
        vecs: Array,
        weights: Optional[Array] = None,
    ) -> Array:
        if self._weighted and weights is None:
            raise ValueError(
                "whvp_batch requires weights of shape (nqoi, 1)"
            )
        if not self._weighted and weights is not None:
            raise ValueError(
                "hvp_batch takes no weights; use whvp_batch for the "
                "weighted form"
            )
        return self._submit(samples, vecs, weights)


class _BlockingHVP(Generic[Array]):
    """Single-sample Hessian-vector product, plain or weighted.

    The shape convention flips between the batch and single forms,
    which is why this is not a pass-through: ``hvp_batch`` returns
    ``(n, nvars)`` with the sample first, while ``hvp`` returns
    ``(nvars, 1)`` for its one sample.
    """

    def __init__(
        self,
        evaluator: EvaluatorProtocol[Array],
        weighted: bool,
    ) -> None:
        self._batch = _BlockingHVPBatch(evaluator, weighted)
        self._weighted = weighted

    def __call__(
        self,
        sample: Array,
        vec: Array,
        weights: Optional[Array] = None,
    ) -> Array:
        name = "whvp" if self._weighted else "hvp"
        _require_single(sample, name, "sample")
        _require_single(vec, name, "vec")
        batch = (
            self._batch(sample, vec, weights)
            if self._weighted
            else self._batch(sample, vec)
        )
        return batch.T


def _require_single(array: Array, method: str, name: str) -> None:
    """Reject a batch where a single sample is required."""
    if array.ndim != 2 or array.shape[1] != 1:
        raise ValueError(
            f"{method} takes a single {name} of shape (nvars, 1), got "
            f"{tuple(array.shape)}"
        )


def _blocking_bundle(
    evaluator: EvaluatorProtocol[Array],
) -> Derivatives[Array]:
    """Mirror an evaluator's capabilities as blocking callables.

    Only the capabilities the evaluator actually advertises are
    populated, so ``None`` continues to mean absent rather than
    unimplemented -- construction-time branching, not runtime probing.
    """
    source = evaluator.derivatives()
    has_jacobian = source.jacobian_batch is not None
    has_hessian = source.hessian_batch is not None
    has_jvp = source.jvp is not None
    has_hvp = source.hvp_batch is not None
    has_whvp = source.whvp_batch is not None
    return Derivatives(
        jacobian=(
            _BlockingJacobian(evaluator) if has_jacobian else None
        ),
        jacobian_batch=(
            _BlockingJacobianBatch(evaluator) if has_jacobian else None
        ),
        hessian=(_BlockingHessian(evaluator) if has_hessian else None),
        hessian_batch=(
            _BlockingHessianBatch(evaluator) if has_hessian else None
        ),
        jvp=(_BlockingJVP(evaluator) if has_jvp else None),
        hvp=(
            _BlockingHVP(evaluator, weighted=False) if has_hvp else None
        ),
        hvp_batch=(
            _BlockingHVPBatch(evaluator, weighted=False)
            if has_hvp
            else None
        ),
        whvp=(
            _BlockingHVP(evaluator, weighted=True) if has_whvp else None
        ),
        whvp_batch=(
            _BlockingHVPBatch(evaluator, weighted=True)
            if has_whvp
            else None
        ),
    )


def _require_complete(result: EvalResult[Array], nsubmitted: int) -> None:
    """Raise unless every submitted sample came back.

    Names the columns and their statuses, since "some samples failed" is
    not actionable and "sample 7 diverged, sample 9 timed out" is.
    """
    nmissing = result.nfailed() + result.ncancelled()
    if nmissing == 0:
        return
    detail = ", ".join(
        f"{index}: {status.value}"
        for index, status in sorted(result.statuses.items())
        if status is not JobStatus.SUCCEEDED
    )
    raise EvaluationFailure(
        f"{nmissing} of {nsubmitted} samples did not return ({detail}). "
        "A blocking call cannot report a partial result; submit through "
        "the evaluator directly to receive failures as data."
    )


def blocking(evaluator: EvaluatorProtocol[Array]) -> BlockingModel[Array]:
    """Wrap an evaluator so it can be called like any other model."""
    return BlockingModel(evaluator)
