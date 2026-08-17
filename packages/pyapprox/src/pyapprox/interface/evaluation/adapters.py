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

from typing import Generic

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
