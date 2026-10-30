"""Refine an adaptive sparse grid to a sample budget, interruptibly.

:func:`refine_with_model` alternates ``step_samples`` and
``step_values`` until the grid holds at least the budgeted number of
unique samples. It can be interrupted while the model evaluates, and
called again to carry on.

The model may raise between the two calls: an
:class:`~pyapprox.interface.evaluation.external_batch.ExternalBatchModel`
behind a ``CachedObjective`` raises ``PendingEvaluations`` whenever a
batch must be evaluated outside pyapprox. The fitter is then waiting
for that batch's values, the same idle point it holds between the two
calls in any run, so nothing is half done. Calling again finishes that
batch through ``pending_samples`` before requesting a new one, so the
interruption costs no step and no evaluation::

    while True:
        try:
            refine_with_model(fitter, model, max_nsamples)
            break
        except PendingEvaluations as pending:
            ...  # evaluate pending.batch_path(), then import the results

The function keeps no state of its own; the fitter does. A new process
rebuilds the fitter and calls it from the start, and the store behind
the model serves every value already evaluated.
"""

from pyapprox.interface.functions.protocols.function import FunctionProtocol
from pyapprox.surrogates.sparsegrids.adaptive_fitter import (
    SingleFidelityAdaptiveSparseGridFitter,
)
from pyapprox.util.backends.protocols import Array


def refine_with_model(
    fitter: SingleFidelityAdaptiveSparseGridFitter[Array],
    model: FunctionProtocol[Array],
    max_nsamples: int,
) -> int:
    """Refine until the grid holds at least ``max_nsamples`` samples.

    The last batch is always evaluated in full, so the count can exceed
    the budget by up to one batch. Refinement also stops when no
    admissible candidate remains.

    Parameters
    ----------
    fitter : SingleFidelityAdaptiveSparseGridFitter[Array]
        The fitter to refine. May be part way through a step: a batch
        awaiting values is finished first.
    model : FunctionProtocol[Array]
        Evaluates a batch, shape (nvars, n) to (nqoi, n). Anything it
        raises propagates, leaving the fitter waiting for that batch.
    max_nsamples : int
        Stop once the grid holds at least this many unique samples.

    Returns
    -------
    int
        The number of unique samples the grid holds.
    """
    if not isinstance(model, FunctionProtocol):
        raise TypeError(
            f"model must satisfy FunctionProtocol, got {type(model).__name__}"
        )
    if max_nsamples < 1:
        raise ValueError(f"max_nsamples must be positive, got {max_nsamples}")
    while True:
        samples = fitter.pending_samples()
        if samples is None:
            # nselected() is 0 only before the first step, when the grid
            # has no samples to count.
            if fitter.nselected() > 0 and _nsamples(fitter) >= max_nsamples:
                break
            samples = fitter.step_samples()
            if samples is None:
                break
        fitter.step_values(model(samples))
    return _nsamples(fitter)


def _nsamples(fitter: SingleFidelityAdaptiveSparseGridFitter[Array]) -> int:
    if fitter.nselected() == 0:
        return 0
    return int(fitter.get_samples("all").shape[1])
