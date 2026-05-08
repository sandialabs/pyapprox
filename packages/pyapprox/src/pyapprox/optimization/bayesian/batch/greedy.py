"""Greedy batch selection strategies for Bayesian optimization."""

from typing import Callable, Generic, Optional

from pyapprox.optimization.bayesian.protocols import (
    AcquisitionContext,
    AcquisitionFunctionProtocol,
    AcquisitionOptimizerProtocol,
    BODomainProtocol,
)
from pyapprox.util.backends.protocols import Array


class KrigingBeliever(Generic[Array]):
    """Kriging Believer batch selection strategy.

    Selects batch points sequentially. After each point, the surrogate
    is updated with the predicted mean at that point (fantasized
    observation) via the context factory.
    """

    def select_batch(
        self,
        batch_size: int,
        acquisition: AcquisitionFunctionProtocol[Array],
        context_factory: Callable[
            [Optional[Array]], AcquisitionContext[Array]
        ],
        acquisition_optimizer: AcquisitionOptimizerProtocol[Array],
        domain: BODomainProtocol[Array],
    ) -> Array:
        """Select a batch of points using Kriging Believer strategy.

        Parameters
        ----------
        batch_size : int
            Number of points to select.
        acquisition : AcquisitionFunctionProtocol[Array]
            Acquisition function to maximize.
        context_factory : Callable
            Creates AcquisitionContext given pending_X (or None).
            Handles fantasization (refitting GP with pending points
            using predicted means as observations).
        acquisition_optimizer : AcquisitionOptimizerProtocol[Array]
            Optimizer for acquisition function maximization.
        domain : BODomainProtocol[Array]
            Search domain.

        Returns
        -------
        Array
            Batch of selected points, shape (nvars, batch_size).
        """
        batch = []
        pending: Optional[Array] = None

        for _ in range(batch_size):
            ctx = context_factory(pending)
            x_new = acquisition_optimizer.maximize(acquisition, ctx, domain)
            batch.append(x_new)

            if pending is None:
                pending = x_new
            else:
                pending = ctx.bkd.hstack([pending, x_new])

        return ctx.bkd.hstack(batch)
