"""Multi-start wrapper around any bindable optimizer.

Runs an inner optimizer from several starting points and returns the
best result. The wrapper itself satisfies ``BindableOptimizerProtocol``
so it can be passed anywhere a single-start optimizer is accepted.
"""

from typing import Callable, Generic, Optional

import numpy as np

from pyapprox.interface.functions.protocols.objective import (
    ObjectiveProtocol,
)
from pyapprox.optimization.minimize.constraints.protocols import (
    SequenceOfConstraintProtocols,
)
from pyapprox.optimization.minimize.protocols import (
    BindableOptimizerProtocol,
)
from pyapprox.optimization.minimize.result_protocol import (
    OptimizerResultProtocol,
)
from pyapprox.util.backends.protocols import Array

# Draws one starting point. Receives the rng and the bound objective
# (for backend and nvars); returns shape (nvars, 1).
StartSampler = Callable[
    [np.random.Generator, ObjectiveProtocol[Array]], Array
]


def _uniform_in_bounds_sampler(
    rng: np.random.Generator,
    objective: ObjectiveProtocol[Array],
    bounds: Array,
) -> Array:
    bkd = objective.bkd()
    bounds_np = bkd.to_numpy(bounds)
    lb, ub = bounds_np[:, 0], bounds_np[:, 1]
    if not (np.all(np.isfinite(lb)) and np.all(np.isfinite(ub))):
        raise ValueError(
            "default start sampler requires finite bounds; provide a "
            "start_sampler for unbounded problems"
        )
    sample = rng.uniform(lb, ub)
    return bkd.reshape(bkd.asarray(sample), (-1, 1))


class MultiStartOptimizer(Generic[Array]):
    """Run a bindable optimizer from multiple starts, keep the best.

    Parameters
    ----------
    optimizer : BindableOptimizerProtocol[Array]
        Unbound template optimizer, cloned per start.
    nstarts : int
        Total number of starts. The caller-provided init guess is the
        first start; the remaining ``nstarts - 1`` come from
        ``start_sampler``.
    start_sampler : StartSampler, optional
        Callable ``(rng, objective) -> (nvars, 1)`` drawing one start.
        Defaults to uniform sampling within the bound box (requires
        finite bounds). Problems with additional constraints (e.g. a
        probability simplex) should supply a sampler that respects
        them.
    seed : int, optional
        Seed for the start-sampling rng. Each ``minimize`` call
        re-seeds, so repeated calls are deterministic.
    """

    def __init__(
        self,
        optimizer: BindableOptimizerProtocol[Array],
        nstarts: int,
        start_sampler: Optional[StartSampler[Array]] = None,
        seed: Optional[int] = None,
    ) -> None:
        if nstarts < 1:
            raise ValueError(f"nstarts must be >= 1, got {nstarts}")
        self._template = optimizer
        self._nstarts = nstarts
        self._start_sampler = start_sampler
        self._seed = seed
        self._objective: Optional[ObjectiveProtocol[Array]] = None
        self._bounds: Optional[Array] = None
        self._constraints: Optional[
            SequenceOfConstraintProtocols[Array]
        ] = None
        self._is_bound = False

    def bind(
        self,
        objective: ObjectiveProtocol[Array],
        bounds: Array,
        constraints: Optional[SequenceOfConstraintProtocols[Array]] = None,
    ) -> "MultiStartOptimizer[Array]":
        """Store the problem; inner optimizers are bound per start."""
        self._objective = objective
        self._bounds = bounds
        self._constraints = constraints
        self._is_bound = True
        return self

    def is_bound(self) -> bool:
        """Return True if bind() has been called."""
        return self._is_bound

    def copy(self) -> "MultiStartOptimizer[Array]":
        """Return an unbound copy with the same options."""
        return MultiStartOptimizer(
            self._template.copy(),
            self._nstarts,
            start_sampler=self._start_sampler,
            seed=self._seed,
        )

    def _draw_start(self, rng: np.random.Generator) -> Array:
        assert self._objective is not None and self._bounds is not None
        if self._start_sampler is not None:
            return self._start_sampler(rng, self._objective)
        return _uniform_in_bounds_sampler(
            rng, self._objective, self._bounds
        )

    def minimize(
        self, init_guess: Array
    ) -> OptimizerResultProtocol[Array]:
        """Run all starts and return the best result by value.

        Any error raised by an inner optimization propagates.
        """
        if not self._is_bound:
            raise RuntimeError("Optimizer not bound. Call bind() first.")
        assert self._objective is not None and self._bounds is not None

        rng = np.random.default_rng(self._seed)
        starts = [init_guess] + [
            self._draw_start(rng) for _ in range(self._nstarts - 1)
        ]

        best_result: Optional[OptimizerResultProtocol[Array]] = None
        best_value = np.inf
        for start in starts:
            inner = self._template.copy()
            inner.bind(self._objective, self._bounds, self._constraints)
            result = inner.minimize(start)
            value = result.fun()
            if value < best_value:
                best_value = value
                best_result = result
        assert best_result is not None
        return best_result
