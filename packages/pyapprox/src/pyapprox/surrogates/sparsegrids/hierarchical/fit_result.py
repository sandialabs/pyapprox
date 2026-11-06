"""Result dataclass for the hierarchical sparse grid fitters."""

from dataclasses import dataclass
from typing import Generic

from pyapprox.surrogates.sparsegrids.hierarchical.hierarchical_surrogate import (
    HierarchicalSurrogate,
)
from pyapprox.util.backends.protocols import Array


@dataclass(frozen=True)
class HierarchicalSparseGridFitResult(Generic[Array]):
    """Result of fitting a hierarchical (locally adaptive) sparse grid.

    The surrogate sums hierarchical surpluses directly, so unlike the
    combination-technique results there are no Smolyak coefficients.

    Attributes
    ----------
    surrogate : HierarchicalSurrogate[Array]
        The fitted surrogate.
    indices : Array
        Promoted subspace indices, shape (nvars, npromoted).
    nsamples : int
        Total number of samples evaluated.
    error : float
        Final error estimate.
    nsteps : int
        Number of refinement steps performed.
    converged : bool
        Whether the fitter converged to tolerance.
    """

    surrogate: HierarchicalSurrogate[Array]
    indices: Array
    nsamples: int
    error: float
    nsteps: int
    converged: bool
