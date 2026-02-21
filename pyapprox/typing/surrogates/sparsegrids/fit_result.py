"""Result dataclasses for sparse grid fitters."""

from dataclasses import dataclass
from typing import Generic

from pyapprox.typing.util.backends.protocols import Array


@dataclass(frozen=True)
class IsotropicSparseGridFitResult(Generic[Array]):
    """Result of fitting an isotropic sparse grid.

    Attributes
    ----------
    surrogate : object
        The fitted CombinationSurrogate.
    indices : Array
        Subspace indices, shape (nvars, nsubspaces).
    coefficients : Array
        Smolyak coefficients, shape (nsubspaces,).
    nsamples : int
        Total number of unique samples used.
    """

    surrogate: object
    indices: Array
    coefficients: Array
    nsamples: int


@dataclass(frozen=True)
class AdaptiveSparseGridFitResult(Generic[Array]):
    """Result of fitting an adaptive sparse grid.

    Attributes
    ----------
    surrogate : object
        The fitted CombinationSurrogate.
    indices : Array
        Selected subspace indices, shape (nvars, nselected).
    coefficients : Array
        Smolyak coefficients, shape (nselected,).
    nsamples : int
        Total number of unique samples used.
    error : float
        Final error estimate.
    nsteps : int
        Number of refinement steps performed.
    converged : bool
        Whether the fitter converged to tolerance.
    """

    surrogate: object
    indices: Array
    coefficients: Array
    nsamples: int
    error: float
    nsteps: int
    converged: bool
