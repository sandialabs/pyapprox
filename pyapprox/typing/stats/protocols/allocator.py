"""Allocator protocol for sample allocation strategies.

Allocators determine how to distribute computational budget across models
to minimize estimator variance.
"""

from typing import Protocol, Generic, Tuple, runtime_checkable

from pyapprox.typing.util.backends.protocols import Array, Backend


@runtime_checkable
class AllocatorProtocol(Protocol, Generic[Array]):
    """Protocol for sample allocation strategies.

    An allocator computes the optimal number of samples per model
    given a target cost and model covariance structure.

    Type Parameters
    ---------------
    Array : TypeVar
        Array type (e.g., numpy.ndarray, torch.Tensor).
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def allocate(
        self, target_cost: float, costs: Array, cov: Array
    ) -> Tuple[Array, Array]:
        """Allocate samples to minimize variance.

        Parameters
        ----------
        target_cost : float
            Total computational budget.
        costs : Array
            Cost per sample for each model. Shape: (nmodels,)
        cov : Array
            Covariance matrix from pilot study. Shape varies by statistic.

        Returns
        -------
        nsamples : Array
            Optimal samples per model. Shape: (nmodels,)
        weights : Array
            Optimal control variate weights. Shape depends on allocator.
        """
        ...

    def variance(
        self, nsamples: Array, costs: Array, cov: Array
    ) -> Array:
        """Compute estimator variance for given allocation.

        Parameters
        ----------
        nsamples : Array
            Samples per model. Shape: (nmodels,)
        costs : Array
            Cost per sample. Shape: (nmodels,)
        cov : Array
            Covariance matrix. Shape varies by statistic.

        Returns
        -------
        Array
            Estimator variance. Shape: (nstats,) or (nstats, nstats)
        """
        ...
