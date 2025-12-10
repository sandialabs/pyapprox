"""Base class for estimator implementations.

Provides common functionality for all estimator types.
"""

from abc import ABC, abstractmethod
from typing import Generic, List, Callable, Optional

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.stats.protocols import StatisticWithCovarianceProtocol


class AbstractEstimator(ABC, Generic[Array]):
    """Abstract base class for estimators.

    Provides common functionality for multifidelity Monte Carlo estimators.

    Parameters
    ----------
    stat : StatisticWithCovarianceProtocol[Array]
        Statistic to estimate.
    costs : Array
        Computational cost per sample for each model. Shape: (nmodels,)
    bkd : Backend[Array]
        Computational backend.

    Attributes
    ----------
    _stat : StatisticWithCovarianceProtocol[Array]
        The statistic being estimated.
    _costs : Array
        Model costs.
    _bkd : Backend[Array]
        Computational backend.
    _nsamples : Optional[Array]
        Allocated samples per model.
    """

    def __init__(
        self,
        stat: StatisticWithCovarianceProtocol[Array],
        costs: Array,
        bkd: Backend[Array],
    ):
        self._stat = stat
        self._costs = costs
        self._bkd = bkd
        self._nsamples: Optional[Array] = None

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nmodels(self) -> int:
        """Return number of models."""
        return self._costs.shape[0]

    def costs(self) -> Array:
        """Return computational costs per sample."""
        return self._costs

    def stat(self) -> StatisticWithCovarianceProtocol[Array]:
        """Return the statistic being estimated."""
        return self._stat

    def nsamples_per_model(self) -> Array:
        """Return allocated samples per model.

        Returns
        -------
        Array
            Number of samples. Shape: (nmodels,)

        Raises
        ------
        ValueError
            If samples have not been allocated.
        """
        if self._nsamples is None:
            raise ValueError(
                "Samples not allocated. Call allocate_samples() first."
            )
        return self._nsamples

    @abstractmethod
    def allocate_samples(self, target_cost: float) -> None:
        """Allocate samples to minimize variance for given cost.

        Parameters
        ----------
        target_cost : float
            Total computational budget.
        """
        ...

    @abstractmethod
    def generate_samples_per_model(
        self, rvs: Callable[[int], Array]
    ) -> List[Array]:
        """Generate samples for each model.

        Parameters
        ----------
        rvs : Callable[[int], Array]
            Random variable sampler. Takes nsamples, returns samples.
            Output shape: (nsamples, nvars)

        Returns
        -------
        List[Array]
            Samples for each model.
        """
        ...

    @abstractmethod
    def __call__(self, values: List[Array]) -> Array:
        """Compute the estimate from model evaluations.

        Parameters
        ----------
        values : List[Array]
            Model outputs. values[m] has shape (nsamples_m, nqoi)

        Returns
        -------
        Array
            Estimated statistic. Shape: (nstats,)
        """
        ...

    @abstractmethod
    def optimized_covariance(self) -> Array:
        """Return covariance of the optimized estimator.

        Returns
        -------
        Array
            Covariance matrix. Shape: (nstats, nstats)
        """
        ...

    def total_cost(self) -> float:
        """Return total computational cost of allocated samples.

        Returns
        -------
        float
            Total cost.
        """
        nsamples = self.nsamples_per_model()
        costs = self._bkd.to_numpy(self._costs)
        nsamples_np = self._bkd.to_numpy(nsamples)
        return float(np.sum(costs * nsamples_np))
