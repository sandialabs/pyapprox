"""Monte Carlo estimator (single model).

The simplest estimator that uses only the high-fidelity model.
"""

from typing import Generic, List, Callable, Any

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.stats.protocols import StatisticWithCovarianceProtocol
from pyapprox.typing.stats.estimators.base import AbstractEstimator
from pyapprox.typing.stats.estimators.bootstrap import BootstrapMixin


class MCEstimator(BootstrapMixin[Array], AbstractEstimator[Array], Generic[Array]):
    """Monte Carlo estimator using only the high-fidelity model.

    This is the simplest estimator that directly estimates the statistic
    using samples from only the high-fidelity (most expensive) model.

    The estimator variance is:
        Var(Q_MC) = Var(Q_0) / n

    where n is the number of high-fidelity samples.

    Parameters
    ----------
    stat : StatisticWithCovarianceProtocol[Array]
        Statistic to estimate.
    costs : Array
        Cost per sample for the high-fidelity model. Shape: (1,) or scalar.
    bkd : Backend[Array]
        Computational backend.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.stats import MultiOutputMean
    >>> bkd = NumpyBkd()
    >>> stat = MultiOutputMean(nqoi=1, bkd=bkd)
    >>> # Set pilot covariance (variance = 1.0)
    >>> stat.set_pilot_quantities(bkd.asarray([[1.0]]))
    >>> costs = bkd.asarray([10.0])  # HF cost
    >>> mc = MCEstimator(stat, costs, bkd)
    >>> mc.allocate_samples(target_cost=100.0)
    >>> mc.nsamples_per_model()
    array([10])
    """

    def __init__(
        self,
        stat: StatisticWithCovarianceProtocol[Array],
        costs: Array,
        bkd: Backend[Array],
    ):
        # Ensure costs is 1D with single element
        costs_np = bkd.to_numpy(costs)
        if costs_np.ndim == 0:
            costs = bkd.asarray([float(costs_np)])
        elif costs_np.shape[0] > 1:
            # Take only HF cost
            costs = bkd.asarray([float(costs_np[0])])

        super().__init__(stat, costs, bkd)

    def allocate_samples(self, target_cost: float) -> None:
        """Allocate all budget to high-fidelity model.

        Parameters
        ----------
        target_cost : float
            Total computational budget.
        """
        hf_cost = float(self._bkd.to_numpy(self._costs)[0])
        nhf = int(target_cost / hf_cost)
        nhf = max(nhf, self._stat.min_nsamples())

        self._nsamples = self._bkd.asarray([nhf], dtype=self._bkd.int64_dtype())

    def generate_samples_per_model(
        self, rvs: Callable[[int], Array]
    ) -> List[Array]:
        """Generate samples for the high-fidelity model.

        Parameters
        ----------
        rvs : Callable[[int], Array]
            Random variable sampler.

        Returns
        -------
        List[Array]
            Single-element list with HF samples.
        """
        nsamples = self.nsamples_per_model()
        nhf = int(self._bkd.to_numpy(nsamples)[0])
        return [rvs(nhf)]

    def __call__(self, values: List[Array]) -> Array:
        """Compute Monte Carlo estimate.

        Parameters
        ----------
        values : List[Array]
            Single-element list with HF model outputs.
            values[0] has shape (nsamples, nqoi)

        Returns
        -------
        Array
            Estimated statistic. Shape: (nstats,)
        """
        if len(values) != 1:
            raise ValueError(
                f"MCEstimator expects 1 model output, got {len(values)}"
            )

        return self._stat.sample_estimate(values[0])

    def optimized_covariance(self) -> Array:
        """Return covariance of the Monte Carlo estimator.

        Returns
        -------
        Array
            Covariance matrix. Shape: (nstats, nstats)
        """
        nsamples = self.nsamples_per_model()
        nhf = int(self._bkd.to_numpy(nsamples)[0])
        return self._stat.high_fidelity_estimator_covariance(nhf)

    def variance_reduction(self) -> float:
        """Return variance reduction factor (always 1.0 for MC).

        Returns
        -------
        float
            Variance reduction = 1.0 (no reduction for single-model MC).
        """
        return 1.0

    def _estimate_with_weights(
        self, values_per_model: List[Array], weights: Any
    ) -> Array:
        """Compute estimate (weights ignored for MC).

        Parameters
        ----------
        values_per_model : List[Array]
            Model values.
        weights : Any
            Ignored for MC estimator.

        Returns
        -------
        Array
            Estimated statistic.
        """
        return self(values_per_model)

    def __repr__(self) -> str:
        nsamples_str = "not allocated"
        if self._nsamples is not None:
            nsamples_str = str(int(self._bkd.to_numpy(self._nsamples)[0]))
        return f"MCEstimator(nsamples={nsamples_str})"
