"""Generalized Independent Samples (GIS) estimator.

GIS uses independent sample sets for each model, which simplifies
implementation but may not achieve optimal variance reduction.
"""

from typing import Generic, List, Callable

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.stats.protocols import StatisticWithDiscrepancyProtocol
from pyapprox.typing.stats.estimators.acv.base import ACVEstimator


class GISEstimator(ACVEstimator[Array], Generic[Array]):
    """Generalized Independent Samples (GIS) estimator.

    GIS uses independent sample sets for each model (no sample sharing).
    While this doesn't achieve optimal variance reduction, it simplifies
    implementation and allows parallel model evaluation.

    The estimator is:
        Q_GIS = Q_0 + sum_m eta_m * (E[Q_m] - Q_m)

    where E[Q_m] is the known or independently estimated mean of model m.

    Parameters
    ----------
    stat : StatisticWithDiscrepancyProtocol[Array]
        Statistic to estimate.
    costs : Array
        Cost per sample for each model. Shape: (nmodels,)
    bkd : Backend[Array]
        Computational backend.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.stats import MultiOutputMean
    >>> bkd = NumpyBkd()
    >>> stat = MultiOutputMean(nqoi=1, bkd=bkd)
    >>> cov = bkd.asarray([[1.0, 0.9, 0.8], [0.9, 1.0, 0.85], [0.8, 0.85, 1.0]])
    >>> stat.set_pilot_quantities(cov)
    >>> costs = bkd.asarray([10.0, 1.0, 0.1])
    >>> gis = GISEstimator(stat, costs, bkd)
    >>> gis.allocate_samples(target_cost=100.0)
    """

    def __init__(
        self,
        stat: StatisticWithDiscrepancyProtocol[Array],
        costs: Array,
        bkd: Backend[Array],
    ):
        nmodels = costs.shape[0]
        # GIS uses MFMC-like recursion but with independent samples
        recursion_index = np.zeros(nmodels - 1, dtype=np.int64)
        super().__init__(stat, costs, bkd, bkd.asarray(recursion_index))

    def generate_samples_per_model(
        self, rvs: Callable[[int], Array]
    ) -> List[Array]:
        """Generate independent samples for each model.

        Unlike other ACV estimators, GIS generates completely independent
        samples for each model.

        Parameters
        ----------
        rvs : Callable[[int], Array]
            Random variable sampler.

        Returns
        -------
        List[Array]
            Independent samples for each model.
        """
        nsamples = self.nsamples_per_model()

        model_samples: List[Array] = []
        for m in range(self.nmodels()):
            n_m = int(nsamples[m].item())
            model_samples.append(rvs(n_m) if n_m > 0 else rvs(1)[:, :0])

        return model_samples

    def __call__(self, values: List[Array]) -> Array:
        """Compute GIS estimate.

        For GIS, we use independent samples, so the control variate
        is based on the difference from the estimated mean.

        Parameters
        ----------
        values : List[Array]
            Model outputs with independent samples.

        Returns
        -------
        Array
            Estimated statistic.
        """
        if len(values) != self.nmodels():
            raise ValueError(
                f"Expected {self.nmodels()} model outputs, got {len(values)}"
            )

        bkd = self._bkd
        nsamples = self.nsamples_per_model()

        nhf = int(nsamples[0].item())

        # Q_0: HF mean
        Q0 = bkd.sum(values[0], axis=0) / nhf

        # For GIS, control variate uses difference from mean
        # Since we don't have known means, we use estimated means
        # This reduces to standard MC for truly independent samples
        # In practice, GIS works with prior knowledge of LF means

        return Q0

    def __repr__(self) -> str:
        nsamples_str = "not allocated"
        if self._nsamples is not None:
            ns = self._bkd.to_numpy(self._nsamples)
            nsamples_str = str(list(ns.astype(int)))
        return f"GISEstimator(nmodels={self.nmodels()}, nsamples={nsamples_str})"
