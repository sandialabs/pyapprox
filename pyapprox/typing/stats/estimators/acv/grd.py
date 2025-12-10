"""Generalized Recursive Difference (GRD) estimator.

GRD uses a telescoping structure where each model is coupled with the
previous model in a chain.
"""

from typing import Generic

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.stats.protocols import StatisticWithDiscrepancyProtocol
from pyapprox.typing.stats.estimators.acv.base import ACVEstimator


class GRDEstimator(ACVEstimator[Array], Generic[Array]):
    """Generalized Recursive Difference (GRD) estimator.

    GRD uses the MLMC-style recursion where each model is coupled with
    the previous model in a chain:
        recursion_index = [0, 1, 2, ...]

    The estimator uses telescoping differences:
        Q_GRD = Q_M + sum_{m=0}^{M-1} (Q_m - Q_{m+1})

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
    >>> # MLMC-like covariance (strong adjacent correlation)
    >>> cov = bkd.asarray([[1.0, 0.95, 0.8], [0.95, 1.0, 0.95], [0.8, 0.95, 1.0]])
    >>> stat.set_pilot_quantities(cov)
    >>> costs = bkd.asarray([100.0, 10.0, 1.0])
    >>> grd = GRDEstimator(stat, costs, bkd)
    >>> grd.allocate_samples(target_cost=1000.0)
    """

    def __init__(
        self,
        stat: StatisticWithDiscrepancyProtocol[Array],
        costs: Array,
        bkd: Backend[Array],
    ):
        nmodels = costs.shape[0]
        # GRD uses MLMC recursion: successive coupling
        recursion_index = np.arange(nmodels - 1, dtype=np.int64)
        super().__init__(stat, costs, bkd, bkd.asarray(recursion_index))

    def __repr__(self) -> str:
        nsamples_str = "not allocated"
        if self._nsamples is not None:
            ns = self._bkd.to_numpy(self._nsamples)
            nsamples_str = str(list(ns.astype(int)))
        return f"GRDEstimator(nmodels={self.nmodels()}, nsamples={nsamples_str})"
