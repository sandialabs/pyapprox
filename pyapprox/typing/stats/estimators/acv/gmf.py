"""Generalized Multifidelity (GMF) estimator.

GMF uses the MFMC-style recursion where all low-fidelity models are coupled
directly with the high-fidelity model.
"""

from typing import Generic, Optional

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.stats.protocols import StatisticWithDiscrepancyProtocol
from pyapprox.typing.stats.estimators.acv.base import ACVEstimator


class GMFEstimator(ACVEstimator[Array], Generic[Array]):
    """Generalized Multifidelity (GMF) estimator.

    GMF uses the allocation structure where all low-fidelity models are
    coupled with the high-fidelity model:
        recursion_index = [0, 0, 0, ...]

    This is equivalent to MFMC but with optimized weights.

    The estimator is:
        Q_GMF = Q_0 + sum_m eta_m * (mu_m - Q_m)

    where all Q_m use shared samples with Q_0.

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
    >>> gmf = GMFEstimator(stat, costs, bkd)
    >>> gmf.allocate_samples(target_cost=100.0)
    """

    def __init__(
        self,
        stat: StatisticWithDiscrepancyProtocol[Array],
        costs: Array,
        bkd: Backend[Array],
    ):
        nmodels = costs.shape[0]
        # GMF uses MFMC recursion: all LF coupled with HF
        recursion_index = np.zeros(nmodels - 1, dtype=np.int64)
        super().__init__(stat, costs, bkd, bkd.asarray(recursion_index))

    def __repr__(self) -> str:
        nsamples_str = "not allocated"
        if self._nsamples is not None:
            ns = self._bkd.to_numpy(self._nsamples)
            nsamples_str = str(list(ns.astype(int)))
        return f"GMFEstimator(nmodels={self.nmodels()}, nsamples={nsamples_str})"
