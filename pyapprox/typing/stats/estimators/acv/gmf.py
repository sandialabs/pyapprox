"""Generalized Multifidelity (GMF) estimator.

GMF uses a flexible recursion structure specified by recursion_index.
"""

from typing import Generic, Optional

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.stats.protocols import StatisticWithDiscrepancyProtocol
from pyapprox.typing.stats.estimators.acv.base import ACVEstimator
from pyapprox.typing.stats.allocation.matrices import get_allocation_matrix_gmf


class GMFEstimator(ACVEstimator[Array], Generic[Array]):
    """Generalized Multifidelity (GMF) estimator.

    GMF uses a flexible allocation structure specified by recursion_index.
    Default is all LF models coupled with HF: recursion_index = [0, 0, 0, ...]

    The estimator is:
        Q_GMF = Q_0 + sum_m eta_m * (mu_m - Q_m)

    where Q_m use shared samples based on the recursion structure.

    Parameters
    ----------
    stat : StatisticWithDiscrepancyProtocol[Array]
        Statistic to estimate.
    costs : Array
        Cost per sample for each model. Shape: (nmodels,)
    bkd : Backend[Array]
        Computational backend.
    recursion_index : Array, optional
        Recursion index. Shape: (nmodels-1,)
        If None, defaults to [0, 0, ...] (all coupled with HF).

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
        recursion_index: Optional[Array] = None,
    ):
        nmodels = costs.shape[0]
        # Default: all LF coupled with HF
        if recursion_index is None:
            recursion_index = bkd.zeros((nmodels - 1,))
        super().__init__(stat, costs, bkd, recursion_index)

    def get_allocation_matrix(self) -> Array:
        """Return the GMF allocation matrix.

        GMF uses hierarchical inclusion (L-shaped pattern per model).

        Returns
        -------
        Array
            Allocation matrix. Shape: (nmodels, 2*nmodels)
        """
        if self._allocation_mat is None:
            self._allocation_mat = get_allocation_matrix_gmf(
                self.nmodels(), self._recursion_index, self._bkd
            )
        return self._allocation_mat

    def __repr__(self) -> str:
        nsamples_str = "not allocated"
        if self._nsamples is not None:
            ns = self._bkd.to_numpy(self._nsamples)
            nsamples_str = str(list(ns.astype(int)))
        return f"GMFEstimator(nmodels={self.nmodels()}, nsamples={nsamples_str})"
