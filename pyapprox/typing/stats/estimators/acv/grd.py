"""Generalized Recursive Difference (GRD) estimator.

GRD uses a telescoping structure where each model is coupled with the
previous model in a chain.
"""

from typing import Generic, Optional

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.stats.protocols import StatisticWithDiscrepancyProtocol
from pyapprox.typing.stats.estimators.acv.base import ACVEstimator
from pyapprox.typing.stats.allocation.matrices import get_allocation_matrix_grd


class GRDEstimator(ACVEstimator[Array], Generic[Array]):
    """Generalized Recursive Difference (GRD) estimator.

    GRD uses a telescoping structure where each model is coupled with
    the previous model in a chain. By default, uses MLMC-style recursion
    [0, 1, 2, ...] but can accept any valid recursion index.

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
    recursion_index : Array, optional
        The recursion index specifying which model each low-fidelity model
        is coupled with. Shape: (nmodels-1,). If None, defaults to MLMC-style
        recursion [0, 1, 2, ...].

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
        recursion_index: Optional[Array] = None,
    ):
        nmodels = costs.shape[0]
        if recursion_index is None:
            # Default: MLMC-style recursion (successive coupling)
            recursion_index = bkd.arange(nmodels - 1)
        super().__init__(stat, costs, bkd, recursion_index)

    def get_allocation_matrix(self) -> Array:
        """Return the GRD allocation matrix.

        GRD keeps even and odd columns separate (no merge step),
        creating disjoint sample sets.

        Returns
        -------
        Array
            Allocation matrix. Shape: (nmodels, 2*nmodels)
        """
        if self._allocation_mat is None:
            self._allocation_mat = get_allocation_matrix_grd(
                self.nmodels(), self._recursion_index, self._bkd
            )
        return self._allocation_mat

    def __repr__(self) -> str:
        nsamples_str = "not allocated"
        if self._nsamples is not None:
            ns = self._bkd.to_numpy(self._nsamples)
            nsamples_str = str(list(ns.astype(int)))
        return f"GRDEstimator(nmodels={self.nmodels()}, nsamples={nsamples_str})"
