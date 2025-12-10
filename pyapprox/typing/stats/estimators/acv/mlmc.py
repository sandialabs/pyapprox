"""Multilevel Monte Carlo (MLMC) estimator.

MLMC uses a telescoping sum of level differences with analytical
optimal allocation.
"""

from typing import Generic, List

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.stats.protocols import StatisticWithDiscrepancyProtocol
from pyapprox.typing.stats.estimators.acv.base import ACVEstimator


class MLMCEstimator(ACVEstimator[Array], Generic[Array]):
    """Multilevel Monte Carlo (MLMC) estimator.

    MLMC uses a telescoping sum structure:
        Q_MLMC = sum_{l=0}^{L} Y_l

    where Y_l = Q_l - Q_{l-1} (with Q_{-1} = 0).

    The optimal allocation minimizes variance for fixed cost:
        n_l propto sqrt(V_l / c_l)

    where V_l = Var(Y_l) and c_l = cost(Y_l).

    Parameters
    ----------
    stat : StatisticWithDiscrepancyProtocol[Array]
        Statistic to estimate.
    costs : Array
        Cost per sample for each level. Shape: (nlevels,)
        For level l > 0, this is cost of (Q_l, Q_{l-1}) pair.
    bkd : Backend[Array]
        Computational backend.

    Notes
    -----
    MLMC assumes a hierarchy of models with decreasing variance
    of differences: Var(Y_l) decreases with l.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.stats import MultiOutputMean
    >>> bkd = NumpyBkd()
    >>> stat = MultiOutputMean(nqoi=1, bkd=bkd)
    >>> # MLMC-like covariance
    >>> cov = bkd.asarray([[1.0, 0.99, 0.9], [0.99, 1.0, 0.99], [0.9, 0.99, 1.0]])
    >>> stat.set_pilot_quantities(cov)
    >>> # Costs increase with level
    >>> costs = bkd.asarray([1.0, 10.0, 100.0])
    >>> mlmc = MLMCEstimator(stat, costs, bkd)
    >>> mlmc.allocate_samples(target_cost=1000.0)
    """

    def __init__(
        self,
        stat: StatisticWithDiscrepancyProtocol[Array],
        costs: Array,
        bkd: Backend[Array],
    ):
        nmodels = costs.shape[0]
        # MLMC uses successive coupling
        recursion_index = np.arange(nmodels - 1, dtype=np.int64)
        super().__init__(stat, costs, bkd, bkd.asarray(recursion_index))

    def _compute_level_variances(self) -> np.ndarray:
        """Compute variance of level differences Y_l = Q_l - Q_{l-1}.

        Returns
        -------
        variances : ndarray
            Variance of Y_l for each level.
        """
        bkd = self._bkd
        cov = self._stat.cov()
        cov_np = bkd.to_numpy(cov)

        nmodels = self.nmodels()
        nqoi = self._stat.nqoi()

        variances = np.zeros(nmodels)

        for l in range(nmodels):
            if nqoi == 1:
                var_l = cov_np[l, l]
                if l == 0:
                    # Y_0 = Q_0
                    variances[l] = var_l
                else:
                    # Y_l = Q_l - Q_{l-1}
                    var_lm1 = cov_np[l - 1, l - 1]
                    cov_l_lm1 = cov_np[l, l - 1]
                    variances[l] = var_l + var_lm1 - 2 * cov_l_lm1
            else:
                var_l = np.trace(cov_np[l*nqoi:(l+1)*nqoi, l*nqoi:(l+1)*nqoi]) / nqoi
                if l == 0:
                    variances[l] = var_l
                else:
                    var_lm1 = np.trace(cov_np[(l-1)*nqoi:l*nqoi, (l-1)*nqoi:l*nqoi]) / nqoi
                    cov_l_lm1 = np.trace(cov_np[l*nqoi:(l+1)*nqoi, (l-1)*nqoi:l*nqoi]) / nqoi
                    variances[l] = var_l + var_lm1 - 2 * cov_l_lm1

        # Ensure positive
        variances = np.maximum(variances, 1e-10)

        return variances

    def _compute_sample_ratios(self) -> np.ndarray:
        """Compute MLMC optimal sample ratios.

        For MLMC: n_l propto sqrt(V_l / c_l)
        """
        bkd = self._bkd
        costs_np = bkd.to_numpy(self._costs)
        nmodels = self.nmodels()

        variances = self._compute_level_variances()

        # Compute raw ratios: sqrt(V_l / c_l)
        raw_ratios = np.sqrt(variances / costs_np)

        # Normalize to get n_l / n_0
        ratios = raw_ratios / raw_ratios[0]

        return ratios

    def __call__(self, values: List[Array]) -> Array:
        """Compute MLMC estimate.

        Q_MLMC = sum_l Y_l = Q_L + sum_{l<L} (Q_l - Q_{l+1})

        Parameters
        ----------
        values : List[Array]
            Model outputs at each level.

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
        nmodels = self.nmodels()
        nsamples_np = bkd.to_numpy(self.nsamples_per_model())

        # Compute level estimators
        # Y_0 = Q_0
        n0 = int(nsamples_np[0])
        Y = bkd.sum(values[0], axis=0) / n0

        # Y_l = Q_l - Q_{l-1} on shared samples
        for l in range(1, nmodels):
            nl = int(nsamples_np[l])
            n_shared = min(nl, int(nsamples_np[l - 1]))

            Q_l = bkd.sum(values[l][:n_shared], axis=0) / n_shared
            Q_lm1 = bkd.sum(values[l - 1][:n_shared], axis=0) / n_shared

            Y_l = Q_l - Q_lm1
            Y = Y + Y_l

        return Y

    def variance_reduction(self) -> float:
        """Return variance reduction factor.

        MLMC achieves variance reduction through level correlation.
        """
        variances = self._compute_level_variances()

        # Effective variance reduction
        # Compare sum of level variances to single-level variance
        total_var = np.sum(variances)
        hf_var = variances[-1]

        return total_var / hf_var if hf_var > 0 else 1.0

    def __repr__(self) -> str:
        nsamples_str = "not allocated"
        if self._nsamples is not None:
            ns = self._bkd.to_numpy(self._nsamples)
            nsamples_str = str(list(ns.astype(int)))
        return f"MLMCEstimator(nlevels={self.nmodels()}, nsamples={nsamples_str})"
