"""Multifidelity Monte Carlo (MFMC) estimator.

MFMC is a specific ACV estimator with analytical optimal allocation
based on model correlations and costs.
"""

from typing import Generic

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.stats.protocols import StatisticWithDiscrepancyProtocol
from pyapprox.typing.stats.estimators.acv.base import ACVEstimator


class MFMCEstimator(ACVEstimator[Array], Generic[Array]):
    """Multifidelity Monte Carlo (MFMC) estimator.

    MFMC uses the structure where all low-fidelity models are coupled
    with the high-fidelity model, with analytical optimal allocation:
        r_m = sqrt((c_0/c_m) * (rho_m^2 - rho_{m+1}^2) / (1 - rho_1^2))

    The estimator is:
        Q_MFMC = Q_0 + sum_m alpha_m * (mu_m - Q_m)

    where alpha_m = rho_0m * sigma_0 / sigma_m (optimal weights).

    Parameters
    ----------
    stat : StatisticWithDiscrepancyProtocol[Array]
        Statistic to estimate.
    costs : Array
        Cost per sample for each model. Shape: (nmodels,)
        Models should be ordered by decreasing cost.
    bkd : Backend[Array]
        Computational backend.

    Notes
    -----
    MFMC requires models to be ordered by decreasing correlation with
    the high-fidelity model (|rho_1| >= |rho_2| >= ...).

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.stats import MultiOutputMean
    >>> bkd = NumpyBkd()
    >>> stat = MultiOutputMean(nqoi=1, bkd=bkd)
    >>> # Ordered by decreasing correlation
    >>> cov = bkd.asarray([[1.0, 0.95, 0.8], [0.95, 1.0, 0.9], [0.8, 0.9, 1.0]])
    >>> stat.set_pilot_quantities(cov)
    >>> costs = bkd.asarray([100.0, 10.0, 1.0])
    >>> mfmc = MFMCEstimator(stat, costs, bkd)
    >>> mfmc.allocate_samples(target_cost=1000.0)
    """

    def __init__(
        self,
        stat: StatisticWithDiscrepancyProtocol[Array],
        costs: Array,
        bkd: Backend[Array],
    ):
        nmodels = costs.shape[0]
        # MFMC uses all-to-HF coupling
        recursion_index = np.zeros(nmodels - 1, dtype=np.int64)
        super().__init__(stat, costs, bkd, bkd.asarray(recursion_index))

    def _compute_mfmc_ratios(self) -> np.ndarray:
        """Compute MFMC optimal sample ratios analytically.

        Returns
        -------
        ratios : ndarray
            Sample ratios n_m / n_0 for each model.
        """
        bkd = self._bkd
        cov = self._stat.cov()
        cov_np = bkd.to_numpy(cov)
        costs_np = bkd.to_numpy(self._costs)

        nmodels = self.nmodels()
        nqoi = self._stat.nqoi()

        # Extract correlations
        if nqoi == 1:
            rhos = np.zeros(nmodels)
            for m in range(nmodels):
                rhos[m] = cov_np[0, m] / np.sqrt(cov_np[0, 0] * cov_np[m, m])
        else:
            rhos = np.zeros(nmodels)
            var0 = np.trace(cov_np[:nqoi, :nqoi]) / nqoi
            for m in range(nmodels):
                var_m = np.trace(cov_np[m*nqoi:(m+1)*nqoi, m*nqoi:(m+1)*nqoi]) / nqoi
                cov_0m = np.trace(cov_np[:nqoi, m*nqoi:(m+1)*nqoi]) / nqoi
                rhos[m] = cov_0m / np.sqrt(var0 * var_m) if var0 > 0 and var_m > 0 else 0

        rhos[0] = 1.0  # Self-correlation

        # Compute ratios using MFMC formula
        rho_sq = rhos ** 2
        ratios = np.ones(nmodels)

        c0 = costs_np[0]

        for m in range(1, nmodels):
            # rho_{m+1}^2 (or 0 if last model)
            rho_sq_next = rho_sq[m + 1] if m + 1 < nmodels else 0

            # Numerator: rho_m^2 - rho_{m+1}^2
            numer = rho_sq[m] - rho_sq_next

            # Denominator: 1 - rho_1^2
            denom = 1 - rho_sq[1]

            if denom > 1e-10 and numer > 0:
                r_m = np.sqrt((c0 / costs_np[m]) * numer / denom)
                ratios[m] = max(r_m, 1.0)
            else:
                ratios[m] = 1.0

        # Ensure monotonicity: r_1 >= r_2 >= ... >= r_M
        for m in range(2, nmodels):
            ratios[m] = max(ratios[m], ratios[m - 1])

        return ratios

    def _compute_sample_ratios(self) -> np.ndarray:
        """Override to use MFMC-specific formula."""
        return self._compute_mfmc_ratios()

    def _compute_optimal_weights(self) -> Array:
        """Compute MFMC optimal weights.

        For MFMC: alpha_m = rho_0m * sigma_0 / sigma_m
        """
        bkd = self._bkd
        cov = self._stat.cov()
        cov_np = bkd.to_numpy(cov)

        nmodels = self.nmodels()
        nqoi = self._stat.nqoi()

        if nqoi == 1:
            sigma0 = np.sqrt(cov_np[0, 0])
            weights = np.zeros(nmodels - 1)

            for m in range(1, nmodels):
                sigma_m = np.sqrt(cov_np[m, m])
                rho_0m = cov_np[0, m] / (sigma0 * sigma_m)
                weights[m - 1] = rho_0m * sigma0 / sigma_m
        else:
            # Multi-QoI: diagonal weights
            weights = np.zeros(nmodels - 1)
            var0 = np.trace(cov_np[:nqoi, :nqoi]) / nqoi
            sigma0 = np.sqrt(var0)

            for m in range(1, nmodels):
                var_m = np.trace(cov_np[m*nqoi:(m+1)*nqoi, m*nqoi:(m+1)*nqoi]) / nqoi
                sigma_m = np.sqrt(var_m)
                cov_0m = np.trace(cov_np[:nqoi, m*nqoi:(m+1)*nqoi]) / nqoi
                rho_0m = cov_0m / (sigma0 * sigma_m) if sigma0 > 0 and sigma_m > 0 else 0
                weights[m - 1] = rho_0m * sigma0 / sigma_m if sigma_m > 0 else 0

        return bkd.asarray(weights)

    def __repr__(self) -> str:
        nsamples_str = "not allocated"
        if self._nsamples is not None:
            ns = self._bkd.to_numpy(self._nsamples)
            nsamples_str = str(list(ns.astype(int)))
        return f"MFMCEstimator(nmodels={self.nmodels()}, nsamples={nsamples_str})"
