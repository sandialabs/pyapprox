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

    def _compute_mfmc_ratios(self) -> Array:
        """Compute MFMC optimal sample ratios analytically.

        Returns
        -------
        ratios : Array
            Sample ratios n_m / n_0 for each model.
        """
        bkd = self._bkd
        cov = self._stat.cov()
        nmodels = self.nmodels()
        nqoi = self._stat.nqoi()

        # Extract variances
        if nqoi == 1:
            variances = bkd.get_diagonal(cov)
        else:
            traces = []
            for m in range(nmodels):
                block = cov[m * nqoi : (m + 1) * nqoi, m * nqoi : (m + 1) * nqoi]
                traces.append(bkd.trace(block) / nqoi)
            variances = bkd.stack(traces)

        # Extract correlations
        if nqoi == 1:
            rhos = cov[0, :] / bkd.sqrt(cov[0, 0] * variances)
        else:
            corrs = []
            var0 = variances[0]
            for m in range(nmodels):
                cov_0m = cov[:nqoi, m * nqoi : (m + 1) * nqoi]
                var_m = variances[m]
                denom = bkd.sqrt(var0 * var_m)
                corr = bkd.where(
                    denom > 1e-14,
                    bkd.trace(cov_0m) / nqoi / denom,
                    bkd.asarray(0.0)
                )
                corrs.append(corr)
            rhos = bkd.stack(corrs)

        # Set self-correlation to 1.0
        rhos = bkd.concatenate([bkd.asarray([1.0]), rhos[1:]])

        # Compute ratios using MFMC formula
        rho_sq = rhos ** 2
        c0 = self._costs[0]

        # Build ratios for each model
        ratios_list = [bkd.asarray([1.0])]  # r_0 = 1

        for m in range(1, nmodels):
            # rho_{m+1}^2 (or 0 if last model)
            if m + 1 < nmodels:
                rho_sq_next = rho_sq[m + 1]
            else:
                rho_sq_next = bkd.asarray(0.0)

            # Numerator: rho_m^2 - rho_{m+1}^2
            numer = rho_sq[m] - rho_sq_next

            # Denominator: 1 - rho_1^2
            denom = 1 - rho_sq[1]

            # Compute ratio where valid
            valid = (denom > 1e-10) & (numer > 0)
            safe_denom = bkd.where(valid, denom, bkd.ones_like(denom))
            r_m = bkd.sqrt((c0 / self._costs[m]) * numer / safe_denom)
            r_m = bkd.maximum(r_m, bkd.asarray(1.0))
            r_m = bkd.where(valid, r_m, bkd.asarray(1.0))

            ratios_list.append(bkd.reshape(r_m, (1,)))

        ratios = bkd.concatenate(ratios_list)

        # Ensure monotonicity: r_1 >= r_2 >= ... >= r_M
        # Use cumulative maximum
        for m in range(2, nmodels):
            ratios = bkd.concatenate([
                ratios[:m],
                bkd.reshape(bkd.maximum(ratios[m], ratios[m - 1]), (1,)),
                ratios[m + 1:] if m + 1 < nmodels else bkd.asarray([])
            ])

        return ratios

    def _compute_sample_ratios(self) -> Array:
        """Override to use MFMC-specific formula."""
        return self._compute_mfmc_ratios()

    def _compute_optimal_weights(self) -> Array:
        """Compute MFMC optimal weights.

        For MFMC: alpha_m = rho_0m * sigma_0 / sigma_m
        """
        bkd = self._bkd
        cov = self._stat.cov()
        nmodels = self.nmodels()
        nqoi = self._stat.nqoi()

        if nqoi == 1:
            sigma0 = bkd.sqrt(cov[0, 0])
            weights_list = []

            for m in range(1, nmodels):
                sigma_m = bkd.sqrt(cov[m, m])
                rho_0m = cov[0, m] / (sigma0 * sigma_m)
                weight = rho_0m * sigma0 / sigma_m
                weights_list.append(bkd.reshape(weight, (1,)))

            weights = bkd.concatenate(weights_list)
        else:
            # Multi-QoI: diagonal weights
            var0 = bkd.trace(cov[:nqoi, :nqoi]) / nqoi
            sigma0 = bkd.sqrt(var0)

            weights_list = []
            for m in range(1, nmodels):
                var_m = bkd.trace(cov[m * nqoi : (m + 1) * nqoi,
                                      m * nqoi : (m + 1) * nqoi]) / nqoi
                sigma_m = bkd.sqrt(var_m)
                cov_0m = bkd.trace(cov[:nqoi, m * nqoi : (m + 1) * nqoi]) / nqoi
                denom = sigma0 * sigma_m
                rho_0m = bkd.where(
                    denom > 1e-14,
                    cov_0m / denom,
                    bkd.asarray(0.0)
                )
                weight = bkd.where(
                    sigma_m > 1e-14,
                    rho_0m * sigma0 / sigma_m,
                    bkd.asarray(0.0)
                )
                weights_list.append(bkd.reshape(weight, (1,)))

            weights = bkd.concatenate(weights_list)

        return weights

    def __repr__(self) -> str:
        nsamples_str = "not allocated"
        if self._nsamples is not None:
            ns = self._bkd.to_numpy(self._nsamples)
            nsamples_str = str(list(ns.astype(int)))
        return f"MFMCEstimator(nmodels={self.nmodels()}, nsamples={nsamples_str})"
