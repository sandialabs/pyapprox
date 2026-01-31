"""Multifidelity Monte Carlo (MFMC) estimator.

MFMC is a specific ACV estimator with analytical optimal allocation
based on model correlations and costs.
"""

from typing import Generic, Tuple

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.stats.protocols import StatisticWithDiscrepancyProtocol
from pyapprox.typing.stats.estimators.acv.base import ACVEstimator
from pyapprox.typing.stats.allocation.matrices import get_allocation_matrix_gmf


class MFMCEstimator(ACVEstimator[Array], Generic[Array]):
    """Multifidelity Monte Carlo (MFMC) estimator.

    MFMC uses successive coupling where each model is coupled with
    the previous model in a chain, with analytical optimal allocation:
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
        # MFMC uses successive coupling: model m coupled with model m-1
        # This gives nested sample sets where each model evaluates on
        # all samples used by higher-fidelity models
        # recursion_index = [0, 1, 2, ...] for nmodels-1 elements
        recursion_index = bkd.arange(nmodels - 1)
        super().__init__(stat, costs, bkd, recursion_index)

    def get_allocation_matrix(self) -> Array:
        """Return the MFMC allocation matrix.

        MFMC uses the GMF allocation structure with successive coupling
        (recursion_index = [0, 1, 2, ...]).

        Returns
        -------
        Array
            Allocation matrix. Shape: (nmodels, 2*nmodels)
        """
        if self._allocation_mat is None:
            # MFMC uses GMF allocation with its recursion index
            self._allocation_mat = get_allocation_matrix_gmf(
                self.nmodels(), self._recursion_index, self._bkd
            )
        return self._allocation_mat

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

    def _native_ratios_to_npartition_ratios(self, model_ratios: Array) -> Array:
        """Convert MFMC model ratios to partition ratios.

        Parameters
        ----------
        model_ratios : Array
            Sample ratios [r_1, ..., r_{M-1}] where n_m = r_m * n_0.
            Shape: (nmodels-1,)

        Returns
        -------
        partition_ratios : Array
            Partition ratios [r_1-1, r_2-r_1, ...]. Shape: (nmodels-1,)
        """
        bkd = self._bkd
        # First partition ratio: r_1 - 1
        first = bkd.reshape(model_ratios[0] - 1, (1,))
        # Remaining partition ratios: differences
        if model_ratios.shape[0] > 1:
            rest = bkd.diff(model_ratios)
            return bkd.concatenate([first, rest])
        return first

    def _get_rsquared_mfmc(self, nsample_ratios: Array) -> Array:
        """Compute r^2 for MFMC variance reduction.

        Matches legacy _get_rsquared_mfmc from _optim.py:64-101.

        Parameters
        ----------
        nsample_ratios : Array
            Sample ratios [r_1, ..., r_{M-1}] where n_m = r_m * n_0.
            Shape: (nmodels-1,)

        Returns
        -------
        rsquared : Array
            The value r^2 for variance reduction.
        """
        bkd = self._bkd
        cov = self._stat.cov()
        nmodels = self.nmodels()

        # First term: (r_1 - 1) / r_1 * cov[0,1]^2 / (cov[0,0] * cov[1,1])
        rsquared = (
            (nsample_ratios[0] - 1)
            / nsample_ratios[0]
            * cov[0, 1]
            / (cov[0, 0] * cov[1, 1])
            * cov[0, 1]
        )

        # Subsequent terms
        for ii in range(1, nmodels - 1):
            p1 = (nsample_ratios[ii] - nsample_ratios[ii - 1]) / (
                nsample_ratios[ii] * nsample_ratios[ii - 1]
            )
            p1 = p1 * (
                cov[0, ii + 1] / (cov[0, 0] * cov[ii + 1, ii + 1]) * cov[0, ii + 1]
            )
            rsquared = rsquared + p1

        return rsquared

    def allocate_samples_analytical(
        self, target_cost: float
    ) -> Tuple[Array, Array]:
        """Compute MFMC allocation analytically, matching legacy exactly.

        Implements Algorithm 2 from Peherstorfer et al. 2016.

        Parameters
        ----------
        target_cost : float
            Total computational budget.

        Returns
        -------
        nsample_ratios : Array
            Sample ratios n_m / n_0 for m = 1, ..., M-1. Shape: (nmodels-1,)
        log_variance : Array
            Log of estimator variance.
        """
        bkd = self._bkd
        cov = self._stat.cov()
        nmodels = self.nmodels()

        # Convert covariance to correlation
        variances = bkd.get_diagonal(cov)
        std_devs = bkd.sqrt(variances)
        corr = cov / bkd.outer(std_devs, std_devs)

        # Compute sample ratios r_m for all models (including HF)
        r_list = []
        for ii in range(nmodels - 1):
            # For models 0 to M-2: use difference of squared correlations
            num = self._costs[0] * (corr[0, ii] ** 2 - corr[0, ii + 1] ** 2)
            den = self._costs[ii] * (1 - corr[0, 1] ** 2)
            r_list.append(bkd.sqrt(num / den))

        # Last model: use rho_{M-1}^2 (no rho_{M})
        num = self._costs[0] * corr[0, -1] ** 2
        den = self._costs[-1] * (1 - corr[0, 1] ** 2)
        r_list.append(bkd.sqrt(num / den))
        r = bkd.stack(r_list)

        # Compute nhf_samples
        nhf_samples = target_cost / bkd.dot(self._costs, r)

        # Model ratios (exclude HF model r_0)
        nsample_ratios = r[1:]

        # Compute variance reduction and log variance
        gamma = 1 - self._get_rsquared_mfmc(nsample_ratios)
        log_variance = bkd.log(gamma) + bkd.log(cov[0, 0]) - bkd.log(nhf_samples)

        return nsample_ratios, log_variance

    def __repr__(self) -> str:
        nsamples_str = "not allocated"
        if self._nsamples is not None:
            ns = self._bkd.to_numpy(self._nsamples)
            nsamples_str = str(list(ns.astype(int)))
        return f"MFMCEstimator(nmodels={self.nmodels()}, nsamples={nsamples_str})"
