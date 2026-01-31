"""Multilevel Monte Carlo (MLMC) estimator.

MLMC uses a telescoping sum of level differences with analytical
optimal allocation.
"""

from typing import Generic, List, Tuple

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.stats.protocols import StatisticWithDiscrepancyProtocol
from pyapprox.typing.stats.estimators.acv.base import ACVEstimator
from pyapprox.typing.stats.allocation.matrices import get_allocation_matrix_mlmc


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
        # MLMC uses successive coupling: model m coupled with model m-1
        recursion_index = bkd.arange(nmodels - 1)
        super().__init__(stat, costs, bkd, recursion_index)

    def get_allocation_matrix(self) -> Array:
        """Return the MLMC allocation matrix.

        MLMC uses a banded structure where adjacent levels share samples.

        Returns
        -------
        Array
            Allocation matrix. Shape: (nmodels, 2*nmodels)
        """
        if self._allocation_mat is None:
            self._allocation_mat = get_allocation_matrix_mlmc(
                self.nmodels(), self._bkd
            )
        return self._allocation_mat

    def _compute_level_variances(self) -> Array:
        """Compute variance of level differences for MLMC.

        Following legacy convention (model 0 = HF, model M-1 = coarsest):
        - var_deltas[l] = Var(f_l - f_{l+1}) for l = 0, ..., M-2
        - var_deltas[M-1] = Var(f_{M-1}) (coarsest level, standalone)

        Returns
        -------
        variances : Array
            Variance of each level difference. Shape: (nmodels,)
        """
        bkd = self._bkd
        cov = self._stat.cov()
        nmodels = self.nmodels()

        var_list = []
        # Levels 0 to M-2: Var(f_l - f_{l+1})
        for l in range(nmodels - 1):
            diff_var = cov[l, l] + cov[l + 1, l + 1] - 2 * cov[l, l + 1]
            var_list.append(bkd.reshape(diff_var, (1,)))
        # Level M-1: Var(f_{M-1})
        var_list.append(bkd.reshape(cov[-1, -1], (1,)))

        variances = bkd.concatenate(var_list)

        # Ensure positive
        variances = bkd.maximum(variances, bkd.asarray(1e-10) * bkd.ones_like(variances))

        return variances

    def _compute_cost_deltas(self) -> Array:
        """Compute cost of each level difference for MLMC.

        Following legacy convention:
        - cost_deltas[l] = cost[l] + cost[l+1] for l = 0, ..., M-2
          (evaluating both f_l and f_{l+1} for the difference)
        - cost_deltas[M-1] = cost[M-1] (just the coarsest model)

        Returns
        -------
        cost_deltas : Array
            Cost for each level difference. Shape: (nmodels,)
        """
        bkd = self._bkd
        nmodels = self.nmodels()
        cost_list = []
        # Levels 0 to M-2: cost of evaluating both models
        for l in range(nmodels - 1):
            cost_list.append(bkd.reshape(self._costs[l] + self._costs[l + 1], (1,)))
        # Level M-1: just the coarsest model
        cost_list.append(bkd.reshape(self._costs[-1], (1,)))
        return bkd.concatenate(cost_list)

    def _compute_sample_ratios(self) -> Array:
        """Compute MLMC optimal sample ratios using correct cost deltas.

        For MLMC:
        - n_l^delta = lambda * sqrt(V_l / c_l^delta)
        - Model ratios: n_l = (n_{l-1}^delta + n_l^delta) / n_0 for l > 0

        Returns model ratios n_m / n_0 for m = 1, ..., M-1.
        """
        bkd = self._bkd
        variances = self._compute_level_variances()
        cost_deltas = self._compute_cost_deltas()

        # Compute optimal samples per delta (proportional)
        var_cost_ratios = variances / cost_deltas
        nsamples_per_delta_prop = bkd.sqrt(var_cost_ratios)

        # Model ratios: for l > 0, n_l = (n_{l-1}^delta + n_l^delta) / n_0^delta
        n0_delta = nsamples_per_delta_prop[0]
        ratios_list = [bkd.asarray([1.0])]  # r_0 = 1
        for l in range(1, self.nmodels()):
            n_l = (
                nsamples_per_delta_prop[l - 1] + nsamples_per_delta_prop[l]
            ) / n0_delta
            ratios_list.append(bkd.reshape(n_l, (1,)))

        return bkd.concatenate(ratios_list)

    def _native_ratios_to_npartition_ratios(self, model_ratios: Array) -> Array:
        """Convert MLMC model ratios to partition ratios.

        Uses the MLMC-specific recurrence from the legacy implementation.

        Parameters
        ----------
        model_ratios : Array
            Sample ratios [r_1, ..., r_{M-1}] where n_l = r_l * n_0.
            Shape: (nmodels-1,)

        Returns
        -------
        partition_ratios : Array
            Partition ratios using MLMC recurrence. Shape: (nmodels-1,)
        """
        bkd = self._bkd
        nratios = model_ratios.shape[0]

        # Build partition ratios using recurrence: p[0] = r_1 - 1, p[i] = r_{i+1} - p[i-1]
        partition_list = []
        prev_p = model_ratios[0] - 1
        partition_list.append(bkd.reshape(prev_p, (1,)))
        for ii in range(1, nratios):
            curr_p = model_ratios[ii] - prev_p
            partition_list.append(bkd.reshape(curr_p, (1,)))
            prev_p = curr_p
        return bkd.concatenate(partition_list)

    def _get_rsquared_mlmc(self, nsample_ratios: Array) -> Array:
        """Compute r^2 for MLMC variance reduction.

        Matches legacy _get_rsquared_mlmc from _optim.py:179-218.

        Parameters
        ----------
        nsample_ratios : Array
            Sample ratios [r_1, ..., r_{M-1}] where n_l = r_l * n_0.
            Shape: (nmodels-1,)

        Returns
        -------
        rsquared : Array
            The r^2 value for variance reduction computation.
        """
        bkd = self._bkd
        cov = self._stat.cov()
        nmodels = self.nmodels()

        # Compute rhat: effective partition sample ratios
        # rhat[0] = 1, rhat[ii] = nsample_ratios[ii-1] - rhat[ii-1]
        rhat_list = [bkd.asarray([1.0])]
        prev_rhat = bkd.asarray(1.0)
        for ii in range(1, nmodels):
            curr_rhat = nsample_ratios[ii - 1] - prev_rhat
            rhat_list.append(bkd.reshape(curr_rhat, (1,)))
            prev_rhat = curr_rhat
        rhat = bkd.concatenate(rhat_list)

        # Sum of vardelta / rhat for levels 0 to M-2
        gamma = bkd.asarray(0.0)
        for ii in range(nmodels - 1):
            vardelta = cov[ii, ii] + cov[ii + 1, ii + 1] - 2 * cov[ii, ii + 1]
            gamma = gamma + vardelta / rhat[ii]

        # Add final level variance
        v = cov[nmodels - 1, nmodels - 1]
        gamma = gamma + v / rhat[-1]

        # Normalize by HF variance
        gamma = gamma / cov[0, 0]

        return 1 - gamma

    def allocate_samples_analytical(
        self, target_cost: float
    ) -> Tuple[Array, Array]:
        """Compute MLMC allocation analytically (Direction 1: min variance s.t. cost).

        Uses Lagrange multiplier method matching legacy _allocate_samples_mlmc.

        Parameters
        ----------
        target_cost : float
            Total computational budget.

        Returns
        -------
        nsample_ratios : Array
            Sample ratios n_m / n_0 for m = 1, ..., M-1.
        log_variance : Array
            Log of estimator variance.
        """
        bkd = self._bkd
        cov = self._stat.cov()
        variances = self._compute_level_variances()
        cost_deltas = self._compute_cost_deltas()

        # Lagrange multiplier
        var_cost_prods = variances * cost_deltas
        lagrange = target_cost / bkd.sum(bkd.sqrt(var_cost_prods))

        # Samples per level difference
        var_cost_ratios = variances / cost_deltas
        nsamples_per_delta = lagrange * bkd.sqrt(var_cost_ratios)

        # Model ratios
        nhf = nsamples_per_delta[0]
        ratio_list = []
        for l in range(self.nmodels() - 1):
            ratio = (nsamples_per_delta[l] + nsamples_per_delta[l + 1]) / nhf
            ratio_list.append(bkd.reshape(ratio, (1,)))
        nsample_ratios = bkd.concatenate(ratio_list)

        # Variance reduction
        gamma = 1 - self._get_rsquared_mlmc(nsample_ratios)
        log_variance = bkd.log(gamma) + bkd.log(cov[0, 0]) - bkd.log(nhf)

        return nsample_ratios, log_variance

    def allocate_samples_for_variance(
        self, target_variance: float
    ) -> Tuple[Array, Array]:
        """Compute MLMC allocation for target variance (Direction 2: min cost s.t. variance).

        Minimizes cost subject to Var(Q_MLMC) = target_variance (denoted ε²).

        Using Lagrange multiplier λ = ε⁻² * Σ sqrt(V_α * C_α):
            N_α = λ * sqrt(V_α / C_α)
                = (1/ε²) * sqrt(V_α / C_α) * Σ sqrt(V * C)

        Total cost: C_tot = ε⁻² * (Σ sqrt(V_α * C_α))²

        Parameters
        ----------
        target_variance : float
            Target variance ε² for the estimator.

        Returns
        -------
        nsamples : Array
            Samples per model. Shape: (nmodels,)
        total_cost : Array
            Total computational cost (scalar).
        """
        bkd = self._bkd
        variances = self._compute_level_variances()
        cost_deltas = self._compute_cost_deltas()
        target_var = bkd.asarray(target_variance)

        # Compute optimal allocation using Lagrange multiplier
        # From tutorial: N_α = λ * sqrt(V_α / C_α) where λ = ε⁻² * Σ sqrt(V*C)
        # and ε² = target_variance
        var_cost_ratios = variances / cost_deltas
        var_cost_prods = variances * cost_deltas

        # n_l^delta = (1/target_variance) * sqrt(V_l/c_l) * sum(sqrt(V*c))
        sum_sqrt_vc = bkd.sum(bkd.sqrt(var_cost_prods))
        nsamples_per_delta = (1 / target_var) * bkd.sqrt(var_cost_ratios) * sum_sqrt_vc

        # Total cost
        total_cost = bkd.sum(cost_deltas * nsamples_per_delta)

        # Convert to samples per model
        nsamples_list = [bkd.reshape(nsamples_per_delta[0], (1,))]
        for l in range(1, self.nmodels()):
            n_l = nsamples_per_delta[l - 1] + nsamples_per_delta[l]
            nsamples_list.append(bkd.reshape(n_l, (1,)))
        nsamples = bkd.concatenate(nsamples_list)

        return nsamples, total_cost

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
        nsamples = self.nsamples_per_model()

        # Compute level estimators
        # Y_0 = Q_0
        n0 = int(nsamples[0].item())
        Y = bkd.sum(values[0], axis=0) / n0

        # Y_l = Q_l - Q_{l-1} on shared samples
        for l in range(1, nmodels):
            nl = int(nsamples[l].item())
            n_shared = min(nl, int(nsamples[l - 1].item()))

            Q_l = bkd.sum(values[l][:n_shared], axis=0) / n_shared
            Q_lm1 = bkd.sum(values[l - 1][:n_shared], axis=0) / n_shared

            Y_l = Q_l - Q_lm1
            Y = Y + Y_l

        return Y

    def variance_reduction(self) -> float:
        """Return variance reduction factor.

        MLMC achieves variance reduction through level correlation.
        """
        bkd = self._bkd
        variances = self._compute_level_variances()

        # Effective variance reduction
        # Compare sum of level variances to single-level variance
        total_var = bkd.sum(variances)
        hf_var = variances[-1]

        result = bkd.where(
            hf_var > 0,
            total_var / hf_var,
            bkd.asarray(1.0)
        )
        return float(bkd.to_numpy(result))

    def __repr__(self) -> str:
        nsamples_str = "not allocated"
        if self._nsamples is not None:
            ns = self._bkd.to_numpy(self._nsamples)
            nsamples_str = str(list(ns.astype(int)))
        return f"MLMCEstimator(nlevels={self.nmodels()}, nsamples={nsamples_str})"
