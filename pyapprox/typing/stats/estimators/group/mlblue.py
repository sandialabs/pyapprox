"""Multilevel Best Linear Unbiased Estimator (MLBLUE).

MLBLUE is a generalization of MLMC that uses optimal linear combinations
of level estimators to minimize variance.
"""

from typing import Generic, List, Callable, Optional

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.stats.protocols import StatisticWithDiscrepancyProtocol
from pyapprox.typing.stats.estimators.group.base import GroupACVEstimator


class MLBLUEEstimator(GroupACVEstimator[Array], Generic[Array]):
    """Multilevel Best Linear Unbiased Estimator.

    MLBLUE extends MLMC by computing the optimal linear combination of
    level estimators to minimize variance subject to the unbiasedness
    constraint.

    The estimator is:
        Q_MLBLUE = sum_g w_g * Y_g

    where Y_g are estimators for each group and w_g are optimal weights
    satisfying sum_g w_g = 1 (unbiasedness).

    Parameters
    ----------
    stat : StatisticWithDiscrepancyProtocol[Array]
        Statistic to estimate.
    costs : Array
        Cost per sample for each model/level. Shape: (nlevels,)
    bkd : Backend[Array]
        Computational backend.
    groups : List[List[int]], optional
        Model groups. If None, uses MLMC-style groups:
        [[0], [0,1], [1,2], ...] for telescoping sum.

    Notes
    -----
    MLBLUE achieves the minimum variance among all linear unbiased
    estimators for the given sample allocation.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.stats import MultiOutputMean
    >>> bkd = NumpyBkd()
    >>> stat = MultiOutputMean(nqoi=1, bkd=bkd)
    >>> # MLMC-like covariance
    >>> cov = bkd.asarray([[1.0, 0.99, 0.9], [0.99, 1.0, 0.99], [0.9, 0.99, 1.0]])
    >>> stat.set_pilot_quantities(cov)
    >>> costs = bkd.asarray([1.0, 10.0, 100.0])
    >>> mlblue = MLBLUEEstimator(stat, costs, bkd)
    >>> mlblue.allocate_samples(target_cost=1000.0)
    """

    def __init__(
        self,
        stat: StatisticWithDiscrepancyProtocol[Array],
        costs: Array,
        bkd: Backend[Array],
        groups: Optional[List[List[int]]] = None,
    ):
        nmodels = costs.shape[0]

        # Default MLBLUE groups: MLMC-style telescoping
        if groups is None:
            groups = []
            # Single-level groups for each level difference
            for l in range(nmodels):
                if l == 0:
                    groups.append([0])  # Coarsest level
                else:
                    groups.append([l - 1, l])  # Level difference

        super().__init__(stat, costs, bkd, groups)

        self._blue_weights: Optional[Array] = None

    def _compute_blue_weights(self) -> Array:
        """Compute optimal BLUE weights.

        The optimal weights minimize variance subject to sum(w) = 1.
        Using Lagrange multipliers:
            w = Sigma^{-1} @ e / (e^T @ Sigma^{-1} @ e)

        where e = (1, 1, ..., 1)^T and Sigma is the covariance of group estimators.
        """
        bkd = self._bkd
        ngroups = self.ngroups()

        # Compute covariance of group estimators
        group_cov = self._compute_group_covariance()

        try:
            # Solve for weights
            e = np.ones(ngroups)
            Sigma_inv_e = np.linalg.solve(group_cov, e)
            denom = e @ Sigma_inv_e

            if denom > 0:
                weights = Sigma_inv_e / denom
            else:
                # Fallback to equal weights
                weights = np.ones(ngroups) / ngroups
        except np.linalg.LinAlgError:
            # Fallback to equal weights
            weights = np.ones(ngroups) / ngroups

        return bkd.asarray(weights)

    def _compute_group_covariance(self) -> np.ndarray:
        """Compute covariance matrix of group estimators."""
        bkd = self._bkd
        cov = self._stat.cov()
        cov_np = bkd.to_numpy(cov)

        ngroups = self.ngroups()
        nqoi = self._stat.nqoi()
        group_samples = self.group_samples()

        # Group covariance matrix
        Sigma = np.zeros((ngroups, ngroups))

        for g1 in range(ngroups):
            for g2 in range(ngroups):
                group1 = self._groups[g1]
                group2 = self._groups[g2]
                n1 = max(group_samples[g1], 1)
                n2 = max(group_samples[g2], 1)

                # Compute covariance based on shared models
                shared = set(group1) & set(group2)
                if shared:
                    # Groups share samples
                    n_shared = min(n1, n2)
                    cov_sum = 0
                    for m in shared:
                        if nqoi == 1:
                            cov_sum += cov_np[m, m]
                        else:
                            cov_sum += np.trace(
                                cov_np[m*nqoi:(m+1)*nqoi, m*nqoi:(m+1)*nqoi]
                            ) / nqoi
                    Sigma[g1, g2] = cov_sum / n_shared
                else:
                    # Independent groups
                    Sigma[g1, g2] = 0

        # Ensure positive definiteness
        Sigma = Sigma + 1e-10 * np.eye(ngroups)

        return Sigma

    def allocate_samples(self, target_cost: float) -> None:
        """Allocate samples using MLBLUE optimization.

        Parameters
        ----------
        target_cost : float
            Total computational budget.
        """
        # First use parent allocation
        super().allocate_samples(target_cost)

        # Then compute BLUE weights
        self._blue_weights = self._compute_blue_weights()

    def blue_weights(self) -> Array:
        """Return BLUE weights for combining group estimators."""
        if self._blue_weights is None:
            raise ValueError("BLUE weights not computed.")
        return self._blue_weights

    def __call__(self, values: List[Array]) -> Array:
        """Compute MLBLUE estimate.

        Q_MLBLUE = sum_g w_g * Y_g

        where Y_g is the group estimator and w_g are BLUE weights.

        Parameters
        ----------
        values : List[Array]
            Model outputs.

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
        blue_weights = bkd.to_numpy(self.blue_weights())
        nstats = self._stat.nstats()

        # Compute group estimators
        group_estimates = []
        for g, group in enumerate(self._groups):
            n_g = self._group_samples[g]
            if n_g == 0:
                group_estimates.append(bkd.zeros(nstats))
                continue

            if len(group) == 1:
                # Single-model group
                m = group[0]
                n_m = min(n_g, values[m].shape[0])
                Y_g = bkd.sum(values[m][:n_m], axis=0) / n_m
            else:
                # Multi-model group (level difference)
                # Y_g = Q_fine - Q_coarse
                m_fine = max(group)
                m_coarse = min(group)
                n_shared = min(n_g, values[m_fine].shape[0], values[m_coarse].shape[0])

                Q_fine = bkd.sum(values[m_fine][:n_shared], axis=0) / n_shared
                Q_coarse = bkd.sum(values[m_coarse][:n_shared], axis=0) / n_shared
                Y_g = Q_fine - Q_coarse

            group_estimates.append(Y_g)

        # Combine with BLUE weights
        result = bkd.zeros(nstats)
        for g in range(self.ngroups()):
            result = result + blue_weights[g] * group_estimates[g]

        return result

    def optimized_covariance(self) -> Array:
        """Return covariance of the MLBLUE estimator.

        The optimal variance is:
            Var(Q_MLBLUE) = 1 / (e^T @ Sigma^{-1} @ e)
        """
        bkd = self._bkd
        group_cov = self._compute_group_covariance()

        try:
            e = np.ones(self.ngroups())
            Sigma_inv_e = np.linalg.solve(group_cov, e)
            denom = e @ Sigma_inv_e
            if denom > 0:
                opt_var = 1.0 / denom
            else:
                # Fallback to HF variance
                nsamples = self.nsamples_per_model()
                nhf = int(bkd.to_numpy(nsamples)[0])
                return self._stat.high_fidelity_estimator_covariance(nhf)
        except np.linalg.LinAlgError:
            # Fallback
            nsamples = self.nsamples_per_model()
            nhf = int(bkd.to_numpy(nsamples)[0])
            return self._stat.high_fidelity_estimator_covariance(nhf)

        # Ensure non-negative
        opt_var = max(opt_var, 0.0)

        nstats = self._stat.nstats()
        return bkd.asarray(np.eye(nstats) * opt_var)

    def __repr__(self) -> str:
        nsamples_str = "not allocated"
        if self._nsamples is not None:
            ns = self._bkd.to_numpy(self._nsamples)
            nsamples_str = str(list(ns.astype(int)))
        return (
            f"MLBLUEEstimator(nlevels={self.nmodels()}, "
            f"ngroups={self.ngroups()}, nsamples={nsamples_str})"
        )
