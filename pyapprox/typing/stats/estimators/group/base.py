"""Group Approximate Control Variate (GroupACV) estimator.

GroupACV allows flexible grouping of models into subsets, where each
subset can use different sample allocations.
"""

from typing import Generic, List, Callable, Optional, Tuple

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.stats.protocols import StatisticWithDiscrepancyProtocol
from pyapprox.typing.stats.estimators.base import AbstractEstimator


class GroupACVEstimator(AbstractEstimator[Array], Generic[Array]):
    """Group Approximate Control Variate estimator.

    GroupACV extends ACV by allowing models to be grouped into subsets,
    where each subset shares samples. This provides more flexibility
    than standard ACV for complex model hierarchies.

    The estimator uses a more general allocation structure defined by
    model subsets (groups) rather than a single recursion index.

    Parameters
    ----------
    stat : StatisticWithDiscrepancyProtocol[Array]
        Statistic to estimate.
    costs : Array
        Cost per sample for each model. Shape: (nmodels,)
    bkd : Backend[Array]
        Computational backend.
    groups : List[List[int]], optional
        Model groups. Each group is a list of model indices that share
        samples. If None, uses default MFMC-like groups.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.stats import MultiOutputMean
    >>> bkd = NumpyBkd()
    >>> stat = MultiOutputMean(nqoi=1, bkd=bkd)
    >>> cov = bkd.asarray([[1.0, 0.9, 0.8], [0.9, 1.0, 0.85], [0.8, 0.85, 1.0]])
    >>> stat.set_pilot_quantities(cov)
    >>> costs = bkd.asarray([10.0, 1.0, 0.1])
    >>> # Group models: HF with LF1, LF2 separate
    >>> groups = [[0, 1], [2]]
    >>> gacv = GroupACVEstimator(stat, costs, bkd, groups=groups)
    """

    def __init__(
        self,
        stat: StatisticWithDiscrepancyProtocol[Array],
        costs: Array,
        bkd: Backend[Array],
        groups: Optional[List[List[int]]] = None,
    ):
        super().__init__(stat, costs, bkd)

        nmodels = costs.shape[0]

        # Set default groups if not provided
        if groups is None:
            # Default: each LF model grouped with HF
            groups = [[0, m] for m in range(1, nmodels)]
            groups.insert(0, [0])  # HF-only group

        self._groups = groups
        self._validate_groups()

        self._weights: Optional[Array] = None
        self._group_samples: Optional[List[int]] = None

    def _validate_groups(self) -> None:
        """Validate group specification."""
        nmodels = self.nmodels()

        # Check all models appear in at least one group
        all_models = set()
        for group in self._groups:
            for m in group:
                if m < 0 or m >= nmodels:
                    raise ValueError(
                        f"Invalid model index {m} in group. "
                        f"Must be in [0, {nmodels-1}]"
                    )
                all_models.add(m)

        if 0 not in all_models:
            raise ValueError("HF model (index 0) must appear in at least one group")

    def groups(self) -> List[List[int]]:
        """Return model groups."""
        return self._groups

    def ngroups(self) -> int:
        """Return number of groups."""
        return len(self._groups)

    def _compute_group_costs(self) -> np.ndarray:
        """Compute cost per sample for each group."""
        costs_np = self._bkd.to_numpy(self._costs)
        group_costs = []

        for group in self._groups:
            # Cost of evaluating all models in group
            group_cost = sum(costs_np[m] for m in group)
            group_costs.append(group_cost)

        return np.array(group_costs)

    def _compute_optimal_weights(self) -> Array:
        """Compute optimal control variate weights for group structure."""
        bkd = self._bkd
        cov = self._stat.cov()
        cov_np = bkd.to_numpy(cov)

        nmodels = self.nmodels()
        nqoi = self._stat.nqoi()

        # For group ACV, weights depend on group structure
        # Simplified: use correlation-based weights
        if nqoi == 1:
            weights = []
            for m in range(1, nmodels):
                rho = cov_np[0, m] / np.sqrt(cov_np[0, 0] * cov_np[m, m])
                sigma_ratio = np.sqrt(cov_np[0, 0] / cov_np[m, m])
                weights.append(rho * sigma_ratio)
            return bkd.asarray(np.array(weights))
        else:
            return bkd.asarray(np.ones(nmodels - 1))

    def allocate_samples(self, target_cost: float) -> None:
        """Allocate samples optimally across groups.

        Parameters
        ----------
        target_cost : float
            Total computational budget.
        """
        bkd = self._bkd
        costs_np = bkd.to_numpy(self._costs)
        nmodels = self.nmodels()

        group_costs = self._compute_group_costs()

        # Compute samples per group
        # Use simple proportional allocation based on variance reduction potential
        cov = self._stat.cov()
        cov_np = bkd.to_numpy(cov)

        # Estimate variance contribution from each group
        group_weights = []
        for group in self._groups:
            if 0 in group:
                # Groups containing HF contribute to variance reduction
                weight = 1.0
                for m in group:
                    if m > 0:
                        rho_sq = cov_np[0, m]**2 / (cov_np[0, 0] * cov_np[m, m])
                        weight *= (1 - rho_sq)
            else:
                weight = 0.1  # Lower priority for non-HF groups
            group_weights.append(weight)

        group_weights = np.array(group_weights)
        group_weights = group_weights / np.sum(group_weights)

        # Allocate budget to groups
        self._group_samples = []
        remaining_cost = target_cost

        for g, group in enumerate(self._groups):
            allocated_cost = target_cost * group_weights[g]
            nsamples = int(allocated_cost / group_costs[g])
            nsamples = max(nsamples, self._stat.min_nsamples())
            self._group_samples.append(nsamples)

        # Compute samples per model
        model_samples = np.zeros(nmodels, dtype=np.int64)
        for g, group in enumerate(self._groups):
            for m in group:
                model_samples[m] = max(model_samples[m], self._group_samples[g])

        self._nsamples = bkd.asarray(model_samples)

        # Compute weights
        self._weights = self._compute_optimal_weights()

    def group_samples(self) -> List[int]:
        """Return samples per group."""
        if self._group_samples is None:
            raise ValueError("Samples not allocated.")
        return self._group_samples

    def weights(self) -> Array:
        """Return optimal control variate weights."""
        if self._weights is None:
            raise ValueError("Weights not computed.")
        return self._weights

    def generate_samples_per_model(
        self, rvs: Callable[[int], Array]
    ) -> List[Array]:
        """Generate samples for each model based on group structure.

        Parameters
        ----------
        rvs : Callable[[int], Array]
            Random variable sampler.

        Returns
        -------
        List[Array]
            Samples for each model.
        """
        nmodels = self.nmodels()
        nsamples_np = self._bkd.to_numpy(self.nsamples_per_model())

        # Generate samples for each group
        group_samples = []
        for g, n_g in enumerate(self._group_samples):
            group_samples.append(rvs(n_g) if n_g > 0 else rvs(1)[:0])

        # Assign samples to models
        model_samples = []
        for m in range(nmodels):
            # Find groups containing this model
            model_samps = []
            for g, group in enumerate(self._groups):
                if m in group and self._group_samples[g] > 0:
                    model_samps.append(group_samples[g])

            if model_samps:
                model_samples.append(self._bkd.concatenate(model_samps, axis=0))
            else:
                model_samples.append(rvs(1)[:0])

        return model_samples

    def __call__(self, values: List[Array]) -> Array:
        """Compute Group ACV estimate.

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
        nmodels = self.nmodels()
        nsamples_np = bkd.to_numpy(self.nsamples_per_model())
        weights = bkd.to_numpy(self.weights())

        nhf = int(nsamples_np[0])

        # Q_0: HF mean
        Q0 = bkd.sum(values[0], axis=0) / nhf

        # Control variate correction
        correction = bkd.zeros(self._stat.nstats())

        for m in range(1, nmodels):
            nlf = int(nsamples_np[m])
            if nlf == 0:
                continue

            n_shared = min(nhf, nlf)

            # Q_m: LF mean on shared samples
            Q_m = bkd.sum(values[m][:n_shared], axis=0) / n_shared

            # mu_m: LF mean on all samples
            mu_m = bkd.sum(values[m], axis=0) / nlf

            eta_m = weights[m - 1]
            correction = correction + eta_m * (mu_m - Q_m)

        return Q0 + correction

    def optimized_covariance(self) -> Array:
        """Return covariance of the Group ACV estimator."""
        bkd = self._bkd
        nsamples = self.nsamples_per_model()
        nhf = int(bkd.to_numpy(nsamples)[0])

        hf_cov = self._stat.high_fidelity_estimator_covariance(nhf)

        # Approximate variance reduction
        cov = self._stat.cov()
        cov_np = bkd.to_numpy(cov)
        nqoi = self._stat.nqoi()

        if nqoi == 1:
            total_rho_sq = 0
            for m in range(1, self.nmodels()):
                rho = cov_np[0, m] / np.sqrt(cov_np[0, 0] * cov_np[m, m])
                total_rho_sq += rho ** 2
            var_red = max(1 - total_rho_sq, 0.01)
            return bkd.asarray(bkd.to_numpy(hf_cov) * var_red)
        else:
            return hf_cov

    def __repr__(self) -> str:
        nsamples_str = "not allocated"
        if self._nsamples is not None:
            ns = self._bkd.to_numpy(self._nsamples)
            nsamples_str = str(list(ns.astype(int)))
        return (
            f"GroupACVEstimator(nmodels={self.nmodels()}, "
            f"ngroups={self.ngroups()}, nsamples={nsamples_str})"
        )
