"""Mean-guided subset fitter for GroupACV estimators.

Uses a cheap Mean-stat solve to identify active partitions, then prunes
dead partitions before solving the target stat (e.g. Variance) on the
reduced problem.  At low budgets where the dead threshold (lb=2 for
Variance) forces many partitions to their lower bound, this frees
budget for the truly useful partitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
)

from pyapprox.statest.groupacv.utils import get_model_subsets
from pyapprox.util.backends.protocols import Array, Backend

if TYPE_CHECKING:
    from pyapprox.optimization.minimize.protocols import (
        BindableOptimizerProtocol,
    )
    from pyapprox.statest.groupacv.base import (
        BaseGroupACVEstimator,
        FittedGroupACVEstimator,
    )
    from pyapprox.statest.groupacv.optimization import GroupACVObjective
    from pyapprox.statest.groupacv.result import GroupACVAllocationResult
    from pyapprox.statest.groupacv.variable_space import AllocationProblemConfig
    from pyapprox.statest.statistics import MultiOutputStatistic


@dataclass(frozen=True)
class MeanGuidedSubsetResult(Generic[Array]):
    """Result of mean-guided subset selection.

    Attributes
    ----------
    best_estimator : FittedGroupACVEstimator
        Fitted estimator using only the active partitions.
    best_allocation : GroupACVAllocationResult
        Allocation result for the reduced problem.
    active_subset_indices : Tuple[int, ...]
        Indices into candidate_subsets for the active partitions.
    candidate_subsets : List[Array]
        The full list of candidate subsets.
    mean_allocation : GroupACVAllocationResult
        Allocation result from the Mean-stat screening solve.
    mean_npartition_samples : Array
        Per-partition sample counts from the Mean solve (for diagnostics).
    """

    best_estimator: "FittedGroupACVEstimator[Array]"
    best_allocation: "GroupACVAllocationResult[Array]"
    active_subset_indices: Tuple[int, ...]
    candidate_subsets: List[Array]
    mean_allocation: "GroupACVAllocationResult[Array]"
    mean_npartition_samples: Array

    def partitions_pruned(self) -> int:
        return len(self.candidate_subsets) - len(self.active_subset_indices)


class MeanGuidedSubsetFitter(Generic[Array]):
    """Two-stage fitter: Mean screening followed by target-stat optimization.

    Stage 1 — Screening: Build the full estimator using a MultiOutputMean
    stat (which has continuous_dead_threshold=0, so partitions can go to
    zero).  Solve the allocation to identify which partitions receive
    samples above a threshold.

    Stage 2 — Reduced solve: Construct a new estimator using only the
    active subsets, then optimize the target stat (e.g. Variance) on
    the reduced problem.

    Parameters
    ----------
    stat : MultiOutputStatistic
        The target statistic (e.g. MultiOutputVariance).
    costs : Array
        Model costs, shape (nmodels,).
    estimator_class : Type[BaseGroupACVEstimator]
        Estimator class to use for both stages.
    candidate_subsets : List[Array], optional
        Candidate model subsets. If None, generates all subsets.
    optimizer : BindableOptimizerProtocol, optional
        Optimizer for allocation. If None, uses default.
    objective : GroupACVObjective, optional
        Objective for allocation. If None, uses default.
    problem_config : AllocationProblemConfig, optional
        Allocation problem config for the target-stat solve.
        The screening solve always uses log/inequality with bounds_lb=1e-8.
    reg_blue : float, optional
        Regularization parameter for BLUE. Default is 0.
    activity_threshold : float, optional
        Partitions with Mean n_m > this value are kept. Default is 1.0
        (at least one sample in the mean-screening solution). Partitions
        below this threshold would round to zero samples when building an
        actual estimator, and forcing them to the target-stat lower bound
        (e.g. 2 for variance) wastes budget without meaningful variance
        reduction.
    """

    def __init__(
        self,
        stat: "MultiOutputStatistic[Array]",
        costs: Array,
        estimator_class: Type["BaseGroupACVEstimator[Array]"],
        candidate_subsets: Optional[List[Array]] = None,
        optimizer: Optional["BindableOptimizerProtocol[Array]"] = None,
        objective: Optional["GroupACVObjective[Array]"] = None,
        problem_config: Optional["AllocationProblemConfig"] = None,
        reg_blue: float = 0,
        activity_threshold: float = 1.0,
    ) -> None:
        from pyapprox.statest.groupacv.variable_space import (
            AllocationProblemConfig,
        )

        self._stat = stat
        self._costs = costs
        self._bkd: Backend[Array] = stat.bkd()
        self._estimator_class = estimator_class
        self._optimizer = optimizer
        self._objective = objective
        self._reg_blue = reg_blue
        self._activity_threshold = activity_threshold

        if problem_config is None:
            problem_config = AllocationProblemConfig()
        self._problem_config = problem_config

        if candidate_subsets is None:
            candidate_subsets = get_model_subsets(
                len(costs), self._bkd
            )
        self._candidate_subsets = candidate_subsets

    def _build_mean_stat(
        self,
    ) -> "MultiOutputStatistic[Array]":
        """Construct a MultiOutputMean stat sharing the target stat's cov."""
        from pyapprox.statest.statistics import MultiOutputMean

        if self._stat._cov is None:
            raise ValueError(
                "Target stat must have pilot quantities set (call "
                "set_pilot_quantities first)"
            )
        mean_stat: MultiOutputMean[Array] = MultiOutputMean(
            self._stat.nqoi(), self._bkd
        )
        mean_stat.set_pilot_quantities(self._stat._cov)
        return mean_stat

    def _screen(
        self,
        target_cost: float,
        min_nhf_samples: int,
    ) -> Tuple["GroupACVAllocationResult[Array]", Tuple[int, ...]]:
        """Run Mean screening to identify active partitions."""
        from pyapprox.statest.groupacv.allocation import (
            GroupACVAllocationOptimizer,
        )
        from pyapprox.statest.groupacv.variable_space import (
            AllocationProblemConfig,
        )

        mean_stat = self._build_mean_stat()

        # Full estimator with all candidate subsets, using Mean stat
        full_est = self._estimator_class(
            mean_stat,
            self._costs,
            reg_blue=self._reg_blue,
            model_subsets=self._candidate_subsets,
        )

        # Screening config: log space, low lb so partitions can vanish
        screen_config = AllocationProblemConfig(
            variable_scaling="log",
            budget_constraint_form="inequality",
            bounds_lb=1e-8,
        )

        allocator: GroupACVAllocationOptimizer[Array] = (
            GroupACVAllocationOptimizer(
                full_est,
                optimizer=self._optimizer,
                objective=self._objective,
                problem_config=screen_config,
            )
        )
        mean_result = allocator.optimize(
            target_cost,
            min_nhf_samples=min_nhf_samples,
            round_nsamples=False,
        )

        if not mean_result.success:
            raise RuntimeError(
                f"Mean screening solve failed: {mean_result.message}"
            )

        # Identify active partitions
        bkd = self._bkd
        nps_np = bkd.to_numpy(mean_result.npartition_samples)
        active_indices = tuple(
            i for i, n in enumerate(nps_np)
            if float(n) > self._activity_threshold
        )

        if len(active_indices) == 0:
            raise RuntimeError(
                "Mean screening found no active partitions — budget may "
                "be too small for any allocation"
            )

        # Ensure at least one subset containing model 0 survives.
        # When the budget is tight the HF constraint is binding and
        # every model-0 subset may fall below the activity threshold.
        # Keep the one with the largest relaxed allocation.
        has_model0 = any(
            0 in bkd.to_numpy(self._candidate_subsets[i]).tolist()
            for i in active_indices
        )
        if not has_model0:
            model0_indices = [
                i for i in range(len(self._candidate_subsets))
                if 0 in bkd.to_numpy(self._candidate_subsets[i]).tolist()
            ]
            best_m0 = max(model0_indices, key=lambda i: float(nps_np[i]))
            active_indices = tuple(sorted(set(active_indices) | {best_m0}))

        return mean_result, active_indices

    def fit(
        self,
        target_cost: float,
        min_nhf_samples: int = 1,
    ) -> MeanGuidedSubsetResult[Array]:
        """Run two-stage mean-guided fitting.

        Parameters
        ----------
        target_cost : float
            Maximum computational budget.
        min_nhf_samples : int, optional
            Minimum high-fidelity samples. Default is 1.

        Returns
        -------
        MeanGuidedSubsetResult
            Contains the fitted estimator on the reduced partition set.

        Raises
        ------
        RuntimeError
            If the Mean screening or target-stat optimization fails.
        """
        from pyapprox.statest.groupacv.allocation import (
            GroupACVAllocationOptimizer,
        )
        from pyapprox.statest.groupacv.base import FittedGroupACVEstimator

        bkd = self._bkd

        # Stage 1: Mean screening
        mean_result, active_indices = self._screen(
            target_cost, min_nhf_samples
        )

        # Stage 2: Reduced target-stat solve
        active_subsets = [
            self._candidate_subsets[i] for i in active_indices
        ]
        reduced_est = self._estimator_class(
            self._stat,
            self._costs,
            reg_blue=self._reg_blue,
            model_subsets=active_subsets,
        )

        # Check budget feasibility for reduced estimator
        bounds_lb = self._problem_config.resolve_bounds_lb(self._stat)
        partition_costs = bkd.einsum(
            "m,mp->p", reduced_est._costs, reduced_est._partitions_per_model
        )
        min_cost = float(bkd.to_numpy(
            bkd.sum(partition_costs) * bounds_lb
            + reduced_est._costs[0] * min_nhf_samples
        ))
        if target_cost < min_cost:
            raise RuntimeError(
                f"Budget {target_cost} too small for reduced pattern "
                f"(min_cost={min_cost:.2f})"
            )

        allocator: GroupACVAllocationOptimizer[Array] = (
            GroupACVAllocationOptimizer(
                reduced_est,
                optimizer=self._optimizer,
                objective=self._objective,
                problem_config=self._problem_config,
            )
        )
        target_result = allocator.optimize(
            target_cost, min_nhf_samples=min_nhf_samples
        )

        if not target_result.success:
            raise RuntimeError(
                f"Target-stat optimization on reduced estimator failed: "
                f"{target_result.message}"
            )

        fitted = FittedGroupACVEstimator(reduced_est, target_result)

        return MeanGuidedSubsetResult(
            best_estimator=fitted,
            best_allocation=target_result,
            active_subset_indices=active_indices,
            candidate_subsets=self._candidate_subsets,
            mean_allocation=mean_result,
            mean_npartition_samples=mean_result.npartition_samples,
        )
