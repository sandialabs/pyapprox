"""Brute-force subset fitter for GroupACV estimators.

Enumerates all subsets of candidate model groups and runs continuous
optimization on each, selecting the pattern with the best objective.
Reference fitter for tests and small problems (K <= 16).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Generic,
    Iterator,
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
class BruteForceSubsetResult(Generic[Array]):
    """Result of brute-force subset enumeration.

    Attributes
    ----------
    best_estimator : FittedGroupACVEstimator
        Fitted estimator for the winning pattern.
    best_allocation : GroupACVAllocationResult
        Allocation result for the winning pattern.
    best_active_indices : Tuple[int, ...]
        Indices into candidate_subsets for the winning pattern.
    all_allocations : List[Tuple[Tuple[int, ...], GroupACVAllocationResult]]
        All (active_indices, result) pairs evaluated.
    candidate_subsets : List[Array]
        The full list of candidate subsets.
    """

    best_estimator: "FittedGroupACVEstimator[Array]"
    best_allocation: "GroupACVAllocationResult[Array]"
    best_active_indices: Tuple[int, ...]
    all_allocations: List[
        Tuple[Tuple[int, ...], "GroupACVAllocationResult[Array]"]
    ]
    candidate_subsets: List[Array]

    def patterns_evaluated(self) -> int:
        return len(self.all_allocations)

    def patterns_successful(self) -> int:
        return sum(1 for _, alloc in self.all_allocations if alloc.success)


class BruteForceSubsetFitter(Generic[Array]):
    """Enumerate subset patterns and optimize each independently.

    For each pattern (subset of candidate_subsets), constructs an estimator
    using only the active subsets and runs allocation optimization. Selects
    the pattern with the lowest objective value.

    Parameters
    ----------
    stat : MultiOutputStatistic
        The statistic containing pilot quantities.
    costs : Array
        Model costs, shape (nmodels,).
    estimator_class : Type[BaseGroupACVEstimator]
        Estimator class to use for each pattern.
    candidate_subsets : List[Array], optional
        Candidate model subsets. If None, generates all subsets.
    optimizer : BindableOptimizerProtocol, optional
        Optimizer for allocation. If None, uses default.
    objective : GroupACVObjective, optional
        Objective for allocation. If None, uses default.
    problem_config : AllocationProblemConfig, optional
        Allocation problem config. If None, uses default with
        bounds_lb="dead_threshold".
    reg_blue : float, optional
        Regularization parameter for BLUE. Default is 0.
    """

    MAX_SUBSETS: int = 16

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
    ) -> None:
        from pyapprox.statest.groupacv.variable_space import (
            AllocationProblemConfig,
        )
        from pyapprox.statest.groupacv.variants import (
            GroupACVEstimatorNested,
        )

        self._stat = stat
        self._costs = costs
        self._bkd: Backend[Array] = stat.bkd()
        self._estimator_class = estimator_class
        self._optimizer = optimizer
        self._objective = objective
        self._reg_blue = reg_blue
        self._is_nested = issubclass(estimator_class, GroupACVEstimatorNested)

        if problem_config is None:
            problem_config = AllocationProblemConfig()
        self._problem_config = problem_config

        if candidate_subsets is None:
            candidate_subsets = get_model_subsets(
                len(costs), self._bkd
            )
        if len(candidate_subsets) > self.MAX_SUBSETS:
            raise ValueError(
                f"Too many candidate subsets ({len(candidate_subsets)} > "
                f"{self.MAX_SUBSETS}). Brute-force enumeration is only "
                f"feasible for K <= {self.MAX_SUBSETS}."
            )
        self._candidate_subsets = candidate_subsets

    def _iter_patterns(self) -> Iterator[Tuple[int, ...]]:
        """Yield all feasible active-index tuples.

        Filters:
        1. At least one active subset must contain model 0.
        2. For nested estimators, at least one active subset must NOT
           be the singleton [0] (so nesting is non-degenerate).
        """
        bkd = self._bkd
        K = len(self._candidate_subsets)

        has_model_0 = []
        is_singleton_0 = []
        for i, subset in enumerate(self._candidate_subsets):
            np_subset = bkd.to_numpy(subset)
            has_model_0.append(0 in np_subset)
            is_singleton_0.append(
                len(np_subset) == 1 and int(np_subset[0]) == 0
            )

        for size in range(1, K + 1):
            for combo in itertools.combinations(range(K), size):
                if not any(has_model_0[i] for i in combo):
                    continue
                if self._is_nested and all(is_singleton_0[i] for i in combo):
                    continue
                yield combo

    def _optimize_pattern(
        self,
        active_indices: Tuple[int, ...],
        target_cost: float,
        min_nhf_samples: int,
    ) -> "GroupACVAllocationResult[Array]":
        """Run allocation optimization for a single pattern."""
        from pyapprox.statest.groupacv.allocation import (
            GroupACVAllocationOptimizer,
        )
        from pyapprox.statest.groupacv.result import GroupACVAllocationResult

        bkd = self._bkd
        model_subsets = [self._candidate_subsets[i] for i in active_indices]

        est = self._estimator_class(
            self._stat,
            self._costs,
            reg_blue=self._reg_blue,
            model_subsets=model_subsets,
        )

        # Pre-check budget feasibility
        bounds_lb = self._problem_config.resolve_bounds_lb(self._stat)
        partition_costs = bkd.einsum(
            "m,mp->p", est._costs, est._partitions_per_model
        )
        min_cost = float(bkd.to_numpy(
            bkd.sum(partition_costs) * bounds_lb
            + est._costs[0] * min_nhf_samples
        ))
        if target_cost < min_cost:
            npartitions = est.npartitions()
            return GroupACVAllocationResult(
                npartition_samples=bkd.zeros((npartitions,)),
                nsamples_per_model=bkd.zeros((est._nmodels,)),
                actual_cost=0.0,
                objective_value=bkd.array([float("inf")]),
                success=False,
                message="Budget too small for pattern",
            )

        allocator: GroupACVAllocationOptimizer[Array] = (
            GroupACVAllocationOptimizer(
                est,
                optimizer=self._optimizer,
                objective=self._objective,
                problem_config=self._problem_config,
            )
        )
        return allocator.optimize(
            target_cost, min_nhf_samples=min_nhf_samples
        )

    def fit(
        self,
        target_cost: float,
        min_nhf_samples: int = 1,
        allow_failures: bool = True,
    ) -> BruteForceSubsetResult[Array]:
        """Enumerate all patterns and select the best.

        Parameters
        ----------
        target_cost : float
            Maximum computational budget.
        min_nhf_samples : int, optional
            Minimum high-fidelity samples. Default is 1.
        allow_failures : bool, optional
            If True, continue on allocation failures. Default is True.

        Returns
        -------
        BruteForceSubsetResult
            Contains the best estimator/allocation and all candidates.

        Raises
        ------
        RuntimeError
            If no patterns succeed, or if allow_failures=False and any
            pattern fails.
        """
        bkd = self._bkd
        all_allocations: List[
            Tuple[Tuple[int, ...], "GroupACVAllocationResult[Array]"]
        ] = []

        for active_indices in self._iter_patterns():
            result = self._optimize_pattern(
                active_indices, target_cost, min_nhf_samples
            )

            # Post-rounding check: ensure min_nhf_samples is satisfied
            if result.success:
                nhf = float(bkd.to_numpy(result.nsamples_per_model[0]))
                if nhf < min_nhf_samples:
                    from pyapprox.statest.groupacv.result import (
                        GroupACVAllocationResult,
                    )

                    result = GroupACVAllocationResult(
                        npartition_samples=result.npartition_samples,
                        nsamples_per_model=result.nsamples_per_model,
                        actual_cost=result.actual_cost,
                        objective_value=result.objective_value,
                        success=False,
                        message="Post-rounding min_nhf_samples violated",
                    )

            all_allocations.append((active_indices, result))

            if not result.success and not allow_failures:
                raise RuntimeError(
                    f"Allocation failed for pattern {active_indices}: "
                    f"{result.message}"
                )

        # Select best: minimum objective among successful, tie-break by sparsity
        successful = [
            (idx, alloc) for idx, alloc in all_allocations if alloc.success
        ]
        if not successful:
            raise RuntimeError("No successful allocations found")

        best_idx, best_alloc = min(
            successful,
            key=lambda r: (
                float(bkd.to_numpy(r[1].objective_value[0])),
                len(r[0]),
            ),
        )

        # Reconstruct fitted estimator for winner
        from pyapprox.statest.groupacv.base import FittedGroupACVEstimator

        model_subsets = [self._candidate_subsets[i] for i in best_idx]
        best_est = self._estimator_class(
            self._stat,
            self._costs,
            reg_blue=self._reg_blue,
            model_subsets=model_subsets,
        )
        fitted = FittedGroupACVEstimator(best_est, best_alloc)

        return BruteForceSubsetResult(
            best_estimator=fitted,
            best_allocation=best_alloc,
            best_active_indices=best_idx,
            all_allocations=all_allocations,
            candidate_subsets=self._candidate_subsets,
        )
