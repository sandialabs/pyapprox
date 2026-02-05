"""Search for optimal GroupACV estimator configurations."""

from dataclasses import dataclass
from typing import (
    Generic,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TYPE_CHECKING,
)

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.statest.strategies import ModelSubsetStrategy, QoISubsetStrategy

if TYPE_CHECKING:
    from pyapprox.typing.statest.groupacv.base import BaseGroupACVEstimator
    from pyapprox.typing.statest.groupacv.allocation import GroupACVAllocationResult
    from pyapprox.typing.statest.statistics import MultiOutputStatistic


@dataclass
class GroupACVSearchResult(Generic[Array]):
    """Result of GroupACV estimator configuration search."""

    estimator: "BaseGroupACVEstimator[Array]"
    allocation: "GroupACVAllocationResult[Array]"
    all_allocations: List[Tuple["BaseGroupACVEstimator[Array]", "GroupACVAllocationResult[Array]"]]

    # Search configuration
    estimator_classes: List[Type["BaseGroupACVEstimator[Array]"]]
    model_strategy: ModelSubsetStrategy
    qoi_strategy: QoISubsetStrategy

    def candidates_evaluated(self) -> int:
        """Return total number of configurations evaluated."""
        return len(self.all_allocations)

    def candidates_successful(self) -> int:
        """Return number of successful allocations."""
        return sum(1 for _, alloc in self.all_allocations if alloc.success)

    def successful_allocations(
        self,
    ) -> List[Tuple["BaseGroupACVEstimator[Array]", "GroupACVAllocationResult[Array]"]]:
        """Return list of (estimator, allocation) pairs for successful allocations."""
        return [(est, alloc) for est, alloc in self.all_allocations if alloc.success]

    def search_description(self) -> str:
        """Return human-readable description of the search configuration."""
        est_names = [c.__name__ for c in self.estimator_classes]
        return (
            f"Estimators: {est_names}; "
            f"Models: {self.model_strategy.description()}; "
            f"QoI: {self.qoi_strategy.description()}"
        )


class GroupACVSearch(Generic[Array]):
    """Search over GroupACV estimator configurations.

    Searches the Cartesian product of:
    - Estimator types (MLBLUE, GroupACVIS, GroupACVNested)
    - Model subsets (which models to include)
    - QoI subsets (which quantities to estimate)

    Parameters
    ----------
    stat : MultiOutputStatistic
        The statistic containing pilot quantities.
    costs : Array
        Model costs, shape (nmodels,).
    estimator_classes : List[Type[BaseGroupACVEstimator]], optional
        Estimator classes to search. Defaults to [MLBLUEEstimator].
    model_strategy : ModelSubsetStrategy, optional
        Strategy for model subset search. Defaults to AllModelsStrategy.
    qoi_strategy : QoISubsetStrategy, optional
        Strategy for QoI subset search. Defaults to AllQoIStrategy.
    optimizer : optional
        Optimizer for allocation.
    objective : optional
        Objective for allocation.
    """

    def __init__(
        self,
        stat: "MultiOutputStatistic[Array]",
        costs: Array,
        estimator_classes: Optional[List[Type["BaseGroupACVEstimator[Array]"]]] = None,
        model_strategy: Optional[ModelSubsetStrategy] = None,
        qoi_strategy: Optional[QoISubsetStrategy] = None,
        optimizer: Optional[object] = None,
        objective: Optional[object] = None,
    ) -> None:
        from pyapprox.typing.statest.strategies import AllModelsStrategy, AllQoIStrategy
        from pyapprox.typing.statest.groupacv.mlblue import MLBLUEEstimator

        self._stat = stat
        self._costs = costs
        self._bkd: Backend[Array] = stat.bkd()
        self._nmodels = len(costs)
        self._nqoi = stat.nqoi()
        self._estimator_classes = estimator_classes or [MLBLUEEstimator]
        self._model_strategy = model_strategy or AllModelsStrategy()
        self._qoi_strategy = qoi_strategy or AllQoIStrategy()
        self._optimizer = optimizer
        self._objective = objective

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def _iter_configs(
        self,
    ) -> Iterator[Tuple[Type["BaseGroupACVEstimator[Array]"], List[int], List[int]]]:
        """Yield all (est_class, model_indices, qoi_indices) tuples."""
        for est_class in self._estimator_classes:
            for model_indices in self._model_strategy.subsets(self._nmodels):
                for qoi_indices in self._qoi_strategy.subsets(self._nqoi):
                    yield est_class, model_indices, qoi_indices

    def _create_estimator(
        self,
        est_class: Type["BaseGroupACVEstimator[Array]"],
        model_indices: List[int],
        qoi_indices: List[int],
    ) -> "BaseGroupACVEstimator[Array]":
        """Create estimator from config."""
        subset_stat = self._stat.subset(model_indices, qoi_indices)
        costs_np = self._bkd.to_numpy(self._costs)
        subset_costs = self._bkd.array([costs_np[i] for i in model_indices])
        return est_class(subset_stat, subset_costs, model_subsets=None)

    def search(
        self,
        target_cost: float,
        allow_failures: bool = False,
    ) -> GroupACVSearchResult[Array]:
        """Search all configuration dimensions.

        Parameters
        ----------
        target_cost : float
            The computational budget.
        allow_failures : bool
            If True, continue searching even when allocations fail.

        Returns
        -------
        GroupACVSearchResult
            Contains the best estimator/allocation and all candidates.
        """
        from pyapprox.typing.statest.groupacv.allocation import (
            GroupACVAllocationOptimizer,
        )

        all_allocations: List[
            Tuple["BaseGroupACVEstimator[Array]", "GroupACVAllocationResult[Array]"]
        ] = []

        for est_class, model_indices, qoi_indices in self._iter_configs():
            estimator = self._create_estimator(est_class, model_indices, qoi_indices)
            allocator = GroupACVAllocationOptimizer(
                estimator,
                optimizer=self._optimizer,
                objective=self._objective,
            )
            result = allocator.optimize(target_cost)
            all_allocations.append((estimator, result))

            if not result.success and not allow_failures:
                raise RuntimeError(
                    f"Allocation failed for {est_class.__name__} with "
                    f"models={model_indices}, qoi={qoi_indices}: {result.message}"
                )

        return self._build_search_result(all_allocations)

    def _build_search_result(
        self,
        all_allocations: List[
            Tuple["BaseGroupACVEstimator[Array]", "GroupACVAllocationResult[Array]"]
        ],
    ) -> GroupACVSearchResult[Array]:
        """Build result, selecting best allocation."""
        sorted_allocs = sorted(
            all_allocations,
            key=lambda x: (
                float(self._bkd.to_numpy(x[1].objective_value)[0])
                if x[1].success
                else float("inf")
            ),
        )

        for estimator, allocation in sorted_allocs:
            if allocation.success:
                estimator.set_allocation(allocation)
                return GroupACVSearchResult(
                    estimator=estimator,
                    allocation=allocation,
                    all_allocations=sorted_allocs,
                    estimator_classes=self._estimator_classes,
                    model_strategy=self._model_strategy,
                    qoi_strategy=self._qoi_strategy,
                )

        raise RuntimeError("No successful allocations found")
