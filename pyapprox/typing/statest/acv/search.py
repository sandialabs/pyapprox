"""Unified search for optimal ACV estimator configurations."""

from dataclasses import dataclass
from typing import (
    Callable,
    Generic,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from pyapprox.typing.statest.acv.base import ACVEstimator

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.statest.statistics import MultiOutputStatistic
from pyapprox.typing.statest.acv.allocation import (
    Allocator,
    ACVAllocationResult,
    default_allocator_factory,
)
from pyapprox.typing.statest.acv.strategies import (
    RecursionIndexStrategy,
    DefaultRecursionStrategy,
)
from pyapprox.typing.statest.strategies import (
    ModelSubsetStrategy,
    AllModelsStrategy,
    QoISubsetStrategy,
    AllQoIStrategy,
)


@dataclass
class SearchResult(Generic[Array]):
    """Result of estimator configuration search."""

    estimator: "ACVEstimator[Array]"
    allocation: ACVAllocationResult[Array]
    all_allocations: List[Tuple["ACVEstimator[Array]", ACVAllocationResult[Array]]]

    # Strategies and config used (for traceability)
    estimator_classes: List[Type["ACVEstimator[Array]"]]
    recursion_strategy: RecursionIndexStrategy
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
    ) -> List[Tuple["ACVEstimator[Array]", ACVAllocationResult[Array]]]:
        """Return list of (estimator, allocation) pairs for successful allocations."""
        return [(est, alloc) for est, alloc in self.all_allocations if alloc.success]

    def search_description(self) -> str:
        """Return human-readable description of the search configuration."""
        est_names = [c.__name__ for c in self.estimator_classes]
        return (
            f"Estimators: {est_names}; "
            f"Models: {self.model_strategy.description()}; "
            f"QoI: {self.qoi_strategy.description()}; "
            f"Recursion: {self.recursion_strategy.description()}"
        )


class ACVSearch(Generic[Array]):
    """Unified search over all ACV configuration dimensions.

    Searches the Cartesian product of:
    - Estimator types
    - Model subsets
    - QoI subsets
    - Recursion indices

    Unprovided strategies default to "no search" (single configuration).

    Parameters
    ----------
    stat : MultiOutputStatistic
        The statistic containing pilot quantities.
    costs : Array
        Model costs, shape (nmodels,).
    estimator_classes : List[Type[ACVEstimator]], optional
        Estimator classes to search. Defaults to [GMFEstimator].
    model_strategy : ModelSubsetStrategy, optional
        Strategy for model subset search. Defaults to AllModelsStrategy.
    qoi_strategy : QoISubsetStrategy, optional
        Strategy for QoI subset search. Defaults to AllQoIStrategy.
    recursion_strategy : RecursionIndexStrategy, optional
        Strategy for recursion index search. Defaults to DefaultRecursionStrategy.
    allocator_factory : Callable, optional
        Factory for creating allocators. Defaults to default_allocator_factory.
    """

    def __init__(
        self,
        stat: MultiOutputStatistic[Array],
        costs: Array,
        estimator_classes: Optional[List[Type["ACVEstimator[Array]"]]] = None,
        model_strategy: Optional[ModelSubsetStrategy] = None,
        qoi_strategy: Optional[QoISubsetStrategy] = None,
        recursion_strategy: Optional[RecursionIndexStrategy] = None,
        allocator_factory: Optional[
            Callable[["ACVEstimator[Array]"], Allocator[Array]]
        ] = None,
    ) -> None:
        from pyapprox.typing.statest.acv.variants import GMFEstimator

        self._stat = stat
        self._costs = costs
        self._bkd: Backend[Array] = stat.bkd()
        self._nmodels = len(costs)
        self._nqoi = stat.nqoi()
        self._estimator_classes = estimator_classes or [GMFEstimator]
        self._model_strategy = model_strategy or AllModelsStrategy()
        self._qoi_strategy = qoi_strategy or AllQoIStrategy()
        self._recursion_strategy = recursion_strategy or DefaultRecursionStrategy()
        self._allocator_factory = allocator_factory or default_allocator_factory

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def _iter_configs(
        self,
    ) -> Iterator[Tuple[Type["ACVEstimator[Array]"], List[int], List[int], Array]]:
        """Yield all (est_class, model_indices, qoi_indices, recursion_idx) tuples."""
        for est_class in self._estimator_classes:
            for model_indices in self._model_strategy.subsets(self._nmodels):
                subset_nmodels = len(model_indices)
                for qoi_indices in self._qoi_strategy.subsets(self._nqoi):
                    for recursion_idx in self._recursion_strategy.indices(
                        subset_nmodels, self._bkd
                    ):
                        yield est_class, model_indices, qoi_indices, recursion_idx

    def _create_estimator(
        self,
        est_class: Type["ACVEstimator[Array]"],
        model_indices: List[int],
        qoi_indices: List[int],
        recursion_idx: Array,
    ) -> "ACVEstimator[Array]":
        """Create estimator from config."""
        subset_stat = self._stat.subset(model_indices, qoi_indices)
        costs_np = self._bkd.to_numpy(self._costs)
        subset_costs = self._bkd.array([costs_np[i] for i in model_indices])
        return est_class(subset_stat, subset_costs, recursion_index=recursion_idx)

    def search(
        self,
        target_cost: float,
        allow_failures: bool = False,
    ) -> SearchResult[Array]:
        """Search all configuration dimensions.

        Parameters
        ----------
        target_cost : float
            The computational budget.
        allow_failures : bool
            If True, continue searching even when allocations fail.
            If False (default), raise RuntimeError on first failure.

        Returns
        -------
        SearchResult
            Contains the best estimator/allocation and all candidates.

        Raises
        ------
        RuntimeError
            If allow_failures is False and any allocation fails, or if no
            successful allocations are found.
        """
        all_allocations: List[
            Tuple["ACVEstimator[Array]", ACVAllocationResult[Array]]
        ] = []

        for est_class, model_indices, qoi_indices, recursion_idx in self._iter_configs():
            estimator = self._create_estimator(
                est_class, model_indices, qoi_indices, recursion_idx
            )
            allocator = self._allocator_factory(estimator)
            result = allocator.allocate(target_cost)
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
            Tuple["ACVEstimator[Array]", ACVAllocationResult[Array]]
        ],
    ) -> SearchResult[Array]:
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
                return SearchResult(
                    estimator=estimator,
                    allocation=allocation,
                    all_allocations=sorted_allocs,
                    estimator_classes=self._estimator_classes,
                    recursion_strategy=self._recursion_strategy,
                    model_strategy=self._model_strategy,
                    qoi_strategy=self._qoi_strategy,
                )

        raise RuntimeError("No successful allocations found")
