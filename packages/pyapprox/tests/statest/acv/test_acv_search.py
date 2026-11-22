"""Tests for ACV search classes.

The allocation optimizer takes its Jacobian from autodiff, so these tests
run on the autodiff backends.
"""

from typing import Tuple

import numpy as np
import pytest

from pyapprox.statest.acv.search import (
    ACVSearch,
)
from pyapprox.statest.acv.strategies import (
    DefaultRecursionStrategy,
    ListRecursionStrategy,
)
from pyapprox.statest.acv.variants import (
    GISEstimator,
    GMFEstimator,
    GRDEstimator,
)
from pyapprox.statest.statistics import MultiOutputMean
from pyapprox.statest.strategies import (
    AllQoIStrategy,
    AllQoISubsetsStrategy,
    AllSubsetsStrategy,
    FixedSubsetStrategy,
)
from pyapprox.util.backends.autodiff import AutodiffBackend
from pyapprox.util.backends.protocols import Array, Backend
from tests._helpers.markers import slow_test

StatAndCosts = Tuple[MultiOutputMean[Array], Array]


def _create_stat_and_costs(
    bkd: Backend[Array], nmodels: int = 3, nqoi: int = 1
) -> Tuple[MultiOutputMean[Array], Array]:
    """Create test statistic and costs."""
    nqoi_nmodels = nqoi * nmodels
    cov = np.eye(nqoi_nmodels)
    for q in range(nqoi):
        for i in range(nmodels):
            for j in range(nmodels):
                if i != j:
                    corr = 0.9 ** abs(i - j)
                    cov[q * nmodels + i, q * nmodels + j] = corr

    stat = MultiOutputMean(nqoi, bkd)
    stat.set_pilot_quantities(bkd.array(cov))
    costs = bkd.array([100.0, 10.0, 1.0][:nmodels])
    return stat, costs


@pytest.fixture
def stat_and_costs(
    autodiff_bkd: AutodiffBackend[Array],
) -> Tuple[MultiOutputMean[Array], Array]:
    """Three-model, one-QoI statistic and costs."""
    return _create_stat_and_costs(autodiff_bkd)


class TestACVSearch:
    """Tests for ACVSearch."""

    @pytest.fixture(autouse=True)
    def _seed(self) -> None:
        np.random.seed(42)

    def test_non_constructor_refused(self, stat_and_costs: StatAndCosts[Array]) -> None:
        """An estimator_classes entry that cannot build an estimator is
        refused at construction."""
        with pytest.raises(TypeError, match="ACVEstimatorConstructor"):
            ACVSearch(
                *stat_and_costs,
                estimator_classes=["GMFEstimator"],  # type: ignore[list-item]
            )

    def test_acv_search_default_strategies(
        self, stat_and_costs: StatAndCosts[Array]
    ) -> None:
        """Default strategies give single configuration."""
        search = ACVSearch(*stat_and_costs)
        result = search.search(target_cost=1000.0)
        assert result.allocation.success
        assert result.candidates_evaluated() == 1

    def test_acv_search_model_subset_only(
        self, autodiff_bkd: AutodiffBackend[Array]
    ) -> None:
        """Model subset search works."""
        stat, costs = _create_stat_and_costs(autodiff_bkd, nmodels=4)
        search = ACVSearch(
            stat,
            costs,
            model_strategy=AllSubsetsStrategy(min_models=2, max_models=3),
        )
        result = search.search(target_cost=10000.0, allow_failures=True)
        assert result.candidates_evaluated() > 1

    @slow_test
    def test_acv_search_qoi_subset_only(
        self, autodiff_bkd: AutodiffBackend[Array]
    ) -> None:
        """QoI subset search works."""
        stat, costs = _create_stat_and_costs(autodiff_bkd, nmodels=3, nqoi=3)
        search = ACVSearch(
            stat,
            costs,
            qoi_strategy=AllQoISubsetsStrategy(min_qoi=1, max_qoi=2),
        )
        result = search.search(target_cost=1000.0, allow_failures=True)
        assert result.candidates_evaluated() > 1

    @slow_test
    def test_acv_search_qoi_with_required(
        self, autodiff_bkd: AutodiffBackend[Array]
    ) -> None:
        """QoI subset search with required_qoi works."""
        stat, costs = _create_stat_and_costs(autodiff_bkd, nmodels=3, nqoi=4)
        search = ACVSearch(
            stat,
            costs,
            qoi_strategy=AllQoISubsetsStrategy(min_qoi=2, required_qoi=(0,)),
        )
        result = search.search(target_cost=1000.0, allow_failures=True)
        assert result.candidates_evaluated() > 1

    def test_acv_search_recursion_only(
        self, stat_and_costs: StatAndCosts[Array]
    ) -> None:
        """Recursion search works."""
        search = ACVSearch(
            *stat_and_costs,
            recursion_strategy=ListRecursionStrategy(
                recursion_indices=((0, 1), (0, 0))
            ),
        )
        result = search.search(target_cost=1000.0, allow_failures=True)
        assert result.candidates_evaluated() == 2

    def test_acv_search_multiple_estimator_types(
        self, stat_and_costs: StatAndCosts[Array]
    ) -> None:
        """Multiple estimator types work."""
        search = ACVSearch(
            *stat_and_costs,
            estimator_classes=[GMFEstimator, GISEstimator],
        )
        result = search.search(target_cost=1000.0, allow_failures=True)
        assert result.candidates_evaluated() == 2

    def test_iter_configs_count(self, stat_and_costs: StatAndCosts[Array]) -> None:
        """_iter_configs yields correct number of configurations."""
        search = ACVSearch(
            *stat_and_costs,
            estimator_classes=[GMFEstimator, GISEstimator],  # 2
            model_strategy=FixedSubsetStrategy(model_indices=(0, 1, 2)),  # 1
            qoi_strategy=AllQoIStrategy(),  # 1
            recursion_strategy=ListRecursionStrategy(
                recursion_indices=((0, 1), (0, 0))
            ),  # 2
        )
        configs = list(search._iter_configs())
        assert len(configs) == 2 * 1 * 1 * 2  # 4

    def test_acv_search_failure_raises(
        self, stat_and_costs: StatAndCosts[Array]
    ) -> None:
        """Raises RuntimeError when allow_failures=False and allocation fails."""
        search = ACVSearch(*stat_and_costs)
        with pytest.raises(RuntimeError, match="(?i)failed"):
            search.search(target_cost=0.1, allow_failures=False)

    def test_acv_search_allow_failures(
        self, stat_and_costs: StatAndCosts[Array]
    ) -> None:
        """Continues on failure when allow_failures=True."""
        search = ACVSearch(*stat_and_costs)
        try:
            search.search(target_cost=0.1, allow_failures=True)
        except RuntimeError as e:
            assert "No successful" in str(e)

    def test_search_result_methods(self, stat_and_costs: StatAndCosts[Array]) -> None:
        """SearchResult methods work correctly."""
        search = ACVSearch(
            *stat_and_costs,
            estimator_classes=[GMFEstimator, GISEstimator],
        )
        result = search.search(target_cost=1000.0, allow_failures=True)

        assert result.candidates_evaluated() == 2
        assert result.candidates_successful() > 0
        assert len(result.successful_allocations()) > 0

        desc = result.search_description()
        assert "GMFEstimator" in desc
        assert "GISEstimator" in desc

    def test_search_result_stores_estimator_classes(
        self, stat_and_costs: StatAndCosts[Array]
    ) -> None:
        """SearchResult stores estimator_classes for traceability."""
        search = ACVSearch(
            *stat_and_costs,
            estimator_classes=[GMFEstimator, GRDEstimator],
        )
        result = search.search(target_cost=1000.0, allow_failures=True)
        assert result.estimator_classes == [GMFEstimator, GRDEstimator]

    def test_search_result_stores_strategies(
        self, stat_and_costs: StatAndCosts[Array]
    ) -> None:
        """SearchResult stores all strategy objects."""
        model_strat = FixedSubsetStrategy(model_indices=(0, 1, 2))
        qoi_strat = AllQoIStrategy()
        rec_strat: DefaultRecursionStrategy[Array] = DefaultRecursionStrategy()

        search = ACVSearch(
            *stat_and_costs,
            model_strategy=model_strat,
            qoi_strategy=qoi_strat,
            recursion_strategy=rec_strat,
        )
        result = search.search(target_cost=1000.0)

        assert result.model_strategy is model_strat
        assert result.qoi_strategy is qoi_strat
        assert result.recursion_strategy is rec_strat

    def test_best_allocation_is_selected(
        self,
        autodiff_bkd: AutodiffBackend[Array],
        stat_and_costs: StatAndCosts[Array],
    ) -> None:
        """Best allocation (lowest objective) is selected."""
        bkd = autodiff_bkd
        search = ACVSearch(
            *stat_and_costs,
            estimator_classes=[GMFEstimator, GISEstimator, GRDEstimator],
        )
        result = search.search(target_cost=1000.0, allow_failures=True)

        # The selected allocation should be the best one
        successful = result.successful_allocations()
        if len(successful) > 1:
            best_obj = float(bkd.to_numpy(result.allocation.objective_value)[0])
            for _, alloc in successful:
                obj = float(bkd.to_numpy(alloc.objective_value)[0])
                assert obj >= best_obj

    def test_best_is_fitted_estimator(
        self, stat_and_costs: StatAndCosts[Array]
    ) -> None:
        """Returned best is a FittedACVEstimator."""
        from pyapprox.statest.acv.base import FittedACVEstimator

        search = ACVSearch(*stat_and_costs)
        result = search.search(target_cost=1000.0)

        assert isinstance(result.best, FittedACVEstimator)
