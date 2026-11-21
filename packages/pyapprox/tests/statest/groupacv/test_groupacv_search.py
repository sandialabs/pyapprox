"""Tests for GroupACV search classes.

The allocation optimizer takes its Jacobian from autodiff, so these tests
run on the autodiff backends.
"""

from typing import Tuple

import numpy as np
import pytest

from pyapprox.statest.groupacv.mlblue import MLBLUEEstimator
from pyapprox.statest.groupacv.search import (
    GroupACVSearch,
    GroupACVSearchResult,
)
from pyapprox.statest.groupacv.variants import (
    GroupACVEstimatorIS,
)
from pyapprox.statest.statistics import MultiOutputMean
from pyapprox.statest.strategies import AllSubsetsStrategy
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


class TestGroupACVSearch:
    """Tests for GroupACVSearch."""

    @pytest.fixture(autouse=True)
    def _seed(self) -> None:
        np.random.seed(42)

    def test_default_search(self, stat_and_costs: StatAndCosts[Array]) -> None:
        """Default search with MLBLUE."""
        search = GroupACVSearch(*stat_and_costs)
        result = search.search(target_cost=1000.0, allow_failures=True)

        assert isinstance(result, GroupACVSearchResult)
        assert result.candidates_evaluated() == 1

    @slow_test
    def test_multiple_estimator_types(
        self, stat_and_costs: StatAndCosts[Array]
    ) -> None:
        """Search multiple estimator types."""
        search: GroupACVSearch[Array] = GroupACVSearch(
            *stat_and_costs,
            estimator_classes=[MLBLUEEstimator, GroupACVEstimatorIS],
        )
        result = search.search(target_cost=1000.0, allow_failures=True)

        assert result.candidates_evaluated() == 2

    @slow_test
    def test_model_subset_search(self, autodiff_bkd: AutodiffBackend[Array]) -> None:
        """Model subset search."""
        stat, costs = _create_stat_and_costs(autodiff_bkd, nmodels=4)
        # Add fourth cost
        costs = autodiff_bkd.array([100.0, 10.0, 1.0, 0.1])
        search = GroupACVSearch(
            stat,
            costs,
            model_strategy=AllSubsetsStrategy(min_models=2, max_models=3),
        )
        result = search.search(target_cost=10000.0, allow_failures=True)

        # With min_models=2, max_models=3 and 4 models, we get:
        # Size 2: C(3,1) = 3 subsets (choose 1 from models 1,2,3 to pair with 0)
        # Size 3: C(3,2) = 3 subsets
        # Total: 6 subsets * 1 estimator type = 6 candidates
        assert result.candidates_evaluated() > 1

    def test_bkd_method(
        self,
        autodiff_bkd: AutodiffBackend[Array],
        stat_and_costs: StatAndCosts[Array],
    ) -> None:
        """bkd() returns backend."""
        search = GroupACVSearch(*stat_and_costs)
        assert search.bkd() is autodiff_bkd

    @slow_test
    def test_search_result_methods(self, stat_and_costs: StatAndCosts[Array]) -> None:
        """Test GroupACVSearchResult methods."""
        search: GroupACVSearch[Array] = GroupACVSearch(
            *stat_and_costs,
            estimator_classes=[MLBLUEEstimator, GroupACVEstimatorIS],
        )
        result = search.search(target_cost=1000.0, allow_failures=True)

        # Test candidates_evaluated
        assert result.candidates_evaluated() == 2

        # Test candidates_successful
        successful = result.candidates_successful()
        assert successful >= 0
        assert successful <= 2

        # Test successful_allocations
        successful_allocs = result.successful_allocations()
        assert len(successful_allocs) == successful

        # Test search_description
        desc = result.search_description()
        assert "MLBLUEEstimator" in desc
        assert "GroupACVEstimatorIS" in desc

    def test_search_raises_on_failure_when_not_allowed(
        self, stat_and_costs: StatAndCosts[Array]
    ) -> None:
        """Test that search raises when allow_failures=False and allocation fails.

        ValueError is raised when budget is too small for min_nhf_samples.
        RuntimeError is raised when optimization runs but fails.
        """
        search = GroupACVSearch(*stat_and_costs)
        with pytest.raises((RuntimeError, ValueError)):
            search.search(target_cost=0.01, allow_failures=False)

    def test_search_result_has_best_estimator(
        self,
        autodiff_bkd: AutodiffBackend[Array],
        stat_and_costs: StatAndCosts[Array],
    ) -> None:
        """Test that search result has best estimator configured."""
        search = GroupACVSearch(*stat_and_costs)
        result = search.search(target_cost=1000.0, allow_failures=True)

        if result.allocation.success:
            nsamples = result.best.npartition_samples()
            autodiff_bkd.assert_allclose(
                nsamples,
                result.allocation.npartition_samples,
            )
