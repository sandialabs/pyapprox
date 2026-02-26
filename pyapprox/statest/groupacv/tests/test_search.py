"""Tests for GroupACV search classes."""

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
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import slow_test


class TestGroupACVSearch:
    """Tests for GroupACVSearch."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self._bkd = TorchBkd()
        np.random.seed(42)
        self._stat, self._costs = self._create_stat_and_costs()

    def _create_stat_and_costs(
        self, nmodels: int = 3, nqoi: int = 1
    ) -> Tuple[MultiOutputMean, ...]:
        """Create test statistic and costs."""
        nqoi_nmodels = nqoi * nmodels
        cov = np.eye(nqoi_nmodels)
        for q in range(nqoi):
            for i in range(nmodels):
                for j in range(nmodels):
                    if i != j:
                        corr = 0.9 ** abs(i - j)
                        cov[q * nmodels + i, q * nmodels + j] = corr

        stat = MultiOutputMean(nqoi, self._bkd)
        stat.set_pilot_quantities(self._bkd.array(cov))
        costs = self._bkd.array([100.0, 10.0, 1.0][:nmodels])
        return stat, costs

    def test_default_search(self) -> None:
        """Default search with MLBLUE."""
        search = GroupACVSearch(self._stat, self._costs)
        result = search.search(target_cost=1000.0, allow_failures=True)

        assert isinstance(result, GroupACVSearchResult)
        assert result.candidates_evaluated() == 1

    @slow_test
    def test_multiple_estimator_types(self) -> None:
        """Search multiple estimator types."""
        search = GroupACVSearch(
            self._stat,
            self._costs,
            estimator_classes=[MLBLUEEstimator, GroupACVEstimatorIS],
        )
        result = search.search(target_cost=1000.0, allow_failures=True)

        assert result.candidates_evaluated() == 2

    @slow_test
    def test_model_subset_search(self) -> None:
        """Model subset search."""
        stat, costs = self._create_stat_and_costs(nmodels=4)
        # Add fourth cost
        costs = self._bkd.array([100.0, 10.0, 1.0, 0.1])
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

    def test_bkd_method(self) -> None:
        """bkd() returns backend."""
        search = GroupACVSearch(self._stat, self._costs)
        assert search.bkd() is self._bkd

    @slow_test
    def test_search_result_methods(self) -> None:
        """Test GroupACVSearchResult methods."""
        search = GroupACVSearch(
            self._stat,
            self._costs,
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

    def test_search_raises_on_failure_when_not_allowed(self) -> None:
        """Test that search raises when allow_failures=False and allocation fails.

        ValueError is raised when budget is too small for min_nhf_samples.
        RuntimeError is raised when optimization runs but fails.
        """
        search = GroupACVSearch(self._stat, self._costs)
        with pytest.raises((RuntimeError, ValueError)):
            search.search(target_cost=0.01, allow_failures=False)

    def test_search_result_has_best_estimator(self) -> None:
        """Test that search result has best estimator configured."""
        search = GroupACVSearch(self._stat, self._costs)
        result = search.search(target_cost=1000.0, allow_failures=True)

        if result.allocation.success:
            # Estimator should have allocation set
            nsamples = result.estimator.npartition_samples()
            self._bkd.assert_allclose(
                nsamples,
                result.allocation.npartition_samples,
            )
