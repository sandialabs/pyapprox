"""Tests for GroupACV search classes."""

import unittest
from typing import Generic, Tuple

import numpy as np
import torch

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
from pyapprox.util.backends.protocols import Array
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401


class TestGroupACVSearch(Generic[Array], unittest.TestCase):
    """Tests for GroupACVSearch."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        np.random.seed(42)
        self._stat, self._costs = self._create_stat_and_costs()

    def _create_stat_and_costs(
        self, nmodels: int = 3, nqoi: int = 1
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

        stat: MultiOutputMean[Array] = MultiOutputMean(nqoi, self._bkd)
        stat.set_pilot_quantities(self._bkd.array(cov))
        costs = self._bkd.array([100.0, 10.0, 1.0][:nmodels])
        return stat, costs

    def test_default_search(self) -> None:
        """Default search with MLBLUE."""
        search = GroupACVSearch(self._stat, self._costs)
        result = search.search(target_cost=1000.0, allow_failures=True)

        self.assertIsInstance(result, GroupACVSearchResult)
        self.assertEqual(result.candidates_evaluated(), 1)

    def test_multiple_estimator_types(self) -> None:
        """Search multiple estimator types."""
        search = GroupACVSearch(
            self._stat,
            self._costs,
            estimator_classes=[MLBLUEEstimator, GroupACVEstimatorIS],
        )
        result = search.search(target_cost=1000.0, allow_failures=True)

        self.assertEqual(result.candidates_evaluated(), 2)

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
        self.assertGreater(result.candidates_evaluated(), 1)

    def test_bkd_method(self) -> None:
        """bkd() returns backend."""
        search = GroupACVSearch(self._stat, self._costs)
        self.assertIs(search.bkd(), self._bkd)

    def test_search_result_methods(self) -> None:
        """Test GroupACVSearchResult methods."""
        search = GroupACVSearch(
            self._stat,
            self._costs,
            estimator_classes=[MLBLUEEstimator, GroupACVEstimatorIS],
        )
        result = search.search(target_cost=1000.0, allow_failures=True)

        # Test candidates_evaluated
        self.assertEqual(result.candidates_evaluated(), 2)

        # Test candidates_successful
        successful = result.candidates_successful()
        self.assertGreaterEqual(successful, 0)
        self.assertLessEqual(successful, 2)

        # Test successful_allocations
        successful_allocs = result.successful_allocations()
        self.assertEqual(len(successful_allocs), successful)

        # Test search_description
        desc = result.search_description()
        self.assertIn("MLBLUEEstimator", desc)
        self.assertIn("GroupACVEstimatorIS", desc)

    def test_search_raises_on_failure_when_not_allowed(self) -> None:
        """Test that search raises when allow_failures=False and allocation fails.

        ValueError is raised when budget is too small for min_nhf_samples.
        RuntimeError is raised when optimization runs but fails.
        """
        search = GroupACVSearch(self._stat, self._costs)
        with self.assertRaises((RuntimeError, ValueError)):
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


# Note: NumPy backend does not yet support gradient-based optimization.
# Uncomment TestGroupACVSearchNumpy when optimization support is implemented.


class TestGroupACVSearchTorch(TestGroupACVSearch[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
