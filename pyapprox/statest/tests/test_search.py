"""Tests for unified multi-fidelity search."""

import unittest
from typing import Generic, Tuple

import numpy as np
import torch

from pyapprox.statest.acv.search import ACVSearch
from pyapprox.statest.acv.variants import GISEstimator, GMFEstimator
from pyapprox.statest.groupacv.mlblue import MLBLUEEstimator
from pyapprox.statest.groupacv.search import GroupACVSearch
from pyapprox.statest.groupacv.variants import GroupACVEstimatorNested
from pyapprox.statest.search import (
    EstimatorFamily,
    UnifiedSearchResult,
    unified_search,
)
from pyapprox.statest.statistics import MultiOutputMean
from pyapprox.util.backends.protocols import Array
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401


class TestUnifiedSearch(Generic[Array], unittest.TestCase):
    """Tests for unified_search."""

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

    def test_acv_only(self) -> None:
        """Search with ACV only."""
        acv_search = ACVSearch(
            self._stat,
            self._costs,
            estimator_classes=[GMFEstimator],
        )
        result = unified_search(acv_search=acv_search, target_cost=1000.0)

        self.assertIsInstance(result, UnifiedSearchResult)
        self.assertEqual(result.best_family, EstimatorFamily.ACV)
        self.assertIsNotNone(result.acv_result)
        self.assertIsNone(result.groupacv_result)
        self.assertIsNotNone(result.acv_objective)
        self.assertIsNone(result.groupacv_objective)

    def test_groupacv_only(self) -> None:
        """Search with GroupACV only."""
        groupacv_search = GroupACVSearch(
            self._stat,
            self._costs,
            estimator_classes=[MLBLUEEstimator],
        )
        result = unified_search(
            groupacv_search=groupacv_search,
            target_cost=1000.0,
            allow_failures=True,
        )

        self.assertIsInstance(result, UnifiedSearchResult)
        self.assertEqual(result.best_family, EstimatorFamily.GROUPACV)
        self.assertIsNone(result.acv_result)
        self.assertIsNotNone(result.groupacv_result)
        self.assertIsNone(result.acv_objective)
        self.assertIsNotNone(result.groupacv_objective)

    def test_both_families(self) -> None:
        """Search with both ACV and GroupACV."""
        acv_search = ACVSearch(
            self._stat,
            self._costs,
            estimator_classes=[GMFEstimator, GISEstimator],
        )
        groupacv_search = GroupACVSearch(
            self._stat,
            self._costs,
            estimator_classes=[MLBLUEEstimator],
        )
        result = unified_search(
            acv_search=acv_search,
            groupacv_search=groupacv_search,
            target_cost=1000.0,
            allow_failures=True,
        )

        self.assertIn(
            result.best_family, [EstimatorFamily.ACV, EstimatorFamily.GROUPACV]
        )
        self.assertIsNotNone(result.acv_result)
        self.assertIsNotNone(result.groupacv_result)

    def test_comparison_summary(self) -> None:
        """comparison_summary returns useful string."""
        acv_search = ACVSearch(self._stat, self._costs)
        result = unified_search(acv_search=acv_search, target_cost=1000.0)

        summary = result.comparison_summary()
        self.assertIn("Best overall", summary)
        self.assertIn("ACV best", summary)

    def test_comparison_summary_both_families(self) -> None:
        """comparison_summary includes both families when both searched."""
        acv_search = ACVSearch(self._stat, self._costs)
        groupacv_search = GroupACVSearch(self._stat, self._costs)
        result = unified_search(
            acv_search=acv_search,
            groupacv_search=groupacv_search,
            target_cost=1000.0,
            allow_failures=True,
        )

        summary = result.comparison_summary()
        self.assertIn("Best overall", summary)
        self.assertIn("ACV best", summary)
        self.assertIn("GroupACV best", summary)

    def test_no_search_raises(self) -> None:
        """Raises ValueError when no search provided."""
        with self.assertRaises(ValueError):
            unified_search(target_cost=1000.0)

    def test_best_objective_is_float(self) -> None:
        """best_objective is a Python float, not array."""
        acv_search = ACVSearch(self._stat, self._costs)
        result = unified_search(acv_search=acv_search, target_cost=1000.0)

        self.assertIsInstance(result.best_objective, float)
        self.assertIsInstance(result.acv_objective, float)

    def test_equivalent_mfmc_covariances_match(self) -> None:
        """Test that equivalent MFMC configs produce same covariance at same allocation.

        For 3-model MFMC:
        - ACV (GMF) with recursion_index=[0, 1]
        - GroupACV (Nested) with model_subsets=[[0,1], [1,2], [2]]

        Both should produce the same covariance (and thus objective) when given
        the same sample allocation. This test isolates the estimator structure
        from optimization differences.
        """
        # Create well-correlated covariance for MFMC
        nmodels = 3
        cov = np.eye(nmodels)
        for i in range(nmodels):
            for j in range(nmodels):
                if i != j:
                    # High correlation for MFMC to work well
                    cov[i, j] = 0.95 ** abs(i - j)

        stat_acv = MultiOutputMean(1, self._bkd)
        stat_acv.set_pilot_quantities(self._bkd.array(cov))

        stat_groupacv = MultiOutputMean(1, self._bkd)
        stat_groupacv.set_pilot_quantities(self._bkd.array(cov))

        # Costs must decrease for MFMC hierarchy
        costs = self._bkd.array([1.0, 0.1, 0.01])

        # MFMC subsets for GroupACV Nested
        mfmc_subsets = [
            self._bkd.asarray([0, 1], dtype=int),
            self._bkd.asarray([1, 2], dtype=int),
            self._bkd.asarray([2], dtype=int),
        ]

        # MFMC recursion index for ACV GMF
        mfmc_recursion = self._bkd.asarray([0, 1], dtype=int)

        # Create both estimators directly
        acv_est = GMFEstimator(stat_acv, costs, recursion_index=mfmc_recursion)
        groupacv_est = GroupACVEstimatorNested(
            stat_groupacv, costs, model_subsets=mfmc_subsets, reg_blue=0
        )

        # Use the same sample allocation for both
        # npartition_samples for MFMC: 3 partitions
        npartition_samples = self._bkd.array([10.0, 50.0, 200.0])

        # Compute covariances at the same allocation
        acv_cov = acv_est._covariance_from_npartition_samples(npartition_samples)
        groupacv_cov = groupacv_est._covariance_from_npartition_samples(
            npartition_samples
        )

        # Covariances should match exactly (same estimator structure)
        self._bkd.assert_allclose(acv_cov, groupacv_cov, rtol=1e-10)


# Note: NumPy backend does not yet support gradient-based optimization.
# Uncomment TestUnifiedSearchNumpy when optimization support is implemented.


class TestUnifiedSearchTorch(TestUnifiedSearch[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
