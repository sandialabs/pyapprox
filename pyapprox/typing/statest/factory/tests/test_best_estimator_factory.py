"""Tests for comprehensive search module.

Tests verify that the BestEstimatorFactory correctly searches over
estimator types, recursion indices, and model subsets.
"""

from typing import Any, Generic
import unittest

import numpy as np
from numpy.typing import NDArray
import torch

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array
from pyapprox.typing.util.test_utils import (
    load_tests,  # noqa: F401
    allocate_with_allocator,
)

from pyapprox.typing.statest.statistics import MultiOutputMean
from pyapprox.typing.statest.factory.best_estimator_factory import (
    BestEstimatorFactory,
)
from pyapprox.typing.statest.factory.registry import (
    compute_objective,
    register_estimator,
    get_registered_estimators,
)


class TestComputeObjective(Generic[Array], unittest.TestCase):
    """Tests for compute_objective function."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def test_scalar_covariance(self):
        """Test objective for scalar covariance."""
        bkd = self._bkd
        cov = bkd.asarray([[0.5]])
        obj = compute_objective(cov, bkd)
        expected = float(bkd.log(bkd.asarray(0.5)))
        self.assertAlmostEqual(obj, expected, places=10)

    def test_matrix_covariance(self):
        """Test objective for matrix covariance."""
        bkd = self._bkd
        cov = bkd.asarray([[1.0, 0.5], [0.5, 1.0]])
        obj = compute_objective(cov, bkd)
        # log(det(cov)) = log(1 - 0.25) = log(0.75)
        expected = float(bkd.log(bkd.asarray(0.75)))
        self.assertAlmostEqual(obj, expected, places=10)


class TestComputeObjectiveNumpy(TestComputeObjective[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestComputeObjectiveTorch(TestComputeObjective[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestBestEstimatorFactory(Generic[Array], unittest.TestCase):
    """Tests for BestEstimatorFactory class."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        np.random.seed(42)

    def _create_test_stat(self, nmodels: int = 3, nqoi: int = 1):
        """Create a test statistic with correlated models."""
        bkd = self._bkd

        # Create covariance with high correlations (good for MF methods)
        cov_np = np.eye(nmodels * nqoi)
        for i in range(nmodels):
            for j in range(nmodels):
                if i != j:
                    # Correlation decreases with model index difference
                    corr = 0.9 ** abs(i - j)
                    for q in range(nqoi):
                        cov_np[i * nqoi + q, j * nqoi + q] = corr

        stat = MultiOutputMean(nqoi=nqoi, bkd=bkd)
        stat.set_pilot_quantities(bkd.asarray(cov_np))
        return stat

    def test_basic_search(self):
        """Test basic search finds valid estimator."""
        bkd = self._bkd
        stat = self._create_test_stat(nmodels=3)
        costs = bkd.asarray([10.0, 1.0, 0.1])

        factory = BestEstimatorFactory(
            stat,
            costs,
            bkd,
            estimator_types=["gmf"],
            max_depth=2,
            max_nmodels=3,
        )
        allocate_with_allocator(factory, target_cost=100.0)

        # Should find a valid estimator
        best_est = factory.best_estimator()
        self.assertIsNotNone(best_est)
        self.assertEqual(factory.best_type(), "gmf")

    def test_search_with_multiple_types(self):
        """Test search with multiple estimator types."""
        bkd = self._bkd
        stat = self._create_test_stat(nmodels=3)
        costs = bkd.asarray([10.0, 1.0, 0.1])

        factory = BestEstimatorFactory(
            stat,
            costs,
            bkd,
            estimator_types=["gmf", "gis", "grd"],
            max_depth=2,
            max_nmodels=3,
        )
        allocate_with_allocator(factory, target_cost=100.0)

        # Best type should be one of the searched types
        self.assertIn(factory.best_type(), ["gmf", "gis", "grd"])

    def test_model_subset_enumeration(self):
        """Test search over model subsets."""
        bkd = self._bkd
        stat = self._create_test_stat(nmodels=4)
        costs = bkd.asarray([100.0, 10.0, 1.0, 0.1])

        factory = BestEstimatorFactory(
            stat,
            costs,
            bkd,
            estimator_types=["gmf"],
            max_depth=2,
            max_nmodels=4,
            min_nmodels=2,
            save_candidates=True,
        )
        allocate_with_allocator(factory, target_cost=1000.0)

        # Should have searched multiple subsets
        subsets_searched = set()
        for result in factory.candidate_results():
            subsets_searched.add(tuple(result.model_indices))
        self.assertGreater(len(subsets_searched), 1)

    def test_depth_limit(self):
        """Test that depth limit affects search."""
        bkd = self._bkd
        stat = self._create_test_stat(nmodels=4)
        costs = bkd.asarray([100.0, 10.0, 1.0, 0.1])

        # Search with depth=1 (only star topology)
        factory1 = BestEstimatorFactory(
            stat,
            costs,
            bkd,
            estimator_types=["gmf"],
            max_depth=1,
            max_nmodels=4,
            save_candidates=True,
        )
        allocate_with_allocator(factory1, target_cost=1000.0)

        # Search with depth=3 (more tree structures)
        factory2 = BestEstimatorFactory(
            stat,
            costs,
            bkd,
            estimator_types=["gmf"],
            max_depth=3,
            max_nmodels=4,
            save_candidates=True,
        )
        allocate_with_allocator(factory2, target_cost=1000.0)

        # More depth should mean more candidates searched
        stats1 = factory1.search_stats()
        stats2 = factory2.search_stats()
        self.assertLessEqual(stats1["total_candidates"], stats2["total_candidates"])

    def test_best_has_smallest_objective(self):
        """Test that best estimator has smallest objective."""
        bkd = self._bkd
        stat = self._create_test_stat(nmodels=3)
        costs = bkd.asarray([10.0, 1.0, 0.1])

        factory = BestEstimatorFactory(
            stat,
            costs,
            bkd,
            estimator_types=["gmf", "gis"],
            max_depth=2,
            max_nmodels=3,
            save_candidates=True,
        )
        allocate_with_allocator(factory, target_cost=100.0)

        # Best objective should be smallest among successful candidates
        best_obj = factory.best_objective_value()
        for result in factory.candidate_results():
            if result.success:
                self.assertGreaterEqual(result.objective_value, best_obj - 1e-10)

    def test_objective_matches_covariance(self):
        """Test that objective matches log-det of covariance."""
        bkd = self._bkd
        stat = self._create_test_stat(nmodels=3)
        costs = bkd.asarray([10.0, 1.0, 0.1])

        factory = BestEstimatorFactory(
            stat, costs, bkd, estimator_types=["gmf"], max_depth=2
        )
        allocate_with_allocator(factory, target_cost=100.0)

        # Objective should equal log-det of optimized_covariance
        cov = factory.optimized_covariance()
        computed_obj = compute_objective(cov, bkd)
        bkd.assert_allclose(
            bkd.asarray([factory.best_objective_value()]),
            bkd.asarray([computed_obj]),
            rtol=1e-10,
        )

    def test_failure_handling(self):
        """Test graceful failure handling."""
        bkd = self._bkd

        # Create a poorly conditioned stat that might cause failures
        stat = MultiOutputMean(nqoi=1, bkd=bkd)
        # Near-singular covariance
        cov = bkd.asarray([[1.0, 0.99], [0.99, 1.0]])
        stat.set_pilot_quantities(cov)
        costs = bkd.asarray([10.0, 1.0])

        factory = BestEstimatorFactory(
            stat,
            costs,
            bkd,
            estimator_types=["gmf"],
            allow_failures=True,
        )
        # Should not raise even if some candidates fail
        allocate_with_allocator(factory, target_cost=100.0)

    def test_require_hf(self):
        """Test that require_hf ensures model 0 is always included."""
        bkd = self._bkd
        stat = self._create_test_stat(nmodels=4)
        costs = bkd.asarray([100.0, 10.0, 1.0, 0.1])

        factory = BestEstimatorFactory(
            stat,
            costs,
            bkd,
            estimator_types=["gmf"],
            max_nmodels=3,
            require_hf=True,
            save_candidates=True,
        )
        allocate_with_allocator(factory, target_cost=1000.0)

        # All subsets should include model 0
        for result in factory.candidate_results():
            self.assertIn(0, result.model_indices)

    def test_nsamples_per_model(self):
        """Test nsamples_per_model delegation."""
        bkd = self._bkd
        stat = self._create_test_stat(nmodels=3)
        costs = bkd.asarray([10.0, 1.0, 0.1])

        factory = BestEstimatorFactory(stat, costs, bkd, estimator_types=["gmf"])
        allocate_with_allocator(factory, target_cost=100.0)

        nsamples = factory.nsamples_per_model()
        # Should have positive samples for each model in best subset
        self.assertEqual(len(nsamples), len(factory.best_models()))
        for n in bkd.to_numpy(nsamples):
            self.assertGreater(n, 0)

    def test_repr(self):
        """Test string representation."""
        bkd = self._bkd
        stat = self._create_test_stat(nmodels=3)
        costs = bkd.asarray([10.0, 1.0, 0.1])

        factory = BestEstimatorFactory(stat, costs, bkd)

        # Before search
        self.assertIn("not searched", repr(factory))

        # After search
        allocate_with_allocator(factory, target_cost=100.0)
        self.assertIn("BestEstimatorFactory", repr(factory))
        self.assertIn(factory.best_type(), repr(factory))


class TestBestEstimatorFactoryTorch(TestBestEstimatorFactory[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestMultiQoI(Generic[Array], unittest.TestCase):
    """Tests for multi-QoI support."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        np.random.seed(42)

    def test_multi_qoi_search(self):
        """Test search with multiple QoIs."""
        bkd = self._bkd
        nmodels = 3
        nqoi = 2

        # Create block-diagonal covariance
        cov_np = np.zeros((nmodels * nqoi, nmodels * nqoi))
        for i in range(nmodels):
            for j in range(nmodels):
                corr = 0.9 ** abs(i - j)
                for q in range(nqoi):
                    cov_np[i * nqoi + q, j * nqoi + q] = (
                        corr if i == j else corr * 0.8
                    )

        stat = MultiOutputMean(nqoi=nqoi, bkd=bkd)
        stat.set_pilot_quantities(bkd.asarray(cov_np))
        costs = bkd.asarray([10.0, 1.0, 0.1])

        factory = BestEstimatorFactory(
            stat, costs, bkd, estimator_types=["gmf"], max_depth=2
        )
        allocate_with_allocator(factory, target_cost=100.0)

        # Should work with multi-QoI
        self.assertIsNotNone(factory.best_estimator())
        cov = factory.optimized_covariance()
        # Covariance should be nqoi x nqoi
        self.assertEqual(cov.shape, (nqoi, nqoi))


class TestMultiQoITorch(TestMultiQoI[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestEstimatorRegistry(unittest.TestCase):
    """Tests for estimator registry functionality."""

    def test_default_estimators_registered(self):
        """Test that default estimators are in the registry."""
        registry = get_registered_estimators()
        self.assertIn("gmf", registry)
        self.assertIn("gis", registry)
        self.assertIn("grd", registry)
        self.assertIn("mfmc", registry)
        self.assertIn("mlmc", registry)

    def test_gmf_requires_recursion(self):
        """Test that GMF is marked as requiring recursion index."""
        registry = get_registered_estimators()
        _, requires_recursion = registry["gmf"]
        self.assertTrue(requires_recursion)

    def test_mfmc_no_recursion(self):
        """Test that MFMC is marked as not requiring recursion index."""
        registry = get_registered_estimators()
        _, requires_recursion = registry["mfmc"]
        self.assertFalse(requires_recursion)

    def test_register_invalid_class_raises(self):
        """Test that registering a class without required methods raises."""

        class InvalidEstimator:
            pass

        with self.assertRaises(TypeError) as context:
            register_estimator("invalid", InvalidEstimator)

        self.assertIn("does not satisfy EstimatorProtocol", str(context.exception))
        self.assertIn("Missing methods", str(context.exception))

    def test_register_partial_class_raises(self):
        """Test that registering a class with some methods still raises."""

        class PartialEstimator:
            def bkd(self):
                pass

            def allocate_samples(self, target_cost):
                pass

        with self.assertRaises(TypeError) as context:
            register_estimator("partial", PartialEstimator)

        self.assertIn("does not satisfy EstimatorProtocol", str(context.exception))


if __name__ == "__main__":
    unittest.main()
