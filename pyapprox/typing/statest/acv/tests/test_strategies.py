"""Tests for ACV search strategies."""

from typing import Any, Generic
import unittest

from numpy.typing import NDArray
import torch

from pyapprox.typing.util.backends.protocols import Array
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401

from pyapprox.typing.statest.acv.strategies import (
    RecursionIndexStrategy,
    DefaultRecursionStrategy,
    FixedRecursionStrategy,
    ListRecursionStrategy,
    TreeDepthRecursionStrategy,
    HierarchicalPermutationRecursionStrategy,
    ModelSubsetStrategy,
    AllModelsStrategy,
    FixedSubsetStrategy,
    AllSubsetsStrategy,
    ListSubsetStrategy,
)


class TestRecursionIndexStrategy(Generic[Array], unittest.TestCase):
    """Tests for recursion index strategies."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def test_default_recursion_strategy(self):
        """Returns single default index."""
        strategy = DefaultRecursionStrategy()
        indices = strategy.indices(nmodels=4, bkd=self._bkd)
        self.assertEqual(len(indices), 1)
        self._bkd.assert_allclose(indices[0], self._bkd.array([0, 1, 2], dtype=int))

    def test_default_recursion_strategy_description(self):
        """Has valid description."""
        strategy = DefaultRecursionStrategy()
        desc = strategy.description()
        self.assertIsInstance(desc, str)
        self.assertIn("default", desc.lower())

    def test_fixed_recursion_strategy(self):
        """Returns specified index."""
        strategy = FixedRecursionStrategy(recursion_index=(0, 0, 1))
        indices = strategy.indices(nmodels=4, bkd=self._bkd)
        self.assertEqual(len(indices), 1)
        self._bkd.assert_allclose(indices[0], self._bkd.array([0, 0, 1], dtype=int))

    def test_fixed_recursion_strategy_description(self):
        """Has valid description."""
        strategy = FixedRecursionStrategy(recursion_index=(0, 0, 1))
        desc = strategy.description()
        self.assertIsInstance(desc, str)
        self.assertIn("fixed", desc.lower())

    def test_list_recursion_strategy(self):
        """Returns custom list."""
        strategy = ListRecursionStrategy(recursion_indices=((0, 1, 2), (0, 0, 0)))
        indices = strategy.indices(nmodels=4, bkd=self._bkd)
        self.assertEqual(len(indices), 2)
        self._bkd.assert_allclose(indices[0], self._bkd.array([0, 1, 2], dtype=int))
        self._bkd.assert_allclose(indices[1], self._bkd.array([0, 0, 0], dtype=int))

    def test_list_recursion_strategy_description(self):
        """Has valid description."""
        strategy = ListRecursionStrategy(recursion_indices=((0, 1, 2), (0, 0, 0)))
        desc = strategy.description()
        self.assertIsInstance(desc, str)
        self.assertIn("2", desc)

    def test_tree_depth_recursion_strategy(self):
        """Returns tree indices."""
        strategy = TreeDepthRecursionStrategy(max_depth=2)
        indices = strategy.indices(nmodels=4, bkd=self._bkd)
        self.assertGreater(len(indices), 1)
        # All should have length nmodels-1
        for idx in indices:
            self.assertEqual(len(idx), 3)

    def test_tree_depth_recursion_strategy_description(self):
        """Has valid description."""
        strategy = TreeDepthRecursionStrategy(max_depth=2)
        desc = strategy.description()
        self.assertIsInstance(desc, str)
        self.assertIn("2", desc)

    def test_hierarchical_permutation_recursion_strategy(self):
        """Generates all permutations of hierarchical indices."""
        strategy = HierarchicalPermutationRecursionStrategy()
        indices = strategy.indices(nmodels=4, bkd=self._bkd)
        # (nmodels-1)! = 3! = 6 permutations
        self.assertEqual(len(indices), 6)
        # Verify all are valid permutations of [0, 1, 2]
        for idx in indices:
            self.assertEqual(len(idx), 3)
            idx_set = set(int(x) for x in self._bkd.to_numpy(idx).tolist())
            self.assertEqual(idx_set, {0, 1, 2})

    def test_hierarchical_permutation_small(self):
        """Works for small nmodels."""
        strategy = HierarchicalPermutationRecursionStrategy()
        indices = strategy.indices(nmodels=3, bkd=self._bkd)
        # (nmodels-1)! = 2! = 2 permutations
        self.assertEqual(len(indices), 2)

    def test_hierarchical_permutation_description(self):
        """Has valid description."""
        strategy = HierarchicalPermutationRecursionStrategy()
        desc = strategy.description()
        self.assertIsInstance(desc, str)
        self.assertIn("permutation", desc.lower())

    def test_all_strategies_have_description(self):
        """All strategies have non-empty description."""
        strategies = [
            DefaultRecursionStrategy(),
            FixedRecursionStrategy(recursion_index=(0, 1)),
            ListRecursionStrategy(recursion_indices=((0, 1),)),
            TreeDepthRecursionStrategy(max_depth=2),
            HierarchicalPermutationRecursionStrategy(),
        ]
        for s in strategies:
            desc = s.description()
            self.assertIsInstance(desc, str)
            self.assertGreater(len(desc), 0)


class TestRecursionIndexStrategyNumpy(TestRecursionIndexStrategy[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestRecursionIndexStrategyTorch(TestRecursionIndexStrategy[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestModelSubsetStrategy(unittest.TestCase):
    """Tests for model subset strategies."""

    def test_all_models_strategy(self):
        """Returns all models."""
        strategy = AllModelsStrategy()
        subsets = strategy.subsets(nmodels=4)
        self.assertEqual(subsets, [[0, 1, 2, 3]])

    def test_all_models_strategy_description(self):
        """Has valid description."""
        strategy = AllModelsStrategy()
        desc = strategy.description()
        self.assertIsInstance(desc, str)
        self.assertIn("all", desc.lower())

    def test_fixed_subset_strategy(self):
        """Returns specified subset."""
        strategy = FixedSubsetStrategy(model_indices=(0, 1, 3))
        subsets = strategy.subsets(nmodels=4)
        self.assertEqual(subsets, [[0, 1, 3]])

    def test_fixed_subset_strategy_no_zero(self):
        """Raises ValueError if 0 not included."""
        with self.assertRaises(ValueError) as ctx:
            FixedSubsetStrategy(model_indices=(1, 2, 3))
        self.assertIn("0", str(ctx.exception))

    def test_fixed_subset_strategy_description(self):
        """Has valid description."""
        strategy = FixedSubsetStrategy(model_indices=(0, 1, 3))
        desc = strategy.description()
        self.assertIsInstance(desc, str)
        self.assertIn("fixed", desc.lower())

    def test_all_subsets_strategy(self):
        """Generates correct subsets."""
        strategy = AllSubsetsStrategy(min_models=2)
        subsets = strategy.subsets(nmodels=4)
        # 2-model: [0,1], [0,2], [0,3] = 3
        # 3-model: [0,1,2], [0,1,3], [0,2,3] = 3
        # 4-model: [0,1,2,3] = 1
        # Total = 7
        self.assertEqual(len(subsets), 7)
        for subset in subsets:
            self.assertIn(0, subset)

    def test_all_subsets_strategy_max_models(self):
        """Respects max_models."""
        strategy = AllSubsetsStrategy(min_models=2, max_models=3)
        subsets = strategy.subsets(nmodels=5)
        for subset in subsets:
            self.assertLessEqual(len(subset), 3)
            self.assertIn(0, subset)

    def test_all_subsets_strategy_description(self):
        """Has valid description."""
        strategy = AllSubsetsStrategy(min_models=2, max_models=3)
        desc = strategy.description()
        self.assertIsInstance(desc, str)
        self.assertIn("2", desc)
        self.assertIn("3", desc)

    def test_list_subset_strategy(self):
        """Returns custom list."""
        strategy = ListSubsetStrategy(model_subsets=((0, 1), (0, 2, 3)))
        subsets = strategy.subsets(nmodels=4)
        self.assertEqual(len(subsets), 2)
        self.assertEqual(subsets[0], [0, 1])
        self.assertEqual(subsets[1], [0, 2, 3])

    def test_list_subset_strategy_no_zero(self):
        """Raises ValueError if any subset missing 0."""
        with self.assertRaises(ValueError) as ctx:
            ListSubsetStrategy(model_subsets=((0, 1), (1, 2, 3)))
        self.assertIn("0", str(ctx.exception))

    def test_list_subset_strategy_description(self):
        """Has valid description."""
        strategy = ListSubsetStrategy(model_subsets=((0, 1), (0, 2, 3)))
        desc = strategy.description()
        self.assertIsInstance(desc, str)
        self.assertIn("2", desc)

    def test_all_strategies_have_description(self):
        """All strategies have non-empty description."""
        strategies = [
            AllModelsStrategy(),
            FixedSubsetStrategy(model_indices=(0, 1)),
            AllSubsetsStrategy(min_models=2),
            ListSubsetStrategy(model_subsets=((0, 1),)),
        ]
        for s in strategies:
            desc = s.description()
            self.assertIsInstance(desc, str)
            self.assertGreater(len(desc), 0)


if __name__ == "__main__":
    unittest.main()
