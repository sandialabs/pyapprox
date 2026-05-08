"""Tests for ACV search strategies."""

import pytest

from pyapprox.statest.acv.strategies import (
    DefaultRecursionStrategy,
    FixedRecursionStrategy,
    HierarchicalPermutationRecursionStrategy,
    ListRecursionStrategy,
    TreeDepthRecursionStrategy,
)
from pyapprox.statest.strategies import (
    AllModelsStrategy,
    AllSubsetsStrategy,
    FixedSubsetStrategy,
    ListSubsetStrategy,
)


class TestRecursionIndexStrategy:
    """Tests for recursion index strategies."""

    def test_default_recursion_strategy(self, bkd):
        """Returns single default index."""
        strategy = DefaultRecursionStrategy()
        indices = strategy.indices(nmodels=4, bkd=bkd)
        assert len(indices) == 1
        bkd.assert_allclose(indices[0], bkd.array([0, 1, 2], dtype=int))

    def test_default_recursion_strategy_description(self, bkd):
        """Has valid description."""
        strategy = DefaultRecursionStrategy()
        desc = strategy.description()
        assert isinstance(desc, str)
        assert "default" in desc.lower()

    def test_fixed_recursion_strategy(self, bkd):
        """Returns specified index."""
        strategy = FixedRecursionStrategy(recursion_index=(0, 0, 1))
        indices = strategy.indices(nmodels=4, bkd=bkd)
        assert len(indices) == 1
        bkd.assert_allclose(indices[0], bkd.array([0, 0, 1], dtype=int))

    def test_fixed_recursion_strategy_description(self, bkd):
        """Has valid description."""
        strategy = FixedRecursionStrategy(recursion_index=(0, 0, 1))
        desc = strategy.description()
        assert isinstance(desc, str)
        assert "fixed" in desc.lower()

    def test_list_recursion_strategy(self, bkd):
        """Returns custom list."""
        strategy = ListRecursionStrategy(recursion_indices=((0, 1, 2), (0, 0, 0)))
        indices = strategy.indices(nmodels=4, bkd=bkd)
        assert len(indices) == 2
        bkd.assert_allclose(indices[0], bkd.array([0, 1, 2], dtype=int))
        bkd.assert_allclose(indices[1], bkd.array([0, 0, 0], dtype=int))

    def test_list_recursion_strategy_description(self, bkd):
        """Has valid description."""
        strategy = ListRecursionStrategy(recursion_indices=((0, 1, 2), (0, 0, 0)))
        desc = strategy.description()
        assert isinstance(desc, str)
        assert "2" in desc

    def test_tree_depth_recursion_strategy(self, bkd):
        """Returns tree indices."""
        strategy = TreeDepthRecursionStrategy(max_depth=2)
        indices = strategy.indices(nmodels=4, bkd=bkd)
        assert len(indices) > 1
        # All should have length nmodels-1
        for idx in indices:
            assert len(idx) == 3

    def test_tree_depth_recursion_strategy_description(self, bkd):
        """Has valid description."""
        strategy = TreeDepthRecursionStrategy(max_depth=2)
        desc = strategy.description()
        assert isinstance(desc, str)
        assert "2" in desc

    def test_hierarchical_permutation_recursion_strategy(self, bkd):
        """Generates all permutations of hierarchical indices."""
        strategy = HierarchicalPermutationRecursionStrategy()
        indices = strategy.indices(nmodels=4, bkd=bkd)
        # (nmodels-1)! = 3! = 6 permutations
        assert len(indices) == 6
        # Verify all are valid permutations of [0, 1, 2]
        for idx in indices:
            assert len(idx) == 3
            idx_set = set(int(x) for x in bkd.to_numpy(idx).tolist())
            assert idx_set == {0, 1, 2}

    def test_hierarchical_permutation_small(self, bkd):
        """Works for small nmodels."""
        strategy = HierarchicalPermutationRecursionStrategy()
        indices = strategy.indices(nmodels=3, bkd=bkd)
        # (nmodels-1)! = 2! = 2 permutations
        assert len(indices) == 2

    def test_hierarchical_permutation_description(self, bkd):
        """Has valid description."""
        strategy = HierarchicalPermutationRecursionStrategy()
        desc = strategy.description()
        assert isinstance(desc, str)
        assert "permutation" in desc.lower()

    def test_all_strategies_have_description(self, bkd):
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
            assert isinstance(desc, str)
            assert len(desc) > 0


class TestModelSubsetStrategy:
    """Tests for model subset strategies."""

    def test_all_models_strategy(self):
        """Returns all models."""
        strategy = AllModelsStrategy()
        subsets = strategy.subsets(nmodels=4)
        assert subsets == [[0, 1, 2, 3]]

    def test_all_models_strategy_description(self):
        """Has valid description."""
        strategy = AllModelsStrategy()
        desc = strategy.description()
        assert isinstance(desc, str)
        assert "all" in desc.lower()

    def test_fixed_subset_strategy(self):
        """Returns specified subset."""
        strategy = FixedSubsetStrategy(model_indices=(0, 1, 3))
        subsets = strategy.subsets(nmodels=4)
        assert subsets == [[0, 1, 3]]

    def test_fixed_subset_strategy_no_zero(self):
        """Raises ValueError if 0 not included."""
        with pytest.raises(ValueError, match="0"):
            FixedSubsetStrategy(model_indices=(1, 2, 3))

    def test_fixed_subset_strategy_description(self):
        """Has valid description."""
        strategy = FixedSubsetStrategy(model_indices=(0, 1, 3))
        desc = strategy.description()
        assert isinstance(desc, str)
        assert "fixed" in desc.lower()

    def test_all_subsets_strategy(self):
        """Generates correct subsets."""
        strategy = AllSubsetsStrategy(min_models=2)
        subsets = strategy.subsets(nmodels=4)
        # 2-model: [0,1], [0,2], [0,3] = 3
        # 3-model: [0,1,2], [0,1,3], [0,2,3] = 3
        # 4-model: [0,1,2,3] = 1
        # Total = 7
        assert len(subsets) == 7
        for subset in subsets:
            assert 0 in subset

    def test_all_subsets_strategy_max_models(self):
        """Respects max_models."""
        strategy = AllSubsetsStrategy(min_models=2, max_models=3)
        subsets = strategy.subsets(nmodels=5)
        for subset in subsets:
            assert len(subset) <= 3
            assert 0 in subset

    def test_all_subsets_strategy_description(self):
        """Has valid description."""
        strategy = AllSubsetsStrategy(min_models=2, max_models=3)
        desc = strategy.description()
        assert isinstance(desc, str)
        assert "2" in desc
        assert "3" in desc

    def test_list_subset_strategy(self):
        """Returns custom list."""
        strategy = ListSubsetStrategy(model_subsets=((0, 1), (0, 2, 3)))
        subsets = strategy.subsets(nmodels=4)
        assert len(subsets) == 2
        assert subsets[0] == [0, 1]
        assert subsets[1] == [0, 2, 3]

    def test_list_subset_strategy_no_zero(self):
        """Raises ValueError if any subset missing 0."""
        with pytest.raises(ValueError, match="0"):
            ListSubsetStrategy(model_subsets=((0, 1), (1, 2, 3)))

    def test_list_subset_strategy_description(self):
        """Has valid description."""
        strategy = ListSubsetStrategy(model_subsets=((0, 1), (0, 2, 3)))
        desc = strategy.description()
        assert isinstance(desc, str)
        assert "2" in desc

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
            assert isinstance(desc, str)
            assert len(desc) > 0
