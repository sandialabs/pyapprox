"""Tests for tree enumeration module.

Tests verify that tree generation matches legacy behavior and produces
correct recursion indices for ACV estimators.
"""

from typing import Any, Generic
import unittest

from numpy.typing import NDArray
import torch

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401

from pyapprox.typing.statest.factory.tree_enumeration import (
    ModelTree,
    generate_all_trees,
    get_acv_recursion_indices,
    count_recursion_indices,
)


class TestModelTree(Generic[Array], unittest.TestCase):
    """Tests for ModelTree class."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def test_simple_tree_2_models(self):
        """Test 2-model tree (single LF)."""
        bkd = self._bkd
        tree = ModelTree(0, [1], bkd)
        index = tree.to_index()
        # Model 1 has parent 0
        bkd.assert_allclose(index, bkd.asarray([0]))

    def test_star_topology_3_models(self):
        """Test star topology (all LF coupled with HF)."""
        bkd = self._bkd
        # Both LF models directly coupled with HF
        tree = ModelTree(0, [ModelTree(1, [], bkd), ModelTree(2, [], bkd)], bkd)
        index = tree.to_index()
        bkd.assert_allclose(index, bkd.asarray([0, 0]))

    def test_chain_topology_3_models(self):
        """Test chain/MLMC topology (successive coupling)."""
        bkd = self._bkd
        # Model 2 coupled with Model 1, Model 1 with HF
        tree = ModelTree(0, [ModelTree(1, [ModelTree(2, [], bkd)], bkd)], bkd)
        index = tree.to_index()
        bkd.assert_allclose(index, bkd.asarray([0, 1]))

    def test_num_nodes(self):
        """Test node counting."""
        bkd = self._bkd
        # Single node
        tree = ModelTree(0, [], bkd)
        self.assertEqual(tree.num_nodes(), 1)

        # Three nodes (star)
        tree = ModelTree(0, [ModelTree(1, [], bkd), ModelTree(2, [], bkd)], bkd)
        self.assertEqual(tree.num_nodes(), 3)

        # Three nodes (chain)
        tree = ModelTree(0, [ModelTree(1, [ModelTree(2, [], bkd)], bkd)], bkd)
        self.assertEqual(tree.num_nodes(), 3)

    def test_int_children_conversion(self):
        """Test that integer children are converted to ModelTree."""
        bkd = self._bkd
        tree = ModelTree(0, [1, 2], bkd)
        index = tree.to_index()
        bkd.assert_allclose(index, bkd.asarray([0, 0]))


class TestModelTreeNumpy(TestModelTree[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestModelTreeTorch(TestModelTree[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestGenerateAllTrees(Generic[Array], unittest.TestCase):
    """Tests for generate_all_trees function."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def test_2_models_single_tree(self):
        """Test 2 models generates single tree."""
        bkd = self._bkd
        trees = list(generate_all_trees([1], 0, 10, bkd))
        self.assertEqual(len(trees), 1)
        bkd.assert_allclose(trees[0].to_index(), bkd.asarray([0]))

    def test_3_models_full_depth(self):
        """Test 3 models generates 3 trees at full depth."""
        bkd = self._bkd
        trees = list(generate_all_trees([1, 2], 0, 2, bkd))
        self.assertEqual(len(trees), 3)

        # Convert to lists for comparison
        indices = sorted(
            [bkd.to_numpy(t.to_index()).astype(int).tolist() for t in trees]
        )
        # [0, 0]: star - both LF connect to HF
        # [0, 1]: chain 0→1→2
        # [2, 0]: chain 0→2→1
        expected = [[0, 0], [0, 1], [2, 0]]
        self.assertEqual(indices, expected)

    def test_3_models_depth_1(self):
        """Test 3 models with depth=1 gives only star topology."""
        bkd = self._bkd
        trees = list(generate_all_trees([1, 2], 0, 1, bkd))
        self.assertEqual(len(trees), 1)
        bkd.assert_allclose(trees[0].to_index(), bkd.asarray([0, 0]))

    def test_4_models_depth_2(self):
        """Test 4 models with depth=2."""
        bkd = self._bkd
        trees = list(generate_all_trees([1, 2, 3], 0, 2, bkd))
        # Should produce a reasonable number of trees
        self.assertGreater(len(trees), 1)
        self.assertLess(len(trees), 100)

    def test_empty_children(self):
        """Test with no children."""
        bkd = self._bkd
        trees = list(generate_all_trees([], 0, 10, bkd))
        self.assertEqual(len(trees), 1)
        self.assertEqual(trees[0].num_nodes(), 1)


class TestGenerateAllTreesNumpy(TestGenerateAllTrees[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGenerateAllTreesTorch(TestGenerateAllTrees[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestGetAcvRecursionIndices(Generic[Array], unittest.TestCase):
    """Tests for get_acv_recursion_indices function."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def test_2_models(self):
        """Test 2-model case has single index."""
        bkd = self._bkd
        indices = list(get_acv_recursion_indices(2, bkd=bkd))
        self.assertEqual(len(indices), 1)
        bkd.assert_allclose(indices[0], bkd.asarray([0]))

    def test_3_models(self):
        """Test 3-model case has 3 indices."""
        bkd = self._bkd
        indices = list(get_acv_recursion_indices(3, bkd=bkd))
        self.assertEqual(len(indices), 3)

        # All valid trees rooted at model 0 (HF)
        idx_lists = sorted(
            [bkd.to_numpy(idx).astype(int).tolist() for idx in indices]
        )
        # [0, 0]: star - both LF connect to HF
        # [0, 1]: chain 0→1→2
        # [2, 0]: chain 0→2→1
        expected = [[0, 0], [0, 1], [2, 0]]
        self.assertEqual(idx_lists, expected)

    def test_4_models_full_depth(self):
        """Test 4 models at full depth."""
        bkd = self._bkd
        indices = list(get_acv_recursion_indices(4, bkd=bkd))
        # Count should be consistent
        self.assertEqual(len(indices), count_recursion_indices(4))

    def test_depth_limit(self):
        """Test that depth limit reduces count."""
        count_full = count_recursion_indices(5)
        count_limited = count_recursion_indices(5, depth=2)
        self.assertLess(count_limited, count_full)

    def test_depth_exceeds_models_raises(self):
        """Test that depth > nmodels-1 raises ValueError."""
        with self.assertRaises(ValueError):
            list(get_acv_recursion_indices(3, depth=3))

    def test_index_values_valid(self):
        """Test that index values are valid parent references."""
        bkd = self._bkd
        nmodels = 4
        for index in get_acv_recursion_indices(nmodels, bkd=bkd):
            index_np = bkd.to_numpy(index).astype(int)
            # Each entry index[m-1] should be in range [0, nmodels-1]
            # and should not equal m (no self-loops)
            # The tree must be connected to model 0 (HF)
            for m in range(1, nmodels):
                parent = index_np[m - 1]
                self.assertGreaterEqual(parent, 0)
                self.assertLess(parent, nmodels)
                self.assertNotEqual(parent, m)  # no self-loops


class TestGetAcvRecursionIndicesNumpy(TestGetAcvRecursionIndices[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGetAcvRecursionIndicesTorch(TestGetAcvRecursionIndices[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestCountRecursionIndices(unittest.TestCase):
    """Tests for count_recursion_indices function."""

    def test_2_models(self):
        """Test count for 2 models."""
        self.assertEqual(count_recursion_indices(2), 1)

    def test_3_models(self):
        """Test count for 3 models."""
        self.assertEqual(count_recursion_indices(3), 3)

    def test_4_models(self):
        """Test count for 4 models."""
        count = count_recursion_indices(4)
        # Should match number of generated indices
        indices = list(get_acv_recursion_indices(4))
        self.assertEqual(count, len(indices))

    def test_depth_limit_reduces_count(self):
        """Test that depth limit reduces count."""
        count_full = count_recursion_indices(5)
        count_d2 = count_recursion_indices(5, depth=2)
        count_d3 = count_recursion_indices(5, depth=3)

        self.assertLess(count_d2, count_d3)
        self.assertLess(count_d3, count_full)


class TestLegacyCompatibility(unittest.TestCase):
    """Tests to verify compatibility with legacy implementation."""

    def test_matches_legacy_3_models(self):
        """Test that 3-model indices match legacy."""
        try:
            from pyapprox.multifidelity._optim import _get_acv_recursion_indices
        except ImportError:
            self.skipTest("Legacy module not available")

        bkd = NumpyBkd()
        legacy_indices = sorted(
            [idx.astype(int).tolist() for idx in _get_acv_recursion_indices(3)]
        )
        new_indices = sorted(
            [
                bkd.to_numpy(idx).astype(int).tolist()
                for idx in get_acv_recursion_indices(3, bkd=bkd)
            ]
        )
        self.assertEqual(legacy_indices, new_indices)

    def test_matches_legacy_4_models(self):
        """Test that 4-model indices match legacy."""
        try:
            from pyapprox.multifidelity._optim import _get_acv_recursion_indices
        except ImportError:
            self.skipTest("Legacy module not available")

        bkd = NumpyBkd()
        legacy_indices = sorted(
            [idx.astype(int).tolist() for idx in _get_acv_recursion_indices(4)]
        )
        new_indices = sorted(
            [
                bkd.to_numpy(idx).astype(int).tolist()
                for idx in get_acv_recursion_indices(4, bkd=bkd)
            ]
        )
        self.assertEqual(legacy_indices, new_indices)

    def test_matches_legacy_4_models_depth_2(self):
        """Test that 4-model indices with depth=2 match legacy."""
        try:
            from pyapprox.multifidelity._optim import _get_acv_recursion_indices
        except ImportError:
            self.skipTest("Legacy module not available")

        bkd = NumpyBkd()
        legacy_indices = sorted(
            [
                idx.astype(int).tolist()
                for idx in _get_acv_recursion_indices(4, depth=2)
            ]
        )
        new_indices = sorted(
            [
                bkd.to_numpy(idx).astype(int).tolist()
                for idx in get_acv_recursion_indices(4, depth=2, bkd=bkd)
            ]
        )
        self.assertEqual(legacy_indices, new_indices)


if __name__ == "__main__":
    unittest.main()
