"""Tests for Phase 7 adaptive indexing modules."""

import unittest
from typing import Type

import numpy as np
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.protocols import Backend

from pyapprox.typing.surrogates.affine.indices.priority_queue import (
    PriorityQueue,
)
from pyapprox.typing.surrogates.affine.indices.refinement import (
    UnitCostFunction,
    LevelCostFunction,
    ExponentialCostFunction,
    LevelRefinementCriteria,
    CostWeightedRefinementCriteria,
)
from pyapprox.typing.surrogates.affine.indices.basis_generator import (
    BasisIndexGenerator,
)
from pyapprox.typing.surrogates.affine.indices.growth_rules import (
    LinearGrowthRule,
    ClenshawCurtisGrowthRule,
)


class _BaseAdaptiveTest:
    """Base class for adaptive tests. Not run directly."""

    __test__ = False
    bkd_class: Type[Backend[NDArray]] = NumpyBkd

    def setUp(self):
        self.bkd = self.bkd_class()


class TestPriorityQueue(_BaseAdaptiveTest, unittest.TestCase):
    """Tests for PriorityQueue class."""

    __test__ = True

    def test_empty_queue(self):
        """Test empty queue behavior."""
        queue = PriorityQueue[NDArray]()
        self.assertTrue(queue.empty())
        self.assertEqual(len(queue), 0)

    def test_put_get_single(self):
        """Test putting and getting a single item."""
        queue = PriorityQueue[NDArray]()
        queue.put(priority=0.5, error=0.1, index_id=42)
        self.assertFalse(queue.empty())
        self.assertEqual(len(queue), 1)

        priority, error, index_id = queue.get()
        self.assertEqual(priority, 0.5)
        self.assertEqual(error, 0.1)
        self.assertEqual(index_id, 42)
        self.assertTrue(queue.empty())

    def test_max_priority_order(self):
        """Test that max priority items come first."""
        queue = PriorityQueue[NDArray](max_priority=True)
        queue.put(priority=0.3, error=0.1, index_id=0)
        queue.put(priority=0.8, error=0.2, index_id=1)
        queue.put(priority=0.5, error=0.15, index_id=2)

        # Should get highest priority first
        _, _, idx0 = queue.get()
        _, _, idx1 = queue.get()
        _, _, idx2 = queue.get()

        self.assertEqual(idx0, 1)  # priority 0.8
        self.assertEqual(idx1, 2)  # priority 0.5
        self.assertEqual(idx2, 0)  # priority 0.3

    def test_min_priority_order(self):
        """Test min priority order when max_priority=False."""
        queue = PriorityQueue[NDArray](max_priority=False)
        queue.put(priority=0.3, error=0.1, index_id=0)
        queue.put(priority=0.8, error=0.2, index_id=1)
        queue.put(priority=0.5, error=0.15, index_id=2)

        # Should get lowest priority first
        _, _, idx0 = queue.get()
        _, _, idx1 = queue.get()
        _, _, idx2 = queue.get()

        self.assertEqual(idx0, 0)  # priority 0.3
        self.assertEqual(idx1, 2)  # priority 0.5
        self.assertEqual(idx2, 1)  # priority 0.8

    def test_peek(self):
        """Test peek without removing."""
        queue = PriorityQueue[NDArray]()
        queue.put(priority=0.5, error=0.1, index_id=42)

        priority1, error1, idx1 = queue.peek()
        priority2, error2, idx2 = queue.peek()

        self.assertEqual(idx1, idx2)
        self.assertEqual(len(queue), 1)

    def test_clear(self):
        """Test clearing the queue."""
        queue = PriorityQueue[NDArray]()
        queue.put(priority=0.5, error=0.1, index_id=0)
        queue.put(priority=0.6, error=0.2, index_id=1)

        queue.clear()
        self.assertTrue(queue.empty())

    def test_get_empty_raises(self):
        """Test that get on empty queue raises IndexError."""
        queue = PriorityQueue[NDArray]()
        with self.assertRaises(IndexError):
            queue.get()

    def test_peek_empty_raises(self):
        """Test that peek on empty queue raises IndexError."""
        queue = PriorityQueue[NDArray]()
        with self.assertRaises(IndexError):
            queue.peek()


class TestCostFunctions(_BaseAdaptiveTest, unittest.TestCase):
    """Tests for cost function classes."""

    __test__ = True

    def test_unit_cost(self):
        """Test UnitCostFunction always returns 1."""
        bkd = self.bkd
        cost_fn = UnitCostFunction(bkd)
        index = bkd.asarray([1, 2, 3], dtype=bkd.int64_dtype())
        self.assertEqual(cost_fn(index), 1.0)

        index2 = bkd.asarray([0, 0], dtype=bkd.int64_dtype())
        self.assertEqual(cost_fn(index2), 1.0)

    def test_level_cost(self):
        """Test LevelCostFunction returns sum + 1."""
        bkd = self.bkd
        cost_fn = LevelCostFunction(bkd)

        index = bkd.asarray([1, 2, 3], dtype=bkd.int64_dtype())
        self.assertEqual(cost_fn(index), 7.0)  # 1+2+3+1

        index2 = bkd.asarray([0, 0], dtype=bkd.int64_dtype())
        self.assertEqual(cost_fn(index2), 1.0)  # 0+0+1

    def test_exponential_cost(self):
        """Test ExponentialCostFunction."""
        bkd = self.bkd
        cost_fn = ExponentialCostFunction(bkd, base=2.0)

        index = bkd.asarray([1, 1], dtype=bkd.int64_dtype())
        self.assertEqual(cost_fn(index), 4.0)  # 2^2

        index2 = bkd.asarray([0, 0], dtype=bkd.int64_dtype())
        self.assertEqual(cost_fn(index2), 1.0)  # 2^0


class TestRefinementCriteria(_BaseAdaptiveTest, unittest.TestCase):
    """Tests for refinement criteria classes."""

    __test__ = True

    def test_level_refinement_priority(self):
        """Test LevelRefinementCriteria priority computation."""
        bkd = self.bkd
        criteria = LevelRefinementCriteria(bkd, max_level=10)

        # Lower level should have higher priority
        low_idx = bkd.asarray([1, 0], dtype=bkd.int64_dtype())
        high_idx = bkd.asarray([3, 2], dtype=bkd.int64_dtype())

        _, low_priority = criteria(low_idx)
        _, high_priority = criteria(high_idx)

        self.assertGreater(low_priority, high_priority)

    def test_level_refinement_cost(self):
        """Test LevelRefinementCriteria cost defaults to unit cost."""
        bkd = self.bkd
        criteria = LevelRefinementCriteria(bkd, max_level=10)

        index = bkd.asarray([2, 3], dtype=bkd.int64_dtype())
        cost = criteria.cost(index)
        self.assertEqual(cost, 1.0)

    def test_cost_weighted_refinement(self):
        """Test CostWeightedRefinementCriteria."""
        bkd = self.bkd
        cost_fn = LevelCostFunction(bkd)
        criteria = CostWeightedRefinementCriteria(bkd, cost_fn)

        # Set error estimate for an index
        criteria.set_error_estimate(0, 10.0)

        index = bkd.asarray([2, 2], dtype=bkd.int64_dtype())
        error, priority = criteria(index, index_id=0)

        self.assertEqual(error, 10.0)
        # Priority = error / cost = 10 / (2+2+1) = 2.0
        self.assertAlmostEqual(priority, 2.0)


class TestBasisIndexGenerator(_BaseAdaptiveTest, unittest.TestCase):
    """Tests for BasisIndexGenerator class."""

    __test__ = True

    def test_linear_growth_basis_count(self):
        """Test basis count with linear growth rule."""
        bkd = self.bkd
        growth_rule = LinearGrowthRule(scale=1, shift=1)
        gen = BasisIndexGenerator(bkd, nvars=2, growth_rules=growth_rule)

        index = bkd.asarray([2, 1], dtype=bkd.int64_dtype())
        counts = gen.nunivariate_basis(index)

        # linear(2) = 1*2+1 = 3, linear(1) = 1*1+1 = 2
        self.assertEqual(counts, [3, 2])

    def test_double_plus_one_growth(self):
        """Test basis count with double plus one growth rule."""
        bkd = self.bkd
        growth_rule = ClenshawCurtisGrowthRule()
        gen = BasisIndexGenerator(bkd, nvars=2, growth_rules=growth_rule)

        index = bkd.asarray([3, 2], dtype=bkd.int64_dtype())
        counts = gen.nunivariate_basis(index)

        # 2^3+1 = 9, 2^2+1 = 5
        self.assertEqual(counts, [9, 5])

    def test_nsubspace_basis(self):
        """Test total subspace basis count."""
        bkd = self.bkd
        growth_rule = LinearGrowthRule(scale=1, shift=1)
        gen = BasisIndexGenerator(bkd, nvars=2, growth_rules=growth_rule)

        index = bkd.asarray([2, 1], dtype=bkd.int64_dtype())
        total = gen.nsubspace_basis(index)

        # 3 * 2 = 6
        self.assertEqual(total, 6)

    def test_refine_subspace_index(self):
        """Test children index generation."""
        bkd = self.bkd
        gen = BasisIndexGenerator(bkd, nvars=3)

        index = bkd.asarray([1, 2, 0], dtype=bkd.int64_dtype())
        children = gen.refine_subspace_index(index)

        # Should have 3 children (one per dimension)
        self.assertEqual(children.shape, (3, 3))

        # Check each child increments exactly one dimension
        for col in range(3):
            child = children[:, col]
            diff = child - index
            self.assertEqual(int(bkd.sum(diff)), 1)
            self.assertEqual(int(bkd.sum(bkd.abs(diff))), 1)

    def test_get_basis_indices(self):
        """Test basis index generation for a subspace."""
        bkd = self.bkd
        growth_rule = LinearGrowthRule(scale=1, shift=1)
        gen = BasisIndexGenerator(bkd, nvars=2, growth_rules=growth_rule)

        # Level [1, 1] -> linear gives [2, 2] basis functions
        index = bkd.asarray([1, 1], dtype=bkd.int64_dtype())
        basis_indices = gen.get_basis_indices(index)

        # Should have 2*2 = 4 basis indices
        self.assertEqual(basis_indices.shape[1], 4)
        # Each index should be in range [0, 2) for both dimensions
        self.assertTrue(bkd.all_bool(basis_indices[0, :] < 2))
        self.assertTrue(bkd.all_bool(basis_indices[1, :] < 2))

    def test_nvars_and_nrefinement_vars(self):
        """Test nvars and nrefinement_vars properties."""
        bkd = self.bkd
        gen = BasisIndexGenerator(bkd, nvars=3, nrefinement_vars=2)

        self.assertEqual(gen.nvars(), 3)
        self.assertEqual(gen.nrefinement_vars(), 2)

        # Children should only have 2 (nrefinement_vars)
        index = bkd.asarray([1, 1, 1], dtype=bkd.int64_dtype())
        children = gen.refine_subspace_index(index)
        self.assertEqual(children.shape[1], 2)


if __name__ == "__main__":
    unittest.main()
