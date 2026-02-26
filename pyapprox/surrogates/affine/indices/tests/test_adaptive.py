"""Tests for Phase 7 adaptive indexing modules."""

import pytest

from pyapprox.surrogates.affine.indices.basis_generator import (
    BasisIndexGenerator,
)
from pyapprox.surrogates.affine.indices.growth_rules import (
    ClenshawCurtisGrowthRule,
    LinearGrowthRule,
)
from pyapprox.surrogates.affine.indices.priority_queue import (
    PriorityQueue,
)
from pyapprox.surrogates.affine.indices.refinement import (
    CostWeightedRefinementCriteria,
    ExponentialCostFunction,
    LevelCostFunction,
    LevelRefinementCriteria,
    UnitCostFunction,
)
from pyapprox.util.backends.numpy import NumpyBkd


class TestPriorityQueue:
    """Tests for PriorityQueue class."""

    def test_empty_queue(self, bkd):
        """Test empty queue behavior."""
        queue = PriorityQueue()
        assert queue.empty()
        assert len(queue) == 0

    def test_put_get_single(self, bkd):
        """Test putting and getting a single item."""
        queue = PriorityQueue()
        queue.put(priority=0.5, error=0.1, index_id=42)
        assert not queue.empty()
        assert len(queue) == 1

        priority, error, index_id = queue.get()
        assert priority == 0.5
        assert error == 0.1
        assert index_id == 42
        assert queue.empty()

    def test_max_priority_order(self, bkd):
        """Test that max priority items come first."""
        queue = PriorityQueue(max_priority=True)
        queue.put(priority=0.3, error=0.1, index_id=0)
        queue.put(priority=0.8, error=0.2, index_id=1)
        queue.put(priority=0.5, error=0.15, index_id=2)

        # Should get highest priority first
        _, _, idx0 = queue.get()
        _, _, idx1 = queue.get()
        _, _, idx2 = queue.get()

        assert idx0 == 1  # priority 0.8
        assert idx1 == 2  # priority 0.5
        assert idx2 == 0  # priority 0.3

    def test_min_priority_order(self, bkd):
        """Test min priority order when max_priority=False."""
        queue = PriorityQueue(max_priority=False)
        queue.put(priority=0.3, error=0.1, index_id=0)
        queue.put(priority=0.8, error=0.2, index_id=1)
        queue.put(priority=0.5, error=0.15, index_id=2)

        # Should get lowest priority first
        _, _, idx0 = queue.get()
        _, _, idx1 = queue.get()
        _, _, idx2 = queue.get()

        assert idx0 == 0  # priority 0.3
        assert idx1 == 2  # priority 0.5
        assert idx2 == 1  # priority 0.8

    def test_peek(self, bkd):
        """Test peek without removing."""
        queue = PriorityQueue()
        queue.put(priority=0.5, error=0.1, index_id=42)

        priority1, error1, idx1 = queue.peek()
        priority2, error2, idx2 = queue.peek()

        assert idx1 == idx2
        assert len(queue) == 1

    def test_clear(self, bkd):
        """Test clearing the queue."""
        queue = PriorityQueue()
        queue.put(priority=0.5, error=0.1, index_id=0)
        queue.put(priority=0.6, error=0.2, index_id=1)

        queue.clear()
        assert queue.empty()

    def test_get_empty_raises(self, bkd):
        """Test that get on empty queue raises IndexError."""
        queue = PriorityQueue()
        with pytest.raises(IndexError):
            queue.get()

    def test_peek_empty_raises(self, bkd):
        """Test that peek on empty queue raises IndexError."""
        queue = PriorityQueue()
        with pytest.raises(IndexError):
            queue.peek()


class TestCostFunctions:
    """Tests for cost function classes."""

    def test_unit_cost(self, bkd):
        """Test UnitCostFunction always returns 1."""
        cost_fn = UnitCostFunction(bkd)
        index = bkd.asarray([1, 2, 3], dtype=bkd.int64_dtype())
        assert cost_fn(index) == 1.0

        index2 = bkd.asarray([0, 0], dtype=bkd.int64_dtype())
        assert cost_fn(index2) == 1.0

    def test_level_cost(self, bkd):
        """Test LevelCostFunction returns sum + 1."""
        cost_fn = LevelCostFunction(bkd)

        index = bkd.asarray([1, 2, 3], dtype=bkd.int64_dtype())
        assert cost_fn(index) == 7.0  # 1+2+3+1

        index2 = bkd.asarray([0, 0], dtype=bkd.int64_dtype())
        assert cost_fn(index2) == 1.0  # 0+0+1

    def test_exponential_cost(self, bkd):
        """Test ExponentialCostFunction."""
        cost_fn = ExponentialCostFunction(bkd, base=2.0)

        index = bkd.asarray([1, 1], dtype=bkd.int64_dtype())
        assert cost_fn(index) == 4.0  # 2^2

        index2 = bkd.asarray([0, 0], dtype=bkd.int64_dtype())
        assert cost_fn(index2) == 1.0  # 2^0


class TestRefinementCriteria:
    """Tests for refinement criteria classes."""

    def test_level_refinement_priority(self, bkd):
        """Test LevelRefinementCriteria priority computation."""
        criteria = LevelRefinementCriteria(bkd, max_level=10)

        # Lower level should have higher priority
        low_idx = bkd.asarray([1, 0], dtype=bkd.int64_dtype())
        high_idx = bkd.asarray([3, 2], dtype=bkd.int64_dtype())

        _, low_priority = criteria(low_idx)
        _, high_priority = criteria(high_idx)

        assert low_priority > high_priority

    def test_level_refinement_cost(self, bkd):
        """Test LevelRefinementCriteria cost defaults to unit cost."""
        criteria = LevelRefinementCriteria(bkd, max_level=10)

        index = bkd.asarray([2, 3], dtype=bkd.int64_dtype())
        cost = criteria.cost(index)
        assert cost == 1.0

    def test_cost_weighted_refinement(self, bkd):
        """Test CostWeightedRefinementCriteria."""
        cost_fn = LevelCostFunction(bkd)
        criteria = CostWeightedRefinementCriteria(bkd, cost_fn)

        # Set error estimate for an index
        criteria.set_error_estimate(0, 10.0)

        index = bkd.asarray([2, 2], dtype=bkd.int64_dtype())
        error, priority = criteria(index, index_id=0)

        assert error == 10.0
        # Priority = error / cost = 10 / (2+2+1) = 2.0
        assert priority == pytest.approx(2.0)


class TestBasisIndexGenerator:
    """Tests for BasisIndexGenerator class."""

    def test_linear_growth_basis_count(self, bkd):
        """Test basis count with linear growth rule."""
        growth_rule = LinearGrowthRule(scale=1, shift=1)
        gen = BasisIndexGenerator(bkd, nvars=2, growth_rules=growth_rule)

        index = bkd.asarray([2, 1], dtype=bkd.int64_dtype())
        counts = gen.nunivariate_basis(index)

        # linear(2) = 1*2+1 = 3, linear(1) = 1*1+1 = 2
        assert counts == [3, 2]

    def test_double_plus_one_growth(self, bkd):
        """Test basis count with double plus one growth rule."""
        growth_rule = ClenshawCurtisGrowthRule()
        gen = BasisIndexGenerator(bkd, nvars=2, growth_rules=growth_rule)

        index = bkd.asarray([3, 2], dtype=bkd.int64_dtype())
        counts = gen.nunivariate_basis(index)

        # 2^3+1 = 9, 2^2+1 = 5
        assert counts == [9, 5]

    def test_nsubspace_basis(self, bkd):
        """Test total subspace basis count."""
        growth_rule = LinearGrowthRule(scale=1, shift=1)
        gen = BasisIndexGenerator(bkd, nvars=2, growth_rules=growth_rule)

        index = bkd.asarray([2, 1], dtype=bkd.int64_dtype())
        total = gen.nsubspace_basis(index)

        # 3 * 2 = 6
        assert total == 6

    def test_refine_subspace_index(self, bkd):
        """Test children index generation."""
        gen = BasisIndexGenerator(bkd, nvars=3)

        index = bkd.asarray([1, 2, 0], dtype=bkd.int64_dtype())
        children = gen.refine_subspace_index(index)

        # Should have 3 children (one per dimension)
        assert children.shape == (3, 3)

        # Check each child increments exactly one dimension
        for col in range(3):
            child = children[:, col]
            diff = child - index
            assert int(bkd.sum(diff)) == 1
            assert int(bkd.sum(bkd.abs(diff))) == 1

    def test_get_basis_indices(self, bkd):
        """Test basis index generation for a subspace."""
        growth_rule = LinearGrowthRule(scale=1, shift=1)
        gen = BasisIndexGenerator(bkd, nvars=2, growth_rules=growth_rule)

        # Level [1, 1] -> linear gives [2, 2] basis functions
        index = bkd.asarray([1, 1], dtype=bkd.int64_dtype())
        basis_indices = gen.get_basis_indices(index)

        # Should have 2*2 = 4 basis indices
        assert basis_indices.shape[1] == 4
        # Each index should be in range [0, 2) for both dimensions
        assert bkd.all_bool(basis_indices[0, :] < 2)
        assert bkd.all_bool(basis_indices[1, :] < 2)

    def test_nvars_and_nrefinement_vars(self, bkd):
        """Test nvars and nrefinement_vars properties."""
        gen = BasisIndexGenerator(bkd, nvars=3, nrefinement_vars=2)

        assert gen.nvars() == 3
        assert gen.nrefinement_vars() == 2

        # Children should only have 2 (nrefinement_vars)
        index = bkd.asarray([1, 1, 1], dtype=bkd.int64_dtype())
        children = gen.refine_subspace_index(index)
        assert children.shape[1] == 2
