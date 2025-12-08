"""Tests for index generation module."""

import unittest
from typing import Type

import numpy as np
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.protocols import Backend

from pyapprox.typing.surrogates.affine.indices.utils import (
    hash_index,
    compute_hyperbolic_level_indices,
    compute_hyperbolic_indices,
    sort_indices_lexiographically,
    indices_pnorm,
)
from pyapprox.typing.surrogates.affine.indices.admissibility import (
    MaxLevelCriteria,
    Max1DLevelsCriteria,
    MaxIndicesCriteria,
    CompositeCriteria,
)
from pyapprox.typing.surrogates.affine.indices.growth_rules import (
    LinearGrowthRule,
    DoublePlusOneGrowthRule,
    ConstantGrowthRule,
    ExponentialGrowthRule,
)
from pyapprox.typing.surrogates.affine.indices.generators import (
    HyperbolicIndexGenerator,
    IterativeIndexGenerator,
)


class _BaseIndexTest:
    """Base class for index tests. Not run directly."""

    __test__ = False
    bkd_class: Type[Backend[NDArray]] = NumpyBkd

    def setUp(self):
        self.bkd = self.bkd_class()


class TestHashIndex(_BaseIndexTest, unittest.TestCase):
    """Test hash_index function."""

    __test__ = True

    def test_hash_consistency(self):
        """Test that same index produces same hash."""
        bkd = self.bkd
        idx1 = bkd.asarray([1, 2, 3], dtype=bkd.int64_dtype())
        idx2 = bkd.asarray([1, 2, 3], dtype=bkd.int64_dtype())
        self.assertEqual(hash_index(idx1, bkd), hash_index(idx2, bkd))

    def test_hash_uniqueness(self):
        """Test that different indices produce different hashes."""
        bkd = self.bkd
        idx1 = bkd.asarray([1, 2, 3], dtype=bkd.int64_dtype())
        idx2 = bkd.asarray([1, 2, 4], dtype=bkd.int64_dtype())
        idx3 = bkd.asarray([3, 2, 1], dtype=bkd.int64_dtype())
        self.assertNotEqual(hash_index(idx1, bkd), hash_index(idx2, bkd))
        self.assertNotEqual(hash_index(idx1, bkd), hash_index(idx3, bkd))


class TestIndexPnorm(_BaseIndexTest, unittest.TestCase):
    """Test indices_pnorm function."""

    __test__ = True

    def test_pnorm_1(self):
        """Test p=1 norm (total degree)."""
        bkd = self.bkd
        indices = bkd.asarray([[0, 1, 2, 1], [0, 0, 0, 1]], dtype=bkd.int64_dtype())
        norms = indices_pnorm(indices, 1.0, bkd)
        expected = bkd.asarray([0.0, 1.0, 2.0, 2.0])
        bkd.assert_allclose(norms, expected)

    def test_pnorm_2(self):
        """Test p=2 norm."""
        bkd = self.bkd
        indices = bkd.asarray([[3, 0], [4, 0]], dtype=bkd.int64_dtype())
        norms = indices_pnorm(indices, 2.0, bkd)
        expected = bkd.asarray([5.0, 0.0])
        bkd.assert_allclose(norms, expected)

    def test_pnorm_half(self):
        """Test p=0.5 norm (hyperbolic)."""
        bkd = self.bkd
        indices = bkd.asarray([[1, 2], [1, 0]], dtype=bkd.int64_dtype())
        norms = indices_pnorm(indices, 0.5, bkd)
        # (1^0.5 + 1^0.5)^2 = 4, (2^0.5 + 0)^2 = 2
        expected = bkd.asarray([4.0, 2.0])
        bkd.assert_allclose(norms, expected)


class TestHyperbolicLevelIndices(_BaseIndexTest, unittest.TestCase):
    """Test compute_hyperbolic_level_indices function."""

    __test__ = True

    def test_level_0(self):
        """Test level 0 indices."""
        bkd = self.bkd
        indices = compute_hyperbolic_level_indices(2, 0, 1.0, bkd)
        expected = bkd.asarray([[0], [0]], dtype=bkd.int64_dtype())
        bkd.assert_allclose(indices, expected)

    def test_level_1_2d(self):
        """Test level 1 indices in 2D."""
        bkd = self.bkd
        indices = compute_hyperbolic_level_indices(2, 1, 1.0, bkd)
        self.assertEqual(indices.shape[1], 2)
        # Should have [1,0] and [0,1]
        norms = indices_pnorm(indices, 1.0, bkd)
        self.assertTrue(bkd.all_bool(norms == 1.0))

    def test_level_2_2d(self):
        """Test level 2 indices in 2D."""
        bkd = self.bkd
        indices = compute_hyperbolic_level_indices(2, 2, 1.0, bkd)
        self.assertEqual(indices.shape[1], 3)
        # Should have [2,0], [1,1], [0,2]
        norms = indices_pnorm(indices, 1.0, bkd)
        self.assertTrue(bkd.all_bool(norms == 2.0))


class TestHyperbolicIndices(_BaseIndexTest, unittest.TestCase):
    """Test compute_hyperbolic_indices function."""

    __test__ = True

    def test_total_degree_2d_level2(self):
        """Test total degree indices in 2D up to level 2."""
        bkd = self.bkd
        indices = compute_hyperbolic_indices(2, 2, 1.0, bkd)
        # Level 0: 1, Level 1: 2, Level 2: 3 -> total 6
        self.assertEqual(indices.shape[1], 6)

    def test_total_degree_3d_level2(self):
        """Test total degree indices in 3D up to level 2."""
        bkd = self.bkd
        indices = compute_hyperbolic_indices(3, 2, 1.0, bkd)
        # Level 0: 1, Level 1: 3, Level 2: 6 -> total 10
        self.assertEqual(indices.shape[1], 10)

    def test_contains_zero_index(self):
        """Test that result contains zero index."""
        bkd = self.bkd
        indices = compute_hyperbolic_indices(2, 3, 1.0, bkd)
        zero_index = bkd.zeros((2,), dtype=bkd.int64_dtype())
        has_zero = False
        for idx in indices.T:
            if bkd.all_bool(idx == zero_index):
                has_zero = True
                break
        self.assertTrue(has_zero)


class TestSortIndices(_BaseIndexTest, unittest.TestCase):
    """Test sort_indices_lexiographically function."""

    __test__ = True

    def test_sort_2d(self):
        """Test sorting 2D indices.

        Sort is by total level first, then lexicographically within level.
        """
        bkd = self.bkd
        # Unsorted indices: [1,0], [0,1], [2,0], [0,0], [1,1]
        indices = bkd.asarray(
            [[1, 0, 2, 0, 1], [0, 1, 0, 0, 1]], dtype=bkd.int64_dtype()
        )
        sorted_idx = sort_indices_lexiographically(indices, bkd)
        # Expected sort order:
        # Level 0: [0,0]
        # Level 1: [0,1], [1,0]
        # Level 2: [0,2], [1,1], [2,0] - but [2,0] not in input, so [1,1] and [2,0]
        # From input: [0,0] (level 0), [0,1] (level 1), [1,0] (level 1), [2,0] (level 2), [1,1] (level 2)
        # Sorted by level first, then lexicographically:
        # [0,0], [1,0], [0,1], [2,0], [1,1]
        expected = bkd.asarray(
            [[0, 1, 0, 2, 1], [0, 0, 1, 0, 1]], dtype=bkd.int64_dtype()
        )
        bkd.assert_allclose(sorted_idx, expected)


class TestMaxLevelCriteria(_BaseIndexTest, unittest.TestCase):
    """Test MaxLevelCriteria admissibility."""

    __test__ = True

    def test_below_level(self):
        """Test index below max level is admissible."""
        bkd = self.bkd
        criteria = MaxLevelCriteria(5, 1.0, bkd)
        idx = bkd.asarray([2, 1], dtype=bkd.int64_dtype())
        self.assertTrue(criteria(idx))

    def test_at_level(self):
        """Test index at max level is admissible."""
        bkd = self.bkd
        criteria = MaxLevelCriteria(5, 1.0, bkd)
        idx = bkd.asarray([3, 2], dtype=bkd.int64_dtype())
        self.assertTrue(criteria(idx))

    def test_above_level(self):
        """Test index above max level is not admissible."""
        bkd = self.bkd
        criteria = MaxLevelCriteria(5, 1.0, bkd)
        idx = bkd.asarray([3, 3], dtype=bkd.int64_dtype())
        self.assertFalse(criteria(idx))

    def test_pnorm_half(self):
        """Test with p=0.5 (hyperbolic cross)."""
        bkd = self.bkd
        criteria = MaxLevelCriteria(4, 0.5, bkd)
        # (1^0.5 + 1^0.5)^2 = 4
        idx = bkd.asarray([1, 1], dtype=bkd.int64_dtype())
        self.assertTrue(criteria(idx))
        # (2^0.5 + 2^0.5)^2 = 8 > 4
        idx = bkd.asarray([2, 2], dtype=bkd.int64_dtype())
        self.assertFalse(criteria(idx))


class TestMax1DLevelsCriteria(_BaseIndexTest, unittest.TestCase):
    """Test Max1DLevelsCriteria admissibility."""

    __test__ = True

    def test_within_limits(self):
        """Test index within 1D limits is admissible."""
        bkd = self.bkd
        max_levels = bkd.asarray([3, 4], dtype=bkd.int64_dtype())
        criteria = Max1DLevelsCriteria(max_levels, bkd)
        idx = bkd.asarray([2, 3], dtype=bkd.int64_dtype())
        self.assertTrue(criteria(idx))

    def test_at_limits(self):
        """Test index at 1D limits is admissible."""
        bkd = self.bkd
        max_levels = bkd.asarray([3, 4], dtype=bkd.int64_dtype())
        criteria = Max1DLevelsCriteria(max_levels, bkd)
        idx = bkd.asarray([3, 4], dtype=bkd.int64_dtype())
        self.assertTrue(criteria(idx))

    def test_exceeds_one_limit(self):
        """Test index exceeding one 1D limit is not admissible."""
        bkd = self.bkd
        max_levels = bkd.asarray([3, 4], dtype=bkd.int64_dtype())
        criteria = Max1DLevelsCriteria(max_levels, bkd)
        idx = bkd.asarray([4, 3], dtype=bkd.int64_dtype())
        self.assertFalse(criteria(idx))


class TestMaxIndicesCriteria(_BaseIndexTest, unittest.TestCase):
    """Test MaxIndicesCriteria admissibility."""

    __test__ = True

    def test_requires_generator(self):
        """Test that criteria requires generator to be set."""
        bkd = self.bkd
        criteria = MaxIndicesCriteria(10, bkd)
        idx = bkd.asarray([1, 2], dtype=bkd.int64_dtype())
        with self.assertRaises(RuntimeError):
            criteria(idx)

    def test_with_generator(self):
        """Test criteria with generator set."""
        bkd = self.bkd
        # Create a generator with few indices
        gen = HyperbolicIndexGenerator(nvars=2, max_level=1, pnorm=1.0, bkd=bkd)
        # Level 0: 1, Level 1: 2 -> total 3 selected indices

        # Create criteria with max_nindices that allows a few more
        criteria = MaxIndicesCriteria(10, bkd)
        criteria.set_generator(gen)

        idx = bkd.asarray([1, 2], dtype=bkd.int64_dtype())
        # Should be True since we're below max
        self.assertTrue(criteria(idx))


class TestCompositeCriteria(_BaseIndexTest, unittest.TestCase):
    """Test CompositeCriteria admissibility."""

    __test__ = True

    def test_all_pass(self):
        """Test composite when all criteria pass."""
        bkd = self.bkd
        criteria = CompositeCriteria(
            MaxLevelCriteria(5, 1.0, bkd),
            Max1DLevelsCriteria(bkd.asarray([3, 3], dtype=bkd.int64_dtype()), bkd),
        )
        idx = bkd.asarray([2, 2], dtype=bkd.int64_dtype())
        self.assertTrue(criteria(idx))

    def test_one_fails(self):
        """Test composite when one criterion fails."""
        bkd = self.bkd
        criteria = CompositeCriteria(
            MaxLevelCriteria(10, 1.0, bkd),  # passes
            Max1DLevelsCriteria(bkd.asarray([2, 2], dtype=bkd.int64_dtype()), bkd),  # fails
        )
        idx = bkd.asarray([3, 1], dtype=bkd.int64_dtype())
        self.assertFalse(criteria(idx))

    def test_add_criteria(self):
        """Test adding criteria dynamically."""
        bkd = self.bkd
        criteria = CompositeCriteria(MaxLevelCriteria(10, 1.0, bkd))
        idx = bkd.asarray([4, 0], dtype=bkd.int64_dtype())
        self.assertTrue(criteria(idx))

        criteria.add_criteria(
            Max1DLevelsCriteria(bkd.asarray([3, 3], dtype=bkd.int64_dtype()), bkd)
        )
        self.assertFalse(criteria(idx))


class TestGrowthRules(unittest.TestCase):
    """Test growth rule classes."""

    def test_linear_growth(self):
        """Test LinearGrowthRule."""
        rule = LinearGrowthRule(scale=2, shift=1)
        self.assertEqual(rule(0), 1)
        self.assertEqual(rule(1), 3)
        self.assertEqual(rule(2), 5)
        self.assertEqual(rule(3), 7)

    def test_double_plus_one(self):
        """Test DoublePlusOneGrowthRule."""
        rule = DoublePlusOneGrowthRule()
        self.assertEqual(rule(0), 1)
        self.assertEqual(rule(1), 3)
        self.assertEqual(rule(2), 5)
        self.assertEqual(rule(3), 9)
        self.assertEqual(rule(4), 17)

    def test_constant_growth(self):
        """Test ConstantGrowthRule."""
        rule = ConstantGrowthRule(value=5)
        self.assertEqual(rule(0), 5)
        self.assertEqual(rule(1), 5)
        self.assertEqual(rule(10), 5)

    def test_exponential_growth(self):
        """Test ExponentialGrowthRule."""
        rule = ExponentialGrowthRule(base=2)
        self.assertEqual(rule(0), 1)
        self.assertEqual(rule(1), 2)
        self.assertEqual(rule(2), 4)
        self.assertEqual(rule(3), 8)


class TestHyperbolicIndexGenerator(_BaseIndexTest, unittest.TestCase):
    """Test HyperbolicIndexGenerator."""

    __test__ = True

    def test_total_degree_2d(self):
        """Test total degree indices in 2D."""
        bkd = self.bkd
        gen = HyperbolicIndexGenerator(nvars=2, max_level=3, pnorm=1.0, bkd=bkd)
        indices = gen.get_selected_indices()
        # Level 0: 1, Level 1: 2, Level 2: 3, Level 3: 4 -> total 10
        self.assertEqual(indices.shape[1], 10)
        self.assertEqual(gen.nvars(), 2)

    def test_total_degree_3d(self):
        """Test total degree indices in 3D."""
        bkd = self.bkd
        gen = HyperbolicIndexGenerator(nvars=3, max_level=2, pnorm=1.0, bkd=bkd)
        indices = gen.get_selected_indices()
        # Level 0: 1, Level 1: 3, Level 2: 6 -> total 10
        self.assertEqual(indices.shape[1], 10)

    def test_hyperbolic_cross(self):
        """Test hyperbolic cross indices with p=0.5."""
        bkd = self.bkd
        gen = HyperbolicIndexGenerator(nvars=2, max_level=4, pnorm=0.5, bkd=bkd)
        indices = gen.get_selected_indices()
        # Check that norms don't exceed max_level
        norms = indices_pnorm(indices, 0.5, bkd)
        self.assertTrue(bkd.all_bool(norms <= 4.0 + 1e-10))

    def test_with_max_1d_levels(self):
        """Test with 1D level constraints."""
        bkd = self.bkd
        max_1d_levels = bkd.asarray([2, 3], dtype=bkd.int64_dtype())
        gen = HyperbolicIndexGenerator(
            nvars=2, max_level=10, pnorm=1.0, bkd=bkd, max_1d_levels=max_1d_levels
        )
        indices = gen.get_selected_indices()
        # Check 1D constraints
        self.assertTrue(bkd.all_bool(indices[0, :] <= 2))
        self.assertTrue(bkd.all_bool(indices[1, :] <= 3))
        # Should have (2+1)*(3+1) = 12 indices
        self.assertEqual(indices.shape[1], 12)

    def test_step(self):
        """Test stepping to increase max level."""
        bkd = self.bkd
        gen = HyperbolicIndexGenerator(nvars=2, max_level=2, pnorm=1.0, bkd=bkd)
        n_before = gen.nselected_indices()

        gen.step()  # Increase to level 3
        n_after = gen.nselected_indices()

        self.assertGreater(n_after, n_before)

    def test_downward_closed(self):
        """Test that generated indices are downward closed."""
        bkd = self.bkd
        gen = HyperbolicIndexGenerator(nvars=3, max_level=3, pnorm=1.0, bkd=bkd)
        indices = gen.get_selected_indices()

        # Check downward closure
        indices_set = set()
        for idx in indices.T:
            indices_set.add(tuple(bkd.to_numpy(idx)))

        for idx in indices.T:
            idx_np = bkd.to_numpy(idx)
            for dim in range(3):
                if idx_np[dim] > 0:
                    neighbor = idx_np.copy()
                    neighbor[dim] -= 1
                    self.assertIn(
                        tuple(neighbor),
                        indices_set,
                        f"Index {idx_np} missing backward neighbor in dim {dim}",
                    )


class TestIterativeIndexGenerator(_BaseIndexTest, unittest.TestCase):
    """Test IterativeIndexGenerator functionality."""

    __test__ = True

    def test_refine_index(self):
        """Test refining a specific index."""
        bkd = self.bkd
        gen = HyperbolicIndexGenerator(nvars=2, max_level=5, pnorm=1.0, bkd=bkd)

        n_sel_before = gen.nselected_indices()
        n_cand_before = gen.ncandidate_indices()

        # All should be selected since we computed all indices
        self.assertEqual(n_cand_before, 0)
        self.assertGreater(n_sel_before, 0)

    def test_selected_candidates_disjoint(self):
        """Test that selected and candidate sets are disjoint."""
        bkd = self.bkd
        # Create a generator at level 2, then step to get candidates
        gen = HyperbolicIndexGenerator(nvars=2, max_level=2, pnorm=1.0, bkd=bkd)

        # Step will move current indices to selected and find new candidates
        gen.step()

        sel = gen.get_selected_indices()
        cand = gen.get_candidate_indices()

        if cand is not None:
            sel_set = set()
            for idx in sel.T:
                sel_set.add(hash_index(idx, bkd))

            for idx in cand.T:
                self.assertNotIn(
                    hash_index(idx, bkd),
                    sel_set,
                    "Candidate index found in selected set",
                )


if __name__ == "__main__":
    unittest.main()
