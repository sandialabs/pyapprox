"""Tests for index generation module."""

import pytest

from pyapprox.surrogates.affine.indices.admissibility import (
    CompositeCriteria,
    Max1DLevelsCriteria,
    MaxIndicesCriteria,
    MaxLevelCriteria,
)
from pyapprox.surrogates.affine.indices.generators import (
    HyperbolicIndexGenerator,
    HyperbolicIndexSequence,
    IsotropicSparseGridBasisIndexGenerator,
    SparseGridIndexSequence,
)
from pyapprox.surrogates.affine.indices.growth_rules import (
    ClenshawCurtisGrowthRule,
    ConstantGrowthRule,
    ExponentialGrowthRule,
    LinearGrowthRule,
    inverse_growth_rule,
)
from pyapprox.surrogates.affine.indices.utils import (
    compute_downward_closure,
    compute_hyperbolic_indices,
    compute_hyperbolic_level_indices,
    hash_index,
    indices_pnorm,
    sort_indices_lexiographically,
)
from pyapprox.surrogates.affine.protocols.index import (
    IndexSequenceProtocol,
)


class TestHashIndex:
    """Test hash_index function."""

    def test_hash_consistency(self, bkd):
        """Test that same index produces same hash."""
        idx1 = bkd.asarray([1, 2, 3], dtype=bkd.int64_dtype())
        idx2 = bkd.asarray([1, 2, 3], dtype=bkd.int64_dtype())
        assert hash_index(idx1, bkd) == hash_index(idx2, bkd)

    def test_hash_uniqueness(self, bkd):
        """Test that different indices produce different hashes."""
        idx1 = bkd.asarray([1, 2, 3], dtype=bkd.int64_dtype())
        idx2 = bkd.asarray([1, 2, 4], dtype=bkd.int64_dtype())
        idx3 = bkd.asarray([3, 2, 1], dtype=bkd.int64_dtype())
        assert hash_index(idx1, bkd) != hash_index(idx2, bkd)
        assert hash_index(idx1, bkd) != hash_index(idx3, bkd)


class TestIndexPnorm:
    """Test indices_pnorm function."""

    def test_pnorm_1(self, bkd):
        """Test p=1 norm (total degree)."""
        indices = bkd.asarray([[0, 1, 2, 1], [0, 0, 0, 1]], dtype=bkd.int64_dtype())
        norms = indices_pnorm(indices, 1.0, bkd)
        expected = bkd.asarray([0.0, 1.0, 2.0, 2.0])
        bkd.assert_allclose(norms, expected)

    def test_pnorm_2(self, bkd):
        """Test p=2 norm."""
        indices = bkd.asarray([[3, 0], [4, 0]], dtype=bkd.int64_dtype())
        norms = indices_pnorm(indices, 2.0, bkd)
        expected = bkd.asarray([5.0, 0.0])
        bkd.assert_allclose(norms, expected)

    def test_pnorm_half(self, bkd):
        """Test p=0.5 norm (hyperbolic)."""
        indices = bkd.asarray([[1, 2], [1, 0]], dtype=bkd.int64_dtype())
        norms = indices_pnorm(indices, 0.5, bkd)
        # (1^0.5 + 1^0.5)^2 = 4, (2^0.5 + 0)^2 = 2
        expected = bkd.asarray([4.0, 2.0])
        bkd.assert_allclose(norms, expected)


class TestHyperbolicLevelIndices:
    """Test compute_hyperbolic_level_indices function."""

    def test_level_0(self, bkd):
        """Test level 0 indices."""
        indices = compute_hyperbolic_level_indices(2, 0, 1.0, bkd)
        expected = bkd.asarray([[0], [0]], dtype=bkd.int64_dtype())
        bkd.assert_allclose(indices, expected)

    def test_level_1_2d(self, bkd):
        """Test level 1 indices in 2D."""
        indices = compute_hyperbolic_level_indices(2, 1, 1.0, bkd)
        assert indices.shape[1] == 2
        # Should have [1,0] and [0,1]
        norms = indices_pnorm(indices, 1.0, bkd)
        assert bkd.all_bool(norms == 1.0)

    def test_level_2_2d(self, bkd):
        """Test level 2 indices in 2D."""
        indices = compute_hyperbolic_level_indices(2, 2, 1.0, bkd)
        assert indices.shape[1] == 3
        # Should have [2,0], [1,1], [0,2]
        norms = indices_pnorm(indices, 1.0, bkd)
        assert bkd.all_bool(norms == 2.0)


class TestHyperbolicIndices:
    """Test compute_hyperbolic_indices function."""

    def test_total_degree_2d_level2(self, bkd):
        """Test total degree indices in 2D up to level 2."""
        indices = compute_hyperbolic_indices(2, 2, 1.0, bkd)
        # Level 0: 1, Level 1: 2, Level 2: 3 -> total 6
        assert indices.shape[1] == 6

    def test_total_degree_3d_level2(self, bkd):
        """Test total degree indices in 3D up to level 2."""
        indices = compute_hyperbolic_indices(3, 2, 1.0, bkd)
        # Level 0: 1, Level 1: 3, Level 2: 6 -> total 10
        assert indices.shape[1] == 10

    def test_contains_zero_index(self, bkd):
        """Test that result contains zero index."""
        indices = compute_hyperbolic_indices(2, 3, 1.0, bkd)
        zero_index = bkd.zeros((2,), dtype=bkd.int64_dtype())
        has_zero = False
        for idx in indices.T:
            if bkd.all_bool(idx == zero_index):
                has_zero = True
                break
        assert has_zero


class TestSortIndices:
    """Test sort_indices_lexiographically function."""

    def test_sort_2d(self, bkd):
        """Test sorting 2D indices.

        Sort is by total level first, then lexicographically within level.
        """
        # Unsorted indices: [1,0], [0,1], [2,0], [0,0], [1,1]
        indices = bkd.asarray(
            [[1, 0, 2, 0, 1], [0, 1, 0, 0, 1]], dtype=bkd.int64_dtype()
        )
        sorted_idx = sort_indices_lexiographically(indices, bkd)
        # Expected sort order:
        # Level 0: [0,0]
        # Level 1: [0,1], [1,0]
        # Level 2: [0,2], [1,1], [2,0] - but [2,0] not in input, so [1,1] and [2,0]
        # From input: [0,0] (level 0), [0,1] (level 1), [1,0] (level 1), [2,0] (level
        # 2), [1,1] (level 2)
        # Sorted by level first, then lexicographically:
        # [0,0], [1,0], [0,1], [2,0], [1,1]
        expected = bkd.asarray(
            [[0, 1, 0, 2, 1], [0, 0, 1, 0, 1]], dtype=bkd.int64_dtype()
        )
        bkd.assert_allclose(sorted_idx, expected)


class TestMaxLevelCriteria:
    """Test MaxLevelCriteria admissibility."""

    def test_below_level(self, bkd):
        """Test index below max level is admissible."""
        criteria = MaxLevelCriteria(5, 1.0, bkd)
        idx = bkd.asarray([2, 1], dtype=bkd.int64_dtype())
        assert criteria(idx)

    def test_at_level(self, bkd):
        """Test index at max level is admissible."""
        criteria = MaxLevelCriteria(5, 1.0, bkd)
        idx = bkd.asarray([3, 2], dtype=bkd.int64_dtype())
        assert criteria(idx)

    def test_above_level(self, bkd):
        """Test index above max level is not admissible."""
        criteria = MaxLevelCriteria(5, 1.0, bkd)
        idx = bkd.asarray([3, 3], dtype=bkd.int64_dtype())
        assert not criteria(idx)

    def test_pnorm_half(self, bkd):
        """Test with p=0.5 (hyperbolic cross)."""
        criteria = MaxLevelCriteria(4, 0.5, bkd)
        # (1^0.5 + 1^0.5)^2 = 4
        idx = bkd.asarray([1, 1], dtype=bkd.int64_dtype())
        assert criteria(idx)
        # (2^0.5 + 2^0.5)^2 = 8 > 4
        idx = bkd.asarray([2, 2], dtype=bkd.int64_dtype())
        assert not criteria(idx)


class TestMax1DLevelsCriteria:
    """Test Max1DLevelsCriteria admissibility."""

    def test_within_limits(self, bkd):
        """Test index within 1D limits is admissible."""
        max_levels = bkd.asarray([3, 4], dtype=bkd.int64_dtype())
        criteria = Max1DLevelsCriteria(max_levels, bkd)
        idx = bkd.asarray([2, 3], dtype=bkd.int64_dtype())
        assert criteria(idx)

    def test_at_limits(self, bkd):
        """Test index at 1D limits is admissible."""
        max_levels = bkd.asarray([3, 4], dtype=bkd.int64_dtype())
        criteria = Max1DLevelsCriteria(max_levels, bkd)
        idx = bkd.asarray([3, 4], dtype=bkd.int64_dtype())
        assert criteria(idx)

    def test_exceeds_one_limit(self, bkd):
        """Test index exceeding one 1D limit is not admissible."""
        max_levels = bkd.asarray([3, 4], dtype=bkd.int64_dtype())
        criteria = Max1DLevelsCriteria(max_levels, bkd)
        idx = bkd.asarray([4, 3], dtype=bkd.int64_dtype())
        assert not criteria(idx)


class TestMaxIndicesCriteria:
    """Test MaxIndicesCriteria admissibility."""

    def test_requires_generator(self, bkd):
        """Test that criteria requires generator to be set."""
        criteria = MaxIndicesCriteria(10, bkd)
        idx = bkd.asarray([1, 2], dtype=bkd.int64_dtype())
        with pytest.raises(RuntimeError):
            criteria(idx)

    def test_with_generator(self, bkd):
        """Test criteria with generator set."""
        # Create a generator with few indices
        gen = HyperbolicIndexGenerator(nvars=2, max_level=1, pnorm=1.0, bkd=bkd)
        # Level 0: 1, Level 1: 2 -> total 3 selected indices

        # Create criteria with max_nindices that allows a few more
        criteria = MaxIndicesCriteria(10, bkd)
        criteria.set_generator(gen)

        idx = bkd.asarray([1, 2], dtype=bkd.int64_dtype())
        # Should be True since we're below max
        assert criteria(idx)


class TestCompositeCriteria:
    """Test CompositeCriteria admissibility."""

    def test_all_pass(self, bkd):
        """Test composite when all criteria pass."""
        criteria = CompositeCriteria(
            MaxLevelCriteria(5, 1.0, bkd),
            Max1DLevelsCriteria(bkd.asarray([3, 3], dtype=bkd.int64_dtype()), bkd),
        )
        idx = bkd.asarray([2, 2], dtype=bkd.int64_dtype())
        assert criteria(idx)

    def test_one_fails(self, bkd):
        """Test composite when one criterion fails."""
        criteria = CompositeCriteria(
            MaxLevelCriteria(10, 1.0, bkd),  # passes
            Max1DLevelsCriteria(
                bkd.asarray([2, 2], dtype=bkd.int64_dtype()), bkd
            ),  # fails
        )
        idx = bkd.asarray([3, 1], dtype=bkd.int64_dtype())
        assert not criteria(idx)

    def test_add_criteria(self, bkd):
        """Test adding criteria dynamically."""
        criteria = CompositeCriteria(MaxLevelCriteria(10, 1.0, bkd))
        idx = bkd.asarray([4, 0], dtype=bkd.int64_dtype())
        assert criteria(idx)

        criteria.add_criteria(
            Max1DLevelsCriteria(bkd.asarray([3, 3], dtype=bkd.int64_dtype()), bkd)
        )
        assert not criteria(idx)


class TestGrowthRules:
    """Test growth rule classes."""

    def test_linear_growth(self):
        """Test LinearGrowthRule."""
        rule = LinearGrowthRule(scale=2, shift=1)
        assert rule(0) == 1
        assert rule(1) == 3
        assert rule(2) == 5
        assert rule(3) == 7

    def test_double_plus_one(self):
        """Test ClenshawCurtisGrowthRule."""
        rule = ClenshawCurtisGrowthRule()
        assert rule(0) == 1
        assert rule(1) == 3
        assert rule(2) == 5
        assert rule(3) == 9
        assert rule(4) == 17

    def test_constant_growth(self):
        """Test ConstantGrowthRule."""
        rule = ConstantGrowthRule(value=5)
        assert rule(0) == 5
        assert rule(1) == 5
        assert rule(10) == 5

    def test_exponential_growth(self):
        """Test ExponentialGrowthRule."""
        rule = ExponentialGrowthRule(base=2)
        assert rule(0) == 1
        assert rule(1) == 2
        assert rule(2) == 4
        assert rule(3) == 8


class TestHyperbolicIndexGenerator:
    """Test HyperbolicIndexGenerator."""

    def test_total_degree_2d(self, bkd):
        """Test total degree indices in 2D."""
        gen = HyperbolicIndexGenerator(nvars=2, max_level=3, pnorm=1.0, bkd=bkd)
        indices = gen.get_selected_indices()
        # Level 0: 1, Level 1: 2, Level 2: 3, Level 3: 4 -> total 10
        assert indices.shape[1] == 10
        assert gen.nvars() == 2

    def test_total_degree_3d(self, bkd):
        """Test total degree indices in 3D."""
        gen = HyperbolicIndexGenerator(nvars=3, max_level=2, pnorm=1.0, bkd=bkd)
        indices = gen.get_selected_indices()
        # Level 0: 1, Level 1: 3, Level 2: 6 -> total 10
        assert indices.shape[1] == 10

    def test_hyperbolic_cross(self, bkd):
        """Test hyperbolic cross indices with p=0.5."""
        gen = HyperbolicIndexGenerator(nvars=2, max_level=4, pnorm=0.5, bkd=bkd)
        indices = gen.get_selected_indices()
        # Check that norms don't exceed max_level
        norms = indices_pnorm(indices, 0.5, bkd)
        assert bkd.all_bool(norms <= 4.0 + 1e-10)

    def test_with_max_1d_levels(self, bkd):
        """Test with 1D level constraints."""
        max_1d_levels = bkd.asarray([2, 3], dtype=bkd.int64_dtype())
        gen = HyperbolicIndexGenerator(
            nvars=2, max_level=10, pnorm=1.0, bkd=bkd, max_1d_levels=max_1d_levels
        )
        indices = gen.get_selected_indices()
        # Check 1D constraints
        assert bkd.all_bool(indices[0, :] <= 2)
        assert bkd.all_bool(indices[1, :] <= 3)
        # Should have (2+1)*(3+1) = 12 indices
        assert indices.shape[1] == 12

    def test_step(self, bkd):
        """Test stepping to increase max level."""
        gen = HyperbolicIndexGenerator(nvars=2, max_level=2, pnorm=1.0, bkd=bkd)
        n_before = gen.nselected_indices()

        gen.step()  # Increase to level 3
        n_after = gen.nselected_indices()

        assert n_after > n_before

    def test_downward_closed(self, bkd):
        """Test that generated indices are downward closed."""
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
                    assert (
                        tuple(neighbor) in indices_set
                    ), f"Index {idx_np} missing backward neighbor in dim {dim}"


class TestIterativeIndexGenerator:
    """Test IterativeIndexGenerator functionality."""

    def test_refine_index(self, bkd):
        """Test refining a specific index."""
        gen = HyperbolicIndexGenerator(nvars=2, max_level=5, pnorm=1.0, bkd=bkd)

        n_sel_before = gen.nselected_indices()
        n_cand_before = gen.ncandidate_indices()

        # All should be selected since we computed all indices
        assert n_cand_before == 0
        assert n_sel_before > 0

    def test_selected_candidates_disjoint(self, bkd):
        """Test that selected and candidate sets are disjoint."""
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
                assert (
                    hash_index(idx, bkd) not in sel_set
                ), "Candidate index found in selected set"


class TestInverseGrowthRule:
    """Test inverse_growth_rule function."""

    def test_linear_growth_rule_11(self):
        """Test LinearGrowthRule(1,1): n(l) = l+1, degree d needs level d."""
        rule = LinearGrowthRule(scale=1, shift=1)
        # n(0) = 1, so level 0 represents degree 0
        # n(1) = 2, so level 1 represents degree 1
        # n(d) = d+1, so level d represents degree d
        assert inverse_growth_rule(0, rule) == 0
        assert inverse_growth_rule(1, rule) == 1
        assert inverse_growth_rule(2, rule) == 2
        assert inverse_growth_rule(5, rule) == 5

    def test_linear_growth_rule_21(self):
        """Test LinearGrowthRule(2,1): n(l) = 2l+1 for l>0."""
        rule = LinearGrowthRule(scale=2, shift=1)
        # n(0) = 1, n(1) = 3, n(2) = 5, n(3) = 7
        # degree 0: need n > 0, n(0) = 1 > 0, level 0
        # degree 1: need n > 1, n(0) = 1 not > 1, n(1) = 3 > 1, level 1
        # degree 2: need n > 2, n(1) = 3 > 2, level 1
        # degree 3: need n > 3, n(1) = 3 not > 3, n(2) = 5 > 3, level 2
        # degree 4: need n > 4, n(2) = 5 > 4, level 2
        assert inverse_growth_rule(0, rule) == 0
        assert inverse_growth_rule(1, rule) == 1
        assert inverse_growth_rule(2, rule) == 1
        assert inverse_growth_rule(3, rule) == 2
        assert inverse_growth_rule(4, rule) == 2

    def test_double_plus_one(self):
        """Test ClenshawCurtisGrowthRule."""
        rule = ClenshawCurtisGrowthRule()
        # n(0) = 1, n(1) = 3, n(2) = 5, n(3) = 9, n(4) = 17
        # degree 0: n(0) = 1 > 0, level 0
        # degree 2: need n > 2, n(1) = 3 > 2, level 1
        # degree 4: need n > 4, n(2) = 5 > 4, level 2
        # degree 8: need n > 8, n(3) = 9 > 8, level 3
        assert inverse_growth_rule(0, rule) == 0
        assert inverse_growth_rule(2, rule) == 1
        assert inverse_growth_rule(4, rule) == 2
        assert inverse_growth_rule(8, rule) == 3

    def test_degree_zero(self):
        """Test that degree 0 returns level 0 for typical growth rules."""
        # Any growth rule with n(0) >= 1 should return level 0 for degree 0
        assert inverse_growth_rule(0, LinearGrowthRule(1, 1)) == 0
        assert inverse_growth_rule(0, ClenshawCurtisGrowthRule()) == 0
        assert inverse_growth_rule(0, ExponentialGrowthRule(2)) == 0

    def test_negative_degree_raises(self):
        """Test that negative degree raises ValueError."""
        rule = LinearGrowthRule(1, 1)
        with pytest.raises(ValueError):
            inverse_growth_rule(-1, rule)


class TestComputeDownwardClosure:
    """Test compute_downward_closure function."""

    def test_single_index_2d(self, bkd):
        """Test closure of {(2,1)} has 6 elements."""
        indices = bkd.asarray([[2], [1]], dtype=bkd.int64_dtype())
        closure = compute_downward_closure(indices, bkd)
        # {(0,0), (1,0), (2,0), (0,1), (1,1), (2,1)} = 6 elements
        assert closure.shape[1] == 6
        assert closure.shape[0] == 2

    def test_multiple_indices(self, bkd):
        """Test closure of {(1,0), (0,2)} has 4 elements."""
        indices = bkd.asarray([[1, 0], [0, 2]], dtype=bkd.int64_dtype())
        closure = compute_downward_closure(indices, bkd)
        # {(0,0), (1,0), (0,1), (0,2)} = 4 elements
        assert closure.shape[1] == 4

    def test_zero_index(self, bkd):
        """Test closure of {(0,0)} is just {(0,0)}."""
        indices = bkd.asarray([[0], [0]], dtype=bkd.int64_dtype())
        closure = compute_downward_closure(indices, bkd)
        assert closure.shape[1] == 1
        expected = bkd.asarray([[0], [0]], dtype=bkd.int64_dtype())
        bkd.assert_allclose(closure, expected)

    def test_3d_closure(self, bkd):
        """Test 3D closure."""
        # Closure of {(1,1,1)} has 2*2*2 = 8 elements
        indices = bkd.asarray([[1], [1], [1]], dtype=bkd.int64_dtype())
        closure = compute_downward_closure(indices, bkd)
        assert closure.shape[1] == 8
        assert closure.shape[0] == 3

    def test_is_sorted(self, bkd):
        """Test that output is sorted lexicographically."""
        indices = bkd.asarray([[2], [1]], dtype=bkd.int64_dtype())
        closure = compute_downward_closure(indices, bkd)

        # Verify sorted: first by total level, then lexicographically
        # Expected order: (0,0), (1,0), (0,1), (2,0), (1,1), (2,1)
        expected = bkd.asarray(
            [[0, 1, 0, 2, 1, 2], [0, 0, 1, 0, 1, 1]], dtype=bkd.int64_dtype()
        )
        bkd.assert_allclose(closure, expected)

    def test_downward_closed_property(self, bkd):
        """Test that all predecessors are present in the closure."""
        indices = bkd.asarray([[3], [2]], dtype=bkd.int64_dtype())
        closure = compute_downward_closure(indices, bkd)

        # Build set of all indices in closure
        closure_set = set()
        for j in range(closure.shape[1]):
            idx = tuple(int(bkd.to_numpy(closure[i, j])) for i in range(2))
            closure_set.add(idx)

        # Verify every index has all its predecessors
        for idx in closure_set:
            for dim in range(2):
                if idx[dim] > 0:
                    predecessor = list(idx)
                    predecessor[dim] -= 1
                    assert (
                        tuple(predecessor) in closure_set
                    ), f"Index {idx} missing predecessor in dim {dim}"


class TestIsotropicSparseGridBasisIndexGenerator:
    """Test IsotropicSparseGridBasisIndexGenerator."""

    def test_linear_11_matches_total_degree(self, bkd):
        """LinearGrowthRule(1,1) should produce same indices as total degree."""
        rule = LinearGrowthRule(scale=1, shift=1)
        for nvars in [2, 3]:
            for max_level in [2, 3, 4]:
                gen = IsotropicSparseGridBasisIndexGenerator(
                    nvars,
                    max_level,
                    bkd,
                    growth_rules=rule,
                )
                sg_indices = gen.get_indices()
                td_indices = compute_hyperbolic_indices(
                    nvars,
                    max_level,
                    1.0,
                    bkd,
                )
                assert sg_indices.shape[1] == td_indices.shape[1], (
                    f"nvars={nvars}, max_level={max_level}: "
                    f"sg={sg_indices.shape[1]} != td={td_indices.shape[1]}"
                )

    def test_linear_21_has_cross_terms(self, bkd):
        """LinearGrowthRule(2,1) should include cross-terms like (1,1,0)."""
        rule = LinearGrowthRule(scale=2, shift=1)
        gen = IsotropicSparseGridBasisIndexGenerator(
            3,
            2,
            bkd,
            growth_rules=rule,
        )
        indices = gen.get_indices()

        # Build set of tuples for easy checking
        idx_set = set()
        for j in range(indices.shape[1]):
            idx_set.add(tuple(int(bkd.to_numpy(indices[i, j])) for i in range(3)))

        # (1,1,0) should be present (from subspace (1,1,0) with
        # growth_rule(1)=3, so max_degree=2 in each active dim)
        assert (1, 1, 0) in idx_set
        assert (2, 0, 0) in idx_set
        assert (0, 2, 0) in idx_set
        assert (0, 0, 2) in idx_set

    def test_intermediate_growth_rate(self, bkd):
        """Sparse grid indices should be between hyperbolic cross and TD."""
        nvars = 3
        max_level = 3

        # Hyperbolic cross (pnorm=0.5)
        hc_indices = compute_hyperbolic_indices(nvars, max_level, 0.5, bkd)
        n_hc = hc_indices.shape[1]

        # Total degree (pnorm=1.0)
        td_indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
        n_td = td_indices.shape[1]

        # Sparse grid with default growth rule (scale=2, shift=1)
        gen = IsotropicSparseGridBasisIndexGenerator(nvars, max_level, bkd)
        n_sg = gen.get_indices().shape[1]

        # SG should have more terms than hyperbolic cross but can differ
        # from total degree. For nvars=3, max_level=3 with (2,1) growth,
        # it should include cross-terms that HC prunes.
        assert n_sg > n_hc
        # With (2,1) growth and level 3, we expect more terms than TD
        # because max degree per dim can be 2*3=6
        assert n_sg > n_td

    def test_downward_closed(self, bkd):
        """Generated indices should be downward closed."""
        gen = IsotropicSparseGridBasisIndexGenerator(3, 3, bkd)
        indices = gen.get_indices()

        idx_set = set()
        for j in range(indices.shape[1]):
            idx_set.add(tuple(int(bkd.to_numpy(indices[i, j])) for i in range(3)))

        for idx in idx_set:
            for dim in range(3):
                if idx[dim] > 0:
                    predecessor = list(idx)
                    predecessor[dim] -= 1
                    assert (
                        tuple(predecessor) in idx_set
                    ), f"Index {idx} missing predecessor in dim {dim}"

    def test_per_dimension_growth_rules(self, bkd):
        """Per-dimension growth rules should work correctly."""
        rules = [
            LinearGrowthRule(scale=1, shift=1),
            LinearGrowthRule(scale=2, shift=1),
        ]
        gen = IsotropicSparseGridBasisIndexGenerator(
            2,
            2,
            bkd,
            growth_rules=rules,
        )
        indices = gen.get_indices()

        # Check that max degree in dim 0 <= max_level (scale=1)
        max_deg_0 = int(bkd.to_numpy(bkd.max(indices[0, :])))
        # Dim 0 uses LinearGrowthRule(1,1): at level l, n=l+1, max_deg=l
        # Max subspace level per dim is max_level=2, so max_deg_0 <= 2
        assert max_deg_0 <= 2

        # Dim 1 uses LinearGrowthRule(2,1): at level l, n=2l+1, max_deg=2l
        # Max subspace level per dim is max_level=2, so max_deg_1 <= 4
        max_deg_1 = int(bkd.to_numpy(bkd.max(indices[1, :])))
        assert max_deg_1 <= 4

        # With the faster growth in dim 1, should have higher degrees there
        assert max_deg_1 > max_deg_0

    def test_wrong_number_growth_rules_raises(self, bkd):
        """Mismatched number of growth rules should raise ValueError."""
        rules = [LinearGrowthRule(1, 1), LinearGrowthRule(2, 1)]
        with pytest.raises(ValueError):
            IsotropicSparseGridBasisIndexGenerator(3, 2, bkd, growth_rules=rules)

    def test_level_zero(self, bkd):
        """Level 0 should produce just the zero index."""
        gen = IsotropicSparseGridBasisIndexGenerator(3, 0, bkd)
        indices = gen.get_indices()
        assert indices.shape[1] == 1
        expected = bkd.zeros((3, 1), dtype=bkd.int64_dtype())
        bkd.assert_allclose(indices, expected)

    def test_repr(self, bkd):
        """Test repr contains class name."""
        gen = IsotropicSparseGridBasisIndexGenerator(2, 2, bkd)
        r = repr(gen)
        assert "IsotropicSparseGridBasisIndexGenerator" in r


class TestHyperbolicIndexSequence:
    """Test HyperbolicIndexSequence."""

    def test_matches_compute_hyperbolic_indices(self, bkd):
        """Output matches compute_hyperbolic_indices for several configs."""
        for nvars in [1, 2, 3]:
            for level in [0, 1, 2, 3]:
                for pnorm in [0.5, 1.0]:
                    seq = HyperbolicIndexSequence(nvars, pnorm, bkd)
                    result = seq(level)
                    expected = compute_hyperbolic_indices(nvars, level, pnorm, bkd)
                    bkd.assert_allclose(result, expected)

    def test_accessors(self, bkd):
        """Test nvars, pnorm, bkd accessors."""
        seq = HyperbolicIndexSequence(3, 0.5, bkd)
        assert seq.nvars() == 3
        assert seq.pnorm() == pytest.approx(0.5)
        assert seq.bkd() is bkd

    def test_satisfies_protocol(self, bkd):
        """HyperbolicIndexSequence satisfies IndexSequenceProtocol."""
        seq = HyperbolicIndexSequence(2, 1.0, bkd)
        assert isinstance(seq, IndexSequenceProtocol)


class TestSparseGridIndexSequence:
    """Test SparseGridIndexSequence."""

    def test_matches_generator(self, bkd):
        """Output matches IsotropicSparseGridBasisIndexGenerator."""
        for nvars in [2, 3]:
            for level in [0, 1, 2, 3]:
                seq = SparseGridIndexSequence(nvars, bkd)
                result = seq(level)
                gen = IsotropicSparseGridBasisIndexGenerator(nvars, level, bkd)
                expected = gen.get_indices()
                bkd.assert_allclose(result, expected)

    def test_with_custom_growth_rule(self, bkd):
        """Output matches generator with custom growth rule."""
        rule = LinearGrowthRule(scale=1, shift=1)
        nvars = 2
        level = 3
        seq = SparseGridIndexSequence(nvars, bkd, growth_rules=rule)
        result = seq(level)
        gen = IsotropicSparseGridBasisIndexGenerator(
            nvars, level, bkd, growth_rules=rule
        )
        expected = gen.get_indices()
        bkd.assert_allclose(result, expected)

    def test_accessors(self, bkd):
        """Test nvars and bkd accessors."""
        seq = SparseGridIndexSequence(3, bkd)
        assert seq.nvars() == 3
        assert seq.bkd() is bkd

    def test_satisfies_protocol(self, bkd):
        """SparseGridIndexSequence satisfies IndexSequenceProtocol."""
        seq = SparseGridIndexSequence(2, bkd)
        assert isinstance(seq, IndexSequenceProtocol)


class TestIndexSequenceProtocolCheckable:
    """Test that IndexSequenceProtocol is runtime checkable."""

    def test_non_conforming_object_fails(self):
        """An object without __call__ does not satisfy the protocol."""
        assert not isinstance("hello", IndexSequenceProtocol)

    def test_callable_without_correct_signature_passes_runtime_check(self):
        """A callable object passes runtime isinstance check."""

        # Note: runtime_checkable only checks method existence, not signature
        class DummySeq:
            def __call__(self, level):
                return None

        assert isinstance(DummySeq(), IndexSequenceProtocol)
