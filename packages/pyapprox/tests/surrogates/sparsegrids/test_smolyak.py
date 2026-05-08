"""Tests for smolyak.py - Smolyak coefficient computation and index utilities."""

from pyapprox.surrogates.sparsegrids import (
    check_admissibility,
    compute_smolyak_coefficients,
    is_downward_closed,
)


class TestSmolyakCoefficients:
    """Tests for Smolyak coefficient computation."""

    def test_1d_level_2(self, bkd):
        """Test Smolyak coefficients for 1D level 2."""
        # 1D: indices [0], [1], [2]
        indices = bkd.asarray([[0, 1, 2]])
        coefs = compute_smolyak_coefficients(indices, bkd)

        # In 1D, Smolyak gives telescoping series: only final level has coef=1
        # c_0 = 1 - 1 = 0 (neighbor 1 exists)
        # c_1 = 1 - 1 = 0 (neighbor 2 exists)
        # c_2 = 1 - 0 = 1 (no neighbor 3)
        expected = bkd.asarray([0.0, 0.0, 1.0])
        bkd.assert_allclose(coefs, expected)

    def test_2d_level_1(self, bkd):
        """Test Smolyak coefficients for 2D level 1."""
        # 2D level 1: (0,0), (1,0), (0,1)
        indices = bkd.asarray([[0, 1, 0], [0, 0, 1]])
        coefs = compute_smolyak_coefficients(indices, bkd)

        # Expected: c_{0,0} = 1 - 1 - 1 = -1, c_{1,0} = 1, c_{0,1} = 1
        # Using inclusion-exclusion: c_k = sum (-1)^|e| indicator(k+e in K)
        # For (0,0): (0,0)+0=in, (0,0)+(1,0)=in, (0,0)+(0,1)=in, (0,0)+(1,1)=not in
        # = 1 - 1 - 1 + 0 = -1
        expected = bkd.asarray([-1.0, 1.0, 1.0])
        bkd.assert_allclose(coefs, expected)

    def test_coefficients_sum_to_one(self, bkd):
        """Test that Smolyak coefficients sum to 1."""
        # 2D level 2
        indices = bkd.asarray([[0, 1, 0, 2, 1, 0], [0, 0, 1, 0, 1, 2]])
        coefs = compute_smolyak_coefficients(indices, bkd)

        # Sum should be 1
        bkd.assert_allclose(
            bkd.asarray([float(bkd.sum(coefs))]), bkd.asarray([1.0])
        )


class TestDownwardClosed:
    """Tests for downward closure checking."""

    def test_downward_closed_set(self, bkd):
        """Test that a proper set is detected as downward closed."""
        # Valid: (0,0), (1,0), (0,1)
        indices = bkd.asarray([[0, 1, 0], [0, 0, 1]])
        assert is_downward_closed(indices, bkd)

    def test_not_downward_closed(self, bkd):
        """Test that an improper set is detected."""
        # Invalid: (0,0), (2,0) - missing (1,0)
        indices = bkd.asarray([[0, 2], [0, 0]])
        assert not is_downward_closed(indices, bkd)


class TestAdmissibility:
    """Tests for admissibility checking."""

    def test_admissible_candidate(self, bkd):
        """Test admissible candidate detection."""
        # Existing: (0,0), (1,0), (0,1)
        existing = bkd.asarray([[0, 1, 0], [0, 0, 1]])

        # (1,1) is admissible: predecessors (0,1) and (1,0) exist
        candidate = bkd.asarray([1, 1])
        assert check_admissibility(candidate, existing, bkd)

    def test_inadmissible_candidate(self, bkd):
        """Test inadmissible candidate detection."""
        # Existing: (0,0), (1,0)
        existing = bkd.asarray([[0, 1], [0, 0]])

        # (1,1) is not admissible: predecessor (0,1) missing
        candidate = bkd.asarray([1, 1])
        assert not check_admissibility(candidate, existing, bkd)


class TestSmolyakMathematicalProperties:
    """Test mathematical properties of Smolyak coefficients."""

    def test_inclusion_exclusion_formula_manual(self, bkd):
        """Verify inclusion-exclusion formula: c_k = sum_e (-1)^|e| * I(k+e in K).

        For index (0,0) in set {(0,0), (1,0), (0,1)}:
        - e=(0,0): (0,0)+(0,0)=(0,0) in K -> +1
        - e=(1,0): (0,0)+(1,0)=(1,0) in K -> -1
        - e=(0,1): (0,0)+(0,1)=(0,1) in K -> -1
        - e=(1,1): (0,0)+(1,1)=(1,1) not in K -> +0
        Total: 1 - 1 - 1 + 0 = -1
        """
        indices = bkd.asarray([[0, 1, 0], [0, 0, 1]])
        coefs = compute_smolyak_coefficients(indices, bkd)

        # Verify (0,0) coefficient is -1
        bkd.assert_allclose(
            bkd.asarray([float(coefs[0])]), bkd.asarray([-1.0])
        )

    def test_telescoping_property_1d_various_levels(self, bkd):
        """In 1D, only the highest level index has non-zero coefficient.

        This is the telescoping series property.
        """
        for max_level in [1, 2, 3, 4, 5]:
            indices = bkd.asarray([list(range(max_level + 1))])
            coefs = compute_smolyak_coefficients(indices, bkd)

            # All coefficients except last should be 0
            expected = bkd.zeros((max_level + 1,))
            expected[max_level] = 1.0

            bkd.assert_allclose(coefs, expected)

    def test_coefficients_sum_property_various_sets(self, bkd):
        """Smolyak coefficients always sum to 1 for any downward-closed set."""
        test_cases = [
            # 1D various levels
            [[0, 1, 2, 3]],
            # 2D level 1
            [[0, 1, 0], [0, 0, 1]],
            # 2D level 2
            [[0, 1, 0, 2, 1, 0], [0, 0, 1, 0, 1, 2]],
            # 2D level 3
            [[0, 1, 0, 2, 1, 0, 3, 2, 1, 0], [0, 0, 1, 0, 1, 2, 0, 1, 2, 3]],
            # 3D level 1
            [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        ]

        for idx_list in test_cases:
            indices = bkd.asarray(idx_list)
            coefs = compute_smolyak_coefficients(indices, bkd)
            coef_sum = float(bkd.sum(coefs))

            bkd.assert_allclose(
                bkd.asarray([coef_sum]), bkd.asarray([1.0])
            )

    def test_boundary_indices_have_coefficient_one(self, bkd):
        """Indices on the boundary (no forward neighbors in set) have coef=1."""
        # 2D level 2: boundary indices are (2,0), (1,1), (0,2)
        indices = bkd.asarray([[0, 1, 0, 2, 1, 0], [0, 0, 1, 0, 1, 2]])
        coefs = compute_smolyak_coefficients(indices, bkd)

        # Indices 3, 4, 5 are boundary: (2,0), (1,1), (0,2)
        bkd.assert_allclose(coefs[3:6], bkd.asarray([1.0, 1.0, 1.0]))

    def test_negative_coefficients_exist(self, bkd):
        """Some interior indices have negative coefficients."""
        # 2D level 1: (0,0) has coefficient -1
        indices = bkd.asarray([[0, 1, 0], [0, 0, 1]])
        coefs = compute_smolyak_coefficients(indices, bkd)

        # At least one coefficient should be negative
        has_negative = bool(bkd.any_bool(coefs < 0))
        assert has_negative, "Smolyak should have negative coefficients"
