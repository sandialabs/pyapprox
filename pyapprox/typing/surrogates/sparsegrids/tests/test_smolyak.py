"""Tests for smolyak.py - Smolyak coefficient computation and index utilities."""

import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray

from pyapprox.typing.surrogates.sparsegrids import (
    check_admissibility,
    compute_smolyak_coefficients,
    is_downward_closed,
)
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.protocols import Array
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests


class TestSmolyakCoefficients(Generic[Array], unittest.TestCase):
    """Tests for Smolyak coefficient computation."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def test_1d_level_2(self):
        """Test Smolyak coefficients for 1D level 2."""
        # 1D: indices [0], [1], [2]
        indices = self._bkd.asarray([[0, 1, 2]])
        coefs = compute_smolyak_coefficients(indices, self._bkd)

        # In 1D, Smolyak gives telescoping series: only final level has coef=1
        # c_0 = 1 - 1 = 0 (neighbor 1 exists)
        # c_1 = 1 - 1 = 0 (neighbor 2 exists)
        # c_2 = 1 - 0 = 1 (no neighbor 3)
        expected = self._bkd.asarray([0.0, 0.0, 1.0])
        self._bkd.assert_allclose(coefs, expected)

    def test_2d_level_1(self):
        """Test Smolyak coefficients for 2D level 1."""
        # 2D level 1: (0,0), (1,0), (0,1)
        indices = self._bkd.asarray([[0, 1, 0],
                                     [0, 0, 1]])
        coefs = compute_smolyak_coefficients(indices, self._bkd)

        # Expected: c_{0,0} = 1 - 1 - 1 = -1, c_{1,0} = 1, c_{0,1} = 1
        # Using inclusion-exclusion: c_k = sum (-1)^|e| indicator(k+e in K)
        # For (0,0): (0,0)+0=in, (0,0)+(1,0)=in, (0,0)+(0,1)=in, (0,0)+(1,1)=not in
        # = 1 - 1 - 1 + 0 = -1
        expected = self._bkd.asarray([-1.0, 1.0, 1.0])
        self._bkd.assert_allclose(coefs, expected)

    def test_coefficients_sum_to_one(self):
        """Test that Smolyak coefficients sum to 1."""
        # 2D level 2
        indices = self._bkd.asarray([[0, 1, 0, 2, 1, 0],
                                     [0, 0, 1, 0, 1, 2]])
        coefs = compute_smolyak_coefficients(indices, self._bkd)

        # Sum should be 1
        self._bkd.assert_allclose(
            self._bkd.asarray([float(self._bkd.sum(coefs))]),
            self._bkd.asarray([1.0])
        )


class TestSmolyakCoefficientsNumpy(TestSmolyakCoefficients[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestSmolyakCoefficientsTorch(TestSmolyakCoefficients[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestDownwardClosed(Generic[Array], unittest.TestCase):
    """Tests for downward closure checking."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def test_downward_closed_set(self):
        """Test that a proper set is detected as downward closed."""
        # Valid: (0,0), (1,0), (0,1)
        indices = self._bkd.asarray([[0, 1, 0],
                                     [0, 0, 1]])
        self.assertTrue(is_downward_closed(indices, self._bkd))

    def test_not_downward_closed(self):
        """Test that an improper set is detected."""
        # Invalid: (0,0), (2,0) - missing (1,0)
        indices = self._bkd.asarray([[0, 2],
                                     [0, 0]])
        self.assertFalse(is_downward_closed(indices, self._bkd))


class TestDownwardClosedNumpy(TestDownwardClosed[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestDownwardClosedTorch(TestDownwardClosed[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestAdmissibility(Generic[Array], unittest.TestCase):
    """Tests for admissibility checking."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def test_admissible_candidate(self):
        """Test admissible candidate detection."""
        # Existing: (0,0), (1,0), (0,1)
        existing = self._bkd.asarray([[0, 1, 0],
                                      [0, 0, 1]])

        # (1,1) is admissible: predecessors (0,1) and (1,0) exist
        candidate = self._bkd.asarray([1, 1])
        self.assertTrue(check_admissibility(candidate, existing, self._bkd))

    def test_inadmissible_candidate(self):
        """Test inadmissible candidate detection."""
        # Existing: (0,0), (1,0)
        existing = self._bkd.asarray([[0, 1],
                                      [0, 0]])

        # (1,1) is not admissible: predecessor (0,1) missing
        candidate = self._bkd.asarray([1, 1])
        self.assertFalse(check_admissibility(candidate, existing, self._bkd))


class TestAdmissibilityNumpy(TestAdmissibility[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestAdmissibilityTorch(TestAdmissibility[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestSmolyakMathematicalProperties(Generic[Array], unittest.TestCase):
    """Test mathematical properties of Smolyak coefficients."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def test_inclusion_exclusion_formula_manual(self):
        """Verify inclusion-exclusion formula: c_k = sum_e (-1)^|e| * I(k+e in K).

        For index (0,0) in set {(0,0), (1,0), (0,1)}:
        - e=(0,0): (0,0)+(0,0)=(0,0) in K -> +1
        - e=(1,0): (0,0)+(1,0)=(1,0) in K -> -1
        - e=(0,1): (0,0)+(0,1)=(0,1) in K -> -1
        - e=(1,1): (0,0)+(1,1)=(1,1) not in K -> +0
        Total: 1 - 1 - 1 + 0 = -1
        """
        indices = self._bkd.asarray([[0, 1, 0],
                                     [0, 0, 1]])
        coefs = compute_smolyak_coefficients(indices, self._bkd)

        # Verify (0,0) coefficient is -1
        self._bkd.assert_allclose(
            self._bkd.asarray([float(coefs[0])]),
            self._bkd.asarray([-1.0])
        )

    def test_telescoping_property_1d_various_levels(self):
        """In 1D, only the highest level index has non-zero coefficient.

        This is the telescoping series property.
        """
        for max_level in [1, 2, 3, 4, 5]:
            indices = self._bkd.asarray([list(range(max_level + 1))])
            coefs = compute_smolyak_coefficients(indices, self._bkd)

            # All coefficients except last should be 0
            expected = self._bkd.zeros((max_level + 1,))
            expected[max_level] = 1.0

            self._bkd.assert_allclose(coefs, expected)

    def test_coefficients_sum_property_various_sets(self):
        """Smolyak coefficients always sum to 1 for any downward-closed set."""
        test_cases = [
            # 1D various levels
            [[0, 1, 2, 3]],
            # 2D level 1
            [[0, 1, 0], [0, 0, 1]],
            # 2D level 2
            [[0, 1, 0, 2, 1, 0], [0, 0, 1, 0, 1, 2]],
            # 2D level 3
            [[0, 1, 0, 2, 1, 0, 3, 2, 1, 0],
             [0, 0, 1, 0, 1, 2, 0, 1, 2, 3]],
            # 3D level 1
            [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        ]

        for idx_list in test_cases:
            indices = self._bkd.asarray(idx_list)
            coefs = compute_smolyak_coefficients(indices, self._bkd)
            coef_sum = float(self._bkd.sum(coefs))

            self._bkd.assert_allclose(
                self._bkd.asarray([coef_sum]),
                self._bkd.asarray([1.0])
            )

    def test_boundary_indices_have_coefficient_one(self):
        """Indices on the boundary (no forward neighbors in set) have coef=1."""
        # 2D level 2: boundary indices are (2,0), (1,1), (0,2)
        indices = self._bkd.asarray([[0, 1, 0, 2, 1, 0],
                                     [0, 0, 1, 0, 1, 2]])
        coefs = compute_smolyak_coefficients(indices, self._bkd)

        # Indices 3, 4, 5 are boundary: (2,0), (1,1), (0,2)
        self._bkd.assert_allclose(
            coefs[3:6],
            self._bkd.asarray([1.0, 1.0, 1.0])
        )

    def test_negative_coefficients_exist(self):
        """Some interior indices have negative coefficients."""
        # 2D level 1: (0,0) has coefficient -1
        indices = self._bkd.asarray([[0, 1, 0],
                                     [0, 0, 1]])
        coefs = compute_smolyak_coefficients(indices, self._bkd)

        # At least one coefficient should be negative
        has_negative = bool(self._bkd.any_bool(coefs < 0))
        self.assertTrue(has_negative, "Smolyak should have negative coefficients")


class TestSmolyakMathematicalPropertiesNumpy(
    TestSmolyakMathematicalProperties[NDArray[Any]]
):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestSmolyakMathematicalPropertiesTorch(
    TestSmolyakMathematicalProperties[torch.Tensor]
):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
