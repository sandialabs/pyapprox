"""Tests for sparse grid module."""

import unittest
import numpy as np

from pyapprox.typing.util.backends.numpy import NumpyBkd


class TestSmolyakCoefficients(unittest.TestCase):
    """Tests for Smolyak coefficient computation."""

    def setUp(self):
        self.bkd = NumpyBkd()

    def test_1d_level_2(self):
        """Test Smolyak coefficients for 1D level 2."""
        from pyapprox.typing.surrogates.sparsegrids import (
            compute_smolyak_coefficients,
        )

        # 1D: indices [0], [1], [2]
        indices = self.bkd.asarray([[0, 1, 2]])
        coefs = compute_smolyak_coefficients(indices, self.bkd)

        # In 1D, Smolyak gives telescoping series: only final level has coef=1
        # c_0 = 1 - 1 = 0 (neighbor 1 exists)
        # c_1 = 1 - 1 = 0 (neighbor 2 exists)
        # c_2 = 1 - 0 = 1 (no neighbor 3)
        expected = np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(self.bkd.to_numpy(coefs), expected)

    def test_2d_level_1(self):
        """Test Smolyak coefficients for 2D level 1."""
        from pyapprox.typing.surrogates.sparsegrids import (
            compute_smolyak_coefficients,
        )

        # 2D level 1: (0,0), (1,0), (0,1)
        indices = self.bkd.asarray([[0, 1, 0],
                                     [0, 0, 1]])
        coefs = compute_smolyak_coefficients(indices, self.bkd)

        # Expected: c_{0,0} = 1 - 1 - 1 = -1, c_{1,0} = 1, c_{0,1} = 1
        # Using inclusion-exclusion: c_k = sum (-1)^|e| indicator(k+e in K)
        # For (0,0): (0,0)+0=in, (0,0)+(1,0)=in, (0,0)+(0,1)=in, (0,0)+(1,1)=not in
        # = 1 - 1 - 1 + 0 = -1
        expected = np.array([-1.0, 1.0, 1.0])
        np.testing.assert_allclose(self.bkd.to_numpy(coefs), expected)

    def test_coefficients_sum_to_one(self):
        """Test that Smolyak coefficients sum to 1."""
        from pyapprox.typing.surrogates.sparsegrids import (
            compute_smolyak_coefficients,
        )

        # 2D level 2
        indices = self.bkd.asarray([[0, 1, 0, 2, 1, 0],
                                     [0, 0, 1, 0, 1, 2]])
        coefs = compute_smolyak_coefficients(indices, self.bkd)

        # Sum should be 1
        self.assertAlmostEqual(float(self.bkd.sum(coefs)), 1.0)


class TestDownwardClosed(unittest.TestCase):
    """Tests for downward closure checking."""

    def setUp(self):
        self.bkd = NumpyBkd()

    def test_downward_closed_set(self):
        """Test that a proper set is detected as downward closed."""
        from pyapprox.typing.surrogates.sparsegrids import is_downward_closed

        # Valid: (0,0), (1,0), (0,1)
        indices = self.bkd.asarray([[0, 1, 0],
                                     [0, 0, 1]])
        self.assertTrue(is_downward_closed(indices, self.bkd))

    def test_not_downward_closed(self):
        """Test that an improper set is detected."""
        from pyapprox.typing.surrogates.sparsegrids import is_downward_closed

        # Invalid: (0,0), (2,0) - missing (1,0)
        indices = self.bkd.asarray([[0, 2],
                                     [0, 0]])
        self.assertFalse(is_downward_closed(indices, self.bkd))


class TestTensorProductSubspace(unittest.TestCase):
    """Tests for TensorProductSubspace."""

    def setUp(self):
        self.bkd = NumpyBkd()

    def test_subspace_samples(self):
        """Test that subspace generates correct number of samples."""
        from pyapprox.typing.surrogates.sparsegrids import TensorProductSubspace
        from pyapprox.typing.surrogates.affine.univariate import LegendrePolynomial1D
        from pyapprox.typing.surrogates.affine.indices import LinearGrowthRule

        basis = LegendrePolynomial1D(self.bkd)
        growth = LinearGrowthRule(scale=1, shift=1)  # n(l) = l + 1

        # Level (1, 2) -> 2 x 3 = 6 samples
        index = self.bkd.asarray([1, 2])
        subspace = TensorProductSubspace(
            self.bkd, index, [basis, basis], growth
        )

        self.assertEqual(subspace.nsamples(), 6)
        self.assertEqual(subspace.get_samples().shape, (2, 6))

    def test_subspace_interpolation(self):
        """Test that subspace interpolates exactly for polynomials."""
        from pyapprox.typing.surrogates.sparsegrids import TensorProductSubspace
        from pyapprox.typing.surrogates.affine.univariate import LegendrePolynomial1D
        from pyapprox.typing.surrogates.affine.indices import LinearGrowthRule

        basis = LegendrePolynomial1D(self.bkd)
        growth = LinearGrowthRule(scale=1, shift=1)  # n(l) = l + 1

        # Level (2, 2) -> 3 x 3 = 9 samples, can interpolate degree 2 exactly
        index = self.bkd.asarray([2, 2])
        subspace = TensorProductSubspace(
            self.bkd, index, [basis, basis], growth
        )

        # Test function: f(x, y) = x^2 + y
        samples = subspace.get_samples()
        values = samples[0:1, :].T ** 2 + samples[1:2, :].T
        subspace.set_values(values)

        # Test at new points
        test_pts = self.bkd.asarray([[0.3, -0.5, 0.7],
                                      [0.2, 0.4, -0.3]])
        result = subspace(test_pts)
        expected = test_pts[0:1, :].T ** 2 + test_pts[1:2, :].T

        np.testing.assert_allclose(
            self.bkd.to_numpy(result),
            self.bkd.to_numpy(expected),
            rtol=1e-10
        )


class TestIsotropicSparseGrid(unittest.TestCase):
    """Tests for IsotropicCombinationSparseGrid."""

    def setUp(self):
        self.bkd = NumpyBkd()

    def test_level_0(self):
        """Test level 0 sparse grid (single point)."""
        from pyapprox.typing.surrogates.sparsegrids import (
            IsotropicCombinationSparseGrid,
        )
        from pyapprox.typing.surrogates.affine.univariate import LegendrePolynomial1D
        from pyapprox.typing.surrogates.affine.indices import LinearGrowthRule

        basis = LegendrePolynomial1D(self.bkd)
        growth = LinearGrowthRule(scale=1, shift=1)  # n(l) = l + 1

        grid = IsotropicCombinationSparseGrid(
            self.bkd, [basis, basis], growth, level=0
        )

        # Level 0: only (0,0) subspace, 1 sample
        self.assertEqual(grid.nsubspaces(), 1)
        self.assertEqual(grid.nsamples(), 1)

    def test_level_2_subspaces(self):
        """Test level 2 sparse grid has correct number of subspaces."""
        from pyapprox.typing.surrogates.sparsegrids import (
            IsotropicCombinationSparseGrid,
        )
        from pyapprox.typing.surrogates.affine.univariate import LegendrePolynomial1D
        from pyapprox.typing.surrogates.affine.indices import LinearGrowthRule

        basis = LegendrePolynomial1D(self.bkd)
        growth = LinearGrowthRule(scale=1, shift=1)  # n(l) = l + 1

        grid = IsotropicCombinationSparseGrid(
            self.bkd, [basis, basis], growth, level=2
        )

        # 2D level 2: indices with |k|_1 <= 2
        # (0,0), (1,0), (0,1), (2,0), (1,1), (0,2) = 6 subspaces
        self.assertEqual(grid.nsubspaces(), 6)

    def test_interpolation(self):
        """Test sparse grid interpolation."""
        from pyapprox.typing.surrogates.sparsegrids import (
            IsotropicCombinationSparseGrid,
        )
        from pyapprox.typing.surrogates.affine.univariate import LegendrePolynomial1D
        from pyapprox.typing.surrogates.affine.indices import LinearGrowthRule

        basis = LegendrePolynomial1D(self.bkd)
        growth = LinearGrowthRule(scale=1, shift=1)  # n(l) = l + 1

        grid = IsotropicCombinationSparseGrid(
            self.bkd, [basis, basis], growth, level=3
        )

        # Test function: f(x, y) = x^2 + x*y + y^2
        samples = grid.get_samples()
        x, y = samples[0, :], samples[1, :]
        values = (x ** 2 + x * y + y ** 2)[:, None]
        grid.set_values(values)

        # Test at new points
        test_pts = self.bkd.asarray([[0.3, -0.5, 0.7],
                                      [0.2, 0.4, -0.3]])
        result = grid(test_pts)

        x_test, y_test = test_pts[0, :], test_pts[1, :]
        expected = (x_test ** 2 + x_test * y_test + y_test ** 2)[:, None]

        np.testing.assert_allclose(
            self.bkd.to_numpy(result),
            self.bkd.to_numpy(expected),
            rtol=1e-8
        )

    def test_smolyak_coefficients_sum(self):
        """Test Smolyak coefficients sum to 1."""
        from pyapprox.typing.surrogates.sparsegrids import (
            IsotropicCombinationSparseGrid,
        )
        from pyapprox.typing.surrogates.affine.univariate import LegendrePolynomial1D
        from pyapprox.typing.surrogates.affine.indices import LinearGrowthRule

        basis = LegendrePolynomial1D(self.bkd)
        growth = LinearGrowthRule(scale=1, shift=1)  # n(l) = l + 1

        for level in [1, 2, 3]:
            grid = IsotropicCombinationSparseGrid(
                self.bkd, [basis, basis], growth, level=level
            )
            coefs = grid.get_smolyak_coefficients()
            self.assertAlmostEqual(float(self.bkd.sum(coefs)), 1.0)


class TestAdmissibility(unittest.TestCase):
    """Tests for admissibility checking."""

    def setUp(self):
        self.bkd = NumpyBkd()

    def test_admissible_candidate(self):
        """Test admissible candidate detection."""
        from pyapprox.typing.surrogates.sparsegrids import check_admissibility

        # Existing: (0,0), (1,0), (0,1)
        existing = self.bkd.asarray([[0, 1, 0],
                                      [0, 0, 1]])

        # (1,1) is admissible: predecessors (0,1) and (1,0) exist
        candidate = self.bkd.asarray([1, 1])
        self.assertTrue(check_admissibility(candidate, existing, self.bkd))

    def test_inadmissible_candidate(self):
        """Test inadmissible candidate detection."""
        from pyapprox.typing.surrogates.sparsegrids import check_admissibility

        # Existing: (0,0), (1,0)
        existing = self.bkd.asarray([[0, 1],
                                      [0, 0]])

        # (1,1) is not admissible: predecessor (0,1) missing
        candidate = self.bkd.asarray([1, 1])
        self.assertFalse(check_admissibility(candidate, existing, self.bkd))


if __name__ == "__main__":
    unittest.main()
