"""Tests for IsotropicCombinationSparseGrid."""

import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray

from pyapprox.typing.surrogates.affine.indices import (
    HyperbolicIndexGenerator,
    LinearGrowthRule,
)
from pyapprox.typing.surrogates.affine.univariate import LegendrePolynomial1D
from pyapprox.typing.surrogates.sparsegrids import (
    IsotropicCombinationSparseGrid,
    is_downward_closed,
)
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.protocols import Array
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests


class TestIsotropicSparseGrid(Generic[Array], unittest.TestCase):
    """Tests for IsotropicCombinationSparseGrid."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def test_level_0(self):
        """Test level 0 sparse grid (single point)."""
        basis = LegendrePolynomial1D(self._bkd)
        growth = LinearGrowthRule(scale=1, shift=1)  # n(l) = l + 1

        grid = IsotropicCombinationSparseGrid(
            self._bkd, [basis, basis], growth, level=0
        )

        # Level 0: only (0,0) subspace, 1 sample
        self.assertEqual(grid.nsubspaces(), 1)
        self.assertEqual(grid.nsamples(), 1)

    def test_level_2_subspaces(self):
        """Test level 2 sparse grid has correct number of subspaces."""
        basis = LegendrePolynomial1D(self._bkd)
        growth = LinearGrowthRule(scale=1, shift=1)  # n(l) = l + 1

        grid = IsotropicCombinationSparseGrid(
            self._bkd, [basis, basis], growth, level=2
        )

        # 2D level 2: indices with |k|_1 <= 2
        # (0,0), (1,0), (0,1), (2,0), (1,1), (0,2) = 6 subspaces
        self.assertEqual(grid.nsubspaces(), 6)

    def test_interpolation(self):
        """Test sparse grid interpolation."""
        basis = LegendrePolynomial1D(self._bkd)
        growth = LinearGrowthRule(scale=1, shift=1)  # n(l) = l + 1

        grid = IsotropicCombinationSparseGrid(
            self._bkd, [basis, basis], growth, level=3
        )

        # Test function: f(x, y) = x^2 + x*y + y^2
        samples = grid.get_samples()
        x, y = samples[0, :], samples[1, :]
        # Values shape: (nqoi, nsamples) = (1, nsamples)
        values = self._bkd.reshape(x ** 2 + x * y + y ** 2, (1, -1))
        grid.set_values(values)

        # Test at new points
        test_pts = self._bkd.asarray([[0.3, -0.5, 0.7],
                                      [0.2, 0.4, -0.3]])
        result = grid(test_pts)

        x_test, y_test = test_pts[0, :], test_pts[1, :]
        expected = self._bkd.reshape(
            x_test ** 2 + x_test * y_test + y_test ** 2, (1, -1)
        )

        self._bkd.assert_allclose(result, expected, rtol=1e-8)

    def test_smolyak_coefficients_sum(self):
        """Test Smolyak coefficients sum to 1."""
        basis = LegendrePolynomial1D(self._bkd)
        growth = LinearGrowthRule(scale=1, shift=1)  # n(l) = l + 1

        for level in [1, 2, 3]:
            grid = IsotropicCombinationSparseGrid(
                self._bkd, [basis, basis], growth, level=level
            )
            coefs = grid.get_smolyak_coefficients()
            self._bkd.assert_allclose(
                self._bkd.asarray([float(self._bkd.sum(coefs))]),
                self._bkd.asarray([1.0])
            )


class TestIsotropicSparseGridNumpy(TestIsotropicSparseGrid[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestIsotropicSparseGridTorch(TestIsotropicSparseGrid[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestIsotropicWithGenerator(Generic[Array], unittest.TestCase):
    """Tests for IsotropicCombinationSparseGrid with index generator integration."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def test_generator_is_accessible(self):
        """Test that the index generator is accessible and correctly typed."""
        basis = LegendrePolynomial1D(self._bkd)
        growth = LinearGrowthRule(scale=1, shift=1)

        grid = IsotropicCombinationSparseGrid(
            self._bkd, [basis, basis], growth, level=2
        )

        # Generator should be accessible
        gen = grid.get_index_generator()
        self.assertIsNotNone(gen)
        self.assertIsInstance(gen, HyperbolicIndexGenerator)

        # Generator should have correct properties
        self.assertEqual(gen.nvars(), 2)

    def test_generator_produces_same_indices(self):
        """Test that generator produces same indices as the grid's subspaces."""
        basis = LegendrePolynomial1D(self._bkd)
        growth = LinearGrowthRule(scale=1, shift=1)

        grid = IsotropicCombinationSparseGrid(
            self._bkd, [basis, basis], growth, level=2
        )

        # Get indices from grid and generator
        grid_indices = grid.get_subspace_indices()
        gen = grid.get_index_generator()
        gen_indices = gen.get_selected_indices()

        # Should have same number of indices
        self.assertEqual(grid_indices.shape[1], gen_indices.shape[1])

        # All grid indices should be in generator indices
        grid_set = set()
        for j in range(grid_indices.shape[1]):
            idx = tuple(int(grid_indices[i, j]) for i in range(2))
            grid_set.add(idx)

        gen_set = set()
        for j in range(gen_indices.shape[1]):
            idx = tuple(int(gen_indices[i, j]) for i in range(2))
            gen_set.add(idx)

        self.assertEqual(grid_set, gen_set)

    def test_generator_index_count_by_level(self):
        """Test that generator produces correct number of indices for each level."""
        basis = LegendrePolynomial1D(self._bkd)
        growth = LinearGrowthRule(scale=1, shift=1)

        # Expected number of indices for 2D isotropic grid at each level
        # Level L: number of (i,j) with i+j <= L = (L+1)(L+2)/2
        expected_counts = {
            0: 1,   # (0,0)
            1: 3,   # (0,0), (1,0), (0,1)
            2: 6,   # + (2,0), (1,1), (0,2)
            3: 10,  # + (3,0), (2,1), (1,2), (0,3)
        }

        for level, expected in expected_counts.items():
            grid = IsotropicCombinationSparseGrid(
                self._bkd, [basis, basis], growth, level=level
            )
            self.assertEqual(
                grid.nsubspaces(), expected,
                f"Level {level}: expected {expected} subspaces, got {grid.nsubspaces()}"
            )

    def test_2d_level_3_index_set(self):
        """Test exact index set for 2D level 3 sparse grid."""
        basis = LegendrePolynomial1D(self._bkd)
        growth = LinearGrowthRule(scale=1, shift=1)

        grid = IsotropicCombinationSparseGrid(
            self._bkd, [basis, basis], growth, level=3
        )

        # Expected indices: all (i,j) with i+j <= 3
        expected_indices = {
            (0, 0), (1, 0), (0, 1),  # level <= 1
            (2, 0), (1, 1), (0, 2),  # level = 2
            (3, 0), (2, 1), (1, 2), (0, 3),  # level = 3
        }

        # Get actual indices from grid
        grid_indices = grid.get_subspace_indices()
        actual_indices = set()
        for j in range(grid_indices.shape[1]):
            idx = tuple(int(grid_indices[i, j]) for i in range(2))
            actual_indices.add(idx)

        self.assertEqual(actual_indices, expected_indices)

    def test_3d_level_2_index_set(self):
        """Test exact index set for 3D level 2 sparse grid."""
        basis = LegendrePolynomial1D(self._bkd)
        growth = LinearGrowthRule(scale=1, shift=1)

        grid = IsotropicCombinationSparseGrid(
            self._bkd, [basis, basis, basis], growth, level=2
        )

        # Expected indices: all (i,j,k) with i+j+k <= 2
        expected_indices = {
            # level 0
            (0, 0, 0),
            # level 1
            (1, 0, 0), (0, 1, 0), (0, 0, 1),
            # level 2
            (2, 0, 0), (0, 2, 0), (0, 0, 2),
            (1, 1, 0), (1, 0, 1), (0, 1, 1),
        }

        # Get actual indices from grid
        grid_indices = grid.get_subspace_indices()
        actual_indices = set()
        for j in range(grid_indices.shape[1]):
            idx = tuple(int(grid_indices[i, j]) for i in range(3))
            actual_indices.add(idx)

        self.assertEqual(actual_indices, expected_indices)
        # 3D level 2: 1 + 3 + 6 = 10 subspaces
        self.assertEqual(grid.nsubspaces(), 10)

    def test_generator_downward_closed(self):
        """Test that generator produces a downward-closed index set."""
        basis = LegendrePolynomial1D(self._bkd)
        growth = LinearGrowthRule(scale=1, shift=1)

        grid = IsotropicCombinationSparseGrid(
            self._bkd, [basis, basis], growth, level=3
        )

        indices = grid.get_subspace_indices()
        self.assertTrue(is_downward_closed(indices, self._bkd))


class TestIsotropicWithGeneratorNumpy(TestIsotropicWithGenerator[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestIsotropicWithGeneratorTorch(TestIsotropicWithGenerator[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
