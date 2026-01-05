"""Dual-backend tests for CombinationSparseGrid and IsotropicCombinationSparseGrid.

Tests run on both NumPy and PyTorch backends using the base class pattern.
"""

import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.test_utils import load_tests

from pyapprox.typing.surrogates.sparsegrids import (
    CombinationSparseGrid,
    IsotropicCombinationSparseGrid,
    TensorProductSubspace,
    compute_smolyak_coefficients,
    is_downward_closed,
    check_admissibility,
)
from pyapprox.typing.surrogates.affine.univariate import LegendrePolynomial1D
from pyapprox.typing.surrogates.affine.indices import LinearGrowthRule


class TestSmolyakCoefficients(Generic[Array], unittest.TestCase):
    """Tests for Smolyak coefficient computation - dual backend."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_1d_level_2(self) -> None:
        """Test Smolyak coefficients for 1D level 2."""
        indices = self._bkd.asarray([[0, 1, 2]])
        coefs = compute_smolyak_coefficients(indices, self._bkd)

        # In 1D, only final level has coef=1
        expected = self._bkd.asarray([0.0, 0.0, 1.0])
        self.assertTrue(self._bkd.allclose(coefs, expected))

    def test_2d_level_1(self) -> None:
        """Test Smolyak coefficients for 2D level 1."""
        indices = self._bkd.asarray([[0, 1, 0], [0, 0, 1]])
        coefs = compute_smolyak_coefficients(indices, self._bkd)

        # c_{0,0} = -1, c_{1,0} = 1, c_{0,1} = 1
        expected = self._bkd.asarray([-1.0, 1.0, 1.0])
        self.assertTrue(self._bkd.allclose(coefs, expected))

    def test_coefficients_sum_to_one(self) -> None:
        """Test that Smolyak coefficients sum to 1."""
        indices = self._bkd.asarray([[0, 1, 0, 2, 1, 0], [0, 0, 1, 0, 1, 2]])
        coefs = compute_smolyak_coefficients(indices, self._bkd)
        self.assertAlmostEqual(float(self._bkd.sum(coefs)), 1.0, places=10)


class TestDownwardClosed(Generic[Array], unittest.TestCase):
    """Tests for downward closure checking - dual backend."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_downward_closed_set(self) -> None:
        """Test that a proper set is detected as downward closed."""
        indices = self._bkd.asarray([[0, 1, 0], [0, 0, 1]])
        self.assertTrue(is_downward_closed(indices, self._bkd))

    def test_not_downward_closed(self) -> None:
        """Test that an improper set is detected."""
        indices = self._bkd.asarray([[0, 2], [0, 0]])
        self.assertFalse(is_downward_closed(indices, self._bkd))


class TestAdmissibility(Generic[Array], unittest.TestCase):
    """Tests for admissibility checking - dual backend."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_admissible_candidate(self) -> None:
        """Test admissible candidate detection."""
        existing = self._bkd.asarray([[0, 1, 0], [0, 0, 1]])
        candidate = self._bkd.asarray([1, 1])
        self.assertTrue(check_admissibility(candidate, existing, self._bkd))

    def test_inadmissible_candidate(self) -> None:
        """Test inadmissible candidate detection."""
        existing = self._bkd.asarray([[0, 1], [0, 0]])
        candidate = self._bkd.asarray([1, 1])
        self.assertFalse(check_admissibility(candidate, existing, self._bkd))


class TestTensorProductSubspace(Generic[Array], unittest.TestCase):
    """Tests for TensorProductSubspace - dual backend."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_subspace_samples(self) -> None:
        """Test that subspace generates correct number of samples."""
        basis = LegendrePolynomial1D(self._bkd)
        growth = LinearGrowthRule(scale=1, shift=1)

        index = self._bkd.asarray([1, 2])
        subspace = TensorProductSubspace(
            self._bkd, index, [basis, basis], growth
        )

        self.assertEqual(subspace.nsamples(), 6)
        self.assertEqual(subspace.get_samples().shape, (2, 6))

    def test_subspace_interpolation(self) -> None:
        """Test that subspace interpolates exactly for polynomials."""
        basis = LegendrePolynomial1D(self._bkd)
        growth = LinearGrowthRule(scale=1, shift=1)

        index = self._bkd.asarray([2, 2])
        subspace = TensorProductSubspace(
            self._bkd, index, [basis, basis], growth
        )

        # f(x, y) = x^2 + y
        samples = subspace.get_samples()
        values = self._bkd.reshape(samples[0, :] ** 2 + samples[1, :], (-1, 1))
        subspace.set_values(values)

        test_pts = self._bkd.asarray([[0.3, -0.5, 0.7], [0.2, 0.4, -0.3]])
        result = subspace(test_pts)
        expected = self._bkd.reshape(
            test_pts[0, :] ** 2 + test_pts[1, :], (-1, 1)
        )

        self.assertTrue(self._bkd.allclose(result, expected, rtol=1e-10))


class TestIsotropicSparseGrid(Generic[Array], unittest.TestCase):
    """Tests for IsotropicCombinationSparseGrid - dual backend."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_level_0(self) -> None:
        """Test level 0 sparse grid (single point)."""
        basis = LegendrePolynomial1D(self._bkd)
        growth = LinearGrowthRule(scale=1, shift=1)

        grid = IsotropicCombinationSparseGrid(
            self._bkd, [basis, basis], growth, level=0
        )

        self.assertEqual(grid.nsubspaces(), 1)
        self.assertEqual(grid.nsamples(), 1)

    def test_level_2_subspaces(self) -> None:
        """Test level 2 sparse grid has correct number of subspaces."""
        basis = LegendrePolynomial1D(self._bkd)
        growth = LinearGrowthRule(scale=1, shift=1)

        grid = IsotropicCombinationSparseGrid(
            self._bkd, [basis, basis], growth, level=2
        )

        # 2D level 2: 6 subspaces
        self.assertEqual(grid.nsubspaces(), 6)

    def test_interpolation(self) -> None:
        """Test sparse grid interpolation."""
        basis = LegendrePolynomial1D(self._bkd)
        growth = LinearGrowthRule(scale=1, shift=1)

        grid = IsotropicCombinationSparseGrid(
            self._bkd, [basis, basis], growth, level=3
        )

        samples = grid.get_samples()
        x, y = samples[0, :], samples[1, :]
        values = self._bkd.reshape(x ** 2 + x * y + y ** 2, (-1, 1))
        grid.set_values(values)

        test_pts = self._bkd.asarray([[0.3, -0.5, 0.7], [0.2, 0.4, -0.3]])
        result = grid(test_pts)

        x_test, y_test = test_pts[0, :], test_pts[1, :]
        expected = self._bkd.reshape(
            x_test ** 2 + x_test * y_test + y_test ** 2, (-1, 1)
        )

        self.assertTrue(self._bkd.allclose(result, expected, rtol=1e-8))

    def test_smolyak_coefficients_sum(self) -> None:
        """Test Smolyak coefficients sum to 1."""
        basis = LegendrePolynomial1D(self._bkd)
        growth = LinearGrowthRule(scale=1, shift=1)

        for level in [1, 2, 3]:
            grid = IsotropicCombinationSparseGrid(
                self._bkd, [basis, basis], growth, level=level
            )
            coefs = grid.get_smolyak_coefficients()
            self.assertAlmostEqual(float(self._bkd.sum(coefs)), 1.0, places=10)

    def test_3d_sparse_grid(self) -> None:
        """Test 3D sparse grid construction and evaluation."""
        bases = [LegendrePolynomial1D(self._bkd) for _ in range(3)]
        growth = LinearGrowthRule(scale=1, shift=1)

        grid = IsotropicCombinationSparseGrid(self._bkd, bases, growth, level=2)

        # f(x, y, z) = x + y + z
        samples = grid.get_samples()
        values = self._bkd.reshape(
            samples[0, :] + samples[1, :] + samples[2, :], (-1, 1)
        )
        grid.set_values(values)

        test_pts = self._bkd.asarray([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [-0.1, -0.2, -0.3]
        ])
        result = grid(test_pts)
        expected = self._bkd.reshape(
            test_pts[0, :] + test_pts[1, :] + test_pts[2, :], (-1, 1)
        )

        self.assertTrue(self._bkd.allclose(result, expected, rtol=1e-10))

    def test_multi_qoi(self) -> None:
        """Test sparse grid with multiple quantities of interest."""
        basis = LegendrePolynomial1D(self._bkd)
        growth = LinearGrowthRule(scale=1, shift=1)

        grid = IsotropicCombinationSparseGrid(
            self._bkd, [basis, basis], growth, level=2
        )

        samples = grid.get_samples()
        x, y = samples[0, :], samples[1, :]
        # Two QoIs: f1 = x, f2 = y
        values = self._bkd.stack([x, y], axis=1)
        grid.set_values(values)

        test_pts = self._bkd.asarray([[0.3, -0.5], [0.2, 0.4]])
        result = grid(test_pts)

        self.assertEqual(result.shape[1], 2)
        self.assertTrue(
            self._bkd.allclose(result[:, 0], test_pts[0, :], rtol=1e-10)
        )
        self.assertTrue(
            self._bkd.allclose(result[:, 1], test_pts[1, :], rtol=1e-10)
        )


# NumPy backend tests
class TestSmolyakCoefficientsNumpy(TestSmolyakCoefficients[NDArray[Any]]):
    """NumPy backend tests for Smolyak coefficients."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestDownwardClosedNumpy(TestDownwardClosed[NDArray[Any]]):
    """NumPy backend tests for downward closure."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestAdmissibilityNumpy(TestAdmissibility[NDArray[Any]]):
    """NumPy backend tests for admissibility."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestTensorProductSubspaceNumpy(TestTensorProductSubspace[NDArray[Any]]):
    """NumPy backend tests for TensorProductSubspace."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestIsotropicSparseGridNumpy(TestIsotropicSparseGrid[NDArray[Any]]):
    """NumPy backend tests for IsotropicCombinationSparseGrid."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


# PyTorch backend tests
class TestSmolyakCoefficientsTorch(TestSmolyakCoefficients[torch.Tensor]):
    """PyTorch backend tests for Smolyak coefficients."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def bkd(self) -> TorchBkd:
        return TorchBkd()


class TestDownwardClosedTorch(TestDownwardClosed[torch.Tensor]):
    """PyTorch backend tests for downward closure."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def bkd(self) -> TorchBkd:
        return TorchBkd()


class TestAdmissibilityTorch(TestAdmissibility[torch.Tensor]):
    """PyTorch backend tests for admissibility."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def bkd(self) -> TorchBkd:
        return TorchBkd()


class TestTensorProductSubspaceTorch(TestTensorProductSubspace[torch.Tensor]):
    """PyTorch backend tests for TensorProductSubspace."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def bkd(self) -> TorchBkd:
        return TorchBkd()


class TestIsotropicSparseGridTorch(TestIsotropicSparseGrid[torch.Tensor]):
    """PyTorch backend tests for IsotropicCombinationSparseGrid."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def bkd(self) -> TorchBkd:
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
