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
        # Values shape: (nqoi, nsamples) = (1, nsamples)
        values = self._bkd.reshape(samples[0, :] ** 2 + samples[1, :], (1, -1))
        subspace.set_values(values)

        test_pts = self._bkd.asarray([[0.3, -0.5, 0.7], [0.2, 0.4, -0.3]])
        result = subspace(test_pts)
        expected = self._bkd.reshape(
            test_pts[0, :] ** 2 + test_pts[1, :], (1, -1)
        )

        self.assertTrue(self._bkd.allclose(result, expected, rtol=1e-10))


class TestExactInterpolation(Generic[Array], unittest.TestCase):
    """Tests for exact polynomial interpolation - dual backend.

    For Legendre basis with linear growth rule:
    - Level L sparse grid can exactly interpolate polynomials up to
      total degree L in each variable.

    The growth rule n(l) = l + 1 gives n points at level l.
    With n Gauss-Legendre points, polynomials up to degree 2n-1 are
    exactly integrated, and up to degree n-1 are exactly interpolated.
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self._basis = LegendrePolynomial1D(self._bkd)
        self._growth = LinearGrowthRule(scale=1, shift=1)

    def test_exact_interp_degree_1_2d(self) -> None:
        """Test exact interpolation of degree-1 polynomial in 2D.

        f(x, y) = 2*x + 3*y + 1 should be exactly interpolated by level >= 1.
        """
        grid = IsotropicCombinationSparseGrid(
            self._bkd, [self._basis, self._basis], self._growth, level=1
        )

        samples = grid.get_samples()
        x, y = samples[0, :], samples[1, :]
        # Values shape: (nqoi, nsamples) = (1, nsamples)
        values = self._bkd.reshape(2.0 * x + 3.0 * y + 1.0, (1, -1))
        grid.set_values(values)

        test_pts = self._bkd.asarray([[0.23, -0.67, 0.11], [0.45, 0.89, -0.33]])
        result = grid(test_pts)
        x_t, y_t = test_pts[0, :], test_pts[1, :]
        expected = self._bkd.reshape(2.0 * x_t + 3.0 * y_t + 1.0, (1, -1))

        self._bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_exact_interp_degree_2_2d(self) -> None:
        """Test exact interpolation of degree-2 polynomial in 2D.

        f(x, y) = x^2 + x*y + y^2 should be exactly interpolated by level >= 2.
        """
        grid = IsotropicCombinationSparseGrid(
            self._bkd, [self._basis, self._basis], self._growth, level=2
        )

        samples = grid.get_samples()
        x, y = samples[0, :], samples[1, :]
        # Values shape: (nqoi, nsamples) = (1, nsamples)
        values = self._bkd.reshape(x**2 + x * y + y**2, (1, -1))
        grid.set_values(values)

        test_pts = self._bkd.asarray([[0.23, -0.67, 0.11], [0.45, 0.89, -0.33]])
        result = grid(test_pts)
        x_t, y_t = test_pts[0, :], test_pts[1, :]
        expected = self._bkd.reshape(x_t**2 + x_t * y_t + y_t**2, (1, -1))

        self._bkd.assert_allclose(result, expected, rtol=1e-10)

    def test_exact_interp_degree_3_2d(self) -> None:
        """Test exact interpolation of degree-3 polynomial in 2D.

        f(x, y) = x^3 + y^3 should be exactly interpolated by level >= 3.
        """
        grid = IsotropicCombinationSparseGrid(
            self._bkd, [self._basis, self._basis], self._growth, level=3
        )

        samples = grid.get_samples()
        x, y = samples[0, :], samples[1, :]
        # Values shape: (nqoi, nsamples) = (1, nsamples)
        values = self._bkd.reshape(x**3 + y**3, (1, -1))
        grid.set_values(values)

        test_pts = self._bkd.asarray([[0.23, -0.67, 0.11], [0.45, 0.89, -0.33]])
        result = grid(test_pts)
        x_t, y_t = test_pts[0, :], test_pts[1, :]
        expected = self._bkd.reshape(x_t**3 + y_t**3, (1, -1))

        self._bkd.assert_allclose(result, expected, rtol=1e-10)

    def test_anisotropic_polynomial_higher_in_dim0(self) -> None:
        """Test polynomial with higher degree in dim 0.

        f(x, y) = x^4 + y should require more refinement in x than y.
        Error should decrease faster as level increases.
        """
        errors = []
        for level in [2, 3, 4]:
            grid = IsotropicCombinationSparseGrid(
                self._bkd, [self._basis, self._basis], self._growth, level=level
            )

            samples = grid.get_samples()
            x, y = samples[0, :], samples[1, :]
            # x^4 + y: high degree in x, low degree in y
            # Values shape: (nqoi, nsamples) = (1, nsamples)
            values = self._bkd.reshape(x**4 + y, (1, -1))
            grid.set_values(values)

            # Test at multiple points
            test_pts = self._bkd.asarray(
                [[0.1, 0.3, 0.5, 0.7, 0.9, -0.2, -0.6],
                 [0.2, 0.4, 0.6, 0.8, -0.1, -0.3, -0.5]]
            )
            result = grid(test_pts)
            x_t, y_t = test_pts[0, :], test_pts[1, :]
            expected = self._bkd.reshape(x_t**4 + y_t, (1, -1))

            error = float(self._bkd.max(self._bkd.abs(result - expected)))
            errors.append(error)

        # Error should decrease as level increases
        self.assertLess(errors[1], errors[0])
        self.assertLess(errors[2], errors[1])

    def test_convergence_rate_smooth_function(self) -> None:
        """Test convergence rate for smooth function.

        For smooth functions, sparse grid error should decrease
        polynomially with number of points.
        """
        errors = []
        npoints_list = []

        for level in [1, 2, 3, 4]:
            grid = IsotropicCombinationSparseGrid(
                self._bkd, [self._basis, self._basis], self._growth, level=level
            )

            samples = grid.get_samples()
            npoints_list.append(grid.nsamples())
            x, y = samples[0, :], samples[1, :]
            # Smooth analytic function
            # Values shape: (nqoi, nsamples) = (1, nsamples)
            values = self._bkd.reshape(
                self._bkd.exp(-x**2 - y**2), (1, -1)
            )
            grid.set_values(values)

            # Dense test grid
            test_x = self._bkd.linspace(-0.9, 0.9, 10)
            test_y = self._bkd.linspace(-0.9, 0.9, 10)
            test_pts_list = []
            for i in range(10):
                for j in range(10):
                    test_pts_list.append([float(test_x[i]), float(test_y[j])])
            test_pts = self._bkd.asarray(test_pts_list).T

            result = grid(test_pts)
            x_t, y_t = test_pts[0, :], test_pts[1, :]
            expected = self._bkd.reshape(
                self._bkd.exp(-x_t**2 - y_t**2), (1, -1)
            )

            error = float(
                self._bkd.sqrt(self._bkd.mean((result - expected)**2))
            )
            errors.append(error)

        # Error should decrease significantly
        self.assertLess(errors[-1], errors[0] / 10)


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
        # Values shape: (nqoi, nsamples) = (1, nsamples)
        values = self._bkd.reshape(x ** 2 + x * y + y ** 2, (1, -1))
        grid.set_values(values)

        test_pts = self._bkd.asarray([[0.3, -0.5, 0.7], [0.2, 0.4, -0.3]])
        result = grid(test_pts)

        x_test, y_test = test_pts[0, :], test_pts[1, :]
        expected = self._bkd.reshape(
            x_test ** 2 + x_test * y_test + y_test ** 2, (1, -1)
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
        # Values shape: (nqoi, nsamples) = (1, nsamples)
        values = self._bkd.reshape(
            samples[0, :] + samples[1, :] + samples[2, :], (1, -1)
        )
        grid.set_values(values)

        test_pts = self._bkd.asarray([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [-0.1, -0.2, -0.3]
        ])
        result = grid(test_pts)
        expected = self._bkd.reshape(
            test_pts[0, :] + test_pts[1, :] + test_pts[2, :], (1, -1)
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
        # Values shape: (nqoi, nsamples) = (2, nsamples), stack along axis=0
        values = self._bkd.stack([x, y], axis=0)
        grid.set_values(values)

        test_pts = self._bkd.asarray([[0.3, -0.5], [0.2, 0.4]])
        result = grid(test_pts)

        # Result shape: (nqoi, nsamples) = (2, 2)
        self.assertEqual(result.shape[0], 2)  # nqoi
        self.assertTrue(
            self._bkd.allclose(result[0, :], test_pts[0, :], rtol=1e-10)
        )
        self.assertTrue(
            self._bkd.allclose(result[1, :], test_pts[1, :], rtol=1e-10)
        )


class TestIncrementalSmolyakUpdate(Generic[Array], unittest.TestCase):
    """Tests for _adjust_smolyak_coefficients incremental update.

    Tests verify the incremental Smolyak coefficient update algorithm
    produces the same results as computing coefficients from scratch.
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self._basis = LegendrePolynomial1D(self._bkd)
        self._growth = LinearGrowthRule(scale=1, shift=1)

    def test_add_single_index_1d(self) -> None:
        """Test incremental update when adding one index in 1D."""
        # Start with indices [0, 1]
        grid = CombinationSparseGrid(
            self._bkd, [self._basis], self._growth
        )
        grid._add_subspace(self._bkd.asarray([0]))
        grid._add_subspace(self._bkd.asarray([1]))

        # Current coefficients
        old_coefs = grid.get_smolyak_coefficients()

        # Add new index [2]
        new_index = self._bkd.asarray([2])
        old_indices = grid.get_subspace_indices()

        # Extend coefficients array for new index
        extended_coefs = self._bkd.hstack((old_coefs, self._bkd.zeros((1,))))
        new_indices = self._bkd.hstack((old_indices, new_index[:, None]))

        # Compute incrementally
        incremental_coefs = grid._adjust_smolyak_coefficients(
            extended_coefs, new_index, new_indices
        )

        # Compute from scratch
        scratch_coefs = compute_smolyak_coefficients(new_indices, self._bkd)

        self._bkd.assert_allclose(incremental_coefs, scratch_coefs)

    def test_add_single_index_2d(self) -> None:
        """Test incremental update when adding one index in 2D."""
        # Start with 2D level 1: {(0,0), (1,0), (0,1)}
        grid = CombinationSparseGrid(
            self._bkd, [self._basis, self._basis], self._growth
        )
        grid._add_subspace(self._bkd.asarray([0, 0]))
        grid._add_subspace(self._bkd.asarray([1, 0]))
        grid._add_subspace(self._bkd.asarray([0, 1]))

        old_coefs = grid.get_smolyak_coefficients()
        old_indices = grid.get_subspace_indices()

        # Add (1,1)
        new_index = self._bkd.asarray([1, 1])
        extended_coefs = self._bkd.hstack((old_coefs, self._bkd.zeros((1,))))
        new_indices = self._bkd.hstack((old_indices, new_index[:, None]))

        incremental_coefs = grid._adjust_smolyak_coefficients(
            extended_coefs, new_index, new_indices
        )
        scratch_coefs = compute_smolyak_coefficients(new_indices, self._bkd)

        self._bkd.assert_allclose(incremental_coefs, scratch_coefs)

    def test_add_boundary_index_2d(self) -> None:
        """Test incremental update when adding boundary index in 2D."""
        # Start with {(0,0), (1,0), (0,1), (1,1)}
        grid = CombinationSparseGrid(
            self._bkd, [self._basis, self._basis], self._growth
        )
        grid._add_subspace(self._bkd.asarray([0, 0]))
        grid._add_subspace(self._bkd.asarray([1, 0]))
        grid._add_subspace(self._bkd.asarray([0, 1]))
        grid._add_subspace(self._bkd.asarray([1, 1]))

        old_coefs = grid.get_smolyak_coefficients()
        old_indices = grid.get_subspace_indices()

        # Add (2,0) - boundary index
        new_index = self._bkd.asarray([2, 0])
        extended_coefs = self._bkd.hstack((old_coefs, self._bkd.zeros((1,))))
        new_indices = self._bkd.hstack((old_indices, new_index[:, None]))

        incremental_coefs = grid._adjust_smolyak_coefficients(
            extended_coefs, new_index, new_indices
        )
        scratch_coefs = compute_smolyak_coefficients(new_indices, self._bkd)

        self._bkd.assert_allclose(incremental_coefs, scratch_coefs)

    def test_add_multiple_indices_sequential(self) -> None:
        """Test sequential incremental updates produce correct result."""
        grid = CombinationSparseGrid(
            self._bkd, [self._basis, self._basis], self._growth
        )

        # Build up indices one by one
        indices_to_add = [
            self._bkd.asarray([0, 0]),
            self._bkd.asarray([1, 0]),
            self._bkd.asarray([0, 1]),
            self._bkd.asarray([2, 0]),
            self._bkd.asarray([1, 1]),
            self._bkd.asarray([0, 2]),
        ]

        # Add first index manually
        grid._add_subspace(indices_to_add[0])
        current_coefs = grid.get_smolyak_coefficients()

        # Add remaining indices incrementally
        for new_index in indices_to_add[1:]:
            old_indices = grid.get_subspace_indices()
            grid._add_subspace(new_index)

            extended_coefs = self._bkd.hstack(
                (current_coefs, self._bkd.zeros((1,)))
            )
            new_indices = self._bkd.hstack((old_indices, new_index[:, None]))

            current_coefs = grid._adjust_smolyak_coefficients(
                extended_coefs, new_index, new_indices
            )

        # Compare with from-scratch computation
        final_indices = grid.get_subspace_indices()
        scratch_coefs = compute_smolyak_coefficients(final_indices, self._bkd)

        self._bkd.assert_allclose(current_coefs, scratch_coefs)

    def test_coefficients_sum_to_one_after_update(self) -> None:
        """Test that coefficients still sum to 1 after incremental update."""
        grid = CombinationSparseGrid(
            self._bkd, [self._basis, self._basis], self._growth
        )
        grid._add_subspace(self._bkd.asarray([0, 0]))
        grid._add_subspace(self._bkd.asarray([1, 0]))
        grid._add_subspace(self._bkd.asarray([0, 1]))

        old_coefs = grid.get_smolyak_coefficients()
        old_indices = grid.get_subspace_indices()

        new_index = self._bkd.asarray([1, 1])
        extended_coefs = self._bkd.hstack((old_coefs, self._bkd.zeros((1,))))
        new_indices = self._bkd.hstack((old_indices, new_index[:, None]))

        incremental_coefs = grid._adjust_smolyak_coefficients(
            extended_coefs, new_index, new_indices
        )

        coef_sum = float(self._bkd.sum(incremental_coefs))
        self.assertAlmostEqual(coef_sum, 1.0, places=12)

    def test_3d_incremental_update(self) -> None:
        """Test incremental update in 3D."""
        grid = CombinationSparseGrid(
            self._bkd, [self._basis, self._basis, self._basis], self._growth
        )

        # Build 3D level 1 set
        grid._add_subspace(self._bkd.asarray([0, 0, 0]))
        grid._add_subspace(self._bkd.asarray([1, 0, 0]))
        grid._add_subspace(self._bkd.asarray([0, 1, 0]))
        grid._add_subspace(self._bkd.asarray([0, 0, 1]))

        old_coefs = grid.get_smolyak_coefficients()
        old_indices = grid.get_subspace_indices()

        # Add (1,1,0)
        new_index = self._bkd.asarray([1, 1, 0])
        extended_coefs = self._bkd.hstack((old_coefs, self._bkd.zeros((1,))))
        new_indices = self._bkd.hstack((old_indices, new_index[:, None]))

        incremental_coefs = grid._adjust_smolyak_coefficients(
            extended_coefs, new_index, new_indices
        )
        scratch_coefs = compute_smolyak_coefficients(new_indices, self._bkd)

        self._bkd.assert_allclose(incremental_coefs, scratch_coefs)


class TestSparseGridQuadrature(Generic[Array], unittest.TestCase):
    """Test sparse grid quadrature (mean computation).

    Tests verify:
    1. Exact integration for polynomials up to the quadrature degree
    2. Correct mean of constant, linear, and polynomial functions
    3. Accuracy tied to sparse grid level
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_mean_constant_1d(self) -> None:
        """Mean of f(x) = c should be c."""
        basis = LegendrePolynomial1D(self._bkd)
        growth = LinearGrowthRule(scale=1, shift=1)
        grid = IsotropicCombinationSparseGrid(
            self._bkd, [basis], growth, level=1
        )

        samples = grid.get_samples()
        c = 3.5
        # Values shape: (nqoi, nsamples) = (1, nsamples)
        values = self._bkd.full((1, samples.shape[1]), c)
        grid.set_values(values)

        mean = grid.mean()
        self._bkd.assert_allclose(mean, self._bkd.asarray([c]), rtol=1e-12)

    def test_mean_constant_2d(self) -> None:
        """Mean of f(x,y) = c should be c in 2D."""
        basis = LegendrePolynomial1D(self._bkd)
        growth = LinearGrowthRule(scale=1, shift=1)
        grid = IsotropicCombinationSparseGrid(
            self._bkd, [basis, basis], growth, level=2
        )

        samples = grid.get_samples()
        c = 2.7
        # Values shape: (nqoi, nsamples) = (1, nsamples)
        values = self._bkd.full((1, samples.shape[1]), c)
        grid.set_values(values)

        mean = grid.mean()
        self._bkd.assert_allclose(mean, self._bkd.asarray([c]), rtol=1e-12)

    def test_mean_linear_is_zero(self) -> None:
        """Mean of f(x) = x should be 0 on [-1,1]."""
        basis = LegendrePolynomial1D(self._bkd)
        growth = LinearGrowthRule(scale=1, shift=1)
        grid = IsotropicCombinationSparseGrid(
            self._bkd, [basis], growth, level=2
        )

        samples = grid.get_samples()
        # Values shape: (nqoi, nsamples) = (1, nsamples)
        values = self._bkd.reshape(samples[0, :], (1, -1))
        grid.set_values(values)

        mean = grid.mean()
        self._bkd.assert_allclose(mean, self._bkd.asarray([0.0]), atol=1e-12)

    def test_mean_quadratic(self) -> None:
        """Mean of f(x) = x^2 should be 1/3 on [-1,1]."""
        basis = LegendrePolynomial1D(self._bkd)
        growth = LinearGrowthRule(scale=1, shift=1)
        grid = IsotropicCombinationSparseGrid(
            self._bkd, [basis], growth, level=2
        )

        samples = grid.get_samples()
        # Values shape: (nqoi, nsamples) = (1, nsamples)
        values = self._bkd.reshape(samples[0, :] ** 2, (1, -1))
        grid.set_values(values)

        mean = grid.mean()
        expected = 1.0 / 3.0  # E[x^2] for uniform on [-1,1]
        self._bkd.assert_allclose(
            mean, self._bkd.asarray([expected]), rtol=1e-12
        )

    def test_mean_product_2d(self) -> None:
        """Mean of f(x,y) = x*y should be 0 on [-1,1]^2."""
        basis = LegendrePolynomial1D(self._bkd)
        growth = LinearGrowthRule(scale=1, shift=1)
        grid = IsotropicCombinationSparseGrid(
            self._bkd, [basis, basis], growth, level=2
        )

        samples = grid.get_samples()
        # Values shape: (nqoi, nsamples) = (1, nsamples)
        values = self._bkd.reshape(
            samples[0, :] * samples[1, :], (1, -1)
        )
        grid.set_values(values)

        mean = grid.mean()
        self._bkd.assert_allclose(mean, self._bkd.asarray([0.0]), atol=1e-12)

    def test_mean_sum_of_squares_2d(self) -> None:
        """Mean of f(x,y) = x^2 + y^2 should be 2/3 on [-1,1]^2."""
        basis = LegendrePolynomial1D(self._bkd)
        growth = LinearGrowthRule(scale=1, shift=1)
        grid = IsotropicCombinationSparseGrid(
            self._bkd, [basis, basis], growth, level=2
        )

        samples = grid.get_samples()
        # Values shape: (nqoi, nsamples) = (1, nsamples)
        values = self._bkd.reshape(
            samples[0, :] ** 2 + samples[1, :] ** 2, (1, -1)
        )
        grid.set_values(values)

        mean = grid.mean()
        expected = 2.0 / 3.0  # E[x^2] + E[y^2] = 1/3 + 1/3
        self._bkd.assert_allclose(
            mean, self._bkd.asarray([expected]), rtol=1e-12
        )

    def test_mean_polynomial_exact(self) -> None:
        """Sparse grid exactly integrates polynomials up to its degree."""
        basis = LegendrePolynomial1D(self._bkd)
        growth = LinearGrowthRule(scale=1, shift=1)

        # Level 2 with growth n(l) = l + 1 gives max degree per dim = 2
        # So can exactly integrate x^2, y^2, x^2*y^2 etc.
        grid = IsotropicCombinationSparseGrid(
            self._bkd, [basis, basis], growth, level=2
        )

        samples = grid.get_samples()
        # f(x,y) = x^2 * y^2, E[f] = E[x^2]*E[y^2] = (1/3)*(1/3) = 1/9
        # Values shape: (nqoi, nsamples) = (1, nsamples)
        values = self._bkd.reshape(
            (samples[0, :] ** 2) * (samples[1, :] ** 2), (1, -1)
        )
        grid.set_values(values)

        mean = grid.mean()
        expected = 1.0 / 9.0
        self._bkd.assert_allclose(
            mean, self._bkd.asarray([expected]), rtol=1e-12
        )

    def test_mean_high_degree_polynomial(self) -> None:
        """Higher level sparse grid can integrate higher degree polynomials."""
        basis = LegendrePolynomial1D(self._bkd)
        growth = LinearGrowthRule(scale=1, shift=1)

        # Level 4 for higher precision
        grid = IsotropicCombinationSparseGrid(
            self._bkd, [basis, basis], growth, level=4
        )

        samples = grid.get_samples()
        # f(x,y) = x^4 + y^4, E[f] = E[x^4] + E[y^4] = 1/5 + 1/5 = 2/5
        # E[x^4] = integral_{-1}^{1} x^4 * (1/2) dx = (1/5)
        # Values shape: (nqoi, nsamples) = (1, nsamples)
        values = self._bkd.reshape(
            samples[0, :] ** 4 + samples[1, :] ** 4, (1, -1)
        )
        grid.set_values(values)

        mean = grid.mean()
        expected = 2.0 / 5.0
        self._bkd.assert_allclose(
            mean, self._bkd.asarray([expected]), rtol=1e-11
        )

    def test_mean_multi_qoi(self) -> None:
        """Mean computation with multiple QoIs."""
        basis = LegendrePolynomial1D(self._bkd)
        growth = LinearGrowthRule(scale=1, shift=1)
        grid = IsotropicCombinationSparseGrid(
            self._bkd, [basis, basis], growth, level=2
        )

        samples = grid.get_samples()
        # QoI 1: f1 = 1 (mean = 1)
        # QoI 2: f2 = x^2 (mean = 1/3)
        qoi1 = self._bkd.ones((samples.shape[1],))
        qoi2 = samples[0, :] ** 2
        # Values shape: (nqoi, nsamples) = (2, nsamples), stack along axis=0
        values = self._bkd.stack([qoi1, qoi2], axis=0)
        grid.set_values(values)

        mean = grid.mean()
        expected = self._bkd.asarray([1.0, 1.0 / 3.0])
        self._bkd.assert_allclose(mean, expected, rtol=1e-12)

    def test_mean_3d(self) -> None:
        """Mean computation in 3D."""
        basis = LegendrePolynomial1D(self._bkd)
        growth = LinearGrowthRule(scale=1, shift=1)
        grid = IsotropicCombinationSparseGrid(
            self._bkd, [basis, basis, basis], growth, level=2
        )

        samples = grid.get_samples()
        # f(x,y,z) = x^2 + y^2 + z^2, E[f] = 3 * 1/3 = 1
        # Values shape: (nqoi, nsamples) = (1, nsamples)
        values = self._bkd.reshape(
            samples[0, :] ** 2 + samples[1, :] ** 2 + samples[2, :] ** 2,
            (1, -1)
        )
        grid.set_values(values)

        mean = grid.mean()
        expected = 1.0
        self._bkd.assert_allclose(
            mean, self._bkd.asarray([expected]), rtol=1e-12
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


class TestExactInterpolationNumpy(TestExactInterpolation[NDArray[Any]]):
    """NumPy backend tests for exact interpolation."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestSparseGridQuadratureNumpy(TestSparseGridQuadrature[NDArray[Any]]):
    """NumPy backend tests for sparse grid quadrature."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestIsotropicSparseGridNumpy(TestIsotropicSparseGrid[NDArray[Any]]):
    """NumPy backend tests for IsotropicCombinationSparseGrid."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestIncrementalSmolyakUpdateNumpy(TestIncrementalSmolyakUpdate[NDArray[Any]]):
    """NumPy backend tests for incremental Smolyak update."""

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


class TestExactInterpolationTorch(TestExactInterpolation[torch.Tensor]):
    """PyTorch backend tests for exact interpolation."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def bkd(self) -> TorchBkd:
        return TorchBkd()


class TestSparseGridQuadratureTorch(TestSparseGridQuadrature[torch.Tensor]):
    """PyTorch backend tests for sparse grid quadrature."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def bkd(self) -> TorchBkd:
        return TorchBkd()


class TestIncrementalSmolyakUpdateTorch(TestIncrementalSmolyakUpdate[torch.Tensor]):
    """PyTorch backend tests for incremental Smolyak update."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def bkd(self) -> TorchBkd:
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
