"""Dual-backend tests for CombinationSparseGrid base class.

Tests run on both NumPy and PyTorch backends using the base class pattern.

This file contains tests specific to CombinationSparseGrid base class:
- TestCombinationSparseGridBase: Core functionality tests
- TestIncrementalSmolyakUpdate: Incremental coefficient updates
- TestCombinationSparseGridLegacy: Legacy comparison tests

Tests for other components have been migrated to:
- test_smolyak.py: Smolyak coefficients, downward closure, admissibility
- test_subspace.py: TensorProductSubspace
- test_isotropic.py: IsotropicCombinationSparseGrid
- test_refinement.py: Refinement criteria
"""

import unittest
from typing import Any, Generic, cast

import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401

from pyapprox.typing.surrogates.sparsegrids import (
    CombinationSparseGrid,
    IsotropicCombinationSparseGrid,
    compute_smolyak_coefficients,
    PrebuiltBasisFactory,
)
from pyapprox.typing.surrogates.sparsegrids.subspace import TensorProductSubspace
from pyapprox.typing.surrogates.affine.univariate import LegendrePolynomial1D
from pyapprox.typing.surrogates.affine.indices import LinearGrowthRule


# =============================================================================
# Tests for CombinationSparseGrid base class functionality
# =============================================================================


class TestCombinationSparseGridBase(Generic[Array], unittest.TestCase):
    """Tests for CombinationSparseGrid base class functionality."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self._basis = LegendrePolynomial1D(self._bkd)
        self._factory = PrebuiltBasisFactory(self._basis)
        self._growth = LinearGrowthRule(scale=1, shift=1)

    def test_unique_samples_collected_correctly(self) -> None:
        """Test that unique samples are collected from all subspaces."""
        grid = CombinationSparseGrid(
            self._bkd, [self._factory, self._factory], self._growth
        )

        # Add subspaces manually
        grid._add_subspace(self._bkd.asarray([0, 0]))  # 1 sample
        grid._add_subspace(self._bkd.asarray([1, 0]))  # 2 samples
        grid._add_subspace(self._bkd.asarray([0, 1]))  # 2 samples

        samples = grid.get_samples()

        # Should have nvars=2
        self.assertEqual(samples.shape[0], 2)

        # Verify no duplicate columns
        for i in range(samples.shape[1]):
            for j in range(i + 1, samples.shape[1]):
                is_same = self._bkd.allclose(
                    samples[:, i:i+1], samples[:, j:j+1], rtol=1e-14, atol=1e-14
                )
                self.assertFalse(
                    is_same,
                    f"Duplicate samples found at indices {i} and {j}"
                )

    def test_values_distributed_to_subspaces(self) -> None:
        """Test that set_values distributes to each subspace correctly."""
        grid = CombinationSparseGrid(
            self._bkd, [self._factory, self._factory], self._growth
        )
        grid._add_subspace(self._bkd.asarray([1, 0]))
        grid._add_subspace(self._bkd.asarray([0, 1]))

        samples = grid.get_samples()
        # f(x,y) = x + y
        values = self._bkd.reshape(samples[0, :] + samples[1, :], (1, -1))
        grid.set_values(values)

        # Each subspace should have values at its sample locations
        for subspace in grid.get_subspaces():
            sub_samples = subspace.get_samples()
            sub_values = subspace.get_values()
            assert sub_values is not None
            expected = self._bkd.reshape(
                sub_samples[0, :] + sub_samples[1, :], (1, -1)
            )
            self._bkd.assert_allclose(sub_values, expected, rtol=1e-12)

    def test_smolyak_combination_evaluation(self) -> None:
        """Test evaluation is weighted sum: sum_k c_k * I_k(f)(x)."""
        grid = CombinationSparseGrid(
            self._bkd, [self._factory, self._factory], self._growth
        )
        grid._add_subspace(self._bkd.asarray([0, 0]))
        grid._add_subspace(self._bkd.asarray([1, 0]))
        grid._add_subspace(self._bkd.asarray([0, 1]))

        samples = grid.get_samples()
        values = self._bkd.reshape(samples[0, :] ** 2, (1, -1))
        grid.set_values(values)

        test_pt = self._bkd.asarray([[0.5], [0.3]])
        result = grid(test_pt)

        # Manually compute weighted sum
        coefs = grid.get_smolyak_coefficients()
        subspaces = grid.get_subspaces()
        manual_result = self._bkd.zeros((1, 1))
        for j, subspace in enumerate(subspaces):
            coef = float(coefs[j])
            if abs(coef) > 1e-14:
                manual_result = manual_result + coef * subspace(test_pt)

        self._bkd.assert_allclose(result, manual_result, rtol=1e-12)

    def test_mean_is_smolyak_weighted_subspace_means(self) -> None:
        """Test mean() computes Smolyak-weighted sum of subspace means."""
        grid = CombinationSparseGrid(
            self._bkd, [self._factory, self._factory], self._growth
        )
        grid._add_subspace(self._bkd.asarray([0, 0]))
        grid._add_subspace(self._bkd.asarray([1, 0]))
        grid._add_subspace(self._bkd.asarray([0, 1]))

        samples = grid.get_samples()
        # f(x,y) = x^2 + y^2
        values = self._bkd.reshape(
            samples[0, :] ** 2 + samples[1, :] ** 2, (1, -1)
        )
        grid.set_values(values)

        mean = grid.mean()

        # Manually compute weighted sum of subspace integrals
        coefs = grid.get_smolyak_coefficients()
        subspaces = grid.get_subspaces()
        manual_mean = self._bkd.zeros((1,))
        for j, subspace in enumerate(subspaces):
            coef = float(coefs[j])
            if abs(coef) > 1e-14:
                tp_subspace = cast(TensorProductSubspace[Array], subspace)
                manual_mean = manual_mean + coef * tp_subspace.integrate()

        self._bkd.assert_allclose(mean, manual_mean, rtol=1e-12)

    def test_variance_is_smolyak_weighted_subspace_variances(self) -> None:
        """Test variance() computes Smolyak-weighted sum of subspace variances."""
        grid = CombinationSparseGrid(
            self._bkd, [self._factory, self._factory], self._growth
        )
        grid._add_subspace(self._bkd.asarray([0, 0]))
        grid._add_subspace(self._bkd.asarray([1, 0]))
        grid._add_subspace(self._bkd.asarray([0, 1]))
        grid._add_subspace(self._bkd.asarray([1, 1]))

        samples = grid.get_samples()
        # f(x,y) = x
        values = self._bkd.reshape(samples[0, :], (1, -1))
        grid.set_values(values)

        variance = grid.variance()

        # Manually compute weighted sum of subspace variances
        coefs = grid.get_smolyak_coefficients()
        subspaces = grid.get_subspaces()
        manual_variance = self._bkd.zeros((1,))
        for j, subspace in enumerate(subspaces):
            coef = float(coefs[j])
            if abs(coef) > 1e-14:
                tp_subspace = cast(TensorProductSubspace[Array], subspace)
                manual_variance = manual_variance + coef * tp_subspace.variance()

        self._bkd.assert_allclose(variance, manual_variance, rtol=1e-12)

    def test_jacobian_is_smolyak_weighted_subspace_jacobians(self) -> None:
        """Test jacobian is weighted sum of subspace jacobians."""
        grid = CombinationSparseGrid(
            self._bkd, [self._factory, self._factory], self._growth
        )
        grid._add_subspace(self._bkd.asarray([0, 0]))
        grid._add_subspace(self._bkd.asarray([1, 0]))
        grid._add_subspace(self._bkd.asarray([0, 1]))
        grid._add_subspace(self._bkd.asarray([1, 1]))

        samples = grid.get_samples()
        # f(x,y) = x^2 + y^2
        values = self._bkd.reshape(
            samples[0, :] ** 2 + samples[1, :] ** 2, (1, -1)
        )
        grid.set_values(values)

        test_pt = self._bkd.asarray([[0.3], [0.5]])
        jacobian = grid.jacobian(test_pt)

        # Manually compute weighted sum of subspace jacobians
        coefs = grid.get_smolyak_coefficients()
        subspaces = grid.get_subspaces()
        manual_jacobian = self._bkd.zeros((1, 2))
        for j, subspace in enumerate(subspaces):
            coef = float(coefs[j])
            if abs(coef) > 1e-14:
                tp_subspace = cast(TensorProductSubspace[Array], subspace)
                manual_jacobian = manual_jacobian + coef * tp_subspace.jacobian(test_pt)

        self._bkd.assert_allclose(jacobian, manual_jacobian, rtol=1e-10)

    def test_hvp_is_smolyak_weighted_subspace_hvps(self) -> None:
        """Test hvp is weighted sum of subspace hvps."""
        grid = CombinationSparseGrid(
            self._bkd, [self._factory, self._factory], self._growth
        )
        grid._add_subspace(self._bkd.asarray([2, 0]))
        grid._add_subspace(self._bkd.asarray([0, 2]))
        grid._add_subspace(self._bkd.asarray([1, 1]))
        grid._add_subspace(self._bkd.asarray([2, 2]))

        samples = grid.get_samples()
        # f(x,y) = x^2 + y^2 (scalar function for hvp)
        values = self._bkd.reshape(
            samples[0, :] ** 2 + samples[1, :] ** 2, (1, -1)
        )
        grid.set_values(values)

        test_pt = self._bkd.asarray([[0.3], [0.5]])
        vec = self._bkd.asarray([[1.0], [2.0]])
        hvp = grid.hvp(test_pt, vec)

        # Manually compute weighted sum of subspace hvps
        coefs = grid.get_smolyak_coefficients()
        subspaces = grid.get_subspaces()
        manual_hvp = self._bkd.zeros((2, 1))
        for j, subspace in enumerate(subspaces):
            coef = float(coefs[j])
            if abs(coef) > 1e-14:
                tp_subspace = cast(TensorProductSubspace[Array], subspace)
                manual_hvp = manual_hvp + coef * tp_subspace.hvp(test_pt, vec)

        self._bkd.assert_allclose(hvp, manual_hvp, rtol=1e-10)

    def test_nqoi_multi_output(self) -> None:
        """Test CombinationSparseGrid with multiple QoIs."""
        grid = CombinationSparseGrid(
            self._bkd, [self._factory, self._factory], self._growth
        )
        grid._add_subspace(self._bkd.asarray([1, 0]))
        grid._add_subspace(self._bkd.asarray([0, 1]))

        samples = grid.get_samples()
        # Two QoIs: f1 = x, f2 = y
        values = self._bkd.stack([samples[0, :], samples[1, :]], axis=0)
        grid.set_values(values)

        self.assertEqual(grid.nqoi(), 2)

        # Test evaluation
        test_pts = self._bkd.asarray([[0.3, -0.5], [0.2, 0.4]])
        result = grid(test_pts)

        # Result shape: (nqoi, nsamples) = (2, 2)
        self.assertEqual(result.shape[0], 2)
        self._bkd.assert_allclose(result[0, :], test_pts[0, :], rtol=1e-10)
        self._bkd.assert_allclose(result[1, :], test_pts[1, :], rtol=1e-10)


# =============================================================================
# Tests for incremental Smolyak coefficient updates
# =============================================================================


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
        self._factory = PrebuiltBasisFactory(self._basis)
        self._growth = LinearGrowthRule(scale=1, shift=1)

    def test_add_single_index_1d(self) -> None:
        """Test incremental update when adding one index in 1D."""
        grid = CombinationSparseGrid(
            self._bkd, [self._factory], self._growth
        )
        grid._add_subspace(self._bkd.asarray([0]))
        grid._add_subspace(self._bkd.asarray([1]))

        old_coefs = grid.get_smolyak_coefficients()
        new_index = self._bkd.asarray([2])
        old_indices = grid.get_subspace_indices()

        extended_coefs = self._bkd.hstack((old_coefs, self._bkd.zeros((1,))))
        new_indices = self._bkd.hstack((old_indices, new_index[:, None]))

        incremental_coefs = grid._adjust_smolyak_coefficients(
            extended_coefs, new_index, new_indices
        )
        scratch_coefs = compute_smolyak_coefficients(new_indices, self._bkd)

        self._bkd.assert_allclose(incremental_coefs, scratch_coefs)

    def test_add_single_index_2d(self) -> None:
        """Test incremental update when adding one index in 2D."""
        grid = CombinationSparseGrid(
            self._bkd, [self._factory, self._factory], self._growth
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
        scratch_coefs = compute_smolyak_coefficients(new_indices, self._bkd)

        self._bkd.assert_allclose(incremental_coefs, scratch_coefs)

    def test_add_boundary_index_2d(self) -> None:
        """Test incremental update when adding boundary index in 2D."""
        grid = CombinationSparseGrid(
            self._bkd, [self._factory, self._factory], self._growth
        )
        grid._add_subspace(self._bkd.asarray([0, 0]))
        grid._add_subspace(self._bkd.asarray([1, 0]))
        grid._add_subspace(self._bkd.asarray([0, 1]))
        grid._add_subspace(self._bkd.asarray([1, 1]))

        old_coefs = grid.get_smolyak_coefficients()
        old_indices = grid.get_subspace_indices()

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
            self._bkd, [self._factory, self._factory], self._growth
        )

        indices_to_add = [
            self._bkd.asarray([0, 0]),
            self._bkd.asarray([1, 0]),
            self._bkd.asarray([0, 1]),
            self._bkd.asarray([2, 0]),
            self._bkd.asarray([1, 1]),
            self._bkd.asarray([0, 2]),
        ]

        grid._add_subspace(indices_to_add[0])
        current_coefs = grid.get_smolyak_coefficients()

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

        final_indices = grid.get_subspace_indices()
        scratch_coefs = compute_smolyak_coefficients(final_indices, self._bkd)

        self._bkd.assert_allclose(current_coefs, scratch_coefs)

    def test_coefficients_sum_to_one_after_update(self) -> None:
        """Test that coefficients still sum to 1 after incremental update."""
        grid = CombinationSparseGrid(
            self._bkd, [self._factory, self._factory], self._growth
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

        coef_sum = float(self._bkd.sum(incremental_coefs).item())
        self._bkd.assert_allclose(
            self._bkd.asarray([coef_sum]),
            self._bkd.asarray([1.0]),
            rtol=1e-12
        )

    def test_3d_incremental_update(self) -> None:
        """Test incremental update in 3D."""
        grid = CombinationSparseGrid(
            self._bkd, [self._factory, self._factory, self._factory], self._growth
        )

        grid._add_subspace(self._bkd.asarray([0, 0, 0]))
        grid._add_subspace(self._bkd.asarray([1, 0, 0]))
        grid._add_subspace(self._bkd.asarray([0, 1, 0]))
        grid._add_subspace(self._bkd.asarray([0, 0, 1]))

        old_coefs = grid.get_smolyak_coefficients()
        old_indices = grid.get_subspace_indices()

        new_index = self._bkd.asarray([1, 1, 0])
        extended_coefs = self._bkd.hstack((old_coefs, self._bkd.zeros((1,))))
        new_indices = self._bkd.hstack((old_indices, new_index[:, None]))

        incremental_coefs = grid._adjust_smolyak_coefficients(
            extended_coefs, new_index, new_indices
        )
        scratch_coefs = compute_smolyak_coefficients(new_indices, self._bkd)

        self._bkd.assert_allclose(incremental_coefs, scratch_coefs)


# =============================================================================
# Legacy comparison tests
# =============================================================================


class TestCombinationSparseGridLegacy(Generic[Array], unittest.TestCase):
    """Legacy comparison tests for CombinationSparseGrid.

    These tests verify that the typing module implementation matches
    the legacy implementation in pyapprox.surrogates.sparsegrids.combination.
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self._basis = LegendrePolynomial1D(self._bkd)
        self._factory = PrebuiltBasisFactory(self._basis)
        self._growth = LinearGrowthRule(scale=1, shift=1)

    def test_smolyak_coefficients_match_legacy_formula(self) -> None:
        """Verify Smolyak coefficients match legacy inclusion-exclusion formula.

        The legacy implementation uses:
        c_k = sum_e (-1)^|e| * I(k+e in K)
        where e iterates over all binary shifts in {0,1}^nvars.
        """
        # Build a 2D level 2 index set
        grid = IsotropicCombinationSparseGrid(
            self._bkd, [self._factory, self._factory], self._growth, level=2
        )

        indices = grid.get_subspace_indices()
        coefs = grid.get_smolyak_coefficients()

        # Manually compute using inclusion-exclusion formula
        nvars = 2
        nindices = indices.shape[1]

        # Create index lookup
        index_set = set()
        for j in range(nindices):
            idx = tuple(int(indices[i, j]) for i in range(nvars))
            index_set.add(idx)

        # Compute expected coefficients
        expected_coefs = []
        for j in range(nindices):
            idx = tuple(int(indices[i, j]) for i in range(nvars))
            coef = 0.0
            # Iterate over all binary shifts
            for shift_bits in range(2 ** nvars):
                shift = tuple((shift_bits >> d) & 1 for d in range(nvars))
                neighbor = tuple(idx[d] + shift[d] for d in range(nvars))
                if neighbor in index_set:
                    coef += (-1.0) ** sum(shift)
            expected_coefs.append(coef)

        expected = self._bkd.asarray(expected_coefs)
        self._bkd.assert_allclose(coefs, expected, rtol=1e-12)

    def test_evaluation_matches_smolyak_formula(self) -> None:
        """Verify evaluation matches Smolyak combination formula.

        I[f](x) = sum_k c_k * I_k[f](x)
        """
        grid = IsotropicCombinationSparseGrid(
            self._bkd, [self._factory, self._factory], self._growth, level=2
        )

        samples = grid.get_samples()
        x, y = samples[0, :], samples[1, :]
        values = self._bkd.reshape(x ** 2 + x * y + y ** 2, (1, -1))
        grid.set_values(values)

        # Test at multiple points
        test_pts = self._bkd.asarray([
            [0.1, 0.3, -0.5, 0.7],
            [0.2, -0.4, 0.6, -0.8]
        ])
        result = grid(test_pts)

        # Manually compute Smolyak combination
        coefs = grid.get_smolyak_coefficients()
        subspaces = grid.get_subspaces()

        manual_result = self._bkd.zeros((1, test_pts.shape[1]))
        for j, subspace in enumerate(subspaces):
            coef = float(coefs[j])
            if abs(coef) > 1e-14:
                manual_result = manual_result + coef * subspace(test_pts)

        self._bkd.assert_allclose(result, manual_result, rtol=1e-12)

    def test_mean_matches_quadrature_formula(self) -> None:
        """Verify mean matches Smolyak quadrature formula.

        E[f] = sum_k c_k * E_k[f]
        where E_k[f] is the tensor product quadrature integral.
        """
        grid = IsotropicCombinationSparseGrid(
            self._bkd, [self._factory, self._factory], self._growth, level=2
        )

        samples = grid.get_samples()
        # f(x,y) = x^2 + y^2, E[f] = E[x^2] + E[y^2] = 1/3 + 1/3 = 2/3
        values = self._bkd.reshape(
            samples[0, :] ** 2 + samples[1, :] ** 2, (1, -1)
        )
        grid.set_values(values)

        mean = grid.mean()
        expected = 2.0 / 3.0

        self._bkd.assert_allclose(
            mean, self._bkd.asarray([expected]), rtol=1e-12
        )

    def test_variance_formula(self) -> None:
        """Verify variance uses correct Smolyak combination.

        Var[f] = sum_k c_k * Var_k[f]
        """
        grid = IsotropicCombinationSparseGrid(
            self._bkd, [self._factory, self._factory], self._growth, level=2
        )

        samples = grid.get_samples()
        # f(x,y) = x, Var[x] = E[x^2] - E[x]^2 = 1/3 - 0 = 1/3
        values = self._bkd.reshape(samples[0, :], (1, -1))
        grid.set_values(values)

        variance = grid.variance()
        expected = 1.0 / 3.0

        self._bkd.assert_allclose(
            variance, self._bkd.asarray([expected]), rtol=1e-10
        )

    def test_jacobian_analytical(self) -> None:
        """Verify Jacobian against analytical derivative.

        f(x, y) = x^2 + y^2
        J = [2x, 2y]
        """
        grid = IsotropicCombinationSparseGrid(
            self._bkd, [self._factory, self._factory], self._growth, level=3
        )

        samples = grid.get_samples()
        values = self._bkd.reshape(
            samples[0, :] ** 2 + samples[1, :] ** 2, (1, -1)
        )
        grid.set_values(values)

        # Test at a specific point
        test_pt = self._bkd.asarray([[0.3], [0.5]])
        jacobian = grid.jacobian(test_pt)

        # Expected: [2*0.3, 2*0.5] = [0.6, 1.0]
        expected = self._bkd.asarray([[0.6, 1.0]])
        self._bkd.assert_allclose(jacobian, expected, rtol=1e-8)

    def test_hvp_analytical(self) -> None:
        """Verify HVP against analytical Hessian-vector product.

        f(x, y) = x^2 + y^2
        H = [[2, 0], [0, 2]]
        H @ [1, 2] = [2, 4]
        """
        grid = IsotropicCombinationSparseGrid(
            self._bkd, [self._factory, self._factory], self._growth, level=3
        )

        samples = grid.get_samples()
        values = self._bkd.reshape(
            samples[0, :] ** 2 + samples[1, :] ** 2, (1, -1)
        )
        grid.set_values(values)

        test_pt = self._bkd.asarray([[0.3], [0.5]])
        vec = self._bkd.asarray([[1.0], [2.0]])
        hvp = grid.hvp(test_pt, vec)

        # Expected: [[2, 0], [0, 2]] @ [[1], [2]] = [[2], [4]]
        expected = self._bkd.asarray([[2.0], [4.0]])
        self._bkd.assert_allclose(hvp, expected, rtol=1e-8)


# =============================================================================
# NumPy backend tests
# =============================================================================


# =============================================================================
# Runtime protocol validation tests
# =============================================================================


class TestCombinationSparseGridProtocolValidation(Generic[Array], unittest.TestCase):
    """Tests for runtime protocol validation in CombinationSparseGrid."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self._basis = LegendrePolynomial1D(self._bkd)
        self._factory = PrebuiltBasisFactory(self._basis)
        self._growth = LinearGrowthRule(scale=1, shift=1)

    def test_invalid_bkd_raises_typeerror(self) -> None:
        """Verify TypeError raised when bkd doesn't satisfy Backend."""
        with self.assertRaises(TypeError) as ctx:
            CombinationSparseGrid(
                "not a backend",  # type: ignore[arg-type]
                [self._factory, self._factory],
                self._growth
            )
        self.assertIn("Backend", str(ctx.exception))
        self.assertIn("str", str(ctx.exception))

    def test_invalid_basis_factories_raises_typeerror(self) -> None:
        """Verify TypeError raised when basis_factories don't satisfy protocol."""
        with self.assertRaises(TypeError) as ctx:
            CombinationSparseGrid(
                self._bkd,
                ["not a factory", "also not"],  # type: ignore[list-item]
                self._growth
            )
        self.assertIn("BasisFactoryProtocol", str(ctx.exception))

    def test_invalid_growth_rules_raises_typeerror(self) -> None:
        """Verify TypeError raised when growth_rules don't satisfy protocol."""
        with self.assertRaises(TypeError) as ctx:
            CombinationSparseGrid(
                self._bkd,
                [self._factory, self._factory],
                "not a growth rule"  # type: ignore[arg-type]
            )
        self.assertIn("IndexGrowthRuleProtocol", str(ctx.exception))

    def test_invalid_growth_rules_list_raises_typeerror(self) -> None:
        """Verify TypeError raised when growth_rules list contains invalid items."""
        with self.assertRaises(TypeError) as ctx:
            CombinationSparseGrid(
                self._bkd,
                [self._factory, self._factory],
                [self._growth, "not a rule"]  # type: ignore[list-item]
            )
        self.assertIn("IndexGrowthRuleProtocol", str(ctx.exception))


class TestCombinationSparseGridProtocolValidationNumpy(
    TestCombinationSparseGridProtocolValidation[NDArray[Any]]
):
    """NumPy backend tests for protocol validation."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestCombinationSparseGridProtocolValidationTorch(
    TestCombinationSparseGridProtocolValidation[torch.Tensor]
):
    """PyTorch backend tests for protocol validation."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def bkd(self) -> TorchBkd:
        return TorchBkd()


# =============================================================================
# NumPy backend tests
# =============================================================================


class TestCombinationSparseGridBaseNumpy(TestCombinationSparseGridBase[NDArray[Any]]):
    """NumPy backend tests for CombinationSparseGrid base class."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestIncrementalSmolyakUpdateNumpy(TestIncrementalSmolyakUpdate[NDArray[Any]]):
    """NumPy backend tests for incremental Smolyak update."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestCombinationSparseGridLegacyNumpy(TestCombinationSparseGridLegacy[NDArray[Any]]):
    """NumPy backend tests for legacy comparison."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


# =============================================================================
# PyTorch backend tests
# =============================================================================


class TestCombinationSparseGridBaseTorch(TestCombinationSparseGridBase[torch.Tensor]):
    """PyTorch backend tests for CombinationSparseGrid base class."""

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


class TestCombinationSparseGridLegacyTorch(TestCombinationSparseGridLegacy[torch.Tensor]):
    """PyTorch backend tests for legacy comparison."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def bkd(self) -> TorchBkd:
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
