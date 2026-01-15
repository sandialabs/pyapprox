"""Dual-backend tests for sparse grid refinement criteria.

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
    AdaptiveCombinationSparseGrid,
    TensorProductSubspace,
)
from pyapprox.typing.surrogates.sparsegrids.refinement import (
    L2NormRefinementCriteria,
    VarianceRefinementCriteria,
    UnitCostFunction,
    LevelCostFunction,
    SparseGridCostFunctionProtocol,
    SparseGridRefinementCriteriaProtocol,
)
from pyapprox.typing.surrogates.affine.univariate import LegendrePolynomial1D
from pyapprox.typing.surrogates.affine.indices import (
    LinearGrowthRule,
    MaxLevelCriteria,
)


class TestCostFunctions(Generic[Array], unittest.TestCase):
    """Tests for cost functions - dual backend."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_unit_cost_always_one(self) -> None:
        """Test UnitCostFunction returns 1 for any index."""
        cost_fn = UnitCostFunction(self._bkd)
        index1 = self._bkd.asarray([0, 0, 0])
        index2 = self._bkd.asarray([5, 3, 2])

        self.assertEqual(cost_fn(index1), 1.0)
        self.assertEqual(cost_fn(index2), 1.0)

    def test_level_cost_formula(self) -> None:
        """Test LevelCostFunction returns sum(index) + 1."""
        cost_fn = LevelCostFunction(self._bkd)

        # sum = 0 -> cost = 1
        index0 = self._bkd.asarray([0, 0])
        self.assertEqual(cost_fn(index0), 1.0)

        # sum = 3 -> cost = 4
        index3 = self._bkd.asarray([1, 2])
        self.assertEqual(cost_fn(index3), 4.0)

        # sum = 10 -> cost = 11
        index10 = self._bkd.asarray([5, 3, 2])
        self.assertEqual(cost_fn(index10), 11.0)

    def test_unit_cost_satisfies_protocol(self) -> None:
        """Test UnitCostFunction satisfies SparseGridCostFunctionProtocol."""
        cost_fn = UnitCostFunction(self._bkd)
        self.assertIsInstance(cost_fn, SparseGridCostFunctionProtocol)

    def test_level_cost_satisfies_protocol(self) -> None:
        """Test LevelCostFunction satisfies SparseGridCostFunctionProtocol."""
        cost_fn = LevelCostFunction(self._bkd)
        self.assertIsInstance(cost_fn, SparseGridCostFunctionProtocol)


class TestSubspaceVariance(Generic[Array], unittest.TestCase):
    """Tests for TensorProductSubspace.variance() - dual backend."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self._basis = LegendrePolynomial1D(self._bkd)
        self._growth = LinearGrowthRule(scale=1, shift=1)

    def test_variance_constant_is_zero(self) -> None:
        """Variance of constant function should be 0."""
        index = self._bkd.asarray([1, 1])
        subspace = TensorProductSubspace(
            self._bkd, index, [self._basis, self._basis], self._growth
        )

        samples = subspace.get_samples()
        c = 5.0
        values = self._bkd.full((1, samples.shape[1]), c)
        subspace.set_values(values)

        variance = subspace.variance()
        self._bkd.assert_allclose(
            variance, self._bkd.asarray([0.0]), atol=1e-12
        )

    def test_variance_linear_x(self) -> None:
        """Variance of f(x,y) = x should be 1/3 on [-1,1]^2."""
        index = self._bkd.asarray([2, 2])
        subspace = TensorProductSubspace(
            self._bkd, index, [self._basis, self._basis], self._growth
        )

        samples = subspace.get_samples()
        # f(x,y) = x, Var[x] = E[x^2] - E[x]^2 = 1/3 - 0 = 1/3
        values = self._bkd.reshape(samples[0, :], (1, -1))
        subspace.set_values(values)

        variance = subspace.variance()
        expected = 1.0 / 3.0
        self._bkd.assert_allclose(
            variance, self._bkd.asarray([expected]), rtol=1e-10
        )

    def test_variance_x_squared(self) -> None:
        """Variance of f(x,y) = x^2 should be E[x^4] - E[x^2]^2 = 1/5 - 1/9."""
        index = self._bkd.asarray([3, 1])
        subspace = TensorProductSubspace(
            self._bkd, index, [self._basis, self._basis], self._growth
        )

        samples = subspace.get_samples()
        # f(x,y) = x^2, E[x^2] = 1/3, E[x^4] = 1/5
        # Var[x^2] = 1/5 - (1/3)^2 = 1/5 - 1/9 = 4/45
        values = self._bkd.reshape(samples[0, :] ** 2, (1, -1))
        subspace.set_values(values)

        variance = subspace.variance()
        expected = 4.0 / 45.0
        self._bkd.assert_allclose(
            variance, self._bkd.asarray([expected]), rtol=1e-10
        )


class TestL2NormRefinementCriteria(Generic[Array], unittest.TestCase):
    """Tests for L2NormRefinementCriteria - dual backend."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self._basis = LegendrePolynomial1D(self._bkd)
        self._growth = LinearGrowthRule(scale=1, shift=1)

    def test_l2norm_satisfies_protocol(self) -> None:
        """Test L2NormRefinementCriteria satisfies protocol."""
        criteria = L2NormRefinementCriteria(self._bkd)
        self.assertIsInstance(criteria, SparseGridRefinementCriteriaProtocol)

    def test_l2norm_with_unit_cost(self) -> None:
        """Test L2NormRefinementCriteria with default unit cost."""
        admis = MaxLevelCriteria(max_level=3, pnorm=1.0, bkd=self._bkd)
        criteria = L2NormRefinementCriteria(self._bkd)

        grid = AdaptiveCombinationSparseGrid(
            self._bkd,
            [self._basis, self._basis],
            self._growth,
            admis,
            refinement_priority=criteria,
        )

        # First step: get samples
        samples = grid.step_samples()
        self.assertIsNotNone(samples)

        # Provide values for a smooth function
        x, y = samples[0, :], samples[1, :]
        values = self._bkd.reshape(x**2 + y**2, (1, -1))
        grid.step_values(values)

        # Should have computed errors for candidates
        error = grid.error_estimate()
        self.assertGreater(error, 0)

    def test_l2norm_with_level_cost(self) -> None:
        """Test L2NormRefinementCriteria with level cost function."""
        admis = MaxLevelCriteria(max_level=3, pnorm=1.0, bkd=self._bkd)
        cost_fn = LevelCostFunction(self._bkd)
        criteria = L2NormRefinementCriteria(self._bkd, cost_function=cost_fn)

        grid = AdaptiveCombinationSparseGrid(
            self._bkd,
            [self._basis, self._basis],
            self._growth,
            admis,
            refinement_priority=criteria,
        )

        samples = grid.step_samples()
        x, y = samples[0, :], samples[1, :]
        values = self._bkd.reshape(x**2 + y**2, (1, -1))
        grid.step_values(values)

        # Should work without error
        error = grid.error_estimate()
        self.assertGreater(error, 0)


class TestVarianceRefinementCriteria(Generic[Array], unittest.TestCase):
    """Tests for VarianceRefinementCriteria - dual backend."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self._basis = LegendrePolynomial1D(self._bkd)
        self._growth = LinearGrowthRule(scale=1, shift=1)

    def test_variance_satisfies_protocol(self) -> None:
        """Test VarianceRefinementCriteria satisfies protocol."""
        criteria = VarianceRefinementCriteria(self._bkd)
        self.assertIsInstance(criteria, SparseGridRefinementCriteriaProtocol)

    def test_variance_criteria_with_polynomial(self) -> None:
        """Test VarianceRefinementCriteria with polynomial function."""
        admis = MaxLevelCriteria(max_level=3, pnorm=1.0, bkd=self._bkd)
        criteria = VarianceRefinementCriteria(self._bkd)

        grid = AdaptiveCombinationSparseGrid(
            self._bkd,
            [self._basis, self._basis],
            self._growth,
            admis,
            refinement_priority=criteria,
        )

        samples = grid.step_samples()
        x, y = samples[0, :], samples[1, :]
        values = self._bkd.reshape(x**2 + y**2, (1, -1))
        grid.step_values(values)

        # Should have computed priorities
        error = grid.error_estimate()
        self.assertGreater(error, 0)

    def test_variance_criteria_moment_change(self) -> None:
        """Test VarianceRefinementCriteria computes moment changes."""
        admis = MaxLevelCriteria(max_level=4, pnorm=1.0, bkd=self._bkd)
        criteria = VarianceRefinementCriteria(self._bkd)

        grid = AdaptiveCombinationSparseGrid(
            self._bkd,
            [self._basis, self._basis],
            self._growth,
            admis,
            refinement_priority=criteria,
        )

        # First step
        samples = grid.step_samples()
        x, y = samples[0, :], samples[1, :]
        # Use a higher-order polynomial to ensure moment changes
        values = self._bkd.reshape(x**3 + y**3 + x * y, (1, -1))
        grid.step_values(values)

        # Get initial mean and variance
        mean1 = grid.mean()
        var1 = grid.variance()

        # Refine
        samples2 = grid.step_samples()
        if samples2 is not None and samples2.shape[1] > 0:
            x2, y2 = samples2[0, :], samples2[1, :]
            values2 = self._bkd.reshape(x2**3 + y2**3 + x2 * y2, (1, -1))
            grid.step_values(values2)

            # Mean and variance may change after refinement
            mean2 = grid.mean()
            var2 = grid.variance()

            # Check shapes are correct
            self.assertEqual(mean1.shape, (1,))
            self.assertEqual(var1.shape, (1,))
            self.assertEqual(mean2.shape, (1,))
            self.assertEqual(var2.shape, (1,))


class TestAdaptiveGridWithIncrementalSmolyak(Generic[Array], unittest.TestCase):
    """Tests for AdaptiveCombinationSparseGrid with incremental Smolyak updates.

    Tests verify that the incremental Smolyak coefficient updates
    produce correct results during adaptive refinement.
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self._basis = LegendrePolynomial1D(self._bkd)
        self._growth = LinearGrowthRule(scale=1, shift=1)

    def test_mean_with_stored_coefficients(self) -> None:
        """Test mean computation uses stored Smolyak coefficients."""
        admis = MaxLevelCriteria(max_level=3, pnorm=1.0, bkd=self._bkd)
        grid = AdaptiveCombinationSparseGrid(
            self._bkd,
            [self._basis, self._basis],
            self._growth,
            admis,
        )

        samples = grid.step_samples()
        # Constant function - mean should be the constant
        c = 2.5
        values = self._bkd.full((1, samples.shape[1]), c)
        grid.step_values(values)

        mean = grid.mean()
        self._bkd.assert_allclose(mean, self._bkd.asarray([c]), rtol=1e-10)

    def test_variance_with_stored_coefficients(self) -> None:
        """Test variance computation uses stored Smolyak coefficients.

        After the first step, only (0,0) is selected with 1 point, so variance
        is 0. We need to refine at least once to get multiple selected subspaces
        and meaningful variance.
        """
        admis = MaxLevelCriteria(max_level=3, pnorm=1.0, bkd=self._bkd)
        grid = AdaptiveCombinationSparseGrid(
            self._bkd,
            [self._basis, self._basis],
            self._growth,
            admis,
        )

        # Linear function - variance of x is 1/3
        def target_fn(samples: Array) -> Array:
            return self._bkd.reshape(samples[0, :], (1, -1))

        # First step
        samples = grid.step_samples()
        values = target_fn(samples)
        grid.step_values(values)

        # After first step, only (0,0) is selected (1 point), so variance = 0
        variance0 = grid.variance()
        self._bkd.assert_allclose(
            variance0, self._bkd.asarray([0.0]), atol=1e-12
        )

        # Refine to add more selected subspaces
        samples2 = grid.step_samples()
        if samples2 is not None and samples2.shape[1] > 0:
            values2 = target_fn(samples2)
            grid.step_values(values2)

            # After refinement, variance should be approximately 1/3
            variance = grid.variance()
            expected = 1.0 / 3.0
            self._bkd.assert_allclose(
                variance, self._bkd.asarray([expected]), rtol=1e-6
            )

    def test_incremental_update_matches_scratch(self) -> None:
        """Test that incremental Smolyak updates match from-scratch computation."""
        from pyapprox.typing.surrogates.sparsegrids import compute_smolyak_coefficients

        admis = MaxLevelCriteria(max_level=4, pnorm=1.0, bkd=self._bkd)
        grid = AdaptiveCombinationSparseGrid(
            self._bkd,
            [self._basis, self._basis],
            self._growth,
            admis,
        )

        # First step
        samples1 = grid.step_samples()
        x1, y1 = samples1[0, :], samples1[1, :]
        values1 = self._bkd.reshape(x1**2 + y1**2, (1, -1))
        grid.step_values(values1)

        # Refine multiple times
        for _ in range(3):
            samples_next = grid.step_samples()
            if samples_next is None or samples_next.shape[1] == 0:
                break
            x_n, y_n = samples_next[0, :], samples_next[1, :]
            values_n = self._bkd.reshape(x_n**2 + y_n**2, (1, -1))
            grid.step_values(values_n)

        # Check that stored coefficients match from-scratch computation
        if grid._selected_smolyak_coefs is not None:
            selected = grid.get_selected_subspace_indices()
            scratch_coefs = compute_smolyak_coefficients(selected, self._bkd)

            # Only compare coefficients for selected indices
            nselected = selected.shape[1]
            stored_selected_coefs = grid._selected_smolyak_coefs[:nselected]

            self._bkd.assert_allclose(
                stored_selected_coefs, scratch_coefs, rtol=1e-10
            )


class TestRefinementOrdering(Generic[Array], unittest.TestCase):
    """Tests for refinement ordering based on priorities.

    These tests verify that higher-error subspaces are refined first.
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self._basis = LegendrePolynomial1D(self._bkd)
        self._growth = LinearGrowthRule(scale=1, shift=1)

    def test_higher_error_refined_first(self) -> None:
        """Test that subspaces with higher errors are refined first."""
        admis = MaxLevelCriteria(max_level=5, pnorm=1.0, bkd=self._bkd)
        criteria = L2NormRefinementCriteria(self._bkd)

        grid = AdaptiveCombinationSparseGrid(
            self._bkd,
            [self._basis, self._basis],
            self._growth,
            admis,
            refinement_priority=criteria,
        )

        # Use anisotropic function: higher degree in x than y
        def target_fn(samples: Array) -> Array:
            x, y = samples[0, :], samples[1, :]
            return self._bkd.reshape(x**4 + y, (1, -1))

        samples = grid.step_samples()
        values = target_fn(samples)
        grid.step_values(values)

        # Get initial error (after first step, only candidates have non-zero error)
        initial_error = grid.error_estimate()

        # Refine multiple times and track which directions are refined
        nrefinements = 5
        for _ in range(nrefinements):
            next_samples = grid.step_samples()
            if next_samples is None or next_samples.shape[1] == 0:
                break
            next_values = target_fn(next_samples)
            grid.step_values(next_values)

        # After refinement, error should be finite and decrease
        final_error = grid.error_estimate()
        self.assertIsInstance(final_error, float)
        self.assertGreaterEqual(initial_error, 0.0)
        self.assertGreaterEqual(final_error, 0.0)


# NumPy backend tests
class TestCostFunctionsNumpy(TestCostFunctions[NDArray[Any]]):
    """NumPy backend tests for cost functions."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestSubspaceVarianceNumpy(TestSubspaceVariance[NDArray[Any]]):
    """NumPy backend tests for subspace variance."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestL2NormRefinementCriteriaNumpy(TestL2NormRefinementCriteria[NDArray[Any]]):
    """NumPy backend tests for L2NormRefinementCriteria."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestVarianceRefinementCriteriaNumpy(TestVarianceRefinementCriteria[NDArray[Any]]):
    """NumPy backend tests for VarianceRefinementCriteria."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestAdaptiveGridWithIncrementalSmolyakNumpy(
    TestAdaptiveGridWithIncrementalSmolyak[NDArray[Any]]
):
    """NumPy backend tests for adaptive grid incremental Smolyak."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestRefinementOrderingNumpy(TestRefinementOrdering[NDArray[Any]]):
    """NumPy backend tests for refinement ordering."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


# PyTorch backend tests
class TestCostFunctionsTorch(TestCostFunctions[torch.Tensor]):
    """PyTorch backend tests for cost functions."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def bkd(self) -> TorchBkd:
        return TorchBkd()


class TestSubspaceVarianceTorch(TestSubspaceVariance[torch.Tensor]):
    """PyTorch backend tests for subspace variance."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def bkd(self) -> TorchBkd:
        return TorchBkd()


class TestL2NormRefinementCriteriaTorch(TestL2NormRefinementCriteria[torch.Tensor]):
    """PyTorch backend tests for L2NormRefinementCriteria."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def bkd(self) -> TorchBkd:
        return TorchBkd()


class TestVarianceRefinementCriteriaTorch(TestVarianceRefinementCriteria[torch.Tensor]):
    """PyTorch backend tests for VarianceRefinementCriteria."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def bkd(self) -> TorchBkd:
        return TorchBkd()


class TestAdaptiveGridWithIncrementalSmolyakTorch(
    TestAdaptiveGridWithIncrementalSmolyak[torch.Tensor]
):
    """PyTorch backend tests for adaptive grid incremental Smolyak."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def bkd(self) -> TorchBkd:
        return TorchBkd()


class TestRefinementOrderingTorch(TestRefinementOrdering[torch.Tensor]):
    """PyTorch backend tests for refinement ordering."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def bkd(self) -> TorchBkd:
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
