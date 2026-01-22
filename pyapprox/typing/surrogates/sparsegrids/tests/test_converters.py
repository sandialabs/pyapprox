"""Dual-backend tests for sparse grid to PCE converter.

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
    IsotropicCombinationSparseGrid,
    SparseGridToPCEConverter,
    TensorProductSubspaceToPCEConverter,
    TensorProductSubspace,
)
from pyapprox.typing.surrogates.sparsegrids.basis_factory import GaussLagrangeFactory
from pyapprox.typing.surrogates.affine.univariate import create_bases_1d
from pyapprox.typing.probability import UniformMarginal
from pyapprox.typing.surrogates.affine.indices import LinearGrowthRule


class TestSparseGridToPCEConverter(Generic[Array], unittest.TestCase):
    """Tests for SparseGridToPCEConverter - dual backend."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_simple_polynomial(self) -> None:
        """Test conversion for a simple polynomial."""
        nvars = 2
        level = 3

        marginals = [UniformMarginal(-1.0, 1.0, self._bkd) for _ in range(nvars)]
        factories = [GaussLagrangeFactory(m, self._bkd) for m in marginals]
        growth = LinearGrowthRule(scale=2, shift=1)

        grid = IsotropicCombinationSparseGrid(
            self._bkd, factories, growth, level
        )

        # f(x, y) = x^2 + 2*x*y + y
        samples = grid.get_samples()
        x, y = samples[0, :], samples[1, :]
        values = self._bkd.reshape(x ** 2 + 2 * x * y + y, (1, -1))
        grid.set_values(values)

        # Convert to PCE
        pce_bases_1d = create_bases_1d(marginals, self._bkd)
        converter = SparseGridToPCEConverter(self._bkd, pce_bases_1d)
        pce = converter.convert(grid)

        # Test evaluation matches
        # Both return (nqoi, nsamples) per CLAUDE.md conventions
        test_pts = self._bkd.asarray([[-0.5, 0.0, 0.5], [0.3, 0.0, -0.3]])
        grid_vals = grid(test_pts)
        pce_vals = pce(test_pts)

        self._bkd.assert_allclose(grid_vals, pce_vals, rtol=1e-10, atol=1e-14)

    def test_pce_mean_variance(self) -> None:
        """Test PCE statistics are correct."""
        nvars = 2
        level = 4

        marginals = [UniformMarginal(-1.0, 1.0, self._bkd) for _ in range(nvars)]
        factories = [GaussLagrangeFactory(m, self._bkd) for m in marginals]
        growth = LinearGrowthRule(scale=2, shift=1)

        grid = IsotropicCombinationSparseGrid(
            self._bkd, factories, growth, level
        )

        # f(x, y) = x^2 + 2*x*y + y
        # E[f] = E[x^2] = 1/3
        # Var[f] = 13/15
        samples = grid.get_samples()
        x, y = samples[0, :], samples[1, :]
        values = self._bkd.reshape(x ** 2 + 2 * x * y + y, (1, -1))
        grid.set_values(values)

        # Convert to PCE
        pce_bases_1d = create_bases_1d(marginals, self._bkd)
        converter = SparseGridToPCEConverter(self._bkd, pce_bases_1d)
        pce = converter.convert(grid)

        # Check mean and variance
        pce_mean = pce.mean()
        pce_var = pce.variance()

        exact_mean = 1.0 / 3.0
        exact_var = 13.0 / 15.0

        self.assertAlmostEqual(float(pce_mean[0]), exact_mean, places=10)
        self.assertAlmostEqual(float(pce_var[0]), exact_var, places=10)

    def test_sobol_indices(self) -> None:
        """Test PCE Sobol indices are correct."""
        nvars = 2
        level = 4

        marginals = [UniformMarginal(-1.0, 1.0, self._bkd) for _ in range(nvars)]
        factories = [GaussLagrangeFactory(m, self._bkd) for m in marginals]
        growth = LinearGrowthRule(scale=2, shift=1)

        grid = IsotropicCombinationSparseGrid(
            self._bkd, factories, growth, level
        )

        # f(x, y) = x^2 + 2*x*y + y
        samples = grid.get_samples()
        x, y = samples[0, :], samples[1, :]
        values = self._bkd.reshape(x ** 2 + 2 * x * y + y, (1, -1))
        grid.set_values(values)

        # Convert to PCE
        pce_bases_1d = create_bases_1d(marginals, self._bkd)
        converter = SparseGridToPCEConverter(self._bkd, pce_bases_1d)
        pce = converter.convert(grid)

        # Check Sobol indices
        total_sobol = pce.total_sobol_indices()
        main_sobol = pce.main_effect_sobol_indices()

        # Exact values
        exact_total_x = 8.0 / 13.0
        exact_total_y = 35.0 / 39.0
        exact_main_x = 4.0 / 39.0
        exact_main_y = 5.0 / 13.0

        self.assertAlmostEqual(
            float(main_sobol[0, 0]), exact_main_x, places=6
        )
        self.assertAlmostEqual(
            float(main_sobol[1, 0]), exact_main_y, places=6
        )
        self.assertAlmostEqual(
            float(total_sobol[0, 0]), exact_total_x, places=6
        )
        self.assertAlmostEqual(
            float(total_sobol[1, 0]), exact_total_y, places=6
        )

    def test_3d_conversion(self) -> None:
        """Test conversion for 3D sparse grid."""
        nvars = 3
        level = 2

        marginals = [UniformMarginal(-1.0, 1.0, self._bkd) for _ in range(nvars)]
        factories = [GaussLagrangeFactory(m, self._bkd) for m in marginals]
        growth = LinearGrowthRule(scale=2, shift=1)

        grid = IsotropicCombinationSparseGrid(
            self._bkd, factories, growth, level
        )

        # f(x, y, z) = x + y + z
        samples = grid.get_samples()
        values = self._bkd.reshape(
            samples[0, :] + samples[1, :] + samples[2, :], (1, -1)
        )
        grid.set_values(values)

        # Convert to PCE
        pce_bases_1d = create_bases_1d(marginals, self._bkd)
        converter = SparseGridToPCEConverter(self._bkd, pce_bases_1d)
        pce = converter.convert(grid)

        # Test evaluation matches
        # Both return (nqoi, nsamples) per CLAUDE.md conventions
        test_pts = self._bkd.asarray([
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6]
        ])
        grid_vals = grid(test_pts)
        pce_vals = pce(test_pts)

        self._bkd.assert_allclose(grid_vals, pce_vals, rtol=1e-10, atol=1e-14)

        # Mean should be 0 for linear function
        pce_mean = pce.mean()
        self.assertAlmostEqual(float(pce_mean[0]), 0.0, places=10)

    def test_non_canonical_domain(self) -> None:
        """Test conversion with non-canonical domain [0, 1].

        Verifies converter works correctly when user domain differs from
        canonical domain. The converter relies on TransformedBasis1D from
        create_bases_1d() to handle domain transforms automatically.
        """
        nvars = 2
        level = 3

        # Use [0, 1] domain instead of canonical [-1, 1]
        marginals = [UniformMarginal(0.0, 1.0, self._bkd) for _ in range(nvars)]
        factories = [GaussLagrangeFactory(m, self._bkd) for m in marginals]
        growth = LinearGrowthRule(scale=2, shift=1)

        grid = IsotropicCombinationSparseGrid(
            self._bkd, factories, growth, level
        )

        # Verify samples are in [0, 1] domain
        samples = grid.get_samples()
        self.assertTrue(self._bkd.all_bool(samples >= 0.0))
        self.assertTrue(self._bkd.all_bool(samples <= 1.0))

        # f(x, y) = x^2 + 2*x*y + y
        x, y = samples[0, :], samples[1, :]
        values = self._bkd.reshape(x ** 2 + 2 * x * y + y, (1, -1))
        grid.set_values(values)

        # Convert to PCE
        pce_bases_1d = create_bases_1d(marginals, self._bkd)
        converter = SparseGridToPCEConverter(self._bkd, pce_bases_1d)
        pce = converter.convert(grid)

        # Test evaluation matches at points in [0, 1] domain
        test_pts = self._bkd.asarray([[0.25, 0.5, 0.75], [0.3, 0.5, 0.7]])
        grid_vals = grid(test_pts)
        pce_vals = pce(test_pts)

        self._bkd.assert_allclose(grid_vals, pce_vals, rtol=1e-10, atol=1e-14)

    def test_multi_qoi_conversion(self) -> None:
        """Test conversion with multiple quantities of interest."""
        nvars = 2
        level = 3

        marginals = [UniformMarginal(-1.0, 1.0, self._bkd) for _ in range(nvars)]
        factories = [GaussLagrangeFactory(m, self._bkd) for m in marginals]
        growth = LinearGrowthRule(scale=2, shift=1)

        grid = IsotropicCombinationSparseGrid(
            self._bkd, factories, growth, level
        )

        # Two QoIs: f1 = x, f2 = y - shape (nqoi, nsamples) = (2, nsamples)
        samples = grid.get_samples()
        values = self._bkd.stack([samples[0, :], samples[1, :]], axis=0)
        grid.set_values(values)

        # Convert to PCE
        pce_bases_1d = create_bases_1d(marginals, self._bkd)
        converter = SparseGridToPCEConverter(self._bkd, pce_bases_1d)
        pce = converter.convert(grid)

        # Test evaluation matches
        # Both return (nqoi, nsamples) per CLAUDE.md conventions
        test_pts = self._bkd.asarray([[0.3, -0.5], [0.2, 0.4]])
        grid_vals = grid(test_pts)
        pce_vals = pce(test_pts)

        self.assertEqual(pce_vals.shape[0], 2)  # nqoi=2 is first dimension
        self._bkd.assert_allclose(grid_vals, pce_vals, rtol=1e-10, atol=1e-14)


class TestTensorProductSubspaceToPCEConverter(Generic[Array], unittest.TestCase):
    """Tests for TensorProductSubspaceToPCEConverter - dual backend."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_subspace_conversion(self) -> None:
        """Test conversion of single tensor product subspace."""
        nvars = 2
        marginals = [UniformMarginal(-1.0, 1.0, self._bkd) for _ in range(nvars)]
        factories = [GaussLagrangeFactory(m, self._bkd) for m in marginals]
        growth = LinearGrowthRule(scale=1, shift=1)

        # Create a subspace
        index = self._bkd.asarray([2, 2])
        subspace = TensorProductSubspace(
            self._bkd, index, factories, growth
        )

        # f(x, y) = x^2 + y
        samples = subspace.get_samples()
        values = self._bkd.reshape(samples[0, :] ** 2 + samples[1, :], (1, -1))
        subspace.set_values(values)

        # Convert to PCE coefficients
        pce_bases_1d = create_bases_1d(marginals, self._bkd)
        converter = TensorProductSubspaceToPCEConverter(self._bkd, pce_bases_1d)
        indices, coefficients = converter.convert_subspace(subspace)

        # Verify shapes - coefficients is (nqoi, ncoefs)
        self.assertEqual(indices.shape[0], nvars)
        self.assertEqual(coefficients.shape[0], 1)  # nqoi is first dimension
        self.assertEqual(indices.shape[1], coefficients.shape[1])


# NumPy backend tests
class TestSparseGridToPCEConverterNumpy(
    TestSparseGridToPCEConverter[NDArray[Any]]
):
    """NumPy backend tests for SparseGridToPCEConverter."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestTensorProductSubspaceToPCEConverterNumpy(
    TestTensorProductSubspaceToPCEConverter[NDArray[Any]]
):
    """NumPy backend tests for TensorProductSubspaceToPCEConverter."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


# PyTorch backend tests
class TestSparseGridToPCEConverterTorch(
    TestSparseGridToPCEConverter[torch.Tensor]
):
    """PyTorch backend tests for SparseGridToPCEConverter."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def bkd(self) -> TorchBkd:
        return TorchBkd()


class TestTensorProductSubspaceToPCEConverterTorch(
    TestTensorProductSubspaceToPCEConverter[torch.Tensor]
):
    """PyTorch backend tests for TensorProductSubspaceToPCEConverter."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def bkd(self) -> TorchBkd:
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
