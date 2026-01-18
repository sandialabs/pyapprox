"""Tests for IsotropicCombinationSparseGrid.

Tests run on both NumPy and PyTorch backends using the base class pattern.

This file contains tests specific to IsotropicCombinationSparseGrid:
- TestIsotropicSparseGrid: Core functionality tests
- TestIsotropicWithGenerator: Index generator integration tests
- TestIsotropicQuadrature: Quadrature/integration tests
- TestIsotropicLegacy: Legacy comparison tests
"""

import unittest
from typing import Any, Generic

from scipy import stats
import torch
from numpy.typing import NDArray

from pyapprox.typing.surrogates.affine.expansions import create_pce
from pyapprox.typing.surrogates.affine.indices import (
    HyperbolicIndexGenerator,
    LinearGrowthRule,
)
from pyapprox.typing.surrogates.affine.univariate import (
    HermitePolynomial1D,
    LegendrePolynomial1D,
)
from pyapprox.typing.probability import (
    ScipyContinuousMarginal,
    IndependentJoint,
)
from pyapprox.typing.surrogates.sparsegrids import (
    IsotropicCombinationSparseGrid,
    is_downward_closed,
    PrebuiltBasisFactory,
)
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests


# =============================================================================
# Core functionality tests
# =============================================================================


class TestIsotropicSparseGrid(Generic[Array], unittest.TestCase):
    """Tests for IsotropicCombinationSparseGrid core functionality."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_level_0(self) -> None:
        """Test level 0 sparse grid (single point)."""
        basis = LegendrePolynomial1D(self._bkd)
        factory = PrebuiltBasisFactory(basis)
        growth = LinearGrowthRule(scale=1, shift=1)  # n(l) = l + 1

        grid = IsotropicCombinationSparseGrid(
            self._bkd, [factory, factory], growth, level=0
        )

        # Level 0: only (0,0) subspace, 1 sample
        self.assertEqual(grid.nsubspaces(), 1)
        self.assertEqual(grid.nsamples(), 1)

    def test_level_2_subspaces(self):
        """Test level 2 sparse grid has correct subspaces."""
        basis = LegendrePolynomial1D(self._bkd)
        factory = PrebuiltBasisFactory(basis)
        growth = LinearGrowthRule(scale=1, shift=1)  # n(l) = l + 1

        grid = IsotropicCombinationSparseGrid(
            self._bkd, [factory, factory], growth, level=2
        )

        # 2D level 2: indices with |k|_1 <= 2
        expected_indices = {
            (0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)
        }

        # Get actual indices from grid
        grid_indices = grid.get_subspace_indices()
        actual_indices = set()
        for j in range(grid_indices.shape[1]):
            idx = tuple(int(grid_indices[i, j]) for i in range(2))
            actual_indices.add(idx)

        self.assertEqual(grid.nsubspaces(), 6)
        self.assertEqual(actual_indices, expected_indices)

    def test_interpolation(self):
        """Test sparse grid interpolation using a PCE with matching index set.

        Creates a PCE with hyperbolic index set that the sparse grid can
        exactly interpolate, then verifies interpolation at random samples
        drawn from the underlying uniform distribution.

        TODO: repeat this for different nvars and level, e.g. (2, 3) (2, 4) (3, 2) and joint variables. reuse code do not create seperate functions with redundant code for each combination. Note that pce.nterms will be much less than what sparse grid is capable of interepolating. level of a sparse grid is used with growth rule which grows much faster than level of hyperbolic index set. Repeat this test with integration but for different random variabls and gauss quadrature rules and leja quadrature rules. reuse code to avoid test file bloat, e.g. for test_gauss_quadrature_integration iterate over different nvars, levels, and joint variables, e.g. all uniform, all gaussian, mix of gaussian and uniform. Do same for test_leja_quadrature_integration. create_pce should be create_hyperbolic_pce to better reflect its contents, should also construct helper function create_tensor_product_pce. also create helper function create_bases_1d that takes in independentrandom variables and returns the correct list of 1d bases. use the legacy code for this as a starting point but update to use new typing infrastrcture and coding convenctions. make sure that backend protocol has all the functions that appear in both torch and nupy backends. This way we can always check backend has what is needed
        """
        nvars = 2
        level = 3
        basis = LegendrePolynomial1D(self._bkd)
        factory = PrebuiltBasisFactory(basis)
        growth = LinearGrowthRule(scale=1, shift=1)  # n(l) = l + 1

        grid = IsotropicCombinationSparseGrid(
            self._bkd, [factory, factory], growth, level=level
        )

        # Create a PCE with hyperbolic index set matching the sparse grid
        # The sparse grid with level L can exactly interpolate polynomials
        # of total degree up to L
        bases_1d = [LegendrePolynomial1D(self._bkd) for _ in range(nvars)]
        pce = create_pce(bases_1d, max_level=level, bkd=self._bkd, nqoi=1)

        # Set random coefficients on the PCE
        nterms = pce.nterms()
        #TODO: randomly generated coefficients from multivariate standard normal
        coefficients = self._bkd.asarray(
            [[0.5], [-0.3], [0.2], [0.1], [-0.15], [0.25]]
            + [[0.0]] * (nterms - 6)  # Zero out higher terms if any
        )
        pce.set_coefficients(coefficients[:nterms, :])

        # Evaluate PCE at sparse grid samples and set values
        samples = grid.get_samples()
        values = pce(samples)
        grid.set_values(values)

        # Generate random test points from uniform distribution on [-1, 1]^2
        uniform_marginal = ScipyContinuousMarginal(
            stats.uniform(-1, 2), self._bkd  # uniform(-1, 2) = U[-1, 1]
        )
        joint = IndependentJoint(
            [uniform_marginal for _ in range(nvars)], self._bkd
        )
        test_pts = joint.rvs(20)

        # Compare sparse grid interpolation to exact PCE evaluation
        result = grid(test_pts)
        expected = pce(test_pts)

        self._bkd.assert_allclose(result, expected, rtol=1e-10)

    def test_smolyak_coefficients_sum(self):
        """Test Smolyak coefficients sum to 1."""
        basis = LegendrePolynomial1D(self._bkd)
        factory = PrebuiltBasisFactory(basis)
        growth = LinearGrowthRule(scale=1, shift=1)  # n(l) = l + 1

        for level in [1, 2, 3]:
            grid = IsotropicCombinationSparseGrid(
                self._bkd, [factory, factory], growth, level=level
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

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


# =============================================================================
# Quadrature and integration tests
# =============================================================================


class TestIsotropicQuadrature(Generic[Array], unittest.TestCase):
    """Tests for IsotropicCombinationSparseGrid quadrature/integration."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_mean_monomial_exact(self) -> None:
        """Test that mean of x^2 + y^2 is exactly 2/3 on [-1,1]^2.

        For uniform distribution on [-1,1], E[x^2] = 1/3.
        So E[x^2 + y^2] = 1/3 + 1/3 = 2/3.
        """
        basis = LegendrePolynomial1D(self._bkd)
        factory = PrebuiltBasisFactory(basis)
        growth = LinearGrowthRule(scale=1, shift=1)

        grid = IsotropicCombinationSparseGrid(
            self._bkd, [factory, factory], growth, level=2
        )

        samples = grid.get_samples()
        x, y = samples[0, :], samples[1, :]
        values = self._bkd.reshape(x**2 + y**2, (1, -1))
        grid.set_values(values)

        mean = grid.mean()
        expected = 2.0 / 3.0
        self._bkd.assert_allclose(mean, self._bkd.asarray([expected]), rtol=1e-12)

    def test_mean_mixed_monomial(self) -> None:
        """Test mean of x^2*y^2 is exactly 1/9 on [-1,1]^2.

        E[x^2*y^2] = E[x^2] * E[y^2] = (1/3) * (1/3) = 1/9.
        """
        basis = LegendrePolynomial1D(self._bkd)
        factory = PrebuiltBasisFactory(basis)
        growth = LinearGrowthRule(scale=1, shift=1)

        grid = IsotropicCombinationSparseGrid(
            self._bkd, [factory, factory], growth, level=3
        )

        samples = grid.get_samples()
        x, y = samples[0, :], samples[1, :]
        values = self._bkd.reshape(x**2 * y**2, (1, -1))
        grid.set_values(values)

        mean = grid.mean()
        expected = 1.0 / 9.0
        self._bkd.assert_allclose(mean, self._bkd.asarray([expected]), rtol=1e-12)

    def test_integration_symmetry_odd_function(self) -> None:
        """Test that odd functions integrate to zero.

        For symmetric distribution, E[x + y] = E[x^3 + y^3] = E[xy] = 0.
        """
        basis = LegendrePolynomial1D(self._bkd)
        factory = PrebuiltBasisFactory(basis)
        growth = LinearGrowthRule(scale=1, shift=1)

        grid = IsotropicCombinationSparseGrid(
            self._bkd, [factory, factory], growth, level=3
        )

        samples = grid.get_samples()
        x, y = samples[0, :], samples[1, :]

        # Test E[x + y] = 0
        values_sum = self._bkd.reshape(x + y, (1, -1))
        grid.set_values(values_sum)
        mean_sum = grid.mean()
        self._bkd.assert_allclose(mean_sum, self._bkd.asarray([0.0]), atol=1e-14)

        # Test E[x^3 + y^3] = 0
        values_cubes = self._bkd.reshape(x**3 + y**3, (1, -1))
        grid.set_values(values_cubes)
        mean_cubes = grid.mean()
        self._bkd.assert_allclose(mean_cubes, self._bkd.asarray([0.0]), atol=1e-14)

        # Test E[x*y] = 0 (product of independent odd functions)
        values_xy = self._bkd.reshape(x * y, (1, -1))
        grid.set_values(values_xy)
        mean_xy = grid.mean()
        self._bkd.assert_allclose(mean_xy, self._bkd.asarray([0.0]), atol=1e-14)

    def test_variance_sum_function(self) -> None:
        """Test variance of f(x,y) = x + y.

        Var[x + y] = Var[x] + Var[y] = 1/3 + 1/3 = 2/3.
        """
        basis = LegendrePolynomial1D(self._bkd)
        factory = PrebuiltBasisFactory(basis)
        growth = LinearGrowthRule(scale=1, shift=1)

        grid = IsotropicCombinationSparseGrid(
            self._bkd, [factory, factory], growth, level=2
        )

        samples = grid.get_samples()
        x, y = samples[0, :], samples[1, :]
        values = self._bkd.reshape(x + y, (1, -1))
        grid.set_values(values)

        variance = grid.variance()
        expected = 2.0 / 3.0
        self._bkd.assert_allclose(
            variance, self._bkd.asarray([expected]), rtol=1e-10
        )

    def test_variance_product_function(self) -> None:
        """Test variance of f(x,y) = x*y.

        E[xy] = 0, E[(xy)^2] = E[x^2]*E[y^2] = 1/9
        Var[xy] = E[(xy)^2] - E[xy]^2 = 1/9 - 0 = 1/9.
        """
        basis = LegendrePolynomial1D(self._bkd)
        factory = PrebuiltBasisFactory(basis)
        growth = LinearGrowthRule(scale=1, shift=1)

        grid = IsotropicCombinationSparseGrid(
            self._bkd, [factory, factory], growth, level=2
        )

        samples = grid.get_samples()
        x, y = samples[0, :], samples[1, :]
        values = self._bkd.reshape(x * y, (1, -1))
        grid.set_values(values)

        variance = grid.variance()
        expected = 1.0 / 9.0
        self._bkd.assert_allclose(
            variance, self._bkd.asarray([expected]), rtol=1e-10
        )


class TestIsotropicQuadratureNumpy(TestIsotropicQuadrature[NDArray[Any]]):
    """NumPy backend tests for quadrature."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestIsotropicQuadratureTorch(TestIsotropicQuadrature[torch.Tensor]):
    """PyTorch backend tests for quadrature."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


# =============================================================================
# Index generator integration tests
# =============================================================================


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
        factory = PrebuiltBasisFactory(basis)
        growth = LinearGrowthRule(scale=1, shift=1)

        grid = IsotropicCombinationSparseGrid(
            self._bkd, [factory, factory], growth, level=2
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
        factory = PrebuiltBasisFactory(basis)
        growth = LinearGrowthRule(scale=1, shift=1)

        grid = IsotropicCombinationSparseGrid(
            self._bkd, [factory, factory], growth, level=2
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
        factory = PrebuiltBasisFactory(basis)
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
                self._bkd, [factory, factory], growth, level=level
            )
            self.assertEqual(
                grid.nsubspaces(), expected,
                f"Level {level}: expected {expected} subspaces, got {grid.nsubspaces()}"
            )

    def test_2d_level_3_index_set(self):
        """Test exact index set for 2D level 3 sparse grid."""
        basis = LegendrePolynomial1D(self._bkd)
        factory = PrebuiltBasisFactory(basis)
        growth = LinearGrowthRule(scale=1, shift=1)

        grid = IsotropicCombinationSparseGrid(
            self._bkd, [factory, factory], growth, level=3
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
        factory = PrebuiltBasisFactory(basis)
        growth = LinearGrowthRule(scale=1, shift=1)

        grid = IsotropicCombinationSparseGrid(
            self._bkd, [factory, factory, factory], growth, level=2
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
        factory = PrebuiltBasisFactory(basis)
        growth = LinearGrowthRule(scale=1, shift=1)

        grid = IsotropicCombinationSparseGrid(
            self._bkd, [factory, factory], growth, level=3
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


# =============================================================================
# Legacy comparison tests
# =============================================================================


class TestIsotropicLegacy(Generic[Array], unittest.TestCase):
    """Legacy comparison tests for IsotropicCombinationSparseGrid.

    These tests verify that the typing implementation produces the same
    results as the legacy implementation when using the same quadrature
    rules (Clenshaw-Curtis).
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def _build_legacy_sparse_grid(self, nvars: int, level: int, nqoi: int):
        """Build a legacy sparse grid with Clenshaw-Curtis quadrature."""
        from pyapprox.surrogates.univariate.base import (
            ClenshawCurtisQuadratureRule as LegacyCCQuadratureRule,
        )
        from pyapprox.surrogates.univariate.lagrange import UnivariateLagrangeBasis
        from pyapprox.surrogates.affine.basis import TensorProductInterpolatingBasis
        from pyapprox.surrogates.affine.multiindex import DoublePlusOneIndexGrowthRule
        from pyapprox.surrogates.sparsegrids.combination import (
            IsotropicCombinationSparseGrid as LegacyIsotropicSG,
        )

        quad_rule = LegacyCCQuadratureRule(
            store=True, bounds=[-1, 1], prob_measure=True
        )
        # Initialize Lagrange basis with valid CC nterms (2^level + 1)
        # The growth rule determines actual points per subspace level
        growth = DoublePlusOneIndexGrowthRule()
        max_nterms = growth(level)  # 2^level + 1
        bases_1d = [
            UnivariateLagrangeBasis(quad_rule, max_nterms) for _ in range(nvars)
        ]
        basis = TensorProductInterpolatingBasis(bases_1d)

        # Don't specify backend - let legacy use its default
        sg = LegacyIsotropicSG(
            nqoi,
            nvars,
            level,
            growth,
            basis,
        )
        return sg

    def _build_typing_sparse_grid(self, nvars: int, level: int):
        """Build a typing sparse grid with Clenshaw-Curtis quadrature."""
        from pyapprox.typing.surrogates.affine.univariate import LagrangeBasis1D
        from pyapprox.typing.surrogates.affine.univariate.globalpoly import (
            ClenshawCurtisQuadratureRule,
        )
        from pyapprox.typing.surrogates.affine.indices import DoublePlusOneGrowthRule
        from pyapprox.typing.surrogates.sparsegrids import (
            IsotropicCombinationSparseGrid,
            PrebuiltBasisFactory,
        )

        cc_quad = ClenshawCurtisQuadratureRule(self._bkd, store=True)
        bases = [LagrangeBasis1D(self._bkd, cc_quad) for _ in range(nvars)]
        factories = [PrebuiltBasisFactory(b) for b in bases]
        growth = DoublePlusOneGrowthRule()

        sg = IsotropicCombinationSparseGrid(self._bkd, factories, growth, level=level)
        return sg

    def test_sample_points_match(self) -> None:
        """Test that sample points match between typing and legacy."""
        nvars, level = 2, 3

        legacy_sg = self._build_legacy_sparse_grid(nvars, level, nqoi=1)
        typing_sg = self._build_typing_sparse_grid(nvars, level)

        legacy_samples = legacy_sg.train_samples()
        typing_samples = self._bkd.to_numpy(typing_sg.get_samples())

        # Both should have the same number of samples
        self.assertEqual(legacy_samples.shape[1], typing_samples.shape[1])

        # Sort samples to ensure we can compare them
        legacy_sorted = legacy_samples[:, legacy_samples[0, :].argsort()]
        typing_sorted = typing_samples[:, typing_samples[0, :].argsort()]

        # Compare sorted samples
        import numpy as np
        np.testing.assert_allclose(legacy_sorted, typing_sorted, rtol=1e-12)

    def test_interpolation_matches_legacy(self) -> None:
        """Test interpolation matches legacy implementation."""
        nvars, level, nqoi = 2, 3, 1

        legacy_sg = self._build_legacy_sparse_grid(nvars, level, nqoi)
        typing_sg = self._build_typing_sparse_grid(nvars, level)

        # Function: f(x, y) = x^2 + x*y + y^2
        def test_func_np(samples):
            x, y = samples[0, :], samples[1, :]
            return (x**2 + x * y + y**2).reshape(1, -1)  # (nqoi, nsamples)

        # Set values on both grids
        # Legacy expects values shape (nsamples, nqoi) - transpose of typing
        legacy_samples = legacy_sg.train_samples()
        legacy_values = test_func_np(legacy_samples).T
        legacy_sg.fit(legacy_samples, legacy_values)

        typing_samples = typing_sg.get_samples()
        typing_values = self._bkd.reshape(
            typing_samples[0, :] ** 2
            + typing_samples[0, :] * typing_samples[1, :]
            + typing_samples[1, :] ** 2,
            (1, -1),
        )
        typing_sg.set_values(typing_values)

        # Evaluate at test points
        import numpy as np
        test_pts_np = np.array([[0.3, -0.5, 0.7], [0.2, 0.4, -0.3]])
        test_pts = self._bkd.asarray(test_pts_np)

        # Legacy returns (nsamples, nqoi), typing returns (nqoi, nsamples)
        legacy_result = legacy_sg(test_pts_np).T
        typing_result = self._bkd.to_numpy(typing_sg(test_pts))

        np.testing.assert_allclose(legacy_result, typing_result, rtol=1e-10)

    def test_mean_matches_legacy(self) -> None:
        """Test mean computation matches legacy implementation."""
        nvars, level, nqoi = 2, 3, 1

        legacy_sg = self._build_legacy_sparse_grid(nvars, level, nqoi)
        typing_sg = self._build_typing_sparse_grid(nvars, level)

        # Function: f(x, y) = x^2 + y^2
        # Legacy expects values shape (nsamples, nqoi) - transpose of typing
        legacy_samples = legacy_sg.train_samples()
        legacy_values = (
            legacy_samples[0, :] ** 2 + legacy_samples[1, :] ** 2
        ).reshape(1, -1).T
        legacy_sg.fit(legacy_samples, legacy_values)

        typing_samples = typing_sg.get_samples()
        typing_values = self._bkd.reshape(
            typing_samples[0, :] ** 2 + typing_samples[1, :] ** 2, (1, -1)
        )
        typing_sg.set_values(typing_values)

        legacy_mean = legacy_sg.mean()
        typing_mean = self._bkd.to_numpy(typing_sg.mean())

        import numpy as np
        np.testing.assert_allclose(legacy_mean, typing_mean, rtol=1e-12)

    def test_variance_matches_legacy(self) -> None:
        """Test variance computation matches legacy implementation."""
        nvars, level, nqoi = 2, 3, 1

        legacy_sg = self._build_legacy_sparse_grid(nvars, level, nqoi)
        typing_sg = self._build_typing_sparse_grid(nvars, level)

        # Function: f(x, y) = x + y
        # Legacy expects values shape (nsamples, nqoi) - transpose of typing
        legacy_samples = legacy_sg.train_samples()
        legacy_values = (legacy_samples[0, :] + legacy_samples[1, :]).reshape(
            1, -1
        ).T
        legacy_sg.fit(legacy_samples, legacy_values)

        typing_samples = typing_sg.get_samples()
        typing_values = self._bkd.reshape(
            typing_samples[0, :] + typing_samples[1, :], (1, -1)
        )
        typing_sg.set_values(typing_values)

        legacy_variance = legacy_sg.variance()
        typing_variance = self._bkd.to_numpy(typing_sg.variance())

        import numpy as np
        np.testing.assert_allclose(legacy_variance, typing_variance, rtol=1e-10)

    def test_higher_dimension_matches_legacy(self) -> None:
        """Test 3D sparse grid matches legacy implementation."""
        nvars, level, nqoi = 3, 2, 1

        legacy_sg = self._build_legacy_sparse_grid(nvars, level, nqoi)
        typing_sg = self._build_typing_sparse_grid(nvars, level)

        # Function: f(x, y, z) = x*y + y*z + x*z
        # Legacy expects values shape (nsamples, nqoi) - transpose of typing
        legacy_samples = legacy_sg.train_samples()
        x, y, z = legacy_samples[0, :], legacy_samples[1, :], legacy_samples[2, :]
        legacy_values = (x * y + y * z + x * z).reshape(1, -1).T
        legacy_sg.fit(legacy_samples, legacy_values)

        typing_samples = typing_sg.get_samples()
        tx, ty, tz = (
            typing_samples[0, :],
            typing_samples[1, :],
            typing_samples[2, :],
        )
        typing_values = self._bkd.reshape(tx * ty + ty * tz + tx * tz, (1, -1))
        typing_sg.set_values(typing_values)

        # Compare at test points
        import numpy as np
        test_pts_np = np.array([[0.3, -0.5], [0.2, 0.4], [-0.1, 0.6]])
        test_pts = self._bkd.asarray(test_pts_np)

        # Legacy returns (nsamples, nqoi), typing returns (nqoi, nsamples)
        legacy_result = legacy_sg(test_pts_np).T
        typing_result = self._bkd.to_numpy(typing_sg(test_pts))

        np.testing.assert_allclose(legacy_result, typing_result, rtol=1e-10)


class TestIsotropicLegacyNumpy(TestIsotropicLegacy[NDArray[Any]]):
    """NumPy backend tests for legacy comparison."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestIsotropicLegacyTorch(TestIsotropicLegacy[torch.Tensor]):
    """PyTorch backend tests for legacy comparison."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


# =============================================================================
# Parametrized tests for systematic coverage
# =============================================================================

import numpy as np
from unittest_parametrize import ParametrizedTestCase, parametrize

from pyapprox.typing.surrogates.sparsegrids.tests.test_helpers import (
    create_test_joint,
    create_test_pce,
    create_test_grid_gauss,
    create_test_grid_leja,
)


# Interpolation test configurations for Gauss quadrature: (name, joint_config, level)
# Gauss quadrature works with all marginal types
GAUSS_INTERPOLATION_CONFIGS = [
    ("2d_uniform_L1", "2d_uniform", 1),
    ("2d_uniform_L2", "2d_uniform", 2),
    ("2d_uniform_L3", "2d_uniform", 3),
    ("2d_gaussian_L2", "2d_gaussian", 2),
    ("2d_gaussian_L3", "2d_gaussian", 3),
    ("2d_beta_L2", "2d_beta", 2),
    ("2d_gamma_L2", "2d_gamma", 2),
    ("2d_gamma_L3", "2d_gamma", 3),
    ("2d_mixed_ug_L3", "2d_mixed_ug", 3),
    ("3d_uniform_L2", "3d_uniform", 2),
    ("3d_mixed_L2", "3d_mixed", 2),
    ("4d_uniform_L2", "4d_uniform", 2),
]

# Leja quadrature only works with bounded marginals (uniform, beta)
# Gaussian and Gamma require CDF/invcdf which needs quadrature rule
LEJA_INTERPOLATION_CONFIGS = [
    ("2d_uniform_L1", "2d_uniform", 1),
    ("2d_uniform_L2", "2d_uniform", 2),
    ("2d_uniform_L3", "2d_uniform", 3),
    ("2d_beta_L2", "2d_beta", 2),
    ("2d_mixed_ub_L2", "2d_mixed_ub", 2),
    ("3d_uniform_L2", "3d_uniform", 2),
    ("4d_uniform_L2", "4d_uniform", 2),
]


class TestIsotropicInterpolation(
    Generic[Array], ParametrizedTestCase, unittest.TestCase
):
    """Parametrized interpolation tests for sparse grids.

    Tests that sparse grids exactly interpolate PCE functions with
    matching polynomial degree across various:
    - Marginal distributions (uniform, gaussian, beta, gamma)
    - Dimensions (2D, 3D, 4D)
    - Levels (1, 2, 3)
    - Quadrature rules (Gauss, Leja)

    Note: Leja quadrature only works with bounded marginals (uniform, beta)
    since unbounded marginals (gaussian, gamma) require CDF/invcdf methods
    that need a quadrature rule for numerical integration.
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    @parametrize(
        "name,joint_config,level",
        GAUSS_INTERPOLATION_CONFIGS,
    )
    def test_gauss_interpolation(
        self, name: str, joint_config: str, level: int
    ) -> None:
        """Test Gauss quadrature sparse grid exactly interpolates PCE."""
        joint = create_test_joint(joint_config, self._bkd)
        pce = create_test_pce(joint, level, nqoi=1, bkd=self._bkd)
        grid = create_test_grid_gauss(joint, level, self._bkd)

        # Set grid values from PCE at sparse grid samples
        samples = grid.get_samples()
        values = pce(samples)
        grid.set_values(values)

        # Test at random points from joint distribution
        np.random.seed(123)
        test_pts = joint.rvs(20)
        result = grid(test_pts)
        expected = pce(test_pts)

        self._bkd.assert_allclose(result, expected, rtol=1e-10)

    @parametrize(
        "name,joint_config,level",
        LEJA_INTERPOLATION_CONFIGS,
    )
    def test_leja_interpolation(
        self, name: str, joint_config: str, level: int
    ) -> None:
        """Test Leja quadrature sparse grid exactly interpolates PCE."""
        joint = create_test_joint(joint_config, self._bkd)
        pce = create_test_pce(joint, level, nqoi=1, bkd=self._bkd)
        grid = create_test_grid_leja(joint, level, self._bkd)

        samples = grid.get_samples()
        values = pce(samples)
        grid.set_values(values)

        np.random.seed(123)
        test_pts = joint.rvs(20)
        result = grid(test_pts)
        expected = pce(test_pts)

        self._bkd.assert_allclose(result, expected, rtol=1e-10)


class TestIsotropicInterpolationNumpy(
    TestIsotropicInterpolation[NDArray[Any]]
):
    """NumPy backend interpolation tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestIsotropicInterpolationTorch(
    TestIsotropicInterpolation[torch.Tensor]
):
    """PyTorch backend interpolation tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


# Integration test configurations for Gauss quadrature: (name, joint_config, level)
# TODO: Beta and gamma marginals fail integration tests - see Phase 3 in plan
GAUSS_INTEGRATION_CONFIGS = [
    ("2d_uniform_L3", "2d_uniform", 3),
    ("2d_uniform_L4", "2d_uniform", 4),
    ("2d_gaussian_L3", "2d_gaussian", 3),
    ("2d_mixed_ug_L4", "2d_mixed_ug", 4),
    ("3d_uniform_L3", "3d_uniform", 3),
]

# Leja integration only tested with uniform marginals
# TODO: Beta fails - see Phase 3 in plan
LEJA_INTEGRATION_CONFIGS = [
    ("2d_uniform_L3", "2d_uniform", 3),
    ("2d_uniform_L4", "2d_uniform", 4),
    ("3d_uniform_L3", "3d_uniform", 3),
]


class TestIsotropicIntegration(
    Generic[Array], ParametrizedTestCase, unittest.TestCase
):
    """Parametrized integration tests for sparse grids.

    Tests that sparse grid quadrature computes correct mean for
    PCE functions. The PCE mean equals its constant coefficient (c_0).

    Note: Leja quadrature only works with bounded marginals (uniform, beta).
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    @parametrize(
        "name,joint_config,level",
        GAUSS_INTEGRATION_CONFIGS,
    )
    def test_gauss_integration_mean(
        self, name: str, joint_config: str, level: int
    ) -> None:
        """Test Gauss quadrature integrates to correct mean."""
        joint = create_test_joint(joint_config, self._bkd)
        pce = create_test_pce(joint, level, nqoi=1, bkd=self._bkd)
        grid = create_test_grid_gauss(joint, level, self._bkd)

        samples = grid.get_samples()
        values = pce(samples)
        grid.set_values(values)

        # PCE mean is c_0 (coefficient of constant term)
        expected_mean = pce.get_coefficients()[0, :]
        grid_mean = grid.mean()

        self._bkd.assert_allclose(grid_mean, expected_mean, rtol=1e-10)

    @parametrize(
        "name,joint_config,level",
        LEJA_INTEGRATION_CONFIGS,
    )
    def test_leja_integration_mean(
        self, name: str, joint_config: str, level: int
    ) -> None:
        """Test Leja quadrature integrates to correct mean."""
        joint = create_test_joint(joint_config, self._bkd)
        pce = create_test_pce(joint, level, nqoi=1, bkd=self._bkd)
        grid = create_test_grid_leja(joint, level, self._bkd)

        samples = grid.get_samples()
        values = pce(samples)
        grid.set_values(values)

        expected_mean = pce.get_coefficients()[0, :]
        grid_mean = grid.mean()

        self._bkd.assert_allclose(grid_mean, expected_mean, rtol=1e-10)


class TestIsotropicIntegrationNumpy(TestIsotropicIntegration[NDArray[Any]]):
    """NumPy backend integration tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestIsotropicIntegrationTorch(TestIsotropicIntegration[torch.Tensor]):
    """PyTorch backend integration tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


# Multi-QoI test configurations: (name, joint_config, level, nqoi)
MULTI_QOI_CONFIGS = [
    ("2d_uniform_L2_nqoi2", "2d_uniform", 2, 2),
    ("2d_uniform_L3_nqoi3", "2d_uniform", 3, 3),
    ("3d_uniform_L2_nqoi2", "3d_uniform", 2, 2),
]


class TestIsotropicMultiQoI(
    Generic[Array], ParametrizedTestCase, unittest.TestCase
):
    """Parametrized multi-QoI tests for sparse grids.

    Tests that sparse grids correctly handle multiple quantities of interest.
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    @parametrize(
        "name,joint_config,level,nqoi",
        MULTI_QOI_CONFIGS,
    )
    def test_multi_qoi_interpolation(
        self, name: str, joint_config: str, level: int, nqoi: int
    ) -> None:
        """Test multi-QoI sparse grid interpolation."""
        joint = create_test_joint(joint_config, self._bkd)
        pce = create_test_pce(joint, level, nqoi=nqoi, bkd=self._bkd)
        grid = create_test_grid_gauss(joint, level, self._bkd)

        samples = grid.get_samples()
        values = pce(samples)
        grid.set_values(values)

        np.random.seed(123)
        test_pts = joint.rvs(10)
        result = grid(test_pts)
        expected = pce(test_pts)

        self.assertEqual(result.shape[0], nqoi)
        self._bkd.assert_allclose(result, expected, rtol=1e-10)


class TestIsotropicMultiQoINumpy(TestIsotropicMultiQoI[NDArray[Any]]):
    """NumPy backend multi-QoI tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestIsotropicMultiQoITorch(TestIsotropicMultiQoI[torch.Tensor]):
    """PyTorch backend multi-QoI tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
