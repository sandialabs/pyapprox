"""Dual-backend tests for AdaptiveCombinationSparseGrid.

Tests run on both NumPy and PyTorch backends using the base class pattern.
"""

import unittest
from typing import Any, Generic, List

import numpy as np
import torch
from numpy.typing import NDArray
from unittest_parametrize import ParametrizedTestCase, parametrize

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.test_utils import load_tests

from pyapprox.typing.surrogates.sparsegrids import (
    AdaptiveCombinationSparseGrid,
    create_basis_factories,
)
from pyapprox.typing.surrogates.sparsegrids.basis_factory import (
    GaussLagrangeFactory,
)
from pyapprox.typing.surrogates.sparsegrids.tests.test_helpers import (
    create_test_joint,
    create_test_pce,
    create_anisotropic_pce,
    create_additive_pce,
    get_required_sg_levels,
    GROWTH_RULES,
)
from pyapprox.typing.probability import UniformMarginal
from pyapprox.typing.surrogates.affine.indices import (
    LinearGrowthRule,
    MaxLevelCriteria,
)


class TestAdaptiveSparseGrid(Generic[Array], unittest.TestCase):
    """Tests for AdaptiveCombinationSparseGrid - dual backend."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_step_samples_values_pattern(self) -> None:
        """Test the step_samples/step_values pattern."""
        marginal = UniformMarginal(-1.0, 1.0, self._bkd)
        factory = GaussLagrangeFactory(marginal, self._bkd)
        growth = LinearGrowthRule(scale=2, shift=1)
        admis = MaxLevelCriteria(max_level=3, pnorm=1.0, bkd=self._bkd)

        grid = AdaptiveCombinationSparseGrid(
            self._bkd, [factory, factory], growth, admis
        )

        # First step should return samples
        samples = grid.step_samples()
        self.assertIsNotNone(samples)
        self.assertGreater(samples.shape[1], 0)

        # Values should be accepted - shape (nqoi, nsamples)
        x, y = samples[0, :], samples[1, :]
        values = self._bkd.reshape(x**2 + y**2, (1, -1))
        grid.step_values(values)

        # Second step should also return samples (candidates exist)
        samples2 = grid.step_samples()
        self.assertIsNotNone(samples2)
        self.assertGreater(samples2.shape[1], 0)

    def test_evaluation_after_refinement(self) -> None:
        """Test that evaluation works after refinement."""
        marginal = UniformMarginal(-1.0, 1.0, self._bkd)
        factory = GaussLagrangeFactory(marginal, self._bkd)
        growth = LinearGrowthRule(scale=2, shift=1)
        admis = MaxLevelCriteria(max_level=3, pnorm=1.0, bkd=self._bkd)

        grid = AdaptiveCombinationSparseGrid(
            self._bkd, [factory, factory], growth, admis
        )

        # Test function: f(x, y) = x^2 + y^2 - shape (nqoi, nsamples)
        def test_func(samples: Array) -> Array:
            x, y = samples[0, :], samples[1, :]
            return self._bkd.reshape(x**2 + y**2, (1, -1))

        # Perform several refinement steps
        for _ in range(3):
            samples = grid.step_samples()
            if samples is None:
                break
            values = test_func(samples)
            grid.step_values(values)

        # Evaluation should work
        test_pts = self._bkd.asarray([[0.3, -0.5], [0.2, 0.4]])
        result = grid(test_pts)
        expected = test_func(test_pts)

        self.assertTrue(self._bkd.allclose(result, expected, rtol=1e-8))

    def test_convergence_on_polynomial(self) -> None:
        """Test that adaptive grid converges for polynomial target."""
        marginal = UniformMarginal(-1.0, 1.0, self._bkd)
        factory = GaussLagrangeFactory(marginal, self._bkd)
        growth = LinearGrowthRule(scale=2, shift=1)
        admis = MaxLevelCriteria(max_level=3, pnorm=1.0, bkd=self._bkd)

        grid = AdaptiveCombinationSparseGrid(
            self._bkd, [factory, factory], growth, admis
        )

        # Polynomial that should be exactly represented - shape (nqoi, nsamples)
        def poly_func(samples: Array) -> Array:
            x, y = samples[0, :], samples[1, :]
            return self._bkd.reshape(x**2 + y**2, (1, -1))

        # Refine until convergence
        max_steps = 10
        for _ in range(max_steps):
            samples = grid.step_samples()
            if samples is None:
                break
            values = poly_func(samples)
            grid.step_values(values)

        # Should have refined at least once
        self.assertGreater(grid.nsubspaces(), 1)

        # Evaluation should be exact
        test_pts = self._bkd.asarray([[0.3, -0.5, 0.8], [0.2, 0.4, -0.7]])
        result = grid(test_pts)
        expected = poly_func(test_pts)

        self.assertTrue(self._bkd.allclose(result, expected, rtol=1e-10))

    def test_multi_qoi_adaptive(self) -> None:
        """Test adaptive grid with multiple quantities of interest."""
        marginal = UniformMarginal(-1.0, 1.0, self._bkd)
        factory = GaussLagrangeFactory(marginal, self._bkd)
        growth = LinearGrowthRule(scale=2, shift=1)
        admis = MaxLevelCriteria(max_level=2, pnorm=1.0, bkd=self._bkd)

        grid = AdaptiveCombinationSparseGrid(
            self._bkd, [factory, factory], growth, admis
        )

        # Two QoIs - shape (nqoi, nsamples)
        def multi_func(samples: Array) -> Array:
            x, y = samples[0, :], samples[1, :]
            return self._bkd.stack([x + y, x * y], axis=0)

        # Refine
        for _ in range(3):
            samples = grid.step_samples()
            if samples is None:
                break
            values = multi_func(samples)
            grid.step_values(values)

        # Test evaluation
        test_pts = self._bkd.asarray([[0.3, -0.2], [0.4, 0.5]])
        result = grid(test_pts)

        self.assertEqual(result.shape[0], 2)  # nqoi is first dimension

    def test_3d_adaptive_grid(self) -> None:
        """Test 3D adaptive sparse grid."""
        marginal = UniformMarginal(-1.0, 1.0, self._bkd)
        factories = [
            GaussLagrangeFactory(marginal, self._bkd) for _ in range(3)
        ]
        growth = LinearGrowthRule(scale=2, shift=1)
        admis = MaxLevelCriteria(max_level=2, pnorm=1.0, bkd=self._bkd)

        grid = AdaptiveCombinationSparseGrid(
            self._bkd, factories, growth, admis
        )

        # Linear function - shape (nqoi, nsamples)
        def linear_func(samples: Array) -> Array:
            return self._bkd.reshape(
                samples[0, :] + samples[1, :] + samples[2, :], (1, -1)
            )

        # Refine until convergence (or max steps)
        for _ in range(10):
            samples = grid.step_samples()
            if samples is None:
                break
            values = linear_func(samples)
            grid.step_values(values)

        # Test evaluation - should be exact for linear
        test_pts = self._bkd.asarray([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        result = grid(test_pts)
        expected = linear_func(test_pts)

        self.assertTrue(self._bkd.allclose(result, expected, rtol=1e-10))


# NumPy backend tests
class TestAdaptiveSparseGridNumpy(TestAdaptiveSparseGrid[NDArray[Any]]):
    """NumPy backend tests for AdaptiveCombinationSparseGrid."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


# PyTorch backend tests
class TestAdaptiveSparseGridTorch(TestAdaptiveSparseGrid[torch.Tensor]):
    """PyTorch backend tests for AdaptiveCombinationSparseGrid."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def bkd(self) -> TorchBkd:
        return TorchBkd()


# =============================================================================
# Parametrized Adaptive Convergence Tests
# =============================================================================

ADAPTIVE_CONFIGS = [
    # (name, joint_config, max_level)
    ("2d_uniform_L3", "2d_uniform", 3),
    ("2d_gaussian_L4", "2d_gaussian", 4),
    ("2d_mixed_ug_L3", "2d_mixed_ug", 3),
    ("3d_uniform_L2", "3d_uniform", 2),
    # Non-canonical domain tests (catches domain mismatch bugs)
    ("2d_beta_L3", "2d_beta", 3),
]


class TestAdaptiveConvergence(
    Generic[Array], ParametrizedTestCase, unittest.TestCase
):
    """Parametrized tests for adaptive sparse grid convergence."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    @parametrize("name,joint_config,max_level", ADAPTIVE_CONFIGS)
    def test_adaptive_converges_to_pce(
        self, name: str, joint_config: str, max_level: int
    ) -> None:
        """Test adaptive grid converges to PCE for polynomial target.

        Note: Error threshold of 1e-12 is for polynomial targets that should be
        exactly representable by the sparse grid. For non-polynomial targets,
        use a larger tolerance based on expected approximation error.
        """
        joint = create_test_joint(joint_config, self._bkd)
        pce = create_test_pce(joint, max_level, nqoi=1, bkd=self._bkd)

        factories = create_basis_factories(
            joint.marginals(), self._bkd, "gauss"
        )
        growth = LinearGrowthRule(scale=1, shift=1)
        admis = MaxLevelCriteria(max_level=max_level, pnorm=1.0, bkd=self._bkd)

        grid = AdaptiveCombinationSparseGrid(
            self._bkd, factories, growth, admis
        )

        # Refine until convergence
        for _ in range(50):
            samples = grid.step_samples()
            if samples is None:
                break
            values = pce(samples)
            grid.step_values(values)

        # Verify interpolation
        np.random.seed(123)
        test_pts = joint.rvs(20)
        result = grid(test_pts)
        expected = pce(test_pts)

        self._bkd.assert_allclose(result, expected, rtol=1e-8)


class TestAdaptiveConvergenceNumpy(TestAdaptiveConvergence[NDArray[Any]]):
    """NumPy backend tests."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestAdaptiveConvergenceTorch(TestAdaptiveConvergence[torch.Tensor]):
    """PyTorch backend tests."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def bkd(self) -> TorchBkd:
        return TorchBkd()


# =============================================================================
# Anisotropic Index Set Recovery Tests
# =============================================================================

ANISOTROPIC_CONFIGS = [
    # (name, joint_config, max_levels_1d, total_degree, growth_type)
    # Linear (1,1) growth: d = l + 1
    ("2d_aniso_31_lin11", "2d_uniform", [3, 1], None, "linear_1_1"),
    ("2d_aniso_32_td4_lin11", "2d_uniform", [3, 2], 4, "linear_1_1"),
    ("3d_aniso_312_lin11", "3d_uniform", [3, 1, 2], None, "linear_1_1"),
    ("3d_aniso_312_td4_lin11", "3d_uniform", [3, 1, 2], 4, "linear_1_1"),
    # Linear (2,1) growth: d = 2*l + 1
    ("2d_aniso_31_lin21", "2d_uniform", [3, 1], None, "linear_2_1"),
    ("3d_aniso_312_lin21", "3d_uniform", [3, 1, 2], None, "linear_2_1"),
]


class TestAdaptiveAnisotropicRecovery(
    Generic[Array], ParametrizedTestCase, unittest.TestCase
):
    """Tests for adaptive recovery of anisotropic index sets without over-refinement."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def _assert_no_over_refinement(
        self,
        grid: AdaptiveCombinationSparseGrid[Array],
        required_levels: List[int],
        tolerance: int = 1,
    ) -> None:
        """Assert adaptive grid didn't refine beyond required levels.

        Parameters
        ----------
        grid : AdaptiveCombinationSparseGrid
            The adaptive grid to check.
        required_levels : List[int]
            Minimum required SG level per dimension.
        tolerance : int
            Allowed over-refinement (default 1 for exploration).
        """
        selected_indices = (
            grid.get_selected_subspace_indices()
        )  # (nvars, nselected)
        for dim, req_level in enumerate(required_levels):
            dim_max = int(
                self._bkd.to_numpy(self._bkd.max(selected_indices[dim, :]))
            )
            self.assertLessEqual(
                dim_max,
                req_level + tolerance,
                f"Over-refinement in dimension {dim}: achieved level {dim_max}, "
                f"required {req_level}, tolerance {tolerance}",
            )

    @parametrize(
        "name,joint_config,max_levels_1d,total_degree,growth_type",
        ANISOTROPIC_CONFIGS,
    )
    def test_recovers_anisotropic_index_set(
        self,
        name: str,
        joint_config: str,
        max_levels_1d: List[int],
        total_degree: int | None,
        growth_type: str,
    ) -> None:
        """Test adaptive SG recovers anisotropic PCE index set without over-refinement.

        Note: Error threshold of 1e-12 is for polynomial targets that should be
        exactly representable by the sparse grid.
        """
        joint = create_test_joint(joint_config, self._bkd)

        # Create anisotropic PCE target
        pce = create_anisotropic_pce(
            joint, max_levels_1d, total_degree, nqoi=1, bkd=self._bkd
        )

        # Get growth rule and compute required SG levels
        growth = GROWTH_RULES[growth_type]
        required_levels = get_required_sg_levels(max_levels_1d, growth)
        max_level = max(required_levels) + 2  # Allow exploration buffer

        factories = create_basis_factories(
            joint.marginals(), self._bkd, "gauss"
        )
        admis = MaxLevelCriteria(max_level=max_level, pnorm=1.0, bkd=self._bkd)
        grid = AdaptiveCombinationSparseGrid(
            self._bkd, factories, growth, admis
        )

        # DEBUG: Print setup info
        print(f"\n{'='*60}")
        print(f"TEST: {name}")
        print(f"max_levels_1d: {max_levels_1d}, total_degree: {total_degree}")
        print(
            f"growth_type: {growth_type}, required_levels: {required_levels}"
        )
        print(f"max_level (with buffer): {max_level}")
        print(f"{'='*60}")

        # Refine until convergence OR error < tolerance
        tol = 1e-12  # For exact polynomial targets
        for i in range(100):
            samples = grid.step_samples()
            if samples is None:
                print(
                    f"Step {i}: step_samples returned None - no more candidates"
                )
                break
            values = pce(samples)
            grid.step_values(values)

            error = grid.error_estimate()
            num_selected = grid.get_selected_subspace_indices().shape[1]

            # Get max level per dimension
            selected_indices = grid.get_selected_subspace_indices()
            max_levels_achieved = [
                int(self._bkd.to_numpy(self._bkd.max(selected_indices[d, :])))
                for d in range(selected_indices.shape[0])
            ]

            # Count candidates with inf error (access internal state for debug)
            if hasattr(grid, "_subspace_errors"):
                inf_count = sum(
                    1 for e in grid._subspace_errors if e == float("inf")
                )
            else:
                inf_count = "N/A"

            print(
                f"Step {i}: error={error:.2e}, selected={num_selected}, "
                f"max_levels={max_levels_achieved}, inf_candidates={inf_count}"
            )

            if error < tol:
                print(f"*** Converged at step {i} with error {error:.2e} ***")
                break
        else:
            print(
                f"*** Did NOT converge after 100 steps, final error={error:.2e} ***"
            )

        # Verify exact interpolation
        np.random.seed(123)
        test_pts = joint.rvs(20)
        self._bkd.assert_allclose(grid(test_pts), pce(test_pts), rtol=1e-10)

        # Verify no over-refinement
        print(
            f"\nFinal check: required_levels={required_levels}, "
            f"achieved={max_levels_achieved}"
        )
        self._assert_no_over_refinement(grid, required_levels, tolerance=1)


class TestAdaptiveAnisotropicRecoveryNumpy(
    TestAdaptiveAnisotropicRecovery[NDArray[Any]]
):
    """NumPy backend tests."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestAdaptiveAnisotropicRecoveryTorch(
    TestAdaptiveAnisotropicRecovery[torch.Tensor]
):
    """PyTorch backend tests."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def bkd(self) -> TorchBkd:
        return TorchBkd()


# =============================================================================
# Additive Function Recovery Tests
# =============================================================================

ADDITIVE_CONFIGS = [
    # (name, joint_config, max_levels_1d, growth_type)
    # Linear (1,1) growth
    ("2d_additive_32_lin11", "2d_uniform", [3, 2], "linear_1_1"),
    ("3d_additive_324_lin11", "3d_uniform", [3, 2, 4], "linear_1_1"),
    # Linear (2,1) growth
    ("2d_additive_32_lin21", "2d_uniform", [3, 2], "linear_2_1"),
    ("3d_additive_324_lin21", "3d_uniform", [3, 2, 4], "linear_2_1"),
]


class TestAdaptiveAdditiveRecovery(
    Generic[Array], ParametrizedTestCase, unittest.TestCase
):
    """Tests for adaptive recovery of additive functions (no cross-terms)."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    @parametrize(
        "name,joint_config,max_levels_1d,growth_type", ADDITIVE_CONFIGS
    )
    def test_additive_function_no_cross_terms(
        self,
        name: str,
        joint_config: str,
        max_levels_1d: List[int],
        growth_type: str,
    ) -> None:
        """Test additive function recovers 1D subspaces only.

        For f(x) = g1(x1) + g2(x2) + g3(x3):
        - SG should have 1D subspaces up to required levels
        - 2D level-1 subspaces like (1,1,0) are explored (admissibility)
        - But (1,1,0) has zero error -> low priority -> (2,1,0) never added

        Note: Error threshold of 1e-12 is for polynomial targets.
        """
        joint = create_test_joint(joint_config, self._bkd)

        # Create additive PCE (only 1D terms)
        pce = create_additive_pce(joint, max_levels_1d, nqoi=1, bkd=self._bkd)

        # Get growth rule and compute required SG levels
        growth = GROWTH_RULES[growth_type]
        required_levels = get_required_sg_levels(max_levels_1d, growth)
        max_level = max(required_levels) + 2

        factories = create_basis_factories(
            joint.marginals(), self._bkd, "gauss"
        )
        admis = MaxLevelCriteria(max_level=max_level, pnorm=1.0, bkd=self._bkd)
        grid = AdaptiveCombinationSparseGrid(
            self._bkd, factories, growth, admis
        )

        # Refine until convergence
        tol = 1e-12
        for _ in range(100):
            samples = grid.step_samples()
            if samples is None:
                break
            values = pce(samples)
            grid.step_values(values)

            if grid.error_estimate() < tol:
                break

        # Verify exact interpolation
        np.random.seed(123)
        test_pts = joint.rvs(20)
        self._bkd.assert_allclose(grid(test_pts), pce(test_pts), rtol=1e-10)

        # Verify structure: 2D subspaces should only be level-1 (explored but rejected)
        sg_indices = grid.get_selected_subspace_indices()

        for j in range(sg_indices.shape[1]):
            idx = sg_indices[:, j]
            nonzero_count = int(self._bkd.to_numpy(self._bkd.sum(idx > 0)))
            if nonzero_count >= 2:
                # Multi-dimensional subspace should be level 1 in each active dim
                active_levels = idx[idx > 0]
                self.assertTrue(
                    self._bkd.all_bool(active_levels == 1),
                    f"Multi-dim subspace {self._bkd.to_numpy(idx)} has levels > 1, "
                    "indicating over-refinement of cross-terms",
                )


class TestAdaptiveAdditiveRecoveryNumpy(
    TestAdaptiveAdditiveRecovery[NDArray[Any]]
):
    """NumPy backend tests."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestAdaptiveAdditiveRecoveryTorch(
    TestAdaptiveAdditiveRecovery[torch.Tensor]
):
    """PyTorch backend tests."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def bkd(self) -> TorchBkd:
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
