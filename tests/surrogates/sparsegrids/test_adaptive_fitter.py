"""Tests for SingleFidelityAdaptiveSparseGridFitter.

Tests verify that adaptive refinement converges for polynomial targets,
recovers anisotropic index sets without over-refinement, and correctly
handles additive functions and variance-based refinement.

Tests run on both NumPy and PyTorch backends.
"""

from typing import List

import numpy as np
import pytest

from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.indices import (
    LinearGrowthRule,
    MaxLevelCriteria,
)
from pyapprox.surrogates.sparsegrids import create_basis_factories
from pyapprox.surrogates.sparsegrids.adaptive_fitter import (
    SingleFidelityAdaptiveSparseGridFitter,
)
from pyapprox.surrogates.sparsegrids.basis_factory import (
    GaussLagrangeFactory,
)
from pyapprox.surrogates.sparsegrids.combination_surrogate import (
    CombinationSurrogate,
)
from pyapprox.surrogates.sparsegrids.error_indicators import (
    VarianceChangeIndicator,
)
from pyapprox.surrogates.sparsegrids.isotropic_fitter import (
    IsotropicSparseGridFitter,
)
from pyapprox.surrogates.sparsegrids.subspace_factory import (
    TensorProductSubspaceFactory,
)
from tests._helpers.sparsegrids_helpers import (
    GROWTH_RULES,
    compute_required_sg_subspaces,
    create_additive_pce,
    create_anisotropic_pce,
    create_test_joint,
    create_test_pce,
    get_required_sg_levels,
)
from tests._helpers.markers import slow_test, slower_test

# =============================================================================
# Core functionality tests
# =============================================================================


class TestAdaptiveFitter:
    """Core tests for SingleFidelityAdaptiveSparseGridFitter."""

    def _make_fitter(self, bkd, nvars=2, max_level=3):
        marginal = UniformMarginal(-1.0, 1.0, bkd)
        factories = [GaussLagrangeFactory(marginal, bkd)] * nvars
        growth = LinearGrowthRule(scale=1, shift=1)
        tp_factory = TensorProductSubspaceFactory(bkd, factories, growth)
        admis = MaxLevelCriteria(max_level=max_level, pnorm=1.0, bkd=bkd)
        return SingleFidelityAdaptiveSparseGridFitter(bkd, tp_factory, admis)

    def test_convergence_on_polynomial(self, bkd) -> None:
        """Adaptive grid converges for polynomial target."""
        fitter = self._make_fitter(bkd, nvars=2, max_level=3)

        def poly_func(samples):
            x, y = samples[0, :], samples[1, :]
            return bkd.reshape(x**2 + y**2, (1, -1))

        result = fitter.refine_to_tolerance(poly_func, tol=1e-12)

        np.random.seed(123)
        marginal = UniformMarginal(-1.0, 1.0, bkd)
        from pyapprox.probability import IndependentJoint

        joint = IndependentJoint([marginal, marginal], bkd)
        test_pts = joint.rvs(20)

        expected = poly_func(test_pts)
        actual = result.surrogate(test_pts)
        bkd.assert_allclose(actual, expected, rtol=1e-10)

    @pytest.mark.slow_on("TorchBkd")
    def test_3d_adaptive_grid(self, bkd) -> None:
        """3D adaptive grid converges for linear target."""
        fitter = self._make_fitter(bkd, nvars=3, max_level=2)

        def linear_func(samples):
            return bkd.reshape(
                samples[0, :] + samples[1, :] + samples[2, :], (1, -1)
            )

        result = fitter.refine_to_tolerance(linear_func, tol=1e-12)

        test_pts = bkd.asarray([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        bkd.assert_allclose(
            result.surrogate(test_pts), linear_func(test_pts), rtol=1e-10
        )

    def test_result_is_adaptive_result(self, bkd) -> None:
        """Result has AdaptiveSparseGridFitResult fields."""
        fitter = self._make_fitter(bkd, nvars=2, max_level=2)

        def func(s):
            return bkd.reshape(s[0, :] + s[1, :], (1, -1))

        result = fitter.refine_to_tolerance(func, tol=1e-12)

        assert isinstance(result.surrogate, CombinationSurrogate)
        assert isinstance(result.nsamples, int)
        assert result.nsamples > 0
        assert isinstance(result.error, float)
        assert isinstance(result.nsteps, int)
        assert isinstance(result.converged, bool)

    def test_step_samples_values_pattern(self, bkd) -> None:
        """Test the step_samples/step_values manual pattern."""
        fitter = self._make_fitter(bkd, nvars=2, max_level=3)

        def func(s):
            return bkd.reshape(s[0, :] ** 2 + s[1, :] ** 2, (1, -1))

        # First step
        samples = fitter.step_samples()
        assert samples is not None
        values = func(samples)
        fitter.step_values(values)

        # Second step
        samples2 = fitter.step_samples()
        assert samples2 is not None
        values2 = func(samples2)
        fitter.step_values(values2)

        # Should have a valid result
        result = fitter.result()
        assert isinstance(result.surrogate, CombinationSurrogate)


# =============================================================================
# Parametrized convergence tests
# =============================================================================

ADAPTIVE_CONFIGS = [
    ("2d_uniform_L3", "2d_uniform", 3),
    ("2d_gaussian_L4", "2d_gaussian", 4),
    ("2d_beta_L3", "2d_beta", 3),
    ("3d_uniform_L2", "3d_uniform", 2),
]


class TestAdaptiveConvergence:
    """Parametrized convergence tests for adaptive fitter."""

    @pytest.mark.parametrize("name,joint_config,max_level", ADAPTIVE_CONFIGS)
    def test_adaptive_converges_to_pce(
        self, name: str, joint_config: str, max_level: int, bkd
    ) -> None:
        """Adaptive grid converges to PCE for polynomial target."""
        joint = create_test_joint(joint_config, bkd)
        pce = create_test_pce(joint, max_level, nqoi=1, bkd=bkd)

        factories = create_basis_factories(joint.marginals(), bkd, "gauss")
        growth = LinearGrowthRule(scale=1, shift=1)
        tp_factory = TensorProductSubspaceFactory(bkd, factories, growth)
        admis = MaxLevelCriteria(max_level=max_level, pnorm=1.0, bkd=bkd)
        fitter = SingleFidelityAdaptiveSparseGridFitter(bkd, tp_factory, admis)

        result = fitter.refine_to_tolerance(lambda s: pce(s), tol=1e-12, max_steps=50)

        np.random.seed(123)
        test_pts = joint.rvs(20)
        bkd.assert_allclose(result.surrogate(test_pts), pce(test_pts), rtol=1e-8)


# =============================================================================
# Mean/Variance matching tests
# =============================================================================


class TestAdaptiveMoments:
    """Tests that adaptive SG mean/variance match PCE."""

    def _build_converged_fitter(self, bkd, nqoi=1):
        """Build an adaptive fitter converged on a PCE target."""
        joint = create_test_joint("2d_uniform", bkd)
        pce = create_test_pce(joint, level=3, nqoi=nqoi, bkd=bkd)

        factories = create_basis_factories(joint.marginals(), bkd, "gauss")
        growth = LinearGrowthRule(scale=1, shift=1)
        tp_factory = TensorProductSubspaceFactory(bkd, factories, growth)
        admis = MaxLevelCriteria(max_level=4, pnorm=1.0, bkd=bkd)
        fitter = SingleFidelityAdaptiveSparseGridFitter(bkd, tp_factory, admis)

        result = fitter.refine_to_tolerance(lambda s: pce(s), tol=1e-12, max_steps=50)
        return result, pce

    def test_adaptive_mean_matches_pce(self, bkd) -> None:
        """Adaptive SG mean matches PCE mean."""
        result, pce = self._build_converged_fitter(bkd, nqoi=2)
        bkd.assert_allclose(result.surrogate.mean(), pce.mean(), rtol=1e-8)

    def test_adaptive_variance_matches_pce(self, bkd) -> None:
        """Adaptive SG variance matches PCE variance."""
        result, pce = self._build_converged_fitter(bkd, nqoi=2)
        bkd.assert_allclose(
            result.surrogate.variance(), pce.variance(), rtol=1e-6
        )


# =============================================================================
# Anisotropic index set recovery
# =============================================================================

ANISOTROPIC_CONFIGS = [
    ("2d_aniso_31_leja", "2d_uniform", [3, 1], None, "linear_1_1", "leja"),
    ("3d_aniso_312_leja", "3d_uniform", [3, 1, 2], None, "linear_1_1", "leja"),
    (
        "2d_aniso_31_cc",
        "2d_uniform",
        [3, 1],
        None,
        "clenshaw_curtis",
        "clenshaw_curtis",
    ),
    (
        "3d_aniso_312_cc",
        "3d_uniform",
        [3, 1, 2],
        None,
        "clenshaw_curtis",
        "clenshaw_curtis",
    ),
]


class TestAdaptiveAnisotropicRecovery:
    """Tests that adaptive SG recovers anisotropic index sets.

    Requires nested quadrature (Leja or Clenshaw-Curtis) for stable
    hierarchical surpluses.
    """

    @pytest.mark.parametrize(
        "name,joint_config,max_levels_1d,total_degree,growth_type,basis_type",
        ANISOTROPIC_CONFIGS,
    )
    @slow_test
    def test_recovers_anisotropic_index_set(
        self,
        name: str,
        joint_config: str,
        max_levels_1d: List[int],
        total_degree,
        growth_type: str,
        basis_type: str,
        bkd,
    ) -> None:
        """Adaptive SG recovers required subspaces without over-refinement."""
        joint = create_test_joint(joint_config, bkd)
        pce = create_anisotropic_pce(
            joint, max_levels_1d, total_degree, nqoi=1, bkd=bkd
        )

        pce_indices = pce.get_indices()
        max_total_degree = max(
            int(bkd.to_numpy(bkd.sum(pce_indices[:, j])))
            for j in range(pce_indices.shape[1])
        )
        max_level = max_total_degree + 2

        growth = GROWTH_RULES[growth_type]
        factories = create_basis_factories(joint.marginals(), bkd, basis_type)
        tp_factory = TensorProductSubspaceFactory(bkd, factories, growth)
        admis = MaxLevelCriteria(max_level=max_level, pnorm=1.0, bkd=bkd)
        fitter = SingleFidelityAdaptiveSparseGridFitter(bkd, tp_factory, admis)

        result = fitter.refine_to_tolerance(lambda s: pce(s), tol=1e-12, max_steps=100)

        # Verify exact interpolation
        np.random.seed(123)
        test_pts = joint.rvs(20)
        bkd.assert_allclose(result.surrogate(test_pts), pce(test_pts), rtol=1e-10)

        # Verify no over-refinement
        required_sg = compute_required_sg_subspaces(pce_indices, growth, bkd)
        selected_sg = result.indices
        nvars = selected_sg.shape[0]

        max_required = bkd.asarray(
            [
                int(bkd.to_numpy(bkd.max(required_sg[d, :])))
                for d in range(nvars)
            ]
        )

        for j in range(selected_sg.shape[1]):
            sel_idx = selected_sg[:, j]
            for d in range(nvars):
                sel_level = int(bkd.to_numpy(sel_idx[d]))
                max_req = int(bkd.to_numpy(max_required[d]))
                assert sel_level <= max_req + 1, f"Over-refinement in dim {d}"


# =============================================================================
# Additive function recovery
# =============================================================================

ADDITIVE_CONFIGS = [
    ("2d_add_32_leja", "2d_uniform", [3, 2], "linear_1_1", "leja"),
    ("3d_add_324_leja", "3d_uniform", [3, 2, 4], "linear_1_1", "leja"),
    ("2d_add_32_cc", "2d_uniform", [3, 2], "clenshaw_curtis", "clenshaw_curtis"),
]


@slower_test
class TestAdaptiveAdditiveRecovery:
    """Tests that additive functions only refine 1D subspaces.

    Requires nested quadrature (Leja or CC) for stable surpluses.
    """

    @pytest.mark.parametrize(
        "name,joint_config,max_levels_1d,growth_type,basis_type",
        ADDITIVE_CONFIGS,
    )
    def test_additive_function_no_cross_terms(
        self,
        name: str,
        joint_config: str,
        max_levels_1d: List[int],
        growth_type: str,
        basis_type: str,
        bkd,
    ) -> None:
        """Additive function recovers 1D subspaces only."""
        joint = create_test_joint(joint_config, bkd)
        pce = create_additive_pce(joint, max_levels_1d, nqoi=1, bkd=bkd)

        growth = GROWTH_RULES[growth_type]
        required_levels = get_required_sg_levels(max_levels_1d, growth)
        max_level = max(required_levels) + 2

        factories = create_basis_factories(joint.marginals(), bkd, basis_type)
        tp_factory = TensorProductSubspaceFactory(bkd, factories, growth)
        admis = MaxLevelCriteria(max_level=max_level, pnorm=1.0, bkd=bkd)
        fitter = SingleFidelityAdaptiveSparseGridFitter(bkd, tp_factory, admis)

        result = fitter.refine_to_tolerance(lambda s: pce(s), tol=1e-12, max_steps=100)

        # Verify exact interpolation
        np.random.seed(123)
        test_pts = joint.rvs(20)
        bkd.assert_allclose(result.surrogate(test_pts), pce(test_pts), rtol=1e-10)

        # Verify structure: multi-dim subspaces should be level 1 only
        sg_indices = result.indices
        for j in range(sg_indices.shape[1]):
            idx = sg_indices[:, j]
            nonzero_count = int(bkd.to_numpy(bkd.sum(idx > 0)))
            if nonzero_count >= 2:
                active_levels = idx[idx > 0]
                assert bkd.all_bool(active_levels == 1), (
                    f"Multi-dim subspace {bkd.to_numpy(idx)} "
                    "has levels > 1 (over-refinement of cross-terms)"
                )


# =============================================================================
# Variance refinement tests
# =============================================================================


class TestAdaptiveVarianceRefinement:
    """End-to-end tests with VarianceChangeIndicator."""

    def test_adaptive_with_variance_refinement(self, bkd) -> None:
        """Adaptive SG converges using variance-based refinement."""
        joint = create_test_joint("2d_uniform", bkd)
        pce = create_test_pce(joint, level=3, nqoi=1, bkd=bkd)

        factories = create_basis_factories(joint.marginals(), bkd, "gauss")
        growth = LinearGrowthRule(scale=1, shift=1)
        tp_factory = TensorProductSubspaceFactory(bkd, factories, growth)
        admis = MaxLevelCriteria(max_level=4, pnorm=1.0, bkd=bkd)
        indicator = VarianceChangeIndicator(bkd)
        fitter = SingleFidelityAdaptiveSparseGridFitter(
            bkd, tp_factory, admis, error_indicator=indicator
        )

        result = fitter.refine_to_tolerance(lambda s: pce(s), tol=1e-12, max_steps=50)

        np.random.seed(123)
        test_pts = joint.rvs(20)
        bkd.assert_allclose(result.surrogate(test_pts), pce(test_pts), rtol=1e-8)
        bkd.assert_allclose(result.surrogate.mean(), pce.mean(), rtol=1e-8)
        bkd.assert_allclose(
            result.surrogate.variance(), pce.variance(), rtol=1e-6
        )

    def test_variance_refinement_multi_qoi(self, bkd) -> None:
        """Variance refinement with multiple QoIs."""
        joint = create_test_joint("2d_uniform", bkd)
        pce = create_test_pce(joint, level=3, nqoi=2, bkd=bkd)

        factories = create_basis_factories(joint.marginals(), bkd, "gauss")
        growth = LinearGrowthRule(scale=1, shift=1)
        tp_factory = TensorProductSubspaceFactory(bkd, factories, growth)
        admis = MaxLevelCriteria(max_level=4, pnorm=1.0, bkd=bkd)
        indicator = VarianceChangeIndicator(bkd)
        fitter = SingleFidelityAdaptiveSparseGridFitter(
            bkd, tp_factory, admis, error_indicator=indicator
        )

        result = fitter.refine_to_tolerance(lambda s: pce(s), tol=1e-12, max_steps=50)

        np.random.seed(123)
        test_pts = joint.rvs(20)
        bkd.assert_allclose(result.surrogate(test_pts), pce(test_pts), rtol=1e-8)


# =============================================================================
# Adaptive recovers isotropic
# =============================================================================


class TestAdaptiveRecoversIsotropic:
    """Test that adaptive grid recovers same result as isotropic grid."""

    def test_recovers_isotropic(self, bkd) -> None:
        """Adaptive fitter with MaxLevel recovers isotropic fitter result."""
        level = 2
        joint = create_test_joint("2d_uniform", bkd)
        pce = create_test_pce(joint, level, nqoi=1, bkd=bkd)

        factories = create_basis_factories(joint.marginals(), bkd, "gauss")
        growth = LinearGrowthRule(scale=1, shift=1)
        tp_factory = TensorProductSubspaceFactory(bkd, factories, growth)

        # Build isotropic
        iso_fitter = IsotropicSparseGridFitter(bkd, tp_factory, level)
        iso_samples = iso_fitter.get_samples()
        iso_result = iso_fitter.fit(pce(iso_samples))

        # Build adaptive with MaxLevel at same level
        admis = MaxLevelCriteria(max_level=level, pnorm=1.0, bkd=bkd)
        tp_factory2 = TensorProductSubspaceFactory(bkd, factories, growth)
        ada_fitter = SingleFidelityAdaptiveSparseGridFitter(
            bkd, tp_factory2, admis
        )
        ada_result = ada_fitter.refine_to_tolerance(
            lambda s: pce(s), tol=1e-14, max_steps=50
        )

        # Both should give same evaluation
        np.random.seed(42)
        test_pts = joint.rvs(20)
        bkd.assert_allclose(
            ada_result.surrogate(test_pts),
            iso_result.surrogate(test_pts),
            rtol=1e-10,
        )

        # Same mean and variance
        bkd.assert_allclose(
            ada_result.surrogate.mean(),
            iso_result.surrogate.mean(),
            rtol=1e-10,
        )
        bkd.assert_allclose(
            ada_result.surrogate.variance(),
            iso_result.surrogate.variance(),
            rtol=1e-8,
        )
