"""Tests for AdaptiveSparseGridFitter.

Tests verify that adaptive refinement converges for polynomial targets,
recovers anisotropic index sets without over-refinement, and correctly
handles additive functions and variance-based refinement.

Tests run on both NumPy and PyTorch backends.
"""

import unittest
from typing import Any, Generic, List

import numpy as np
import torch
from numpy.typing import NDArray
from unittest_parametrize import ParametrizedTestCase, parametrize

from pyapprox.typing.probability import UniformMarginal
from pyapprox.typing.surrogates.affine.indices import (
    LinearGrowthRule,
    MaxLevelCriteria,
)
from pyapprox.typing.surrogates.sparsegrids import create_basis_factories
from pyapprox.typing.surrogates.sparsegrids.adaptive_fitter import (
    AdaptiveSparseGridFitter,
)
from pyapprox.typing.surrogates.sparsegrids.basis_factory import (
    GaussLagrangeFactory,
)
from pyapprox.typing.surrogates.sparsegrids.combination_surrogate import (
    CombinationSurrogate,
)
from pyapprox.typing.surrogates.sparsegrids.error_indicators import (
    L2SurrogateDifferenceIndicator,
    VarianceChangeIndicator,
)
from pyapprox.typing.surrogates.sparsegrids.isotropic_fitter import (
    IsotropicSparseGridFitter,
)
from pyapprox.typing.surrogates.sparsegrids.subspace_factory import (
    TensorProductSubspaceFactory,
)
from pyapprox.typing.surrogates.sparsegrids.tests.test_helpers import (
    create_test_joint,
    create_test_pce,
    create_anisotropic_pce,
    create_additive_pce,
    get_required_sg_levels,
    compute_required_sg_subspaces,
    GROWTH_RULES,
)
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401


# =============================================================================
# Core functionality tests
# =============================================================================


class TestAdaptiveFitter(Generic[Array], unittest.TestCase):
    """Core tests for AdaptiveSparseGridFitter."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def _make_fitter(
        self, nvars: int = 2, max_level: int = 3
    ) -> AdaptiveSparseGridFitter[Array]:
        marginal = UniformMarginal(-1.0, 1.0, self._bkd)
        factories = [GaussLagrangeFactory(marginal, self._bkd)] * nvars
        growth = LinearGrowthRule(scale=1, shift=1)
        tp_factory = TensorProductSubspaceFactory(
            self._bkd, factories, growth
        )
        admis = MaxLevelCriteria(
            max_level=max_level, pnorm=1.0, bkd=self._bkd
        )
        return AdaptiveSparseGridFitter(self._bkd, tp_factory, admis)

    def test_convergence_on_polynomial(self) -> None:
        """Adaptive grid converges for polynomial target."""
        fitter = self._make_fitter(nvars=2, max_level=3)

        def poly_func(samples: Array) -> Array:
            x, y = samples[0, :], samples[1, :]
            return self._bkd.reshape(x ** 2 + y ** 2, (1, -1))

        result = fitter.refine_to_tolerance(poly_func, tol=1e-12)

        np.random.seed(123)
        marginal = UniformMarginal(-1.0, 1.0, self._bkd)
        from pyapprox.typing.probability import IndependentJoint
        joint = IndependentJoint([marginal, marginal], self._bkd)
        test_pts = joint.rvs(20)

        expected = poly_func(test_pts)
        actual = result.surrogate(test_pts)
        self._bkd.assert_allclose(actual, expected, rtol=1e-10)

    def test_3d_adaptive_grid(self) -> None:
        """3D adaptive grid converges for linear target."""
        fitter = self._make_fitter(nvars=3, max_level=2)

        def linear_func(samples: Array) -> Array:
            return self._bkd.reshape(
                samples[0, :] + samples[1, :] + samples[2, :], (1, -1)
            )

        result = fitter.refine_to_tolerance(linear_func, tol=1e-12)

        test_pts = self._bkd.asarray(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        )
        self._bkd.assert_allclose(
            result.surrogate(test_pts), linear_func(test_pts), rtol=1e-10
        )

    def test_result_is_adaptive_result(self) -> None:
        """Result has AdaptiveSparseGridFitResult fields."""
        fitter = self._make_fitter(nvars=2, max_level=2)

        def func(s: Array) -> Array:
            return self._bkd.reshape(s[0, :] + s[1, :], (1, -1))

        result = fitter.refine_to_tolerance(func, tol=1e-12)

        self.assertIsInstance(result.surrogate, CombinationSurrogate)
        self.assertIsInstance(result.nsamples, int)
        self.assertGreater(result.nsamples, 0)
        self.assertIsInstance(result.error, float)
        self.assertIsInstance(result.nsteps, int)
        self.assertIsInstance(result.converged, bool)

    def test_step_samples_values_pattern(self) -> None:
        """Test the step_samples/step_values manual pattern."""
        fitter = self._make_fitter(nvars=2, max_level=3)

        def func(s: Array) -> Array:
            return self._bkd.reshape(s[0, :] ** 2 + s[1, :] ** 2, (1, -1))

        # First step
        samples = fitter.step_samples()
        self.assertIsNotNone(samples)
        values = func(samples)
        fitter.step_values(values)

        # Second step
        samples2 = fitter.step_samples()
        self.assertIsNotNone(samples2)
        values2 = func(samples2)
        fitter.step_values(values2)

        # Should have a valid result
        result = fitter.result()
        self.assertIsInstance(result.surrogate, CombinationSurrogate)


class TestAdaptiveFitterNumpy(TestAdaptiveFitter[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestAdaptiveFitterTorch(TestAdaptiveFitter[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


# =============================================================================
# Parametrized convergence tests
# =============================================================================

ADAPTIVE_CONFIGS = [
    ("2d_uniform_L3", "2d_uniform", 3),
    ("2d_gaussian_L4", "2d_gaussian", 4),
    ("2d_beta_L3", "2d_beta", 3),
    ("3d_uniform_L2", "3d_uniform", 2),
]


class TestAdaptiveConvergence(
    Generic[Array], ParametrizedTestCase, unittest.TestCase
):
    """Parametrized convergence tests for adaptive fitter."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    @parametrize("name,joint_config,max_level", ADAPTIVE_CONFIGS)
    def test_adaptive_converges_to_pce(
        self, name: str, joint_config: str, max_level: int
    ) -> None:
        """Adaptive grid converges to PCE for polynomial target."""
        joint = create_test_joint(joint_config, self._bkd)
        pce = create_test_pce(joint, max_level, nqoi=1, bkd=self._bkd)

        factories = create_basis_factories(
            joint.marginals(), self._bkd, "gauss"
        )
        growth = LinearGrowthRule(scale=1, shift=1)
        tp_factory = TensorProductSubspaceFactory(
            self._bkd, factories, growth
        )
        admis = MaxLevelCriteria(
            max_level=max_level, pnorm=1.0, bkd=self._bkd
        )
        fitter = AdaptiveSparseGridFitter(self._bkd, tp_factory, admis)

        result = fitter.refine_to_tolerance(
            lambda s: pce(s), tol=1e-12, max_steps=50
        )

        np.random.seed(123)
        test_pts = joint.rvs(20)
        self._bkd.assert_allclose(
            result.surrogate(test_pts), pce(test_pts), rtol=1e-8
        )


class TestAdaptiveConvergenceNumpy(TestAdaptiveConvergence[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestAdaptiveConvergenceTorch(TestAdaptiveConvergence[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


# =============================================================================
# Mean/Variance matching tests
# =============================================================================


class TestAdaptiveMoments(Generic[Array], unittest.TestCase):
    """Tests that adaptive SG mean/variance match PCE."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def _build_converged_fitter(self, nqoi: int = 1):
        """Build an adaptive fitter converged on a PCE target."""
        joint = create_test_joint("2d_uniform", self._bkd)
        pce = create_test_pce(joint, level=3, nqoi=nqoi, bkd=self._bkd)

        factories = create_basis_factories(
            joint.marginals(), self._bkd, "gauss"
        )
        growth = LinearGrowthRule(scale=1, shift=1)
        tp_factory = TensorProductSubspaceFactory(
            self._bkd, factories, growth
        )
        admis = MaxLevelCriteria(
            max_level=4, pnorm=1.0, bkd=self._bkd
        )
        fitter = AdaptiveSparseGridFitter(self._bkd, tp_factory, admis)

        result = fitter.refine_to_tolerance(
            lambda s: pce(s), tol=1e-12, max_steps=50
        )
        return result, pce

    def test_adaptive_mean_matches_pce(self) -> None:
        """Adaptive SG mean matches PCE mean."""
        result, pce = self._build_converged_fitter(nqoi=2)
        self._bkd.assert_allclose(
            result.surrogate.mean(), pce.mean(), rtol=1e-8
        )

    def test_adaptive_variance_matches_pce(self) -> None:
        """Adaptive SG variance matches PCE variance."""
        result, pce = self._build_converged_fitter(nqoi=2)
        self._bkd.assert_allclose(
            result.surrogate.variance(), pce.variance(), rtol=1e-6
        )


class TestAdaptiveMomentsNumpy(TestAdaptiveMoments[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestAdaptiveMomentsTorch(TestAdaptiveMoments[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


# =============================================================================
# Anisotropic index set recovery
# =============================================================================

ANISOTROPIC_CONFIGS = [
    ("2d_aniso_31_leja", "2d_uniform", [3, 1], None, "linear_1_1", "leja"),
    ("3d_aniso_312_leja", "3d_uniform", [3, 1, 2], None, "linear_1_1", "leja"),
    ("2d_aniso_31_cc", "2d_uniform", [3, 1], None, "clenshaw_curtis", "clenshaw_curtis"),
    ("3d_aniso_312_cc", "3d_uniform", [3, 1, 2], None, "clenshaw_curtis", "clenshaw_curtis"),
]


class TestAdaptiveAnisotropicRecovery(
    Generic[Array], ParametrizedTestCase, unittest.TestCase
):
    """Tests that adaptive SG recovers anisotropic index sets.

    Requires nested quadrature (Leja or Clenshaw-Curtis) for stable
    hierarchical surpluses.
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    @parametrize(
        "name,joint_config,max_levels_1d,total_degree,growth_type,basis_type",
        ANISOTROPIC_CONFIGS,
    )
    def test_recovers_anisotropic_index_set(
        self,
        name: str,
        joint_config: str,
        max_levels_1d: List[int],
        total_degree,
        growth_type: str,
        basis_type: str,
    ) -> None:
        """Adaptive SG recovers required subspaces without over-refinement."""
        joint = create_test_joint(joint_config, self._bkd)
        pce = create_anisotropic_pce(
            joint, max_levels_1d, total_degree, nqoi=1, bkd=self._bkd
        )

        pce_indices = pce.get_indices()
        max_total_degree = max(
            int(self._bkd.to_numpy(self._bkd.sum(pce_indices[:, j])))
            for j in range(pce_indices.shape[1])
        )
        max_level = max_total_degree + 2

        growth = GROWTH_RULES[growth_type]
        factories = create_basis_factories(
            joint.marginals(), self._bkd, basis_type
        )
        tp_factory = TensorProductSubspaceFactory(
            self._bkd, factories, growth
        )
        admis = MaxLevelCriteria(
            max_level=max_level, pnorm=1.0, bkd=self._bkd
        )
        fitter = AdaptiveSparseGridFitter(self._bkd, tp_factory, admis)

        result = fitter.refine_to_tolerance(
            lambda s: pce(s), tol=1e-12, max_steps=100
        )

        # Verify exact interpolation
        np.random.seed(123)
        test_pts = joint.rvs(20)
        self._bkd.assert_allclose(
            result.surrogate(test_pts), pce(test_pts), rtol=1e-10
        )

        # Verify no over-refinement
        required_sg = compute_required_sg_subspaces(
            pce_indices, growth, self._bkd
        )
        selected_sg = result.indices
        nvars = selected_sg.shape[0]

        max_required = self._bkd.asarray([
            int(self._bkd.to_numpy(self._bkd.max(required_sg[d, :])))
            for d in range(nvars)
        ])

        for j in range(selected_sg.shape[1]):
            sel_idx = selected_sg[:, j]
            for d in range(nvars):
                sel_level = int(self._bkd.to_numpy(sel_idx[d]))
                max_req = int(self._bkd.to_numpy(max_required[d]))
                self.assertLessEqual(
                    sel_level, max_req + 1,
                    f"Over-refinement in dim {d}",
                )


class TestAdaptiveAnisotropicRecoveryNumpy(
    TestAdaptiveAnisotropicRecovery[NDArray[Any]]
):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestAdaptiveAnisotropicRecoveryTorch(
    TestAdaptiveAnisotropicRecovery[torch.Tensor]
):
    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


# =============================================================================
# Additive function recovery
# =============================================================================

ADDITIVE_CONFIGS = [
    ("2d_add_32_leja", "2d_uniform", [3, 2], "linear_1_1", "leja"),
    ("3d_add_324_leja", "3d_uniform", [3, 2, 4], "linear_1_1", "leja"),
    ("2d_add_32_cc", "2d_uniform", [3, 2], "clenshaw_curtis", "clenshaw_curtis"),
]


class TestAdaptiveAdditiveRecovery(
    Generic[Array], ParametrizedTestCase, unittest.TestCase
):
    """Tests that additive functions only refine 1D subspaces.

    Requires nested quadrature (Leja or CC) for stable surpluses.
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    @parametrize(
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
    ) -> None:
        """Additive function recovers 1D subspaces only."""
        joint = create_test_joint(joint_config, self._bkd)
        pce = create_additive_pce(
            joint, max_levels_1d, nqoi=1, bkd=self._bkd
        )

        growth = GROWTH_RULES[growth_type]
        required_levels = get_required_sg_levels(max_levels_1d, growth)
        max_level = max(required_levels) + 2

        factories = create_basis_factories(
            joint.marginals(), self._bkd, basis_type
        )
        tp_factory = TensorProductSubspaceFactory(
            self._bkd, factories, growth
        )
        admis = MaxLevelCriteria(
            max_level=max_level, pnorm=1.0, bkd=self._bkd
        )
        fitter = AdaptiveSparseGridFitter(self._bkd, tp_factory, admis)

        result = fitter.refine_to_tolerance(
            lambda s: pce(s), tol=1e-12, max_steps=100
        )

        # Verify exact interpolation
        np.random.seed(123)
        test_pts = joint.rvs(20)
        self._bkd.assert_allclose(
            result.surrogate(test_pts), pce(test_pts), rtol=1e-10
        )

        # Verify structure: multi-dim subspaces should be level 1 only
        sg_indices = result.indices
        for j in range(sg_indices.shape[1]):
            idx = sg_indices[:, j]
            nonzero_count = int(
                self._bkd.to_numpy(self._bkd.sum(idx > 0))
            )
            if nonzero_count >= 2:
                active_levels = idx[idx > 0]
                self.assertTrue(
                    self._bkd.all_bool(active_levels == 1),
                    f"Multi-dim subspace {self._bkd.to_numpy(idx)} "
                    "has levels > 1 (over-refinement of cross-terms)",
                )


class TestAdaptiveAdditiveRecoveryNumpy(
    TestAdaptiveAdditiveRecovery[NDArray[Any]]
):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestAdaptiveAdditiveRecoveryTorch(
    TestAdaptiveAdditiveRecovery[torch.Tensor]
):
    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


# =============================================================================
# Variance refinement tests
# =============================================================================


class TestAdaptiveVarianceRefinement(Generic[Array], unittest.TestCase):
    """End-to-end tests with VarianceChangeIndicator."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_adaptive_with_variance_refinement(self) -> None:
        """Adaptive SG converges using variance-based refinement."""
        joint = create_test_joint("2d_uniform", self._bkd)
        pce = create_test_pce(joint, level=3, nqoi=1, bkd=self._bkd)

        factories = create_basis_factories(
            joint.marginals(), self._bkd, "gauss"
        )
        growth = LinearGrowthRule(scale=1, shift=1)
        tp_factory = TensorProductSubspaceFactory(
            self._bkd, factories, growth
        )
        admis = MaxLevelCriteria(
            max_level=4, pnorm=1.0, bkd=self._bkd
        )
        indicator = VarianceChangeIndicator(self._bkd)
        fitter = AdaptiveSparseGridFitter(
            self._bkd, tp_factory, admis, error_indicator=indicator
        )

        result = fitter.refine_to_tolerance(
            lambda s: pce(s), tol=1e-12, max_steps=50
        )

        np.random.seed(123)
        test_pts = joint.rvs(20)
        self._bkd.assert_allclose(
            result.surrogate(test_pts), pce(test_pts), rtol=1e-8
        )
        self._bkd.assert_allclose(
            result.surrogate.mean(), pce.mean(), rtol=1e-8
        )
        self._bkd.assert_allclose(
            result.surrogate.variance(), pce.variance(), rtol=1e-6
        )

    def test_variance_refinement_multi_qoi(self) -> None:
        """Variance refinement with multiple QoIs."""
        joint = create_test_joint("2d_uniform", self._bkd)
        pce = create_test_pce(joint, level=3, nqoi=2, bkd=self._bkd)

        factories = create_basis_factories(
            joint.marginals(), self._bkd, "gauss"
        )
        growth = LinearGrowthRule(scale=1, shift=1)
        tp_factory = TensorProductSubspaceFactory(
            self._bkd, factories, growth
        )
        admis = MaxLevelCriteria(
            max_level=4, pnorm=1.0, bkd=self._bkd
        )
        indicator = VarianceChangeIndicator(self._bkd)
        fitter = AdaptiveSparseGridFitter(
            self._bkd, tp_factory, admis, error_indicator=indicator
        )

        result = fitter.refine_to_tolerance(
            lambda s: pce(s), tol=1e-12, max_steps=50
        )

        np.random.seed(123)
        test_pts = joint.rvs(20)
        self._bkd.assert_allclose(
            result.surrogate(test_pts), pce(test_pts), rtol=1e-8
        )


class TestAdaptiveVarianceRefinementNumpy(
    TestAdaptiveVarianceRefinement[NDArray[Any]]
):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestAdaptiveVarianceRefinementTorch(
    TestAdaptiveVarianceRefinement[torch.Tensor]
):
    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


# =============================================================================
# Adaptive recovers isotropic
# =============================================================================


class TestAdaptiveRecoversIsotropic(Generic[Array], unittest.TestCase):
    """Test that adaptive grid recovers same result as isotropic grid."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_recovers_isotropic(self) -> None:
        """Adaptive fitter with MaxLevel recovers isotropic fitter result."""
        level = 2
        joint = create_test_joint("2d_uniform", self._bkd)
        pce = create_test_pce(joint, level, nqoi=1, bkd=self._bkd)

        factories = create_basis_factories(
            joint.marginals(), self._bkd, "gauss"
        )
        growth = LinearGrowthRule(scale=1, shift=1)
        tp_factory = TensorProductSubspaceFactory(
            self._bkd, factories, growth
        )

        # Build isotropic
        iso_fitter = IsotropicSparseGridFitter(
            self._bkd, tp_factory, level
        )
        iso_samples = iso_fitter.get_samples()
        iso_result = iso_fitter.fit(pce(iso_samples))

        # Build adaptive with MaxLevel at same level
        admis = MaxLevelCriteria(
            max_level=level, pnorm=1.0, bkd=self._bkd
        )
        tp_factory2 = TensorProductSubspaceFactory(
            self._bkd, factories, growth
        )
        ada_fitter = AdaptiveSparseGridFitter(
            self._bkd, tp_factory2, admis
        )
        ada_result = ada_fitter.refine_to_tolerance(
            lambda s: pce(s), tol=1e-14, max_steps=50
        )

        # Both should give same evaluation
        np.random.seed(42)
        test_pts = joint.rvs(20)
        self._bkd.assert_allclose(
            ada_result.surrogate(test_pts),
            iso_result.surrogate(test_pts),
            rtol=1e-10,
        )

        # Same mean and variance
        self._bkd.assert_allclose(
            ada_result.surrogate.mean(),
            iso_result.surrogate.mean(),
            rtol=1e-10,
        )
        self._bkd.assert_allclose(
            ada_result.surrogate.variance(),
            iso_result.surrogate.variance(),
            rtol=1e-8,
        )


class TestAdaptiveRecoversIsotropicNumpy(
    TestAdaptiveRecoversIsotropic[NDArray[Any]]
):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestAdaptiveRecoversIsotropicTorch(
    TestAdaptiveRecoversIsotropic[torch.Tensor]
):
    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
