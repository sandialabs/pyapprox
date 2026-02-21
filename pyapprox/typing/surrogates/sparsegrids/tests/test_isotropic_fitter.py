"""Tests for IsotropicSparseGridFitter + CombinationSurrogate.

Tests that the new fitter API produces exact interpolation and integration
for functions that can be exactly represented by the sparse grid.

Tests run on both NumPy and PyTorch backends using parametrized configs
across different dimensions, basis types (Gauss, Leja, Clenshaw-Curtis,
piecewise), and marginal distributions.
"""

import unittest
from typing import Any, Generic, List, Union

import numpy as np
import torch
from numpy.typing import NDArray
from unittest_parametrize import ParametrizedTestCase, parametrize

from pyapprox.typing.probability import (
    IndependentJoint,
    UniformMarginal,
)
from pyapprox.typing.surrogates.affine.indices import (
    LinearGrowthRule,
    ClenshawCurtisGrowthRule,
    CubicNestedGrowthRule,
)
from pyapprox.typing.surrogates.affine.protocols import (
    IndexGrowthRuleProtocol,
)
from pyapprox.typing.surrogates.sparsegrids import (
    create_basis_factories,
    is_downward_closed,
)
from pyapprox.typing.surrogates.sparsegrids.basis_factory import (
    BasisFactoryProtocol,
    GaussLagrangeFactory,
    LejaLagrangeFactory,
    ClenshawCurtisLagrangeFactory,
    PiecewiseFactory,
)
from pyapprox.typing.surrogates.sparsegrids.subspace_factory import (
    TensorProductSubspaceFactory,
)
from pyapprox.typing.surrogates.sparsegrids.isotropic_fitter import (
    IsotropicSparseGridFitter,
)
from pyapprox.typing.surrogates.sparsegrids.combination_surrogate import (
    CombinationSurrogate,
)
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401
from pyapprox.typing.surrogates.sparsegrids.tests.test_helpers import (
    create_test_joint,
    create_test_pce,
    create_smooth_test_function,
    _get_default_growth_rule,
    BASIS_TYPE_CONFIGS,
    GROWTH_RULES,
)


# =============================================================================
# Parametrized test configurations
# =============================================================================

# Gauss interpolation: (name, joint_config, level)
GAUSS_INTERPOLATION_CONFIGS = [
    ("2d_uniform_L1", "2d_uniform", 1),
    ("2d_uniform_L2", "2d_uniform", 2),
    ("2d_gaussian_L2", "2d_gaussian", 2),
    ("2d_beta_L2", "2d_beta", 2),
    ("2d_gamma_L2", "2d_gamma", 2),
    ("2d_mixed_ug_L3", "2d_mixed_ug", 3),
    ("3d_uniform_L2", "3d_uniform", 2),
    ("3d_mixed_L2", "3d_mixed", 2),
]

# Leja interpolation: bounded marginals only
LEJA_INTERPOLATION_CONFIGS = [
    ("2d_uniform_L1", "2d_uniform", 1),
    ("2d_uniform_L2", "2d_uniform", 2),
    ("2d_beta_L2", "2d_beta", 2),
    ("2d_mixed_ub_L2", "2d_mixed_ub", 2),
    ("3d_uniform_L2", "3d_uniform", 2),
]

# Clenshaw-Curtis interpolation: bounded marginals only
CC_INTERPOLATION_CONFIGS = [
    ("2d_uniform_L1", "2d_uniform", 1),
    ("2d_uniform_L2", "2d_uniform", 2),
    ("2d_beta_L2", "2d_beta", 2),
    ("3d_uniform_L2", "3d_uniform", 2),
]

# Gauss integration: (name, joint_config, level)
GAUSS_INTEGRATION_CONFIGS = [
    ("2d_uniform_L3", "2d_uniform", 3),
    ("2d_gaussian_L3", "2d_gaussian", 3),
    ("2d_beta_L3", "2d_beta", 3),
    ("2d_gamma_L3", "2d_gamma", 3),
    ("2d_mixed_ug_L4", "2d_mixed_ug", 4),
    ("2d_mixed_ub_L3", "2d_mixed_ub", 3),
    ("3d_uniform_L3", "3d_uniform", 3),
]

# Leja integration: bounded marginals only
LEJA_INTEGRATION_CONFIGS = [
    ("2d_uniform_L3", "2d_uniform", 3),
    ("2d_beta_L3", "2d_beta", 3),
    ("2d_mixed_ub_L3", "2d_mixed_ub", 3),
    ("3d_uniform_L3", "3d_uniform", 3),
]

# CC integration: uniform marginals only (CC quadrature weights assume
# uniform measure; beta/gamma PDF is not incorporated in the weights)
CC_INTEGRATION_CONFIGS = [
    ("2d_uniform_L3", "2d_uniform", 3),
    ("3d_uniform_L3", "3d_uniform", 3),
]

# Multi-QoI: (name, joint_config, level, nqoi)
MULTI_QOI_CONFIGS = [
    ("2d_uniform_L2_nqoi2", "2d_uniform", 2, 2),
    ("2d_uniform_L3_nqoi3", "2d_uniform", 3, 3),
    ("3d_uniform_L2_nqoi2", "3d_uniform", 2, 2),
]

# Piecewise interpolation: (name, joint_config, level, basis_type, tol)
PIECEWISE_INTERPOLATION_CONFIGS = [
    ("2d_uniform_L8_linear", "2d_uniform", 8, "piecewise_linear", 1e-3),
    ("2d_beta_L8_linear", "2d_beta", 8, "piecewise_linear", 1e-3),
    ("2d_uniform_L8_quadratic", "2d_uniform", 8, "piecewise_quadratic", 1e-5),
    ("2d_uniform_L6_cubic", "2d_uniform", 6, "piecewise_cubic", 1e-5),
    ("2d_mixed_ub_L8_linear", "2d_mixed_ub", 8, "piecewise_linear", 1e-3),
]

# Mixed basis interpolation: (name, joint_config, level, basis_types_per_dim)
MIXED_BASIS_INTERPOLATION_CONFIGS = [
    ("2d_gauss_pwlinear_L7", "2d_uniform", 7, ["gauss", "piecewise_linear"]),
    ("2d_cc_pwlinear_L7", "2d_uniform", 7, ["clenshaw_curtis", "piecewise_linear"]),
    ("2d_leja_pwcubic_L4", "2d_uniform", 4, ["leja", "piecewise_cubic"]),
]

# Piecewise integration: (name, joint_config, level, basis_type)
PIECEWISE_INTEGRATION_CONFIGS = [
    ("2d_uniform_L3_linear", "2d_uniform", 3, "piecewise_linear"),
    ("2d_uniform_L3_quadratic", "2d_uniform", 3, "piecewise_quadratic"),
    ("2d_uniform_L3_cubic", "2d_uniform", 3, "piecewise_cubic"),
]

# Mixed basis integration: (name, joint_config, level, basis_types)
MIXED_BASIS_INTEGRATION_CONFIGS = [
    ("2d_gauss_pwlinear_L3", "2d_uniform", 3, ["gauss", "piecewise_linear"]),
    ("2d_cc_pwlinear_L3", "2d_uniform", 3, ["clenshaw_curtis", "piecewise_linear"]),
]


# =============================================================================
# Helper: create fitter from joint + basis_type
# =============================================================================


def _create_fitter(
    joint: IndependentJoint[Array],
    level: int,
    bkd: Backend[Array],
    basis_type: str = "gauss",
    growth: Union[IndexGrowthRuleProtocol, None] = None,
) -> IsotropicSparseGridFitter[Array]:
    """Create fitter with specified basis type and optional growth rule."""
    factories = create_basis_factories(joint.marginals(), bkd, basis_type)
    if growth is None:
        growth_name = None
        for config in BASIS_TYPE_CONFIGS:
            if config[0] == basis_type:
                growth_name = config[2]
                break
        if growth_name is None:
            growth_name = "linear_1_1"
        growth = GROWTH_RULES[growth_name]
    factory = TensorProductSubspaceFactory(bkd, factories, growth)
    return IsotropicSparseGridFitter(bkd, factory, level)


def _create_fitter_mixed(
    joint: IndependentJoint[Array],
    level: int,
    bkd: Backend[Array],
    basis_types: List[str],
) -> IsotropicSparseGridFitter[Array]:
    """Create fitter with mixed basis types per dimension."""
    marginals = joint.marginals()
    factories_list: List[BasisFactoryProtocol[Array]] = []
    for marginal, btype in zip(marginals, basis_types):
        factory: BasisFactoryProtocol[Array]
        if btype == "gauss":
            factory = GaussLagrangeFactory(marginal, bkd)
        elif btype == "leja":
            factory = LejaLagrangeFactory(marginal, bkd)
        elif btype == "clenshaw_curtis":
            factory = ClenshawCurtisLagrangeFactory(marginal, bkd)
        elif btype.startswith("piecewise_"):
            poly_type = btype.replace("piecewise_", "")
            factory = PiecewiseFactory(marginal, bkd, poly_type=poly_type)
        else:
            raise ValueError(f"Unknown basis_type: {btype}")
        factories_list.append(factory)

    growth_rules: List[IndexGrowthRuleProtocol] = [
        _get_default_growth_rule(bt) for bt in basis_types
    ]
    tp_factory = TensorProductSubspaceFactory(bkd, factories_list, growth_rules)
    return IsotropicSparseGridFitter(bkd, tp_factory, level)


# =============================================================================
# Core functionality tests
# =============================================================================


class TestIsotropicFitter(Generic[Array], unittest.TestCase):
    """Tests for IsotropicSparseGridFitter core functionality."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_interpolation(self) -> None:
        """Test sparse grid interpolation via PCE with matching index set."""
        from pyapprox.typing.surrogates.affine.expansions import (
            create_pce_from_marginals,
        )
        from scipy import stats
        from pyapprox.typing.probability import ScipyContinuousMarginal

        nvars = 2
        level = 3
        marginal = UniformMarginal(-1.0, 1.0, self._bkd)
        factories = [GaussLagrangeFactory(marginal, self._bkd)] * nvars
        growth = LinearGrowthRule(scale=1, shift=1)
        tp_factory = TensorProductSubspaceFactory(
            self._bkd, factories, growth
        )
        fitter = IsotropicSparseGridFitter(self._bkd, tp_factory, level)

        marginals = [
            UniformMarginal(-1.0, 1.0, self._bkd) for _ in range(nvars)
        ]
        pce = create_pce_from_marginals(
            marginals, max_level=level, bkd=self._bkd, nqoi=1
        )
        nterms = pce.nterms()
        coefficients = self._bkd.asarray(
            [[0.5], [-0.3], [0.2], [0.1], [-0.15], [0.25]]
            + [[0.0]] * (nterms - 6)
        )
        pce.set_coefficients(coefficients[:nterms, :])

        samples = fitter.get_samples()
        values = pce(samples)
        result = fitter.fit(values)
        surrogate = result.surrogate

        uniform_marginal = ScipyContinuousMarginal(
            stats.uniform(-1, 2), self._bkd
        )
        joint = IndependentJoint(
            [uniform_marginal for _ in range(nvars)], self._bkd
        )
        np.random.seed(42)
        test_pts = joint.rvs(20)

        sg_result = surrogate(test_pts)
        expected = pce(test_pts)
        self._bkd.assert_allclose(sg_result, expected, rtol=1e-10)

    def test_smolyak_coefficients_sum(self) -> None:
        """Test Smolyak coefficients sum to 1."""
        marginal = UniformMarginal(-1.0, 1.0, self._bkd)
        factories = [GaussLagrangeFactory(marginal, self._bkd)] * 2
        growth = LinearGrowthRule(scale=1, shift=1)
        tp_factory = TensorProductSubspaceFactory(
            self._bkd, factories, growth
        )

        for level in [1, 2, 3]:
            fitter = IsotropicSparseGridFitter(self._bkd, tp_factory, level)
            samples = fitter.get_samples()
            values = self._bkd.zeros((1, samples.shape[1]))
            result = fitter.fit(values)

            coefs = result.coefficients
            self._bkd.assert_allclose(
                self._bkd.asarray([float(self._bkd.sum(coefs))]),
                self._bkd.asarray([1.0]),
            )

    def test_result_nsamples(self) -> None:
        """Test that result.nsamples equals number of unique samples."""
        marginal = UniformMarginal(-1.0, 1.0, self._bkd)
        factories = [GaussLagrangeFactory(marginal, self._bkd)] * 2
        growth = LinearGrowthRule(scale=1, shift=1)
        tp_factory = TensorProductSubspaceFactory(
            self._bkd, factories, growth
        )
        fitter = IsotropicSparseGridFitter(self._bkd, tp_factory, level=2)

        samples = fitter.get_samples()
        values = self._bkd.zeros((1, samples.shape[1]))
        result = fitter.fit(values)

        self.assertEqual(result.nsamples, samples.shape[1])

    def test_surrogate_is_combination_surrogate(self) -> None:
        """Test that result.surrogate is a CombinationSurrogate."""
        marginal = UniformMarginal(-1.0, 1.0, self._bkd)
        factories = [GaussLagrangeFactory(marginal, self._bkd)] * 2
        growth = LinearGrowthRule(scale=1, shift=1)
        tp_factory = TensorProductSubspaceFactory(
            self._bkd, factories, growth
        )
        fitter = IsotropicSparseGridFitter(self._bkd, tp_factory, level=2)

        samples = fitter.get_samples()
        values = self._bkd.zeros((1, samples.shape[1]))
        result = fitter.fit(values)

        self.assertIsInstance(result.surrogate, CombinationSurrogate)

    def test_indices_downward_closed(self) -> None:
        """Test that result indices are downward closed."""
        marginal = UniformMarginal(-1.0, 1.0, self._bkd)
        factories = [GaussLagrangeFactory(marginal, self._bkd)] * 2
        growth = LinearGrowthRule(scale=1, shift=1)
        tp_factory = TensorProductSubspaceFactory(
            self._bkd, factories, growth
        )
        fitter = IsotropicSparseGridFitter(self._bkd, tp_factory, level=3)

        samples = fitter.get_samples()
        values = self._bkd.zeros((1, samples.shape[1]))
        result = fitter.fit(values)

        self.assertTrue(is_downward_closed(result.indices, self._bkd))


class TestIsotropicFitterNumpy(TestIsotropicFitter[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestIsotropicFitterTorch(TestIsotropicFitter[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


# =============================================================================
# Quadrature / moment tests
# =============================================================================


class TestFitterQuadrature(Generic[Array], unittest.TestCase):
    """Tests for mean/variance via IsotropicSparseGridFitter."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def _make_fitter(self, level: int) -> IsotropicSparseGridFitter[Array]:
        marginal = UniformMarginal(-1.0, 1.0, self._bkd)
        factories = [GaussLagrangeFactory(marginal, self._bkd)] * 2
        growth = LinearGrowthRule(scale=1, shift=1)
        tp_factory = TensorProductSubspaceFactory(
            self._bkd, factories, growth
        )
        return IsotropicSparseGridFitter(self._bkd, tp_factory, level)

    def test_mean_monomial_exact(self) -> None:
        """E[x^2 + y^2] = 2/3 on [-1,1]^2."""
        fitter = self._make_fitter(level=2)
        samples = fitter.get_samples()
        x, y = samples[0, :], samples[1, :]
        values = self._bkd.reshape(x ** 2 + y ** 2, (1, -1))
        result = fitter.fit(values)

        mean = result.surrogate.mean()
        self._bkd.assert_allclose(
            mean, self._bkd.asarray([2.0 / 3.0]), rtol=1e-12
        )

    def test_mean_mixed_monomial(self) -> None:
        """E[x^2*y^2] = 1/9 on [-1,1]^2."""
        fitter = self._make_fitter(level=3)
        samples = fitter.get_samples()
        x, y = samples[0, :], samples[1, :]
        values = self._bkd.reshape(x ** 2 * y ** 2, (1, -1))
        result = fitter.fit(values)

        mean = result.surrogate.mean()
        self._bkd.assert_allclose(
            mean, self._bkd.asarray([1.0 / 9.0]), rtol=1e-12
        )

    def test_integration_symmetry_odd_function(self) -> None:
        """Odd functions integrate to zero on symmetric domain."""
        fitter = self._make_fitter(level=3)
        samples = fitter.get_samples()
        x, y = samples[0, :], samples[1, :]

        values = self._bkd.reshape(x + y, (1, -1))
        result = fitter.fit(values)
        self._bkd.assert_allclose(
            result.surrogate.mean(), self._bkd.asarray([0.0]), atol=1e-14
        )

    def test_variance_sum_function(self) -> None:
        """Var[x + y] = 2/3 on [-1,1]^2."""
        fitter = self._make_fitter(level=2)
        samples = fitter.get_samples()
        x, y = samples[0, :], samples[1, :]
        values = self._bkd.reshape(x + y, (1, -1))
        result = fitter.fit(values)

        variance = result.surrogate.variance()
        self._bkd.assert_allclose(
            variance, self._bkd.asarray([2.0 / 3.0]), rtol=1e-10
        )

    def test_variance_product_function(self) -> None:
        """Var[x*y] = 1/9 on [-1,1]^2."""
        fitter = self._make_fitter(level=2)
        samples = fitter.get_samples()
        x, y = samples[0, :], samples[1, :]
        values = self._bkd.reshape(x * y, (1, -1))
        result = fitter.fit(values)

        variance = result.surrogate.variance()
        self._bkd.assert_allclose(
            variance, self._bkd.asarray([1.0 / 9.0]), rtol=1e-10
        )


class TestFitterQuadratureNumpy(TestFitterQuadrature[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestFitterQuadratureTorch(TestFitterQuadrature[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


# =============================================================================
# Parametrized exact interpolation tests
# =============================================================================


class TestFitterInterpolation(
    Generic[Array], ParametrizedTestCase, unittest.TestCase
):
    """Parametrized exact interpolation tests.

    For each basis type (Gauss, Leja, CC) and each dimension/marginal combo,
    creates a PCE that the sparse grid can exactly represent at the given
    level, then verifies interpolation is exact at random test points.
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
        """Test Gauss quadrature fitter exactly interpolates PCE."""
        joint = create_test_joint(joint_config, self._bkd)
        pce = create_test_pce(joint, level, nqoi=1, bkd=self._bkd)
        fitter = _create_fitter(joint, level, self._bkd, "gauss")

        samples = fitter.get_samples()
        values = pce(samples)
        result = fitter.fit(values)
        surrogate = result.surrogate

        np.random.seed(123)
        test_pts = joint.rvs(20)
        self._bkd.assert_allclose(
            surrogate(test_pts), pce(test_pts), rtol=1e-10
        )

    @parametrize(
        "name,joint_config,level",
        LEJA_INTERPOLATION_CONFIGS,
    )
    def test_leja_interpolation(
        self, name: str, joint_config: str, level: int
    ) -> None:
        """Test Leja quadrature fitter exactly interpolates PCE."""
        joint = create_test_joint(joint_config, self._bkd)
        pce = create_test_pce(joint, level, nqoi=1, bkd=self._bkd)
        fitter = _create_fitter(joint, level, self._bkd, "leja")

        samples = fitter.get_samples()
        values = pce(samples)
        result = fitter.fit(values)
        surrogate = result.surrogate

        np.random.seed(123)
        test_pts = joint.rvs(20)
        self._bkd.assert_allclose(
            surrogate(test_pts), pce(test_pts), rtol=1e-10
        )

    @parametrize(
        "name,joint_config,level",
        CC_INTERPOLATION_CONFIGS,
    )
    def test_cc_interpolation(
        self, name: str, joint_config: str, level: int
    ) -> None:
        """Test Clenshaw-Curtis fitter exactly interpolates PCE."""
        joint = create_test_joint(joint_config, self._bkd)
        pce = create_test_pce(joint, level, nqoi=1, bkd=self._bkd)
        fitter = _create_fitter(joint, level, self._bkd, "clenshaw_curtis")

        samples = fitter.get_samples()
        values = pce(samples)
        result = fitter.fit(values)
        surrogate = result.surrogate

        np.random.seed(123)
        test_pts = joint.rvs(20)
        self._bkd.assert_allclose(
            surrogate(test_pts), pce(test_pts), rtol=1e-10
        )


class TestFitterInterpolationNumpy(TestFitterInterpolation[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestFitterInterpolationTorch(TestFitterInterpolation[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


# =============================================================================
# Parametrized exact integration tests
# =============================================================================


class TestFitterIntegration(
    Generic[Array], ParametrizedTestCase, unittest.TestCase
):
    """Parametrized integration (mean) tests.

    For Lagrange-type bases (Gauss, Leja, CC), the PCE mean equals its
    constant coefficient c_0. This verifies the sparse grid quadrature
    computes the exact mean.
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
        """Test Gauss quadrature mean equals PCE constant coefficient."""
        joint = create_test_joint(joint_config, self._bkd)
        pce = create_test_pce(joint, level, nqoi=1, bkd=self._bkd)
        fitter = _create_fitter(joint, level, self._bkd, "gauss")

        samples = fitter.get_samples()
        values = pce(samples)
        result = fitter.fit(values)

        expected_mean = pce.get_coefficients()[0, :]
        self._bkd.assert_allclose(
            result.surrogate.mean(), expected_mean, rtol=1e-10
        )

    @parametrize(
        "name,joint_config,level",
        LEJA_INTEGRATION_CONFIGS,
    )
    def test_leja_integration_mean(
        self, name: str, joint_config: str, level: int
    ) -> None:
        """Test Leja quadrature mean equals PCE constant coefficient."""
        joint = create_test_joint(joint_config, self._bkd)
        pce = create_test_pce(joint, level, nqoi=1, bkd=self._bkd)
        fitter = _create_fitter(joint, level, self._bkd, "leja")

        samples = fitter.get_samples()
        values = pce(samples)
        result = fitter.fit(values)

        expected_mean = pce.get_coefficients()[0, :]
        self._bkd.assert_allclose(
            result.surrogate.mean(), expected_mean, rtol=1e-10
        )

    @parametrize(
        "name,joint_config,level",
        CC_INTEGRATION_CONFIGS,
    )
    def test_cc_integration_mean(
        self, name: str, joint_config: str, level: int
    ) -> None:
        """Test Clenshaw-Curtis quadrature mean equals PCE constant coeff."""
        joint = create_test_joint(joint_config, self._bkd)
        pce = create_test_pce(joint, level, nqoi=1, bkd=self._bkd)
        fitter = _create_fitter(joint, level, self._bkd, "clenshaw_curtis")

        samples = fitter.get_samples()
        values = pce(samples)
        result = fitter.fit(values)

        expected_mean = pce.get_coefficients()[0, :]
        self._bkd.assert_allclose(
            result.surrogate.mean(), expected_mean, rtol=1e-10
        )


class TestFitterIntegrationNumpy(TestFitterIntegration[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestFitterIntegrationTorch(TestFitterIntegration[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


# =============================================================================
# Multi-QoI tests
# =============================================================================


class TestFitterMultiQoI(
    Generic[Array], ParametrizedTestCase, unittest.TestCase
):
    """Multi-QoI interpolation tests using IsotropicSparseGridFitter."""

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
        """Test multi-QoI fitter exactly interpolates PCE."""
        joint = create_test_joint(joint_config, self._bkd)
        pce = create_test_pce(joint, level, nqoi=nqoi, bkd=self._bkd)
        fitter = _create_fitter(joint, level, self._bkd, "gauss")

        samples = fitter.get_samples()
        values = pce(samples)
        result = fitter.fit(values)
        surrogate = result.surrogate

        np.random.seed(123)
        test_pts = joint.rvs(10)
        sg_result = surrogate(test_pts)
        expected = pce(test_pts)

        self.assertEqual(sg_result.shape[0], nqoi)
        self._bkd.assert_allclose(sg_result, expected, rtol=1e-10)


class TestFitterMultiQoINumpy(TestFitterMultiQoI[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestFitterMultiQoITorch(TestFitterMultiQoI[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


# =============================================================================
# Piecewise interpolation tests (convergence, not exact)
# =============================================================================


class TestFitterPiecewiseInterpolation(
    Generic[Array], ParametrizedTestCase, unittest.TestCase
):
    """Piecewise interpolation tests: convergence on smooth functions."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    @parametrize(
        "name,joint_config,level,basis_type,tol",
        PIECEWISE_INTERPOLATION_CONFIGS,
    )
    def test_piecewise_interpolation_error(
        self,
        name: str,
        joint_config: str,
        level: int,
        basis_type: str,
        tol: float,
    ) -> None:
        """Test piecewise fitter achieves expected interpolation error."""
        joint = create_test_joint(joint_config, self._bkd)
        fitter = _create_fitter(joint, level, self._bkd, basis_type)
        test_func = create_smooth_test_function(joint, self._bkd)

        samples = fitter.get_samples()
        values = test_func(samples)
        result = fitter.fit(values)
        surrogate = result.surrogate

        np.random.seed(123)
        test_pts = joint.rvs(50)
        error = float(
            self._bkd.to_numpy(
                self._bkd.max(
                    self._bkd.abs(surrogate(test_pts) - test_func(test_pts))
                )
            )
        )
        self.assertLess(error, tol)

    @parametrize(
        "name,basis_type,expected_min_ratio",
        [
            ("linear_convergence", "piecewise_linear", 2.5),
            ("quadratic_convergence", "piecewise_quadratic", 5.0),
            ("cubic_convergence", "piecewise_cubic", 10.0),
        ],
    )
    def test_piecewise_convergence_rate(
        self, name: str, basis_type: str, expected_min_ratio: float
    ) -> None:
        """Test piecewise convergence rates O(h^p)."""
        joint = create_test_joint("2d_uniform", self._bkd)
        test_func = create_smooth_test_function(joint, self._bkd)

        level1, level2 = 5, 6
        errors = []
        for level in [level1, level2]:
            fitter = _create_fitter(joint, level, self._bkd, basis_type)
            samples = fitter.get_samples()
            result = fitter.fit(test_func(samples))
            surrogate = result.surrogate

            np.random.seed(123)
            test_pts = joint.rvs(200)
            error = float(
                self._bkd.to_numpy(
                    self._bkd.max(
                        self._bkd.abs(
                            surrogate(test_pts) - test_func(test_pts)
                        )
                    )
                )
            )
            errors.append(error)

        ratio = errors[0] / errors[1]
        self.assertGreater(
            ratio,
            expected_min_ratio,
            f"{basis_type}: error ratio {ratio:.2f} < {expected_min_ratio}",
        )


class TestFitterPiecewiseInterpolationNumpy(
    TestFitterPiecewiseInterpolation[NDArray[Any]]
):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestFitterPiecewiseInterpolationTorch(
    TestFitterPiecewiseInterpolation[torch.Tensor]
):
    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


# =============================================================================
# Mixed basis interpolation tests
# =============================================================================


class TestFitterMixedInterpolation(
    Generic[Array], ParametrizedTestCase, unittest.TestCase
):
    """Mixed Lagrange + piecewise interpolation tests."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    @parametrize(
        "name,joint_config,level,basis_types",
        MIXED_BASIS_INTERPOLATION_CONFIGS,
    )
    def test_mixed_interpolation_error(
        self,
        name: str,
        joint_config: str,
        level: int,
        basis_types: List[str],
    ) -> None:
        """Test mixed basis fitter achieves small interpolation error."""
        joint = create_test_joint(joint_config, self._bkd)
        fitter = _create_fitter_mixed(joint, level, self._bkd, basis_types)
        test_func = create_smooth_test_function(joint, self._bkd)

        samples = fitter.get_samples()
        result = fitter.fit(test_func(samples))
        surrogate = result.surrogate

        np.random.seed(123)
        test_pts = joint.rvs(50)
        error = float(
            self._bkd.to_numpy(
                self._bkd.max(
                    self._bkd.abs(surrogate(test_pts) - test_func(test_pts))
                )
            )
        )
        self.assertLess(error, 1e-3)


class TestFitterMixedInterpolationNumpy(
    TestFitterMixedInterpolation[NDArray[Any]]
):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestFitterMixedInterpolationTorch(
    TestFitterMixedInterpolation[torch.Tensor]
):
    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


# =============================================================================
# Piecewise / mixed integration tests
# =============================================================================


class TestFitterPiecewiseIntegration(
    Generic[Array], ParametrizedTestCase, unittest.TestCase
):
    """Piecewise integration tests (mean against Gauss reference)."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    @parametrize(
        "name,joint_config,level,basis_type",
        PIECEWISE_INTEGRATION_CONFIGS,
    )
    def test_piecewise_integration(
        self, name: str, joint_config: str, level: int, basis_type: str
    ) -> None:
        """Test piecewise quadrature mean against Gauss reference."""
        joint = create_test_joint(joint_config, self._bkd)
        fitter = _create_fitter(joint, level, self._bkd, basis_type)
        test_func = create_smooth_test_function(joint, self._bkd)

        samples = fitter.get_samples()
        result = fitter.fit(test_func(samples))
        grid_mean = result.surrogate.mean()

        ref_fitter = _create_fitter(joint, level + 2, self._bkd, "gauss")
        ref_samples = ref_fitter.get_samples()
        ref_result = ref_fitter.fit(test_func(ref_samples))
        ref_mean = ref_result.surrogate.mean()

        self._bkd.assert_allclose(grid_mean, ref_mean, atol=1e-10)

    @parametrize(
        "name,joint_config,level,basis_types",
        MIXED_BASIS_INTEGRATION_CONFIGS,
    )
    def test_mixed_integration(
        self,
        name: str,
        joint_config: str,
        level: int,
        basis_types: List[str],
    ) -> None:
        """Test mixed basis quadrature mean against Gauss reference."""
        joint = create_test_joint(joint_config, self._bkd)
        fitter = _create_fitter_mixed(joint, level, self._bkd, basis_types)
        test_func = create_smooth_test_function(joint, self._bkd)

        samples = fitter.get_samples()
        result = fitter.fit(test_func(samples))
        grid_mean = result.surrogate.mean()

        ref_fitter = _create_fitter(joint, level + 2, self._bkd, "gauss")
        ref_samples = ref_fitter.get_samples()
        ref_result = ref_fitter.fit(test_func(ref_samples))
        ref_mean = ref_result.surrogate.mean()

        self._bkd.assert_allclose(grid_mean, ref_mean, atol=1e-10)


class TestFitterPiecewiseIntegrationNumpy(
    TestFitterPiecewiseIntegration[NDArray[Any]]
):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestFitterPiecewiseIntegrationTorch(
    TestFitterPiecewiseIntegration[torch.Tensor]
):
    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


# =============================================================================
# Legacy comparison tests
# =============================================================================


class TestFitterLegacy(Generic[Array], unittest.TestCase):
    """Compare IsotropicSparseGridFitter against legacy implementation."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def _build_legacy_sparse_grid(self, nvars: int, level: int, nqoi: int):
        from pyapprox.surrogates.univariate.base import (
            ClenshawCurtisQuadratureRule as LegacyCCQuadratureRule,
        )
        from pyapprox.surrogates.univariate.lagrange import (
            UnivariateLagrangeBasis,
        )
        from pyapprox.surrogates.affine.basis import (
            TensorProductInterpolatingBasis,
        )
        from pyapprox.surrogates.affine.multiindex import (
            DoublePlusOneIndexGrowthRule,
        )
        from pyapprox.surrogates.sparsegrids.combination import (
            IsotropicCombinationSparseGrid as LegacyIsotropicSG,
        )

        quad_rule = LegacyCCQuadratureRule(
            store=True, bounds=[-1, 1], prob_measure=True
        )
        growth = DoublePlusOneIndexGrowthRule()
        max_nterms = growth(level)
        bases_1d = [
            UnivariateLagrangeBasis(quad_rule, max_nterms)
            for _ in range(nvars)
        ]
        basis = TensorProductInterpolatingBasis(bases_1d)
        sg = LegacyIsotropicSG(nqoi, nvars, level, growth, basis)
        return sg

    def _build_fitter(
        self, nvars: int, level: int
    ) -> IsotropicSparseGridFitter[Array]:
        from pyapprox.typing.surrogates.affine.univariate import (
            LagrangeBasis1D,
        )
        from pyapprox.typing.surrogates.affine.univariate.globalpoly import (
            ClenshawCurtisQuadratureRule,
        )
        from pyapprox.typing.surrogates.sparsegrids import (
            PrebuiltBasisFactory,
        )

        cc_quad = ClenshawCurtisQuadratureRule(self._bkd, store=True)
        bases = [LagrangeBasis1D(self._bkd, cc_quad) for _ in range(nvars)]
        factories = [PrebuiltBasisFactory(b) for b in bases]
        growth = ClenshawCurtisGrowthRule()
        tp_factory = TensorProductSubspaceFactory(
            self._bkd, factories, growth
        )
        return IsotropicSparseGridFitter(self._bkd, tp_factory, level)

    def test_interpolation_matches_legacy(self) -> None:
        """Test interpolation matches legacy implementation."""
        nvars, level = 2, 3

        legacy_sg = self._build_legacy_sparse_grid(nvars, level, nqoi=1)
        fitter = self._build_fitter(nvars, level)

        def test_func_np(samples):
            x, y = samples[0, :], samples[1, :]
            return (x ** 2 + x * y + y ** 2).reshape(1, -1)

        legacy_samples = legacy_sg.train_samples()
        legacy_values = test_func_np(legacy_samples).T
        legacy_sg.fit(legacy_samples, legacy_values)

        samples = fitter.get_samples()
        x, y = samples[0, :], samples[1, :]
        values = self._bkd.reshape(x ** 2 + x * y + y ** 2, (1, -1))
        result = fitter.fit(values)
        surrogate = result.surrogate

        test_pts_np = np.array([[0.3, -0.5, 0.7], [0.2, 0.4, -0.3]])
        test_pts = self._bkd.asarray(test_pts_np)

        legacy_result = legacy_sg(test_pts_np).T
        typing_result = self._bkd.to_numpy(surrogate(test_pts))

        self._bkd.assert_allclose(
            self._bkd.asarray(legacy_result),
            self._bkd.asarray(typing_result),
            rtol=1e-10,
        )

    def test_mean_matches_legacy(self) -> None:
        """Test mean computation matches legacy."""
        nvars, level = 2, 3

        legacy_sg = self._build_legacy_sparse_grid(nvars, level, nqoi=1)
        fitter = self._build_fitter(nvars, level)

        legacy_samples = legacy_sg.train_samples()
        legacy_values = (
            legacy_samples[0, :] ** 2 + legacy_samples[1, :] ** 2
        ).reshape(1, -1).T
        legacy_sg.fit(legacy_samples, legacy_values)

        samples = fitter.get_samples()
        values = self._bkd.reshape(
            samples[0, :] ** 2 + samples[1, :] ** 2, (1, -1)
        )
        result = fitter.fit(values)

        legacy_mean = legacy_sg.mean()
        typing_mean = self._bkd.to_numpy(result.surrogate.mean())

        self._bkd.assert_allclose(
            self._bkd.asarray(legacy_mean),
            self._bkd.asarray(typing_mean),
            rtol=1e-12,
        )

    def test_variance_matches_legacy(self) -> None:
        """Test variance computation matches legacy."""
        nvars, level = 2, 3

        legacy_sg = self._build_legacy_sparse_grid(nvars, level, nqoi=1)
        fitter = self._build_fitter(nvars, level)

        legacy_samples = legacy_sg.train_samples()
        legacy_values = (
            legacy_samples[0, :] + legacy_samples[1, :]
        ).reshape(1, -1).T
        legacy_sg.fit(legacy_samples, legacy_values)

        samples = fitter.get_samples()
        values = self._bkd.reshape(
            samples[0, :] + samples[1, :], (1, -1)
        )
        result = fitter.fit(values)

        legacy_variance = legacy_sg.variance()
        typing_variance = self._bkd.to_numpy(result.surrogate.variance())

        self._bkd.assert_allclose(
            self._bkd.asarray(legacy_variance),
            self._bkd.asarray(typing_variance),
            rtol=1e-10,
        )


class TestFitterLegacyNumpy(TestFitterLegacy[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestFitterLegacyTorch(TestFitterLegacy[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


# =============================================================================
# DerivativeChecker tests on CombinationSurrogate
# =============================================================================


class TestFitterDerivatives(Generic[Array], unittest.TestCase):
    """DerivativeChecker tests on CombinationSurrogate from fitter."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def _make_surrogate(self, func_name: str) -> CombinationSurrogate:
        """Create a fitted surrogate for a polynomial test function."""
        marginal = UniformMarginal(-1.0, 1.0, self._bkd)
        factories = [GaussLagrangeFactory(marginal, self._bkd)] * 2
        growth = LinearGrowthRule(scale=1, shift=1)
        tp_factory = TensorProductSubspaceFactory(
            self._bkd, factories, growth
        )
        fitter = IsotropicSparseGridFitter(self._bkd, tp_factory, level=3)
        samples = fitter.get_samples()
        x, y = samples[0, :], samples[1, :]

        if func_name == "quadratic":
            values = self._bkd.reshape(x ** 2 + y ** 2, (1, -1))
        elif func_name == "cubic":
            values = self._bkd.reshape(x ** 3 + x * y ** 2, (1, -1))
        else:
            raise ValueError(func_name)
        return fitter.fit(values).surrogate

    def test_jacobian_quadratic(self) -> None:
        """Test jacobian of interpolated x^2 + y^2."""
        surrogate = self._make_surrogate("quadratic")
        sample = self._bkd.asarray([[0.3], [0.5]])
        jac = surrogate.jacobian(sample)
        expected_jac = self._bkd.asarray([[0.6, 1.0]])
        self._bkd.assert_allclose(jac, expected_jac, rtol=1e-10)

    def test_hvp_quadratic(self) -> None:
        """Test HVP of interpolated x^2 + y^2."""
        surrogate = self._make_surrogate("quadratic")
        sample = self._bkd.asarray([[0.3], [0.5]])
        vec = self._bkd.asarray([[1.0], [0.0]])
        hvp = surrogate.hvp(sample, vec)
        expected_hvp = self._bkd.asarray([[2.0], [0.0]])
        self._bkd.assert_allclose(hvp, expected_hvp, atol=1e-12)

    def test_derivative_checker(self) -> None:
        """Test DerivativeChecker passes for jacobian and HVP."""
        from pyapprox.typing.interface.functions.derivative_checks.derivative_checker import DerivativeChecker

        surrogate = self._make_surrogate("cubic")
        sample = self._bkd.asarray([[0.3], [0.5]])
        checker = DerivativeChecker(surrogate)
        errors = checker.check_derivatives(sample)
        # errors[0] is jacobian FD errors, errors[1] is HVP FD errors
        for err in errors:
            ratio = float(self._bkd.to_numpy(checker.error_ratio(err)))
            self.assertLess(ratio, 1e-6)


class TestFitterDerivativesNumpy(TestFitterDerivatives[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestFitterDerivativesTorch(TestFitterDerivatives[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
