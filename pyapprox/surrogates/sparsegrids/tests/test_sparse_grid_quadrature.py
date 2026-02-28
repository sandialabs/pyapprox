"""Tests for sparse grid quadrature weights and parameterized rule."""

import pytest

from pyapprox.probability.univariate import UniformMarginal
from pyapprox.surrogates.affine.indices import LinearGrowthRule
from pyapprox.surrogates.sparsegrids import (
    GaussLagrangeFactory,
    IsotropicSparseGridFitter,
    ParameterizedIsotropicSparseGridQuadratureRule,
    TensorProductSubspaceFactory,
)


def _make_fitter(level, bkd):
    """Create a 2D isotropic sparse grid fitter on [-1,1]^2."""
    marginal = UniformMarginal(-1.0, 1.0, bkd)
    factories = [GaussLagrangeFactory(marginal, bkd)] * 2
    growth = LinearGrowthRule(scale=1, shift=1)
    tp_factory = TensorProductSubspaceFactory(bkd, factories, growth)
    return IsotropicSparseGridFitter(bkd, tp_factory, level)


def _make_factory(bkd, nvars=2):
    """Create a SubspaceFactory for nvars dims on [-1,1]^nvars."""
    marginal = UniformMarginal(-1.0, 1.0, bkd)
    factories = [GaussLagrangeFactory(marginal, bkd)] * nvars
    growth = LinearGrowthRule(scale=1, shift=1)
    return TensorProductSubspaceFactory(bkd, factories, growth)


class TestSparseGridQuadratureWeights:
    """Tests for IsotropicSparseGridFitter.get_quadrature_weights()."""

    def test_shape(self, bkd) -> None:
        """Weights shape matches number of unique samples."""
        fitter = _make_fitter(level=2, bkd=bkd)
        samples = fitter.get_samples()
        weights = fitter.get_quadrature_weights()
        assert weights.shape == (samples.shape[1],)

    def test_weights_sum_to_one(self, bkd) -> None:
        """Weights sum to 1 for probability measure (Gauss-Legendre on U(-1,1))."""
        fitter = _make_fitter(level=3, bkd=bkd)
        weights = fitter.get_quadrature_weights()
        bkd.assert_allclose(
            bkd.sum(weights, axis=0),
            bkd.asarray(1.0),
            rtol=1e-12,
        )

    def test_mean_monomial_exact(self, bkd) -> None:
        """E[x^2 + y^2] = 2/3 on [-1,1]^2 via direct weight summation."""
        fitter = _make_fitter(level=2, bkd=bkd)
        samples = fitter.get_samples()
        weights = fitter.get_quadrature_weights()
        x, y = samples[0, :], samples[1, :]
        f_vals = x**2 + y**2
        integral = bkd.sum(weights * f_vals, axis=0)
        bkd.assert_allclose(
            bkd.asarray([integral]),
            bkd.asarray([2.0 / 3.0]),
            rtol=1e-12,
        )

    def test_mean_mixed_monomial(self, bkd) -> None:
        """E[x^2*y^2] = 1/9 on [-1,1]^2 via direct weight summation."""
        fitter = _make_fitter(level=3, bkd=bkd)
        samples = fitter.get_samples()
        weights = fitter.get_quadrature_weights()
        x, y = samples[0, :], samples[1, :]
        f_vals = x**2 * y**2
        integral = bkd.sum(weights * f_vals, axis=0)
        bkd.assert_allclose(
            bkd.asarray([integral]),
            bkd.asarray([1.0 / 9.0]),
            rtol=1e-12,
        )

    def test_odd_function_zero(self, bkd) -> None:
        """Odd function integrates to zero on symmetric domain."""
        fitter = _make_fitter(level=3, bkd=bkd)
        samples = fitter.get_samples()
        weights = fitter.get_quadrature_weights()
        x, y = samples[0, :], samples[1, :]
        f_vals = x + y
        integral = bkd.sum(weights * f_vals, axis=0)
        bkd.assert_allclose(
            bkd.asarray([integral]),
            bkd.asarray([0.0]),
            atol=1e-14,
        )

    def test_consistency_with_surrogate_mean(self, bkd) -> None:
        """Direct quadrature matches CombinationSurrogate.mean()."""
        fitter = _make_fitter(level=3, bkd=bkd)
        samples = fitter.get_samples()
        weights = fitter.get_quadrature_weights()

        x, y = samples[0, :], samples[1, :]
        f_vals = x**2 * y + x * y**2
        values = bkd.reshape(f_vals, (1, -1))

        # Via surrogate
        result = fitter.fit(values)
        surrogate_mean = result.surrogate.mean()

        # Via direct quadrature
        quad_mean = bkd.sum(weights * f_vals, axis=0)

        bkd.assert_allclose(
            bkd.asarray([quad_mean]),
            surrogate_mean,
            rtol=1e-12,
        )


class TestParameterizedSparseGridRule:
    """Tests for ParameterizedIsotropicSparseGridQuadratureRule."""

    def test_rule_interface(self, bkd) -> None:
        """Rule returns same samples/weights as building fitter manually."""
        factory = _make_factory(bkd)
        rule = ParameterizedIsotropicSparseGridQuadratureRule(bkd, factory)
        level = 2

        samples_rule, weights_rule = rule(level)

        fitter = IsotropicSparseGridFitter(bkd, factory, level)
        samples_fitter = fitter.get_samples()
        weights_fitter = fitter.get_quadrature_weights()

        bkd.assert_allclose(samples_rule, samples_fitter, rtol=1e-14)
        bkd.assert_allclose(weights_rule, weights_fitter, rtol=1e-14)

    def test_polynomial_exactness_via_rule(self, bkd) -> None:
        """E[x^2 + y^2] = 2/3 computed through the rule interface."""
        factory = _make_factory(bkd)
        rule = ParameterizedIsotropicSparseGridQuadratureRule(bkd, factory)
        samples, weights = rule(2)
        x, y = samples[0, :], samples[1, :]
        f_vals = x**2 + y**2
        integral = bkd.sum(weights * f_vals, axis=0)
        bkd.assert_allclose(
            bkd.asarray([integral]),
            bkd.asarray([2.0 / 3.0]),
            rtol=1e-12,
        )

    def test_nvars(self, bkd) -> None:
        """nvars() returns correct value."""
        factory = _make_factory(bkd, nvars=3)
        rule = ParameterizedIsotropicSparseGridQuadratureRule(bkd, factory)
        assert rule.nvars() == 3

    def test_increasing_samples_with_level(self, bkd) -> None:
        """Higher level produces more quadrature points."""
        factory = _make_factory(bkd)
        rule = ParameterizedIsotropicSparseGridQuadratureRule(bkd, factory)
        _, w1 = rule(1)
        _, w2 = rule(2)
        assert len(w2) > len(w1)

    def test_fewer_samples_than_tensor_product(self, bkd) -> None:
        """Sparse grid has fewer points than tensor product at higher dimension."""
        from pyapprox.surrogates.quadrature import (
            gauss_quadrature_rule,
        )
        from pyapprox.surrogates.quadrature.tensor_product import (
            ParameterizedTensorProductQuadratureRule,
        )

        marginal = UniformMarginal(-1.0, 1.0, bkd)
        growth = LinearGrowthRule(scale=1, shift=1)
        nvars = 5
        level = 3

        # Sparse grid
        factory = _make_factory(bkd, nvars=nvars)
        sg_rule = ParameterizedIsotropicSparseGridQuadratureRule(bkd, factory)
        sg_samples, _ = sg_rule(level)

        # Tensor product: npts = growth(level) = 4 per dim, 4^5 = 1024
        def univar_rule(npts):
            return gauss_quadrature_rule(marginal, npts, bkd)

        tp_rule = ParameterizedTensorProductQuadratureRule(
            bkd, [univar_rule] * nvars, growth,
        )
        tp_samples, _ = tp_rule(level)

        assert sg_samples.shape[1] < tp_samples.shape[1]

    def test_satisfies_protocol(self, bkd) -> None:
        """Rule satisfies ParameterizedQuadratureRuleProtocol."""
        from pyapprox.surrogates.quadrature.protocols import (
            ParameterizedQuadratureRuleProtocol,
        )

        factory = _make_factory(bkd)
        rule = ParameterizedIsotropicSparseGridQuadratureRule(bkd, factory)
        assert isinstance(rule, ParameterizedQuadratureRuleProtocol)
