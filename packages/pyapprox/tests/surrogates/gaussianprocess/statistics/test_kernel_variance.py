"""
Tests for kernel variance scaling in GP statistics.

Verifies that PolynomialScaling * SeparableProductKernel is correctly
handled: decomposition extracts base kernel and s², and statistics
formulas properly apply variance scaling.
"""

import math

import numpy as np
import pytest

from pyapprox.probability.univariate.uniform import UniformMarginal
from pyapprox.surrogates.gaussianprocess import ExactGaussianProcess
from pyapprox.surrogates.gaussianprocess.statistics import (
    GaussianProcessStatistics,
    SeparableKernelIntegralCalculator,
)
from pyapprox.surrogates.gaussianprocess.statistics.decompose import (
    _decompose_kernel,
)
from pyapprox.surrogates.gaussianprocess.statistics.sensitivity import (
    GaussianProcessSensitivity,
)
from pyapprox.surrogates.kernels.composition import (
    SeparableProductKernel,
)
from pyapprox.surrogates.kernels.matern import (
    SquaredExponentialKernel,
)
from pyapprox.surrogates.kernels.protocols import (
    SeparableKernelProtocol,
)
from pyapprox.surrogates.kernels.scalings import PolynomialScaling
from pyapprox.surrogates.sparsegrids.basis_factory import (
    create_basis_factories,
)


def _create_quadrature_bases(
    marginals, nquad_points, bkd,
):
    """Helper to create quadrature bases from marginals."""
    factories = create_basis_factories(marginals, bkd, "gauss")
    bases = [f.create_basis() for f in factories]
    for b in bases:
        b.set_nterms(nquad_points)
    return bases


def _create_separable_kernel(
    length_scales, bkd,
):
    """Create a SeparableProductKernel from length scales."""
    kernels_1d = [
        SquaredExponentialKernel([ls], (0.1, 10.0), 1, bkd) for ls in length_scales
    ]
    return SeparableProductKernel(kernels_1d, bkd)


def _create_scaled_kernel(
    s_value,
    length_scales,
    bkd,
):
    """Create PolynomialScaling([s]) * SeparableProductKernel."""
    base = _create_separable_kernel(length_scales, bkd)
    nvars = len(length_scales)
    scaling = PolynomialScaling([s_value], (0.01, 100.0), bkd, nvars=nvars, fixed=False)
    return scaling * base


def _fit_gp(
    kernel, nvars, X_train, y_train, bkd,
):
    """Create and fit a GP with the given kernel."""
    gp = ExactGaussianProcess(
        kernel, nvars=nvars, bkd=bkd, nugget=1e-6
    )
    gp.hyp_list().set_all_inactive()
    gp.fit(X_train, y_train)
    return gp


def _create_stats(
    gp, marginals, nquad, bkd,
):
    """Create GaussianProcessStatistics from a fitted GP."""
    bases = _create_quadrature_bases(marginals, nquad, bkd)
    calc = SeparableKernelIntegralCalculator(
        gp, bases, marginals, bkd=bkd
    )
    return GaussianProcessStatistics(gp, calc)


def _create_sensitivity(
    gp, marginals, nquad, bkd,
):
    """Create GaussianProcessSensitivity from a fitted GP."""
    bases = _create_quadrature_bases(marginals, nquad, bkd)
    calc = SeparableKernelIntegralCalculator(
        gp, bases, marginals, bkd=bkd
    )
    stats = GaussianProcessStatistics(gp, calc)
    return GaussianProcessSensitivity(stats)


# ===========================================================================
# Test 1: Kernel Decomposition
# ===========================================================================


class TestKernelDecomposition:
    """Unit tests for _decompose_kernel()."""

    def test_bare_separable_returns_unit_variance(self, bkd) -> None:
        """Bare SeparableProductKernel decomposes to (kernel, 1.0)."""
        kernel = _create_separable_kernel([1.0, 0.5], bkd)
        base, s2 = _decompose_kernel(kernel, bkd)
        assert isinstance(base, SeparableKernelProtocol)
        bkd.assert_allclose(s2, bkd.asarray(1.0))
        assert base is kernel

    def test_scaled_separable_extracts_s_squared(self, bkd) -> None:
        """PolynomialScaling([s]) * SeparableKernel -> (base, s**2)."""
        scaled = _create_scaled_kernel(3.0, [1.0, 0.5], bkd)
        base, s2 = _decompose_kernel(scaled, bkd)
        assert isinstance(base, SeparableKernelProtocol)
        bkd.assert_allclose(s2, bkd.asarray(9.0))

    def test_reversed_order(self, bkd) -> None:
        """SeparableKernel * PolynomialScaling([s]) -> (base, s**2)."""
        base_kernel = _create_separable_kernel([1.0], bkd)
        scaling = PolynomialScaling(
            [2.0], (0.01, 100.0), bkd, nvars=1, fixed=False
        )
        # Reversed: base * scaling
        product = base_kernel * scaling
        base, s2 = _decompose_kernel(product, bkd)
        assert isinstance(base, SeparableKernelProtocol)
        bkd.assert_allclose(s2, bkd.asarray(4.0))

    def test_non_constant_scaling_raises(self, bkd) -> None:
        """Non-constant PolynomialScaling raises TypeError."""
        base_kernel = _create_separable_kernel([1.0, 0.5], bkd)
        # Linear scaling (degree 1, 3 coefficients for 2 vars)
        scaling = PolynomialScaling(
            [1.0, 0.5, 0.3], (0.01, 100.0), bkd, fixed=False
        )
        product = scaling * base_kernel
        with pytest.raises(TypeError):
            _decompose_kernel(product, bkd)

    def test_non_separable_base_raises(self, bkd) -> None:
        """Non-separable base kernel raises TypeError."""
        # A plain SE kernel with nvars=2 satisfies SeparableKernelProtocol,
        # so test with something that doesn't. Use a raw ProductKernel of
        # two non-protocol kernels.

        k1 = PolynomialScaling([2.0], (0.01, 100.0), bkd, nvars=2, fixed=False)
        k2 = PolynomialScaling([3.0], (0.01, 100.0), bkd, nvars=2, fixed=False)
        product = k1 * k2
        with pytest.raises(TypeError):
            _decompose_kernel(product, bkd)


# ===========================================================================
# Test 2: Kernel Diagonal
# ===========================================================================


class TestKernelDiagonal:
    """Verify k(x,x) = s**2 for scaled kernel, 1.0 for unscaled."""

    def test_unscaled_diagonal_is_one(self, bkd) -> None:
        """Unscaled separable kernel has k(x,x) = 1."""
        kernel = _create_separable_kernel([1.0, 0.5], bkd)
        x = bkd.array([[0.3], [0.7]])
        kxx = kernel(x, x)
        bkd.assert_allclose(kxx, bkd.array([[1.0]]))

    def test_scaled_diagonal_is_s_squared(self, bkd) -> None:
        """Scaled kernel has k(x,x) = s**2."""
        s = 2.5
        kernel = _create_scaled_kernel(s, [1.0, 0.5], bkd)
        x = bkd.array([[0.3], [0.7]])
        kxx = kernel(x, x)
        bkd.assert_allclose(kxx, bkd.array([[s * s]]))


# ===========================================================================
# Test 3: Unit Scaling Equivalence
# ===========================================================================


class TestUnitScalingEquivalence:
    """PolynomialScaling([1.0]) * base gives identical statistics to base."""

    def _setup(self, bkd):
        np.random.seed(42)

        length_scales = [1.0, 0.5]
        nvars = 2
        nquad = 30

        # Training data
        X_train = bkd.array(np.random.rand(nvars, 10) * 2 - 1)
        y_train = bkd.reshape(
            bkd.sin(math.pi * X_train[0, :])
            * bkd.cos(math.pi * X_train[1, :]),
            (1, -1),
        )

        marginals = [
            UniformMarginal(-1.0, 1.0, bkd),
            UniformMarginal(-1.0, 1.0, bkd),
        ]

        # Unscaled GP
        kernel_bare = _create_separable_kernel(length_scales, bkd)
        gp_bare = _fit_gp(kernel_bare, nvars, X_train, y_train, bkd)
        stats_bare = _create_stats(gp_bare, marginals, nquad, bkd)
        sens_bare = _create_sensitivity(
            gp_bare, marginals, nquad, bkd
        )

        # Scaled with s=1.0 (should be equivalent)
        kernel_unit = _create_scaled_kernel(1.0, length_scales, bkd)
        gp_unit = _fit_gp(kernel_unit, nvars, X_train, y_train, bkd)
        stats_unit = _create_stats(gp_unit, marginals, nquad, bkd)
        sens_unit = _create_sensitivity(
            gp_unit, marginals, nquad, bkd
        )

        return stats_bare, stats_unit, sens_bare, sens_unit

    def test_input_mean_of_posterior_mean(self, bkd) -> None:
        stats_bare, stats_unit, _, _ = self._setup(bkd)
        bkd.assert_allclose(
            bkd.asarray([stats_unit.input_mean_of_posterior_mean()]),
            bkd.asarray([stats_bare.input_mean_of_posterior_mean()]),
            rtol=1e-10,
        )

    def test_gp_variance_of_posterior_mean(self, bkd) -> None:
        stats_bare, stats_unit, _, _ = self._setup(bkd)
        bkd.assert_allclose(
            bkd.asarray([stats_unit.gp_variance_of_posterior_mean()]),
            bkd.asarray([stats_bare.gp_variance_of_posterior_mean()]),
            rtol=1e-10,
        )

    def test_input_mean_of_posterior_variance(self, bkd) -> None:
        stats_bare, stats_unit, _, _ = self._setup(bkd)
        bkd.assert_allclose(
            bkd.asarray([stats_unit.input_mean_of_posterior_variance()]),
            bkd.asarray([stats_bare.input_mean_of_posterior_variance()]),
            rtol=1e-10,
        )

    def test_gp_variance_of_posterior_variance(self, bkd) -> None:
        stats_bare, stats_unit, _, _ = self._setup(bkd)
        bkd.assert_allclose(
            bkd.asarray([stats_unit.gp_variance_of_posterior_variance()]),
            bkd.asarray([stats_bare.gp_variance_of_posterior_variance()]),
            rtol=1e-10,
        )

    def test_main_effect_indices(self, bkd) -> None:
        _, _, sens_bare, sens_unit = self._setup(bkd)
        bare = sens_bare.main_effect_indices()
        unit = sens_unit.main_effect_indices()
        for dim in bare:
            bkd.assert_allclose(
                bkd.asarray([unit[dim]]),
                bkd.asarray([bare[dim]]),
                rtol=1e-10,
            )

    def test_total_effect_indices(self, bkd) -> None:
        _, _, sens_bare, sens_unit = self._setup(bkd)
        bare = sens_bare.total_effect_indices()
        unit = sens_unit.total_effect_indices()
        for dim in bare:
            bkd.assert_allclose(
                bkd.asarray([unit[dim]]),
                bkd.asarray([bare[dim]]),
                rtol=1e-10,
            )


# ===========================================================================
# Test 4: GP Prior Variance at Far Points
# ===========================================================================


class TestGPPredictions:
    """Prior variance at far-away points should approximate s**2."""

    def test_prior_variance_scaled(self, bkd) -> None:
        """GP prior std at far points ~ s."""
        np.random.seed(42)
        s = 3.0
        nvars = 2
        kernel = _create_scaled_kernel(s, [0.1, 0.1], bkd)
        gp = ExactGaussianProcess(
            kernel, nvars=nvars, bkd=bkd, nugget=1e-6
        )
        gp.hyp_list().set_all_inactive()

        # Train near origin with short length scales
        X_train = bkd.array([[0.0, 0.01], [0.0, 0.01]])
        y_train = bkd.array([[1.0, 1.1]])
        gp.fit(X_train, y_train)

        # Predict far from training data
        X_far = bkd.array([[10.0], [10.0]])
        std = gp.predict_std(X_far)

        # Prior std should be close to s = 3.0
        bkd.assert_allclose(std, bkd.array([[s]]), rtol=0.01)

    def test_prior_variance_unscaled(self, bkd) -> None:
        """Unscaled GP prior std at far points ~ 1.0."""
        np.random.seed(42)
        nvars = 2
        kernel = _create_separable_kernel([0.1, 0.1], bkd)
        gp = ExactGaussianProcess(
            kernel, nvars=nvars, bkd=bkd, nugget=1e-6
        )
        gp.hyp_list().set_all_inactive()

        X_train = bkd.array([[0.0, 0.01], [0.0, 0.01]])
        y_train = bkd.array([[1.0, 1.1]])
        gp.fit(X_train, y_train)

        X_far = bkd.array([[10.0], [10.0]])
        std = gp.predict_std(X_far)

        bkd.assert_allclose(std, bkd.array([[1.0]]), rtol=0.01)
