"""
Tests for kernel variance scaling in GP statistics.

Verifies that PolynomialScaling * SeparableProductKernel is correctly
handled: decomposition extracts base kernel and s², and statistics
formulas properly apply variance scaling.
"""

import math
import unittest
from typing import Generic, Any, Dict, List

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.protocols import Backend, Array
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401
from pyapprox.typing.surrogates.kernels.matern import (
    SquaredExponentialKernel,
)
from pyapprox.typing.surrogates.kernels.composition import (
    SeparableProductKernel,
)
from pyapprox.typing.surrogates.kernels.scalings import PolynomialScaling
from pyapprox.typing.surrogates.gaussianprocess import ExactGaussianProcess
from pyapprox.typing.probability.univariate.uniform import UniformMarginal
from pyapprox.typing.surrogates.sparsegrids.basis_factory import (
    create_basis_factories,
)
from pyapprox.typing.surrogates.gaussianprocess.statistics import (
    SeparableKernelIntegralCalculator,
    GaussianProcessStatistics,
)
from pyapprox.typing.surrogates.gaussianprocess.statistics.sensitivity import (
    GaussianProcessSensitivity,
)
from pyapprox.typing.surrogates.gaussianprocess.statistics.decompose import (
    _decompose_kernel,
)
from pyapprox.typing.surrogates.kernels.protocols import (
    SeparableKernelProtocol,
)


def _create_quadrature_bases(
    marginals: List[Any], nquad_points: int, bkd: Backend[Array]
) -> List[Any]:
    """Helper to create quadrature bases from marginals."""
    factories = create_basis_factories(marginals, bkd, "gauss")
    bases = [f.create_basis() for f in factories]
    for b in bases:
        b.set_nterms(nquad_points)
    return bases


def _create_separable_kernel(
    length_scales: List[float], bkd: Backend[Array]
) -> SeparableProductKernel[Array]:
    """Create a SeparableProductKernel from length scales."""
    kernels_1d = [
        SquaredExponentialKernel([ls], (0.1, 10.0), 1, bkd)
        for ls in length_scales
    ]
    return SeparableProductKernel(kernels_1d, bkd)


def _create_scaled_kernel(
    s_value: float,
    length_scales: List[float],
    bkd: Backend[Array],
) -> Any:
    """Create PolynomialScaling([s]) * SeparableProductKernel."""
    base = _create_separable_kernel(length_scales, bkd)
    nvars = len(length_scales)
    scaling = PolynomialScaling(
        [s_value], (0.01, 100.0), bkd, nvars=nvars, fixed=False
    )
    return scaling * base


def _fit_gp(
    kernel: Any, nvars: int, X_train: Any, y_train: Any, bkd: Backend[Array]
) -> ExactGaussianProcess[Array]:
    """Create and fit a GP with the given kernel."""
    gp: ExactGaussianProcess[Array] = ExactGaussianProcess(
        kernel, nvars=nvars, bkd=bkd, nugget=1e-6
    )
    gp.hyp_list().set_all_inactive()
    gp.fit(X_train, y_train)
    return gp


def _create_stats(
    gp: Any, marginals: List[Any], nquad: int, bkd: Backend[Array]
) -> GaussianProcessStatistics[Array]:
    """Create GaussianProcessStatistics from a fitted GP."""
    bases = _create_quadrature_bases(marginals, nquad, bkd)
    calc: SeparableKernelIntegralCalculator[Array] = (
        SeparableKernelIntegralCalculator(gp, bases, marginals, bkd=bkd)
    )
    return GaussianProcessStatistics(gp, calc)


def _create_sensitivity(
    gp: Any, marginals: List[Any], nquad: int, bkd: Backend[Array]
) -> GaussianProcessSensitivity[Array]:
    """Create GaussianProcessSensitivity from a fitted GP."""
    bases = _create_quadrature_bases(marginals, nquad, bkd)
    calc: SeparableKernelIntegralCalculator[Array] = (
        SeparableKernelIntegralCalculator(gp, bases, marginals, bkd=bkd)
    )
    stats = GaussianProcessStatistics(gp, calc)
    return GaussianProcessSensitivity(stats)


# ===========================================================================
# Test 1: Kernel Decomposition
# ===========================================================================


class TestKernelDecomposition(Generic[Array], unittest.TestCase):
    """Unit tests for _decompose_kernel()."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_bare_separable_returns_unit_variance(self) -> None:
        """Bare SeparableProductKernel decomposes to (kernel, 1.0)."""
        kernel = _create_separable_kernel([1.0, 0.5], self._bkd)
        base, s2 = _decompose_kernel(kernel, self._bkd)
        self.assertIsInstance(base, SeparableKernelProtocol)
        self._bkd.assert_allclose(s2, self._bkd.asarray(1.0))
        self.assertIs(base, kernel)

    def test_scaled_separable_extracts_s_squared(self) -> None:
        """PolynomialScaling([s]) * SeparableKernel → (base, s²)."""
        scaled = _create_scaled_kernel(3.0, [1.0, 0.5], self._bkd)
        base, s2 = _decompose_kernel(scaled, self._bkd)
        self.assertIsInstance(base, SeparableKernelProtocol)
        self._bkd.assert_allclose(s2, self._bkd.asarray(9.0))

    def test_reversed_order(self) -> None:
        """SeparableKernel * PolynomialScaling([s]) → (base, s²)."""
        base_kernel = _create_separable_kernel([1.0], self._bkd)
        scaling = PolynomialScaling(
            [2.0], (0.01, 100.0), self._bkd, nvars=1, fixed=False
        )
        # Reversed: base * scaling
        product = base_kernel * scaling
        base, s2 = _decompose_kernel(product, self._bkd)
        self.assertIsInstance(base, SeparableKernelProtocol)
        self._bkd.assert_allclose(s2, self._bkd.asarray(4.0))

    def test_non_constant_scaling_raises(self) -> None:
        """Non-constant PolynomialScaling raises TypeError."""
        base_kernel = _create_separable_kernel([1.0, 0.5], self._bkd)
        # Linear scaling (degree 1, 3 coefficients for 2 vars)
        scaling = PolynomialScaling(
            [1.0, 0.5, 0.3], (0.01, 100.0), self._bkd, fixed=False
        )
        product = scaling * base_kernel
        with self.assertRaises(TypeError):
            _decompose_kernel(product, self._bkd)

    def test_non_separable_base_raises(self) -> None:
        """Non-separable base kernel raises TypeError."""
        # A plain SE kernel with nvars=2 satisfies SeparableKernelProtocol,
        # so test with something that doesn't. Use a raw ProductKernel of
        # two non-protocol kernels.
        from pyapprox.typing.surrogates.kernels.composition import (
            ProductKernel,
        )

        k1 = PolynomialScaling(
            [2.0], (0.01, 100.0), self._bkd, nvars=2, fixed=False
        )
        k2 = PolynomialScaling(
            [3.0], (0.01, 100.0), self._bkd, nvars=2, fixed=False
        )
        product = k1 * k2
        with self.assertRaises(TypeError):
            _decompose_kernel(product, self._bkd)


class TestKernelDecompositionNumpy(TestKernelDecomposition[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestKernelDecompositionTorch(TestKernelDecomposition[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


# ===========================================================================
# Test 2: Kernel Diagonal
# ===========================================================================


class TestKernelDiagonal(Generic[Array], unittest.TestCase):
    """Verify k(x,x) = s² for scaled kernel, 1.0 for unscaled."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_unscaled_diagonal_is_one(self) -> None:
        """Unscaled separable kernel has k(x,x) = 1."""
        kernel = _create_separable_kernel([1.0, 0.5], self._bkd)
        x = self._bkd.array([[0.3], [0.7]])
        kxx = kernel(x, x)
        self._bkd.assert_allclose(kxx, self._bkd.array([[1.0]]))

    def test_scaled_diagonal_is_s_squared(self) -> None:
        """Scaled kernel has k(x,x) = s²."""
        s = 2.5
        kernel = _create_scaled_kernel(s, [1.0, 0.5], self._bkd)
        x = self._bkd.array([[0.3], [0.7]])
        kxx = kernel(x, x)
        self._bkd.assert_allclose(kxx, self._bkd.array([[s * s]]))


class TestKernelDiagonalNumpy(TestKernelDiagonal[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestKernelDiagonalTorch(TestKernelDiagonal[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


# ===========================================================================
# Test 3: Unit Scaling Equivalence
# ===========================================================================


class TestUnitScalingEquivalence(Generic[Array], unittest.TestCase):
    """PolynomialScaling([1.0]) * base gives identical statistics to base."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        np.random.seed(42)

        length_scales = [1.0, 0.5]
        nvars = 2
        nquad = 30

        # Training data
        X_train = self._bkd.array(np.random.rand(nvars, 10) * 2 - 1)
        y_train = self._bkd.reshape(
            self._bkd.sin(math.pi * X_train[0, :])
            * self._bkd.cos(math.pi * X_train[1, :]),
            (1, -1),
        )

        self._marginals: List[Any] = [
            UniformMarginal(-1.0, 1.0, self._bkd),
            UniformMarginal(-1.0, 1.0, self._bkd),
        ]

        # Unscaled GP
        kernel_bare = _create_separable_kernel(length_scales, self._bkd)
        gp_bare = _fit_gp(kernel_bare, nvars, X_train, y_train, self._bkd)
        self._stats_bare = _create_stats(
            gp_bare, self._marginals, nquad, self._bkd
        )
        self._sens_bare = _create_sensitivity(
            gp_bare, self._marginals, nquad, self._bkd
        )

        # Scaled with s=1.0 (should be equivalent)
        kernel_unit = _create_scaled_kernel(1.0, length_scales, self._bkd)
        gp_unit = _fit_gp(kernel_unit, nvars, X_train, y_train, self._bkd)
        self._stats_unit = _create_stats(
            gp_unit, self._marginals, nquad, self._bkd
        )
        self._sens_unit = _create_sensitivity(
            gp_unit, self._marginals, nquad, self._bkd
        )

    def test_mean_of_mean(self) -> None:
        self._bkd.assert_allclose(
            self._bkd.asarray([self._stats_unit.mean_of_mean()]),
            self._bkd.asarray([self._stats_bare.mean_of_mean()]),
            rtol=1e-10,
        )

    def test_variance_of_mean(self) -> None:
        self._bkd.assert_allclose(
            self._bkd.asarray([self._stats_unit.variance_of_mean()]),
            self._bkd.asarray([self._stats_bare.variance_of_mean()]),
            rtol=1e-10,
        )

    def test_mean_of_variance(self) -> None:
        self._bkd.assert_allclose(
            self._bkd.asarray([self._stats_unit.mean_of_variance()]),
            self._bkd.asarray([self._stats_bare.mean_of_variance()]),
            rtol=1e-10,
        )

    def test_variance_of_variance(self) -> None:
        self._bkd.assert_allclose(
            self._bkd.asarray([self._stats_unit.variance_of_variance()]),
            self._bkd.asarray([self._stats_bare.variance_of_variance()]),
            rtol=1e-10,
        )

    def test_main_effect_indices(self) -> None:
        bare = self._sens_bare.main_effect_indices()
        unit = self._sens_unit.main_effect_indices()
        for dim in bare:
            self._bkd.assert_allclose(
                self._bkd.asarray([unit[dim]]),
                self._bkd.asarray([bare[dim]]),
                rtol=1e-10,
            )

    def test_total_effect_indices(self) -> None:
        bare = self._sens_bare.total_effect_indices()
        unit = self._sens_unit.total_effect_indices()
        for dim in bare:
            self._bkd.assert_allclose(
                self._bkd.asarray([unit[dim]]),
                self._bkd.asarray([bare[dim]]),
                rtol=1e-10,
            )


class TestUnitScalingEquivalenceNumpy(
    TestUnitScalingEquivalence[NDArray[Any]]
):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestUnitScalingEquivalenceTorch(
    TestUnitScalingEquivalence[torch.Tensor]
):
    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


# ===========================================================================
# Test 4: GP Prior Variance at Far Points
# ===========================================================================


class TestGPPredictions(Generic[Array], unittest.TestCase):
    """Prior variance at far-away points should approximate s²."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        np.random.seed(42)

    def test_prior_variance_scaled(self) -> None:
        """GP prior std at far points ~ s."""
        s = 3.0
        nvars = 2
        kernel = _create_scaled_kernel(s, [0.1, 0.1], self._bkd)
        gp: ExactGaussianProcess[Array] = ExactGaussianProcess(
            kernel, nvars=nvars, bkd=self._bkd, nugget=1e-6
        )
        gp.hyp_list().set_all_inactive()

        # Train near origin with short length scales
        X_train = self._bkd.array([[0.0, 0.01], [0.0, 0.01]])
        y_train = self._bkd.array([[1.0, 1.1]])
        gp.fit(X_train, y_train)

        # Predict far from training data
        X_far = self._bkd.array([[10.0], [10.0]])
        std = gp.predict_std(X_far)

        # Prior std should be close to s = 3.0
        self._bkd.assert_allclose(std, self._bkd.array([[s]]), rtol=0.01)

    def test_prior_variance_unscaled(self) -> None:
        """Unscaled GP prior std at far points ~ 1.0."""
        nvars = 2
        kernel = _create_separable_kernel([0.1, 0.1], self._bkd)
        gp: ExactGaussianProcess[Array] = ExactGaussianProcess(
            kernel, nvars=nvars, bkd=self._bkd, nugget=1e-6
        )
        gp.hyp_list().set_all_inactive()

        X_train = self._bkd.array([[0.0, 0.01], [0.0, 0.01]])
        y_train = self._bkd.array([[1.0, 1.1]])
        gp.fit(X_train, y_train)

        X_far = self._bkd.array([[10.0], [10.0]])
        std = gp.predict_std(X_far)

        self._bkd.assert_allclose(std, self._bkd.array([[1.0]]), rtol=0.01)


class TestGPPredictionsNumpy(TestGPPredictions[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGPPredictionsTorch(TestGPPredictions[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
