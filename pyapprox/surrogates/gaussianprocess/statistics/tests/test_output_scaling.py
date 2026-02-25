"""
Tests for output scaling in GP statistics.

Verifies that statistics in original space match the expected transformation
of statistics computed in scaled space:

- mean_of_mean(B) ≈ σ_y * mean_of_mean(A) + μ_y
- variance_of_mean(B) ≈ σ_y² * variance_of_mean(A)
- mean_of_variance(B) ≈ σ_y² * mean_of_variance(A)
- variance_of_variance(B) ≈ σ_y⁴ * variance_of_variance(A)
- conditional_variance(B) ≈ σ_y² * conditional_variance(A)
- Sobol indices are invariant to output scaling

Test strategy: GP A trains on manually pre-scaled data (no transform).
GP B trains on original data with OutputStandardScaler. Same kernel params.
"""

import math
import unittest
from typing import Any, Generic, List

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.probability.univariate.uniform import UniformMarginal
from pyapprox.surrogates.gaussianprocess import ExactGaussianProcess
from pyapprox.surrogates.gaussianprocess.output_transform import (
    OutputStandardScaler,
)
from pyapprox.surrogates.gaussianprocess.statistics import (
    GaussianProcessStatistics,
    SeparableKernelIntegralCalculator,
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
from pyapprox.surrogates.sparsegrids.basis_factory import (
    create_basis_factories,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401

_NUGGET = 1e-10
_NQUAD = 20


def _create_kernel(bkd: Backend[Array]) -> SeparableProductKernel[Array]:
    k1 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, bkd)
    k2 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, bkd)
    return SeparableProductKernel([k1, k2], bkd)


def _make_training_data(
    bkd: Backend[Array],
) -> tuple[Any, Any]:
    np.random.seed(42)
    X = bkd.array(np.random.rand(2, 5) * 2 - 1)
    y = bkd.reshape(
        10.0 * bkd.sin(math.pi * X[0, :]) * bkd.cos(math.pi * X[1, :]) + 50.0,
        (1, -1),
    )
    return X, y


def _create_quadrature_bases(marginals: List[Any], bkd: Backend[Array]) -> List[Any]:
    factories = create_basis_factories(marginals, bkd, "gauss")
    bases = [f.create_basis() for f in factories]
    for b in bases:
        b.set_nterms(_NQUAD)
    return bases


def _create_stats(
    gp: Any, marginals: List[Any], bkd: Backend[Array]
) -> GaussianProcessStatistics[Array]:
    bases = _create_quadrature_bases(marginals, bkd)
    calc: SeparableKernelIntegralCalculator[Array] = SeparableKernelIntegralCalculator(
        gp, bases, marginals, bkd=bkd
    )
    return GaussianProcessStatistics(gp, calc)


def _create_sensitivity(
    gp: Any, marginals: List[Any], bkd: Backend[Array]
) -> GaussianProcessSensitivity[Array]:
    bases = _create_quadrature_bases(marginals, bkd)
    calc: SeparableKernelIntegralCalculator[Array] = SeparableKernelIntegralCalculator(
        gp, bases, marginals, bkd=bkd
    )
    stats = GaussianProcessStatistics(gp, calc)
    return GaussianProcessSensitivity(stats)


class TestOutputScaling(Generic[Array], unittest.TestCase):
    """Verify that statistics transform correctly with output scaling."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        bkd = self._bkd

        X_train, y_train_orig = _make_training_data(bkd)
        self._scaler = OutputStandardScaler.from_data(y_train_orig, bkd)
        y_train_scaled = self._scaler.inverse_transform(y_train_orig)

        marginals = [
            UniformMarginal(-1.0, 1.0, bkd),
            UniformMarginal(-1.0, 1.0, bkd),
        ]

        # GP A: trains on pre-scaled data, no transform
        kernel_a = _create_kernel(bkd)
        gp_a: ExactGaussianProcess[Array] = ExactGaussianProcess(
            kernel_a, nvars=2, bkd=bkd, nugget=_NUGGET
        )
        gp_a.hyp_list().set_all_inactive()
        gp_a.fit(X_train, y_train_scaled)

        # GP B: trains on original data, with transform
        kernel_b = _create_kernel(bkd)
        gp_b: ExactGaussianProcess[Array] = ExactGaussianProcess(
            kernel_b, nvars=2, bkd=bkd, nugget=_NUGGET
        )
        gp_b.hyp_list().set_all_inactive()
        gp_b.fit(X_train, y_train_orig, output_transform=self._scaler)

        self._stats_a = _create_stats(gp_a, marginals, bkd)
        self._stats_b = _create_stats(gp_b, marginals, bkd)
        self._sens_a = _create_sensitivity(gp_a, marginals, bkd)
        self._sens_b = _create_sensitivity(gp_b, marginals, bkd)

    def test_mean_of_mean(self) -> None:
        """mean_of_mean(B) ≈ σ_y * mean_of_mean(A) + μ_y"""
        bkd = self._bkd
        eta_a = self._stats_a.mean_of_mean()
        eta_b = self._stats_b.mean_of_mean()

        sigma_y = self._scaler.scale()[0]
        mu_y = self._scaler.shift()[0]
        expected = sigma_y * eta_a + mu_y

        bkd.assert_allclose(bkd.asarray([eta_b]), bkd.asarray([expected]), rtol=1e-10)

    def test_variance_of_mean(self) -> None:
        """variance_of_mean(B) ≈ σ_y² * variance_of_mean(A)"""
        bkd = self._bkd
        var_a = self._stats_a.variance_of_mean()
        var_b = self._stats_b.variance_of_mean()

        sigma_y_sq = self._scaler.scale()[0] ** 2
        expected = sigma_y_sq * var_a

        bkd.assert_allclose(bkd.asarray([var_b]), bkd.asarray([expected]), rtol=1e-10)

    def test_mean_of_variance(self) -> None:
        """mean_of_variance(B) ≈ σ_y² * mean_of_variance(A)"""
        bkd = self._bkd
        mov_a = self._stats_a.mean_of_variance()
        mov_b = self._stats_b.mean_of_variance()

        sigma_y_sq = self._scaler.scale()[0] ** 2
        expected = sigma_y_sq * mov_a

        bkd.assert_allclose(bkd.asarray([mov_b]), bkd.asarray([expected]), rtol=1e-10)

    def test_variance_of_variance(self) -> None:
        """variance_of_variance(B) ≈ σ_y⁴ * variance_of_variance(A)"""
        bkd = self._bkd
        vov_a = self._stats_a.variance_of_variance()
        vov_b = self._stats_b.variance_of_variance()

        sigma_y_4 = self._scaler.scale()[0] ** 4
        expected = sigma_y_4 * vov_a

        bkd.assert_allclose(bkd.asarray([vov_b]), bkd.asarray([expected]), rtol=1e-8)

    def test_conditional_variance(self) -> None:
        """conditional_variance(B) ≈ σ_y² * conditional_variance(A)"""
        bkd = self._bkd

        # Main effect index for variable 0
        index = bkd.asarray([1.0, 0.0])
        cv_a = self._sens_a.conditional_variance(index)
        cv_b = self._sens_b.conditional_variance(index)

        sigma_y_sq = self._scaler.scale()[0] ** 2
        expected = sigma_y_sq * cv_a

        bkd.assert_allclose(bkd.asarray([cv_b]), bkd.asarray([expected]), rtol=1e-10)

    def test_sobol_indices_invariant(self) -> None:
        """Sobol indices are invariant to output scaling."""
        bkd = self._bkd

        main_a = self._sens_a.main_effect_indices()
        main_b = self._sens_b.main_effect_indices()

        for i in range(2):
            bkd.assert_allclose(
                bkd.asarray([main_a[i]]),
                bkd.asarray([main_b[i]]),
                rtol=1e-10,
            )

        total_a = self._sens_a.total_effect_indices()
        total_b = self._sens_b.total_effect_indices()

        for i in range(2):
            bkd.assert_allclose(
                bkd.asarray([total_a[i]]),
                bkd.asarray([total_b[i]]),
                rtol=1e-10,
            )

    def test_predict_and_stats_same_space(self) -> None:
        """Verify gp.predict() and stats are in the same (original) space."""
        bkd = self._bkd

        # mean_of_mean is the integral of predict() over input space.
        # For a GP fitted to data with mean ~50, mean_of_mean should be ~50.
        eta = self._stats_b.mean_of_mean()

        # The original y has mean ~50, so eta should be in that ballpark.
        # Just verify it's not in scaled space (which would be ~0).
        float(bkd.to_numpy(self._scaler.shift()[0:1])[0])
        self.assertGreater(
            abs(float(bkd.to_numpy(bkd.asarray([eta]))[0])),
            1.0,
            "mean_of_mean appears to be in scaled space, not original",
        )


class TestOutputScalingNumpy(TestOutputScaling[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestOutputScalingTorch(TestOutputScaling[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
