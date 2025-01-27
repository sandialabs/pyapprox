"""
Tests for input affine transformation classes and GP integration.

Tests verify:
- InputStandardScaler computes correct mean/std from data
- Zero-variance edge case (constant dimension) is handled
- transform() and inverse_transform() are inverses
- IdentityInputTransform is a no-op
- InputBoundsScaler maps corners correctly
- GP with input_transform returns correct predictions
- GP predictions match manual scaling of raw GP predictions
- GP jacobian applies chain rule correctly
- GP HVP applies chain rule correctly
- GP statistics with both input and output transforms match MC
"""

import math
import unittest
from itertools import product as iterproduct
from typing import Generic, Any, List

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.protocols import Backend, Array
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401
from pyapprox.typing.util.test_utils import slow_test

from pyapprox.typing.surrogates.gaussianprocess.input_transform import (
    InputStandardScaler,
    InputBoundsScaler,
    IdentityInputTransform,
)
from pyapprox.typing.surrogates.gaussianprocess.output_transform import (
    OutputStandardScaler,
)
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


_NUGGET = 1e-10


def _create_kernel(
    bkd: Backend[Array],
) -> SeparableProductKernel[Array]:
    k1 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, bkd)
    k2 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, bkd)
    return SeparableProductKernel([k1, k2], bkd)


def _make_training_data(
    bkd: Backend[Array], nvars: int = 2, n_train: int = 8
) -> tuple[Any, Any]:
    np.random.seed(42)
    X_train = bkd.array(np.random.rand(nvars, n_train) * 2 - 1)
    y_train = bkd.reshape(
        10.0 * bkd.sin(math.pi * X_train[0, :]) + 5.0,
        (1, -1),
    )
    return X_train, y_train


def _make_training_data_different_scales(
    bkd: Backend[Array],
) -> tuple[Any, Any]:
    """Training data with very different input scales."""
    np.random.seed(42)
    # x1 in [0, 1000], x2 in [0, 0.01]
    X_raw = np.random.rand(2, 8)
    X_raw[0, :] *= 1000.0
    X_raw[1, :] *= 0.01
    X_train = bkd.array(X_raw)
    y_train = bkd.reshape(
        bkd.sin(X_train[0, :] / 1000.0 * math.pi)
        + 100.0 * X_train[1, :],
        (1, -1),
    )
    return X_train, y_train


# ===================================================================
# Unit tests for transform classes
# ===================================================================


class TestInputTransformUnit(Generic[Array], unittest.TestCase):
    """Unit tests for input transform classes."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_identity_is_noop(self) -> None:
        """IdentityInputTransform does not change values."""
        bkd = self._bkd
        identity = IdentityInputTransform(3, bkd)

        z = bkd.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        bkd.assert_allclose(identity.transform(z), z)
        bkd.assert_allclose(identity.inverse_transform(z), z)
        bkd.assert_allclose(identity.jacobian_factor(), bkd.ones(3))
        bkd.assert_allclose(identity.hessian_factor(), bkd.ones((3, 3)))
        self.assertEqual(identity.nvars(), 3)

    def test_standard_scaler_from_data(self) -> None:
        """from_data computes correct mean and std."""
        bkd = self._bkd
        z = bkd.array([[1.0, 3.0, 5.0, 7.0],
                        [10.0, 20.0, 30.0, 40.0]])
        scaler = InputStandardScaler.from_data(z, bkd)

        expected_mean = bkd.mean(z, axis=1)
        expected_std = bkd.std(z, axis=1)

        bkd.assert_allclose(scaler.shift(), expected_mean)
        bkd.assert_allclose(scaler.scale(), expected_std)

    def test_standard_scaler_zero_variance(self) -> None:
        """Constant dimension gets std=1.0."""
        bkd = self._bkd
        z = bkd.array([[5.0, 5.0, 5.0],
                        [1.0, 2.0, 3.0]])
        scaler = InputStandardScaler.from_data(z, bkd)

        bkd.assert_allclose(
            scaler.scale(),
            bkd.array([1.0, bkd.std(z[1:2, :], axis=1)[0]])
        )

    def test_standard_scaler_roundtrip(self) -> None:
        """transform then inverse_transform gives back original."""
        bkd = self._bkd
        np.random.seed(42)
        z = bkd.array(np.random.randn(3, 20) * np.array([[1], [100], [0.01]])
                       + np.array([[0], [50], [0.5]]))
        scaler = InputStandardScaler.from_data(z, bkd)

        z_scaled = scaler.transform(z)
        z_recovered = scaler.inverse_transform(z_scaled)
        bkd.assert_allclose(z_recovered, z, rtol=1e-12)

    def test_standard_scaler_produces_unit_stats(self) -> None:
        """Scaled data has mean≈0, std≈1."""
        bkd = self._bkd
        np.random.seed(42)
        z = bkd.array(np.random.randn(2, 100) * np.array([[10], [0.1]])
                       + np.array([[50], [-3]]))
        scaler = InputStandardScaler.from_data(z, bkd)
        z_scaled = scaler.transform(z)

        bkd.assert_allclose(
            bkd.mean(z_scaled, axis=1), bkd.zeros(2), atol=1e-10
        )
        bkd.assert_allclose(
            bkd.std(z_scaled, axis=1), bkd.ones(2), atol=1e-10
        )

    def test_bounds_scaler_maps_corners(self) -> None:
        """BoundsScaler maps [lb,ub] to [0,1]."""
        bkd = self._bkd
        lb = bkd.array([0.0, -10.0])
        ub = bkd.array([100.0, 10.0])
        scaler = InputBoundsScaler(lb, ub, bkd, target_range=(0.0, 1.0))

        lb_2d = bkd.reshape(lb, (2, 1))
        ub_2d = bkd.reshape(ub, (2, 1))
        bkd.assert_allclose(scaler.transform(lb_2d),
                            bkd.zeros((2, 1)), atol=1e-12)
        bkd.assert_allclose(scaler.transform(ub_2d),
                            bkd.ones((2, 1)), atol=1e-12)

    def test_bounds_scaler_roundtrip(self) -> None:
        """transform then inverse_transform gives back original."""
        bkd = self._bkd
        lb = bkd.array([-5.0, 0.0])
        ub = bkd.array([5.0, 100.0])
        scaler = InputBoundsScaler(lb, ub, bkd, target_range=(-1.0, 1.0))

        np.random.seed(42)
        z = bkd.array(np.random.rand(2, 10))
        z_recovered = scaler.inverse_transform(scaler.transform(z))
        bkd.assert_allclose(z_recovered, z, rtol=1e-12)

    def test_jacobian_factor(self) -> None:
        """jacobian_factor returns 1/scale."""
        bkd = self._bkd
        mean = bkd.array([0.0, 0.0])
        std = bkd.array([2.0, 5.0])
        scaler = InputStandardScaler(mean, std, bkd)

        bkd.assert_allclose(
            scaler.jacobian_factor(), bkd.array([0.5, 0.2])
        )

    def test_hessian_factor(self) -> None:
        """hessian_factor returns outer product of 1/scale."""
        bkd = self._bkd
        mean = bkd.array([0.0, 0.0])
        std = bkd.array([2.0, 5.0])
        scaler = InputStandardScaler(mean, std, bkd)

        inv_std = bkd.array([0.5, 0.2])
        expected = bkd.outer(inv_std, inv_std)
        bkd.assert_allclose(scaler.hessian_factor(), expected)


class TestInputTransformUnitNumpy(TestInputTransformUnit[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestInputTransformUnitTorch(TestInputTransformUnit[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


# ===================================================================
# GP integration tests
# ===================================================================


class TestGPInputTransform(Generic[Array], unittest.TestCase):
    """Test GP predictions with input transform."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_gp_predict_matches_manual_scaling(self) -> None:
        """GP with input_transform matches manually scaling inputs."""
        bkd = self._bkd
        X_train, y_train = _make_training_data(bkd)

        scaler = InputStandardScaler.from_data(X_train, bkd)
        X_scaled = scaler.transform(X_train)

        # GP with transform
        kernel_t = _create_kernel(bkd)
        gp_t = ExactGaussianProcess(
            kernel_t, nvars=2, bkd=bkd, nugget=_NUGGET
        )
        gp_t.hyp_list().set_all_inactive()
        gp_t.fit(X_train, y_train, input_transform=scaler)

        # Raw GP on manually scaled data
        kernel_r = _create_kernel(bkd)
        gp_r = ExactGaussianProcess(
            kernel_r, nvars=2, bkd=bkd, nugget=_NUGGET
        )
        gp_r.hyp_list().set_all_inactive()
        gp_r.fit(X_scaled, y_train)

        # Test points in original space
        np.random.seed(99)
        X_test = bkd.array(np.random.rand(2, 5) * 2 - 1)
        X_test_scaled = scaler.transform(X_test)

        pred_t = gp_t.predict(X_test)
        pred_r = gp_r.predict(X_test_scaled)
        bkd.assert_allclose(pred_t, pred_r, rtol=1e-10)

    def test_gp_predict_std_matches_manual(self) -> None:
        """GP predict_std with input_transform matches manual."""
        bkd = self._bkd
        X_train, y_train = _make_training_data(bkd)

        scaler = InputStandardScaler.from_data(X_train, bkd)
        X_scaled = scaler.transform(X_train)

        kernel_t = _create_kernel(bkd)
        gp_t = ExactGaussianProcess(
            kernel_t, nvars=2, bkd=bkd, nugget=_NUGGET
        )
        gp_t.hyp_list().set_all_inactive()
        gp_t.fit(X_train, y_train, input_transform=scaler)

        kernel_r = _create_kernel(bkd)
        gp_r = ExactGaussianProcess(
            kernel_r, nvars=2, bkd=bkd, nugget=_NUGGET
        )
        gp_r.hyp_list().set_all_inactive()
        gp_r.fit(X_scaled, y_train)

        np.random.seed(99)
        X_test = bkd.array(np.random.rand(2, 5) * 2 - 1)
        X_test_scaled = scaler.transform(X_test)

        std_t = gp_t.predict_std(X_test)
        std_r = gp_r.predict_std(X_test_scaled)
        bkd.assert_allclose(std_t, std_r, rtol=1e-10)

    def test_gp_predict_covariance_matches_manual(self) -> None:
        """GP predict_covariance with input_transform matches manual."""
        bkd = self._bkd
        X_train, y_train = _make_training_data(bkd)

        scaler = InputStandardScaler.from_data(X_train, bkd)
        X_scaled = scaler.transform(X_train)

        kernel_t = _create_kernel(bkd)
        gp_t = ExactGaussianProcess(
            kernel_t, nvars=2, bkd=bkd, nugget=_NUGGET
        )
        gp_t.hyp_list().set_all_inactive()
        gp_t.fit(X_train, y_train, input_transform=scaler)

        kernel_r = _create_kernel(bkd)
        gp_r = ExactGaussianProcess(
            kernel_r, nvars=2, bkd=bkd, nugget=_NUGGET
        )
        gp_r.hyp_list().set_all_inactive()
        gp_r.fit(X_scaled, y_train)

        np.random.seed(99)
        X_test = bkd.array(np.random.rand(2, 3) * 2 - 1)
        X_test_scaled = scaler.transform(X_test)

        cov_t = gp_t.predict_covariance(X_test)
        cov_r = gp_r.predict_covariance(X_test_scaled)
        bkd.assert_allclose(cov_t, cov_r, rtol=1e-10)

    def test_gp_jacobian_matches_manual(self) -> None:
        """GP jacobian with input_transform applies chain rule correctly."""
        bkd = self._bkd
        X_train, y_train = _make_training_data(bkd)

        scaler = InputStandardScaler.from_data(X_train, bkd)
        X_scaled = scaler.transform(X_train)

        kernel_t = _create_kernel(bkd)
        gp_t = ExactGaussianProcess(
            kernel_t, nvars=2, bkd=bkd, nugget=_NUGGET
        )
        gp_t.hyp_list().set_all_inactive()
        gp_t.fit(X_train, y_train, input_transform=scaler)

        kernel_r = _create_kernel(bkd)
        gp_r = ExactGaussianProcess(
            kernel_r, nvars=2, bkd=bkd, nugget=_NUGGET
        )
        gp_r.hyp_list().set_all_inactive()
        gp_r.fit(X_scaled, y_train)

        np.random.seed(99)
        sample = bkd.array(np.random.rand(2, 1) * 2 - 1)
        sample_scaled = scaler.transform(sample)

        jac_t = gp_t.jacobian(sample)
        jac_r = gp_r.jacobian(sample_scaled)

        # Chain rule: ∂f/∂z = (1/σ_z) * ∂f/∂z̃
        jac_factor = scaler.jacobian_factor()  # (nvars,)
        jac_manual = jac_r * jac_factor[None, :]

        bkd.assert_allclose(jac_t, jac_manual, rtol=1e-10)

    def test_gp_jacobian_batch_matches_manual(self) -> None:
        """GP jacobian_batch with input_transform applies chain rule."""
        bkd = self._bkd
        X_train, y_train = _make_training_data(bkd)

        scaler = InputStandardScaler.from_data(X_train, bkd)
        X_scaled = scaler.transform(X_train)

        kernel_t = _create_kernel(bkd)
        gp_t = ExactGaussianProcess(
            kernel_t, nvars=2, bkd=bkd, nugget=_NUGGET
        )
        gp_t.hyp_list().set_all_inactive()
        gp_t.fit(X_train, y_train, input_transform=scaler)

        kernel_r = _create_kernel(bkd)
        gp_r = ExactGaussianProcess(
            kernel_r, nvars=2, bkd=bkd, nugget=_NUGGET
        )
        gp_r.hyp_list().set_all_inactive()
        gp_r.fit(X_scaled, y_train)

        np.random.seed(99)
        samples = bkd.array(np.random.rand(2, 4) * 2 - 1)
        samples_scaled = scaler.transform(samples)

        jac_t = gp_t.jacobian_batch(samples)
        jac_r = gp_r.jacobian_batch(samples_scaled)

        jac_factor = scaler.jacobian_factor()
        jac_manual = jac_r * jac_factor[None, None, :]

        bkd.assert_allclose(jac_t, jac_manual, rtol=1e-10)

    def test_gp_hvp_matches_manual(self) -> None:
        """GP HVP with input_transform applies chain rule correctly."""
        bkd = self._bkd
        X_train, y_train = _make_training_data(bkd)

        scaler = InputStandardScaler.from_data(X_train, bkd)
        X_scaled = scaler.transform(X_train)

        kernel_t = _create_kernel(bkd)
        gp_t = ExactGaussianProcess(
            kernel_t, nvars=2, bkd=bkd, nugget=_NUGGET
        )
        gp_t.hyp_list().set_all_inactive()
        gp_t.fit(X_train, y_train, input_transform=scaler)

        kernel_r = _create_kernel(bkd)
        gp_r = ExactGaussianProcess(
            kernel_r, nvars=2, bkd=bkd, nugget=_NUGGET
        )
        gp_r.hyp_list().set_all_inactive()
        gp_r.fit(X_scaled, y_train)

        # Check hvp is available
        if not hasattr(gp_t, 'hvp'):
            self.skipTest("Kernel does not support HVP")

        np.random.seed(99)
        sample = bkd.array(np.random.rand(2, 1) * 2 - 1)
        vec = bkd.array(np.random.randn(2, 1))

        hvp_t = gp_t.hvp(sample, vec)

        # Manual: (Hv)_j = (1/σ_j) Σ_k H̃_jk (v_k/σ_k)
        # = jac_factor * hvp_scaled(sample_scaled, jac_factor * vec)
        jac_factor = scaler.jacobian_factor()
        vec_scaled = jac_factor[:, None] * vec
        sample_scaled = scaler.transform(sample)
        hvp_r = gp_r.hvp(sample_scaled, vec_scaled)
        hvp_manual = jac_factor[:, None] * hvp_r

        bkd.assert_allclose(hvp_t, hvp_manual, rtol=1e-10)

    def test_gp_no_transform_unchanged(self) -> None:
        """GP without input transform behaves identically."""
        bkd = self._bkd
        X_train, y_train = _make_training_data(bkd)

        gp = ExactGaussianProcess(
            _create_kernel(bkd), nvars=2, bkd=bkd, nugget=_NUGGET
        )
        gp.hyp_list().set_all_inactive()
        gp.fit(X_train, y_train)

        # input_transform should be identity (never None)
        self.assertIsInstance(gp.input_transform(), IdentityInputTransform)

        np.random.seed(99)
        X_test = bkd.array(np.random.rand(2, 5) * 2 - 1)
        pred = gp.predict(X_test)
        self.assertEqual(pred.shape, (1, 5))

    def test_gp_predict_at_training_points(self) -> None:
        """GP with input_transform recovers training y at training X."""
        bkd = self._bkd
        X_train, y_train = _make_training_data(bkd)

        scaler = InputStandardScaler.from_data(X_train, bkd)

        gp = ExactGaussianProcess(
            _create_kernel(bkd), nvars=2, bkd=bkd, nugget=_NUGGET
        )
        gp.hyp_list().set_all_inactive()
        gp.fit(X_train, y_train, input_transform=scaler)

        pred = gp.predict(X_train)
        bkd.assert_allclose(pred, y_train, rtol=1e-4, atol=1e-4)


class TestGPInputTransformNumpy(TestGPInputTransform[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGPInputTransformTorch(TestGPInputTransform[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


# ===================================================================
# Statistics integration test: MC verification with both transforms
# ===================================================================


def _create_quadrature_bases(
    marginals: List[Any], nquad_points: int, bkd: Backend[Array]
) -> List[Any]:
    factories = create_basis_factories(marginals, bkd, "gauss")
    bases = [f.create_basis() for f in factories]
    for b in bases:
        b.set_nterms(nquad_points)
    return bases


class TestStatisticsWithBothTransformsVsMC(
    Generic[Array], unittest.TestCase
):
    """Verify GP statistics with both input and output transforms vs MC.

    This is the strongest integration test: fits a GP with both
    InputStandardScaler and OutputStandardScaler, then compares
    statistics against Monte Carlo estimates.
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        bkd = self._bkd

        # Training data with different input scales
        X_train, y_train = _make_training_data_different_scales(bkd)
        self._nvars = 2
        self._nquad = 30
        self._n_mc = 10000

        # Create scalers
        input_scaler = InputStandardScaler.from_data(X_train, bkd)
        output_scaler = OutputStandardScaler.from_data(y_train, bkd)

        # Fit GP with both transforms
        s = 2.0
        X_scaled = input_scaler.transform(X_train)

        # Create kernel with scaling
        k1 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, bkd)
        k2 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, bkd)
        base_kernel = SeparableProductKernel([k1, k2], bkd)
        scaling = PolynomialScaling(
            [s], (0.01, 100.0), bkd, nvars=2, fixed=False
        )
        kernel = scaling * base_kernel

        gp = ExactGaussianProcess(
            kernel, nvars=2, bkd=bkd, nugget=_NUGGET
        )
        gp.hyp_list().set_all_inactive()
        gp.fit(
            X_train, y_train,
            input_transform=input_scaler,
            output_transform=output_scaler,
        )
        self._gp = gp
        self._input_scaler = input_scaler

        # Marginals defined in original input space
        # The training data has x1 in [0, 1000], x2 in [0, 0.01]
        # Use uniform marginals covering the training range
        x1_min = float(bkd.to_numpy(bkd.min(X_train[0:1, :])))
        x1_max = float(bkd.to_numpy(bkd.max(X_train[0:1, :])))
        x2_min = float(bkd.to_numpy(bkd.min(X_train[1:2, :])))
        x2_max = float(bkd.to_numpy(bkd.max(X_train[1:2, :])))
        self._marginals: List[Any] = [
            UniformMarginal(x1_min, x1_max, bkd),
            UniformMarginal(x2_min, x2_max, bkd),
        ]

        # Create statistics
        bases = _create_quadrature_bases(
            self._marginals, self._nquad, bkd
        )
        calc: SeparableKernelIntegralCalculator[Array] = (
            SeparableKernelIntegralCalculator(
                gp, bases, self._marginals, bkd=bkd
            )
        )
        self._stats = GaussianProcessStatistics(gp, calc)

        # MC samples in original space
        np.random.seed(123)
        mc_x1 = np.random.uniform(x1_min, x1_max, self._n_mc)
        mc_x2 = np.random.uniform(x2_min, x2_max, self._n_mc)
        self._X_mc = bkd.array(np.vstack([mc_x1, mc_x2]))

        # Build reference GP: manually scale X, no input_transform,
        # with scaled-space marginals. This is the ground truth.
        X_scaled = input_scaler.transform(X_train)
        k1_ref = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, bkd)
        k2_ref = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, bkd)
        base_kernel_ref = SeparableProductKernel([k1_ref, k2_ref], bkd)
        scaling_ref = PolynomialScaling(
            [s], (0.01, 100.0), bkd, nvars=2, fixed=False
        )
        kernel_ref = scaling_ref * base_kernel_ref
        gp_ref = ExactGaussianProcess(
            kernel_ref, nvars=2, bkd=bkd, nugget=_NUGGET
        )
        gp_ref.hyp_list().set_all_inactive()
        gp_ref.fit(X_scaled, y_train, output_transform=output_scaler)

        sh = input_scaler.shift()
        sc = input_scaler.scale()
        s_x1_min = (x1_min - float(bkd.to_numpy(sh[0:1]))) / float(
            bkd.to_numpy(sc[0:1])
        )
        s_x1_max = (x1_max - float(bkd.to_numpy(sh[0:1]))) / float(
            bkd.to_numpy(sc[0:1])
        )
        s_x2_min = (x2_min - float(bkd.to_numpy(sh[1:2]))) / float(
            bkd.to_numpy(sc[1:2])
        )
        s_x2_max = (x2_max - float(bkd.to_numpy(sh[1:2]))) / float(
            bkd.to_numpy(sc[1:2])
        )
        marginals_ref: List[Any] = [
            UniformMarginal(s_x1_min, s_x1_max, bkd),
            UniformMarginal(s_x2_min, s_x2_max, bkd),
        ]
        bases_ref = _create_quadrature_bases(
            marginals_ref, self._nquad, bkd
        )
        calc_ref: SeparableKernelIntegralCalculator[Array] = (
            SeparableKernelIntegralCalculator(
                gp_ref, bases_ref, marginals_ref, bkd=bkd
            )
        )
        self._stats_ref = GaussianProcessStatistics(gp_ref, calc_ref)

    def test_mean_of_mean_vs_mc(self) -> None:
        """E[μ(X)] ≈ (1/N) Σ μ(X_i) by Monte Carlo."""
        mu_mc = self._gp.predict(self._X_mc)  # (nqoi, n_mc)
        mc_estimate = self._bkd.mean(mu_mc)
        formula_value = self._stats.mean_of_mean()
        self._bkd.assert_allclose(
            self._bkd.asarray([formula_value]),
            self._bkd.asarray([mc_estimate]),
            rtol=0.05,
        )

    def test_mean_of_mean_matches_manual_scaling(self) -> None:
        """mean_of_mean with input_transform matches manually-scaled GP."""
        self._bkd.assert_allclose(
            self._bkd.asarray([self._stats.mean_of_mean()]),
            self._bkd.asarray([self._stats_ref.mean_of_mean()]),
            rtol=1e-8,
        )

    def test_variance_of_mean_matches_manual_scaling(self) -> None:
        """variance_of_mean with input_transform matches manually-scaled GP."""
        self._bkd.assert_allclose(
            self._bkd.asarray([self._stats.variance_of_mean()]),
            self._bkd.asarray([self._stats_ref.variance_of_mean()]),
            rtol=1e-8,
        )

    def test_mean_of_variance_matches_manual_scaling(self) -> None:
        """mean_of_variance with input_transform matches manually-scaled GP."""
        self._bkd.assert_allclose(
            self._bkd.asarray([self._stats.mean_of_variance()]),
            self._bkd.asarray([self._stats_ref.mean_of_variance()]),
            rtol=1e-8,
        )

    def test_variance_of_variance_matches_manual_scaling(self) -> None:
        """variance_of_variance with input_transform matches manually-scaled GP."""
        self._bkd.assert_allclose(
            self._bkd.asarray([self._stats.variance_of_variance()]),
            self._bkd.asarray([self._stats_ref.variance_of_variance()]),
            rtol=1e-8,
        )


class TestStatisticsWithBothTransformsVsMCNumpy(
    TestStatisticsWithBothTransformsVsMC[NDArray[Any]]
):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestStatisticsWithBothTransformsVsMCTorch(
    TestStatisticsWithBothTransformsVsMC[torch.Tensor]
):
    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
