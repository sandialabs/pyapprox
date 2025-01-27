"""
Tests for output affine transformation classes and GP integration.

Tests verify:
- OutputStandardScaler computes correct mean/std from data
- Zero-variance edge case (constant output) is handled
- transform() and inverse_transform() are inverses
- IdentityOutputTransform is a no-op
- GP with output_transform returns predictions in original space
- GP predictions match manual scaling of unscaled GP predictions
"""

import math
import unittest
from typing import Generic, Any

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.protocols import Backend, Array
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401

from pyapprox.typing.surrogates.gaussianprocess.output_transform import (
    OutputStandardScaler,
    IdentityOutputTransform,
)
from pyapprox.typing.surrogates.kernels.matern import (
    SquaredExponentialKernel,
)
from pyapprox.typing.surrogates.kernels.composition import (
    SeparableProductKernel,
)
from pyapprox.typing.surrogates.gaussianprocess import ExactGaussianProcess


_NUGGET = 1e-10


def _make_training_data(
    bkd: Backend[Array], nvars: int = 2, n_train: int = 8
) -> tuple[Any, Any]:
    np.random.seed(42)
    X_train = bkd.array(np.random.rand(nvars, n_train) * 2 - 1)
    y_train = bkd.reshape(
        10.0 * bkd.sin(math.pi * X_train[0, :])
        + 5.0,  # non-trivial mean and std
        (1, -1),
    )
    return X_train, y_train


def _create_kernel(
    bkd: Backend[Array],
) -> SeparableProductKernel[Array]:
    k1 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, bkd)
    k2 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, bkd)
    return SeparableProductKernel([k1, k2], bkd)


class TestOutputTransform(Generic[Array], unittest.TestCase):
    """Base class for output transform tests."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_standard_scaler_from_data(self) -> None:
        """Test that from_data computes correct mean and std."""
        bkd = self._bkd
        y = bkd.array([[1.0, 2.0, 3.0, 4.0, 5.0]])  # (1, 5)
        scaler = OutputStandardScaler.from_data(y, bkd)

        expected_mean = bkd.mean(y, axis=1)
        expected_std = bkd.std(y, axis=1)

        bkd.assert_allclose(scaler.shift(), expected_mean)
        bkd.assert_allclose(scaler.scale(), expected_std)

    def test_standard_scaler_zero_variance(self) -> None:
        """Test that constant output gets std=1.0 (no scaling)."""
        bkd = self._bkd
        y = bkd.array([[7.0, 7.0, 7.0, 7.0]])  # constant
        scaler = OutputStandardScaler.from_data(y, bkd)

        bkd.assert_allclose(scaler.scale(), bkd.array([1.0]))
        bkd.assert_allclose(scaler.shift(), bkd.array([7.0]))

        # inverse_transform should subtract mean, divide by 1.0
        y_scaled = scaler.inverse_transform(y)
        bkd.assert_allclose(y_scaled, bkd.zeros((1, 4)))

    def test_round_trip(self) -> None:
        """Test that transform(inverse_transform(y)) == y."""
        bkd = self._bkd
        y = bkd.array([[1.0, 3.0, 5.0, 7.0, 9.0]])
        scaler = OutputStandardScaler.from_data(y, bkd)

        y_scaled = scaler.inverse_transform(y)
        y_recovered = scaler.transform(y_scaled)
        bkd.assert_allclose(y_recovered, y, rtol=1e-12)

    def test_inverse_round_trip(self) -> None:
        """Test that inverse_transform(transform(y)) == y."""
        bkd = self._bkd
        y_scaled = bkd.array([[-1.0, 0.0, 1.0]])
        mean = bkd.array([5.0])
        std = bkd.array([2.0])
        scaler = OutputStandardScaler(mean, std, bkd)

        y_orig = scaler.transform(y_scaled)
        y_recovered = scaler.inverse_transform(y_orig)
        bkd.assert_allclose(y_recovered, y_scaled, rtol=1e-12)

    def test_identity_transform(self) -> None:
        """Test that IdentityOutputTransform is a no-op."""
        bkd = self._bkd
        identity = IdentityOutputTransform(1, bkd)

        y = bkd.array([[1.0, 2.0, 3.0]])
        bkd.assert_allclose(identity.transform(y), y)
        bkd.assert_allclose(identity.inverse_transform(y), y)
        bkd.assert_allclose(identity.scale(), bkd.ones(1))
        bkd.assert_allclose(identity.shift(), bkd.zeros(1))

    def test_gp_predict_with_transform(self) -> None:
        """Test that GP with transform returns original-space predictions."""
        bkd = self._bkd
        X_train, y_train = _make_training_data(bkd)
        kernel = _create_kernel(bkd)

        scaler = OutputStandardScaler.from_data(y_train, bkd)

        # GP with transform
        gp_t = ExactGaussianProcess(kernel, nvars=2, bkd=bkd, nugget=_NUGGET)
        gp_t.hyp_list().set_all_inactive()
        gp_t.fit(X_train, y_train, output_transform=scaler)

        # At training points, prediction should approximate original y
        pred = gp_t.predict(X_train)
        bkd.assert_allclose(pred, y_train, rtol=1e-4, atol=1e-4)

    def test_gp_predict_matches_manual_scaling(self) -> None:
        """Test GP with transform matches manual scale/shift of raw GP."""
        bkd = self._bkd
        X_train, y_train = _make_training_data(bkd)
        kernel_t = _create_kernel(bkd)
        kernel_r = _create_kernel(bkd)

        scaler = OutputStandardScaler.from_data(y_train, bkd)
        y_scaled = scaler.inverse_transform(y_train)

        # GP with transform (fits on original, internally scales)
        gp_t = ExactGaussianProcess(
            kernel_t, nvars=2, bkd=bkd, nugget=_NUGGET
        )
        gp_t.hyp_list().set_all_inactive()
        gp_t.fit(X_train, y_train, output_transform=scaler)

        # Raw GP (fits on manually scaled data)
        gp_r = ExactGaussianProcess(
            kernel_r, nvars=2, bkd=bkd, nugget=_NUGGET
        )
        gp_r.hyp_list().set_all_inactive()
        gp_r.fit(X_train, y_scaled)

        # Test points
        np.random.seed(99)
        X_test = bkd.array(np.random.rand(2, 5) * 2 - 1)

        pred_t = gp_t.predict(X_test)
        pred_r = gp_r.predict(X_test)
        sigma_y = scaler.scale()
        mu_y = scaler.shift()
        pred_manual = sigma_y[:, None] * pred_r + mu_y[:, None]

        bkd.assert_allclose(pred_t, pred_manual, rtol=1e-10)

    def test_gp_predict_std_with_transform(self) -> None:
        """Test that predict_std scales by σ_y."""
        bkd = self._bkd
        X_train, y_train = _make_training_data(bkd)
        kernel_t = _create_kernel(bkd)
        kernel_r = _create_kernel(bkd)

        scaler = OutputStandardScaler.from_data(y_train, bkd)
        y_scaled = scaler.inverse_transform(y_train)

        gp_t = ExactGaussianProcess(
            kernel_t, nvars=2, bkd=bkd, nugget=_NUGGET
        )
        gp_t.hyp_list().set_all_inactive()
        gp_t.fit(X_train, y_train, output_transform=scaler)

        gp_r = ExactGaussianProcess(
            kernel_r, nvars=2, bkd=bkd, nugget=_NUGGET
        )
        gp_r.hyp_list().set_all_inactive()
        gp_r.fit(X_train, y_scaled)

        np.random.seed(99)
        X_test = bkd.array(np.random.rand(2, 5) * 2 - 1)

        std_t = gp_t.predict_std(X_test)
        std_r = gp_r.predict_std(X_test)
        sigma_y = scaler.scale()
        std_manual = sigma_y[:, None] * std_r

        bkd.assert_allclose(std_t, std_manual, rtol=1e-10)

    def test_gp_predict_covariance_with_transform(self) -> None:
        """Test that predict_covariance scales by σ_y²."""
        bkd = self._bkd
        X_train, y_train = _make_training_data(bkd)
        kernel_t = _create_kernel(bkd)
        kernel_r = _create_kernel(bkd)

        scaler = OutputStandardScaler.from_data(y_train, bkd)
        y_scaled = scaler.inverse_transform(y_train)

        gp_t = ExactGaussianProcess(
            kernel_t, nvars=2, bkd=bkd, nugget=_NUGGET
        )
        gp_t.hyp_list().set_all_inactive()
        gp_t.fit(X_train, y_train, output_transform=scaler)

        gp_r = ExactGaussianProcess(
            kernel_r, nvars=2, bkd=bkd, nugget=_NUGGET
        )
        gp_r.hyp_list().set_all_inactive()
        gp_r.fit(X_train, y_scaled)

        np.random.seed(99)
        X_test = bkd.array(np.random.rand(2, 3) * 2 - 1)

        cov_t = gp_t.predict_covariance(X_test)
        cov_r = gp_r.predict_covariance(X_test)
        sigma_y_sq = scaler.scale()[0] ** 2
        cov_manual = sigma_y_sq * cov_r

        bkd.assert_allclose(cov_t, cov_manual, rtol=1e-10)

    def test_gp_no_transform_unchanged(self) -> None:
        """Test that GP without transform behaves identically to before."""
        bkd = self._bkd
        X_train, y_train = _make_training_data(bkd)
        kernel = _create_kernel(bkd)

        gp = ExactGaussianProcess(kernel, nvars=2, bkd=bkd, nugget=_NUGGET)
        gp.hyp_list().set_all_inactive()
        gp.fit(X_train, y_train)

        self.assertIsNone(gp.output_transform())

        np.random.seed(99)
        X_test = bkd.array(np.random.rand(2, 5) * 2 - 1)
        pred = gp.predict(X_test)
        # Just verify it runs without error and returns correct shape
        self.assertEqual(pred.shape, (1, 5))


class TestOutputTransformNumpy(TestOutputTransform[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestOutputTransformTorch(TestOutputTransform[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
