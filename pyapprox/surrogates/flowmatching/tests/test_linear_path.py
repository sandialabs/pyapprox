"""Tests for LinearPath probability path."""

import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray

from pyapprox.surrogates.flowmatching.linear_path import LinearPath
from pyapprox.surrogates.flowmatching.protocols import (
    ProbabilityPathProtocol,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd


class TestLinearPath(Generic[Array], unittest.TestCase):
    """Dual-backend tests for LinearPath."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self._path = LinearPath(self._bkd)

    def test_satisfies_protocol(self) -> None:
        self.assertIsInstance(self._path, ProbabilityPathProtocol)

    def test_alpha_boundary(self) -> None:
        t0 = self._bkd.array([[0.0]])
        t1 = self._bkd.array([[1.0]])
        self._bkd.assert_allclose(
            self._path.alpha(t0), self._bkd.array([[0.0]]), rtol=1e-12
        )
        self._bkd.assert_allclose(
            self._path.alpha(t1), self._bkd.array([[1.0]]), rtol=1e-12
        )

    def test_sigma_boundary(self) -> None:
        t0 = self._bkd.array([[0.0]])
        t1 = self._bkd.array([[1.0]])
        self._bkd.assert_allclose(
            self._path.sigma(t0), self._bkd.array([[1.0]]), rtol=1e-12
        )
        self._bkd.assert_allclose(
            self._path.sigma(t1), self._bkd.array([[0.0]]), rtol=1e-12
        )

    def test_alpha_plus_sigma_equals_one(self) -> None:
        t = self._bkd.array([[0.0, 0.25, 0.5, 0.75, 1.0]])
        result = self._path.alpha(t) + self._path.sigma(t)
        self._bkd.assert_allclose(result, self._bkd.ones_like(t), rtol=1e-12)

    def test_interpolate_at_t0_returns_x0(self) -> None:
        t = self._bkd.array([[0.0, 0.0]])
        x0 = self._bkd.array([[1.0, 2.0], [3.0, 4.0]])
        x1 = self._bkd.array([[5.0, 6.0], [7.0, 8.0]])
        result = self._path.interpolate(t, x0, x1)
        self._bkd.assert_allclose(result, x0, rtol=1e-12)

    def test_interpolate_at_t1_returns_x1(self) -> None:
        t = self._bkd.array([[1.0, 1.0]])
        x0 = self._bkd.array([[1.0, 2.0], [3.0, 4.0]])
        x1 = self._bkd.array([[5.0, 6.0], [7.0, 8.0]])
        result = self._path.interpolate(t, x0, x1)
        self._bkd.assert_allclose(result, x1, rtol=1e-12)

    def test_interpolate_midpoint(self) -> None:
        t = self._bkd.array([[0.5]])
        x0 = self._bkd.array([[0.0], [0.0]])
        x1 = self._bkd.array([[2.0], [4.0]])
        result = self._path.interpolate(t, x0, x1)
        expected = self._bkd.array([[1.0], [2.0]])
        self._bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_interpolate_shape(self) -> None:
        d, ns = 3, 5
        t = self._bkd.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
        x0 = self._bkd.zeros((d, ns))
        x1 = self._bkd.ones_like(x0)
        result = self._path.interpolate(t, x0, x1)
        self.assertEqual(result.shape, (d, ns))

    def test_target_field_shape(self) -> None:
        d, ns = 3, 5
        t = self._bkd.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
        x0 = self._bkd.zeros((d, ns))
        x1 = self._bkd.ones_like(x0)
        result = self._path.target_field(t, x0, x1)
        self.assertEqual(result.shape, (d, ns))

    def test_target_field_equals_x1_minus_x0(self) -> None:
        t = self._bkd.array([[0.3, 0.7]])
        x0 = self._bkd.array([[1.0, 2.0], [3.0, 4.0]])
        x1 = self._bkd.array([[5.0, 6.0], [7.0, 8.0]])
        result = self._path.target_field(t, x0, x1)
        expected = x1 - x0
        self._bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_target_field_is_time_derivative_of_interpolate(self) -> None:
        """Verify u_t = d/dt interpolate(t, x0, x1) via finite differences."""
        x0 = self._bkd.array([[1.0, 2.0], [3.0, 4.0]])
        x1 = self._bkd.array([[5.0, 6.0], [7.0, 8.0]])

        t_val = 0.4
        dt = 1e-7
        t = self._bkd.array([[t_val, t_val]])
        t_plus = self._bkd.array([[t_val + dt, t_val + dt]])
        t_minus = self._bkd.array([[t_val - dt, t_val - dt]])

        fd_deriv = (
            self._path.interpolate(t_plus, x0, x1)
            - self._path.interpolate(t_minus, x0, x1)
        ) / (2.0 * dt)

        analytical = self._path.target_field(t, x0, x1)
        self._bkd.assert_allclose(fd_deriv, analytical, rtol=1e-6)

    def test_d_alpha_via_finite_differences(self) -> None:
        t_val = 0.3
        dt = 1e-7
        t = self._bkd.array([[t_val]])
        t_plus = self._bkd.array([[t_val + dt]])
        t_minus = self._bkd.array([[t_val - dt]])

        fd = (self._path.alpha(t_plus) - self._path.alpha(t_minus)) / (2.0 * dt)
        analytical = self._path.d_alpha(t)
        self._bkd.assert_allclose(fd, analytical, rtol=1e-6)

    def test_d_sigma_via_finite_differences(self) -> None:
        t_val = 0.6
        dt = 1e-7
        t = self._bkd.array([[t_val]])
        t_plus = self._bkd.array([[t_val + dt]])
        t_minus = self._bkd.array([[t_val - dt]])

        fd = (self._path.sigma(t_plus) - self._path.sigma(t_minus)) / (2.0 * dt)
        analytical = self._path.d_sigma(t)
        self._bkd.assert_allclose(fd, analytical, rtol=1e-6)

    def test_target_field_consistent_with_coefficients(self) -> None:
        """Verify u_t = d_alpha*x1 + d_sigma*x0 (general formula)."""
        t = self._bkd.array([[0.2, 0.5, 0.8]])
        x0 = self._bkd.array([[1.0, 2.0, 3.0]])
        x1 = self._bkd.array([[4.0, 5.0, 6.0]])

        u_general = self._path.d_alpha(t) * x1 + self._path.d_sigma(t) * x0
        u_target = self._path.target_field(t, x0, x1)
        self._bkd.assert_allclose(u_general, u_target, rtol=1e-12)

    def test_1d_single_sample(self) -> None:
        """Test with d=1, ns=1."""
        t = self._bkd.array([[0.3]])
        x0 = self._bkd.array([[2.0]])
        x1 = self._bkd.array([[5.0]])
        result = self._path.interpolate(t, x0, x1)
        expected = self._bkd.array([[0.7 * 2.0 + 0.3 * 5.0]])
        self._bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_batch_with_varying_times(self) -> None:
        """Test batch with different time values per sample."""
        t = self._bkd.array([[0.0, 0.5, 1.0]])
        x0 = self._bkd.array([[1.0, 1.0, 1.0]])
        x1 = self._bkd.array([[3.0, 3.0, 3.0]])
        result = self._path.interpolate(t, x0, x1)
        expected = self._bkd.array([[1.0, 2.0, 3.0]])
        self._bkd.assert_allclose(result, expected, rtol=1e-12)


class TestLinearPathNumpy(TestLinearPath[NDArray[Any]]):
    """NumPy backend tests for LinearPath."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestLinearPathTorch(TestLinearPath[torch.Tensor]):
    """Torch backend tests for LinearPath."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


from pyapprox.util.test_utils import load_tests  # noqa: F401
