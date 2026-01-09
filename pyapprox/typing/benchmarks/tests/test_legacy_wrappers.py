"""Tests for legacy benchmark wrappers."""

import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.test_utils import load_tests

from pyapprox.typing.interface.functions.protocols.function import (
    FunctionProtocol,
)
from pyapprox.typing.benchmarks.legacy.adapter import (
    LegacyFunctionAdapter,
    LegacyFunctionWithJacobianAdapter,
)
from pyapprox.typing.benchmarks.legacy.wrappers import (
    wrap_legacy_ishigami,
    wrap_legacy_genz,
)


class TestLegacyIshigamiWrapper(Generic[Array], unittest.TestCase):
    """Tests for wrapped legacy Ishigami benchmark."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_wrapper_creates_benchmark(self) -> None:
        """Test that wrapper creates a valid benchmark."""
        benchmark = wrap_legacy_ishigami(self._bkd)
        self.assertEqual(benchmark.name(), "ishigami_3d_legacy")

    def test_function_protocol_compliance(self) -> None:
        """Test that wrapped function satisfies FunctionProtocol."""
        benchmark = wrap_legacy_ishigami(self._bkd)
        func = benchmark.function()
        self.assertIsInstance(func, FunctionProtocol)

    def test_nvars(self) -> None:
        """Test nvars returns 3."""
        benchmark = wrap_legacy_ishigami(self._bkd)
        func = benchmark.function()
        self.assertEqual(func.nvars(), 3)

    def test_nqoi(self) -> None:
        """Test nqoi returns 1."""
        benchmark = wrap_legacy_ishigami(self._bkd)
        func = benchmark.function()
        self.assertEqual(func.nqoi(), 1)

    def test_evaluation_at_origin(self) -> None:
        """Test evaluation at origin."""
        benchmark = wrap_legacy_ishigami(self._bkd)
        func = benchmark.function()
        sample = self._bkd.array([[0.0], [0.0], [0.0]])
        result = func(sample)
        self.assertEqual(result.shape, (1, 1))
        # f(0,0,0) = sin(0) + 7*sin^2(0) + 0.1*0^4*sin(0) = 0
        self._bkd.assert_allclose(result, self._bkd.zeros((1, 1)), atol=1e-12)

    def test_evaluation_batch(self) -> None:
        """Test batch evaluation."""
        benchmark = wrap_legacy_ishigami(self._bkd)
        func = benchmark.function()
        samples = self._bkd.array([
            [0.0, 1.0, -1.0],
            [0.0, 0.5, 0.5],
            [0.0, 0.0, 0.0],
        ])
        result = func(samples)
        self.assertEqual(result.shape, (1, 3))

    def test_ground_truth_mean(self) -> None:
        """Test ground truth mean value."""
        benchmark = wrap_legacy_ishigami(self._bkd)
        gt = benchmark.ground_truth()
        # Mean should be a/2 = 7/2 = 3.5
        self._bkd.assert_allclose(
            self._bkd.asarray([gt.mean]),
            self._bkd.asarray([3.5]),
            rtol=1e-10,
        )

    def test_ground_truth_available(self) -> None:
        """Test that expected ground truth values are available."""
        benchmark = wrap_legacy_ishigami(self._bkd)
        gt = benchmark.ground_truth()
        available = gt.available()
        self.assertIn("mean", available)
        self.assertIn("variance", available)
        self.assertIn("main_effects", available)
        self.assertIn("total_effects", available)

    def test_jacobian_shape(self) -> None:
        """Test Jacobian has correct shape."""
        benchmark = wrap_legacy_ishigami(self._bkd)
        func = benchmark.function()
        sample = self._bkd.array([[0.5], [0.3], [-0.2]])
        jac = func.jacobian(sample)
        self.assertEqual(jac.shape, (1, 3))

    def test_jacobian_at_origin(self) -> None:
        """Test Jacobian at origin."""
        benchmark = wrap_legacy_ishigami(self._bkd)
        func = benchmark.function()
        sample = self._bkd.array([[0.0], [0.0], [0.0]])
        jac = func.jacobian(sample)
        # At origin: df/dx1 = cos(0)*(1 + 0) = 1, df/dx2 = 0, df/dx3 = 0
        expected = self._bkd.array([[1.0, 0.0, 0.0]])
        self._bkd.assert_allclose(jac, expected, atol=1e-12)

    def test_jacobian_invalid_shape(self) -> None:
        """Test Jacobian raises for invalid input shape."""
        benchmark = wrap_legacy_ishigami(self._bkd)
        func = benchmark.function()
        sample = self._bkd.array([[0.5, 0.1], [0.3, 0.2], [-0.2, 0.3]])
        with self.assertRaises(ValueError):
            func.jacobian(sample)

    def test_hvp_shape(self) -> None:
        """Test HVP has correct shape."""
        benchmark = wrap_legacy_ishigami(self._bkd)
        func = benchmark.function()
        sample = self._bkd.array([[0.5], [0.3], [-0.2]])
        vec = self._bkd.array([[1.0], [0.0], [0.0]])
        hvp = func.hvp(sample, vec)
        self.assertEqual(hvp.shape, (3, 1))

    def test_hvp_invalid_sample_shape(self) -> None:
        """Test HVP raises for invalid sample shape."""
        benchmark = wrap_legacy_ishigami(self._bkd)
        func = benchmark.function()
        sample = self._bkd.array([[0.5, 0.1], [0.3, 0.2], [-0.2, 0.3]])
        vec = self._bkd.array([[1.0], [0.0], [0.0]])
        with self.assertRaises(ValueError):
            func.hvp(sample, vec)

    def test_domain(self) -> None:
        """Test domain bounds."""
        import math
        benchmark = wrap_legacy_ishigami(self._bkd)
        domain = benchmark.domain()
        self.assertEqual(domain.nvars(), 3)
        bounds = domain.bounds()
        expected = self._bkd.array([
            [-math.pi, math.pi],
            [-math.pi, math.pi],
            [-math.pi, math.pi],
        ])
        self._bkd.assert_allclose(bounds, expected, atol=1e-12)


class TestLegacyGenzWrapper(Generic[Array], unittest.TestCase):
    """Tests for wrapped legacy Genz benchmarks."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_wrapper_creates_benchmark(self) -> None:
        """Test that wrapper creates a valid benchmark."""
        benchmark = wrap_legacy_genz(self._bkd, nvars=2, genz_type="oscillatory")
        self.assertEqual(benchmark.name(), "genz_oscillatory_2d_legacy")

    def test_function_protocol_compliance(self) -> None:
        """Test that wrapped function satisfies FunctionProtocol."""
        benchmark = wrap_legacy_genz(self._bkd, nvars=2, genz_type="oscillatory")
        func = benchmark.function()
        self.assertIsInstance(func, FunctionProtocol)

    def test_nvars(self) -> None:
        """Test nvars returns expected value."""
        benchmark = wrap_legacy_genz(self._bkd, nvars=3, genz_type="oscillatory")
        func = benchmark.function()
        self.assertEqual(func.nvars(), 3)

    def test_nqoi(self) -> None:
        """Test nqoi returns 1."""
        benchmark = wrap_legacy_genz(self._bkd, nvars=2, genz_type="oscillatory")
        func = benchmark.function()
        self.assertEqual(func.nqoi(), 1)

    def test_evaluation_single(self) -> None:
        """Test evaluation at single sample."""
        benchmark = wrap_legacy_genz(self._bkd, nvars=2, genz_type="oscillatory")
        func = benchmark.function()
        sample = self._bkd.array([[0.5], [0.5]])
        result = func(sample)
        self.assertEqual(result.shape, (1, 1))

    def test_evaluation_batch(self) -> None:
        """Test batch evaluation."""
        benchmark = wrap_legacy_genz(self._bkd, nvars=2, genz_type="oscillatory")
        func = benchmark.function()
        samples = self._bkd.array([
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 1.0],
        ])
        result = func(samples)
        self.assertEqual(result.shape, (1, 3))

    def test_ground_truth_integral(self) -> None:
        """Test that ground truth has integral."""
        benchmark = wrap_legacy_genz(self._bkd, nvars=2, genz_type="oscillatory")
        gt = benchmark.ground_truth()
        self.assertIn("integral", gt.available())
        # Integral should be a finite number
        self.assertTrue(abs(gt.integral) < float('inf'))

    def test_domain(self) -> None:
        """Test domain bounds."""
        benchmark = wrap_legacy_genz(self._bkd, nvars=2, genz_type="oscillatory")
        domain = benchmark.domain()
        self.assertEqual(domain.nvars(), 2)
        bounds = domain.bounds()
        expected = self._bkd.array([[0.0, 1.0], [0.0, 1.0]])
        self._bkd.assert_allclose(bounds, expected, atol=1e-12)

    def test_product_peak_type(self) -> None:
        """Test product peak Genz type."""
        benchmark = wrap_legacy_genz(self._bkd, nvars=2, genz_type="product_peak")
        self.assertEqual(benchmark.name(), "genz_product_peak_2d_legacy")

    def test_corner_peak_type(self) -> None:
        """Test corner peak Genz type."""
        benchmark = wrap_legacy_genz(self._bkd, nvars=2, genz_type="corner_peak")
        self.assertEqual(benchmark.name(), "genz_corner_peak_2d_legacy")

    def test_gaussian_type(self) -> None:
        """Test gaussian Genz type."""
        benchmark = wrap_legacy_genz(self._bkd, nvars=2, genz_type="gaussian")
        self.assertEqual(benchmark.name(), "genz_gaussian_2d_legacy")


class TestLegacyIshigamiWrapperNumpy(TestLegacyIshigamiWrapper[NDArray[Any]]):
    """NumPy backend tests for legacy Ishigami wrapper."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestLegacyIshigamiWrapperTorch(TestLegacyIshigamiWrapper[torch.Tensor]):
    """PyTorch backend tests for legacy Ishigami wrapper."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestLegacyGenzWrapperNumpy(TestLegacyGenzWrapper[NDArray[Any]]):
    """NumPy backend tests for legacy Genz wrapper."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestLegacyGenzWrapperTorch(TestLegacyGenzWrapper[torch.Tensor]):
    """PyTorch backend tests for legacy Genz wrapper."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
