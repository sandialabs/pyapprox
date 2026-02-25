"""Tests for quadrature benchmark instances."""

import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray

from pyapprox.benchmarks.instances.quadrature import (
    genz_corner_peak_2d,
    genz_gaussian_peak_2d,
    genz_gaussian_peak_5d,
    genz_oscillatory_2d,
    genz_oscillatory_5d,
    genz_product_peak_2d,
)
from pyapprox.benchmarks.registry import BenchmarkRegistry
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd


class TestGenzOscillatory2D(Generic[Array], unittest.TestCase):
    """Tests for genz_oscillatory_2d benchmark instance."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_name(self) -> None:
        """Test benchmark name."""
        benchmark = genz_oscillatory_2d(self._bkd)
        self.assertEqual(benchmark.name(), "genz_oscillatory_2d")

    def test_domain_nvars(self) -> None:
        """Test domain has 2 variables."""
        benchmark = genz_oscillatory_2d(self._bkd)
        self.assertEqual(benchmark.domain().nvars(), 2)

    def test_domain_bounds(self) -> None:
        """Test domain bounds are [0, 1]^2."""
        benchmark = genz_oscillatory_2d(self._bkd)
        bounds = benchmark.domain().bounds()
        expected = self._bkd.array([[0.0, 1.0], [0.0, 1.0]])
        self._bkd.assert_allclose(bounds, expected, rtol=1e-12)

    def test_function_nvars(self) -> None:
        """Test function has 2 input variables."""
        benchmark = genz_oscillatory_2d(self._bkd)
        self.assertEqual(benchmark.function().nvars(), 2)

    def test_function_nqoi(self) -> None:
        """Test function has 1 output."""
        benchmark = genz_oscillatory_2d(self._bkd)
        self.assertEqual(benchmark.function().nqoi(), 1)

    def test_ground_truth_integral(self) -> None:
        """Test ground truth has integral."""
        benchmark = genz_oscillatory_2d(self._bkd)
        gt = benchmark.ground_truth()
        self.assertIn("integral", gt.available())
        # Integral should be a finite number
        self.assertTrue(abs(gt.integral) < float("inf"))

    def test_function_evaluation(self) -> None:
        """Test function can be evaluated."""
        benchmark = genz_oscillatory_2d(self._bkd)
        func = benchmark.function()
        sample = self._bkd.array([[0.5], [0.5]])
        result = func(sample)
        self.assertEqual(result.shape, (1, 1))


class TestGenzProductPeak2D(Generic[Array], unittest.TestCase):
    """Tests for genz_product_peak_2d benchmark instance."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_name(self) -> None:
        """Test benchmark name."""
        benchmark = genz_product_peak_2d(self._bkd)
        self.assertEqual(benchmark.name(), "genz_product_peak_2d")

    def test_domain_nvars(self) -> None:
        """Test domain has 2 variables."""
        benchmark = genz_product_peak_2d(self._bkd)
        self.assertEqual(benchmark.domain().nvars(), 2)

    def test_ground_truth_integral(self) -> None:
        """Test ground truth has integral."""
        benchmark = genz_product_peak_2d(self._bkd)
        gt = benchmark.ground_truth()
        self.assertIn("integral", gt.available())


class TestGenzCornerPeak2D(Generic[Array], unittest.TestCase):
    """Tests for genz_corner_peak_2d benchmark instance."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_name(self) -> None:
        """Test benchmark name."""
        benchmark = genz_corner_peak_2d(self._bkd)
        self.assertEqual(benchmark.name(), "genz_corner_peak_2d")

    def test_domain_nvars(self) -> None:
        """Test domain has 2 variables."""
        benchmark = genz_corner_peak_2d(self._bkd)
        self.assertEqual(benchmark.domain().nvars(), 2)

    def test_ground_truth_integral(self) -> None:
        """Test ground truth has integral."""
        benchmark = genz_corner_peak_2d(self._bkd)
        gt = benchmark.ground_truth()
        self.assertIn("integral", gt.available())


class TestGenzGaussianPeak2D(Generic[Array], unittest.TestCase):
    """Tests for genz_gaussian_peak_2d benchmark instance."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_name(self) -> None:
        """Test benchmark name."""
        benchmark = genz_gaussian_peak_2d(self._bkd)
        self.assertEqual(benchmark.name(), "genz_gaussian_peak_2d")

    def test_domain_nvars(self) -> None:
        """Test domain has 2 variables."""
        benchmark = genz_gaussian_peak_2d(self._bkd)
        self.assertEqual(benchmark.domain().nvars(), 2)

    def test_ground_truth_integral(self) -> None:
        """Test ground truth has integral."""
        benchmark = genz_gaussian_peak_2d(self._bkd)
        gt = benchmark.ground_truth()
        self.assertIn("integral", gt.available())


class TestGenzOscillatory5D(Generic[Array], unittest.TestCase):
    """Tests for genz_oscillatory_5d benchmark instance."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_name(self) -> None:
        """Test benchmark name."""
        benchmark = genz_oscillatory_5d(self._bkd)
        self.assertEqual(benchmark.name(), "genz_oscillatory_5d")

    def test_domain_nvars(self) -> None:
        """Test domain has 5 variables."""
        benchmark = genz_oscillatory_5d(self._bkd)
        self.assertEqual(benchmark.domain().nvars(), 5)

    def test_function_nvars(self) -> None:
        """Test function has 5 input variables."""
        benchmark = genz_oscillatory_5d(self._bkd)
        self.assertEqual(benchmark.function().nvars(), 5)

    def test_ground_truth_integral(self) -> None:
        """Test ground truth has integral."""
        benchmark = genz_oscillatory_5d(self._bkd)
        gt = benchmark.ground_truth()
        self.assertIn("integral", gt.available())


class TestGenzGaussianPeak5D(Generic[Array], unittest.TestCase):
    """Tests for genz_gaussian_peak_5d benchmark instance."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_name(self) -> None:
        """Test benchmark name."""
        benchmark = genz_gaussian_peak_5d(self._bkd)
        self.assertEqual(benchmark.name(), "genz_gaussian_peak_5d")

    def test_domain_nvars(self) -> None:
        """Test domain has 5 variables."""
        benchmark = genz_gaussian_peak_5d(self._bkd)
        self.assertEqual(benchmark.domain().nvars(), 5)

    def test_ground_truth_integral(self) -> None:
        """Test ground truth has integral."""
        benchmark = genz_gaussian_peak_5d(self._bkd)
        gt = benchmark.ground_truth()
        self.assertIn("integral", gt.available())


class TestBenchmarkRegistryQuadrature(unittest.TestCase):
    """Tests for BenchmarkRegistry quadrature category."""

    def test_oscillatory_2d_registered(self) -> None:
        """Test genz_oscillatory_2d is registered."""
        self.assertIn("genz_oscillatory_2d", BenchmarkRegistry.list_all())

    def test_product_peak_2d_registered(self) -> None:
        """Test genz_product_peak_2d is registered."""
        self.assertIn("genz_product_peak_2d", BenchmarkRegistry.list_all())

    def test_corner_peak_2d_registered(self) -> None:
        """Test genz_corner_peak_2d is registered."""
        self.assertIn("genz_corner_peak_2d", BenchmarkRegistry.list_all())

    def test_gaussian_peak_2d_registered(self) -> None:
        """Test genz_gaussian_peak_2d is registered."""
        self.assertIn("genz_gaussian_peak_2d", BenchmarkRegistry.list_all())

    def test_analytic_category(self) -> None:
        """Test Genz benchmarks are in analytic category."""
        analytic = BenchmarkRegistry.list_category("analytic")
        self.assertIn("genz_oscillatory_2d", analytic)
        self.assertIn("genz_product_peak_2d", analytic)
        self.assertIn("genz_corner_peak_2d", analytic)
        self.assertIn("genz_gaussian_peak_2d", analytic)


# NumPy backend tests
class TestGenzOscillatory2DNumpy(TestGenzOscillatory2D[NDArray[Any]]):
    """NumPy backend tests."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGenzProductPeak2DNumpy(TestGenzProductPeak2D[NDArray[Any]]):
    """NumPy backend tests."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGenzCornerPeak2DNumpy(TestGenzCornerPeak2D[NDArray[Any]]):
    """NumPy backend tests."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGenzGaussianPeak2DNumpy(TestGenzGaussianPeak2D[NDArray[Any]]):
    """NumPy backend tests."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGenzOscillatory5DNumpy(TestGenzOscillatory5D[NDArray[Any]]):
    """NumPy backend tests."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGenzGaussianPeak5DNumpy(TestGenzGaussianPeak5D[NDArray[Any]]):
    """NumPy backend tests."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


# PyTorch backend tests
class TestGenzOscillatory2DTorch(TestGenzOscillatory2D[torch.Tensor]):
    """PyTorch backend tests."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestGenzProductPeak2DTorch(TestGenzProductPeak2D[torch.Tensor]):
    """PyTorch backend tests."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestGenzCornerPeak2DTorch(TestGenzCornerPeak2D[torch.Tensor]):
    """PyTorch backend tests."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestGenzGaussianPeak2DTorch(TestGenzGaussianPeak2D[torch.Tensor]):
    """PyTorch backend tests."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestGenzOscillatory5DTorch(TestGenzOscillatory5D[torch.Tensor]):
    """PyTorch backend tests."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestGenzGaussianPeak5DTorch(TestGenzGaussianPeak5D[torch.Tensor]):
    """PyTorch backend tests."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
