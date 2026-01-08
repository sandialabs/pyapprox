"""Tests for optimization benchmark instances."""

import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.test_utils import load_tests

from pyapprox.typing.benchmarks.protocols import BenchmarkProtocol
from pyapprox.typing.benchmarks.registry import BenchmarkRegistry
from pyapprox.typing.benchmarks.instances.optimization import (
    rosenbrock_2d,
    rosenbrock_10d,
)


class TestRosenbrock2DBenchmark(Generic[Array], unittest.TestCase):
    """Tests for rosenbrock_2d benchmark instance."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_protocol_compliance(self) -> None:
        """Test that rosenbrock_2d satisfies BenchmarkProtocol."""
        benchmark = rosenbrock_2d(self._bkd)
        self.assertIsInstance(benchmark, BenchmarkProtocol)

    def test_name(self) -> None:
        """Test benchmark name."""
        benchmark = rosenbrock_2d(self._bkd)
        self.assertEqual(benchmark.name(), "rosenbrock_2d")

    def test_function_nvars(self) -> None:
        """Test function nvars."""
        benchmark = rosenbrock_2d(self._bkd)
        self.assertEqual(benchmark.function().nvars(), 2)

    def test_function_nqoi(self) -> None:
        """Test function nqoi."""
        benchmark = rosenbrock_2d(self._bkd)
        self.assertEqual(benchmark.function().nqoi(), 1)

    def test_domain_nvars(self) -> None:
        """Test domain nvars."""
        benchmark = rosenbrock_2d(self._bkd)
        self.assertEqual(benchmark.domain().nvars(), 2)

    def test_domain_bounds(self) -> None:
        """Test domain bounds."""
        benchmark = rosenbrock_2d(self._bkd)
        bounds = benchmark.domain().bounds()
        expected = self._bkd.array([[-5.0, 10.0], [-5.0, 10.0]])
        self._bkd.assert_allclose(bounds, expected, atol=1e-14)

    def test_ground_truth_global_minimum(self) -> None:
        """Test ground truth global minimum is 0."""
        benchmark = rosenbrock_2d(self._bkd)
        gt = benchmark.ground_truth()
        self._bkd.assert_allclose(
            self._bkd.asarray([gt.get("global_minimum")]),
            self._bkd.asarray([0.0]),
            atol=1e-14,
        )

    def test_ground_truth_global_minimizer(self) -> None:
        """Test ground truth global minimizer is (1, 1)."""
        benchmark = rosenbrock_2d(self._bkd)
        gt = benchmark.ground_truth()
        minimizer = gt.get("global_minimizers")
        expected = self._bkd.asarray([[1.0], [1.0]])
        self._bkd.assert_allclose(
            self._bkd.asarray(minimizer),
            expected,
            atol=1e-14,
        )

    def test_function_at_minimizer(self) -> None:
        """Test function value at minimizer is 0."""
        benchmark = rosenbrock_2d(self._bkd)
        func = benchmark.function()
        minimizer = self._bkd.array([[1.0], [1.0]])
        value = func(minimizer)
        self._bkd.assert_allclose(value, self._bkd.zeros((1, 1)), atol=1e-14)


class TestRosenbrock2DBenchmarkNumpy(TestRosenbrock2DBenchmark[NDArray[Any]]):
    """NumPy backend tests."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestRosenbrock2DBenchmarkTorch(TestRosenbrock2DBenchmark[torch.Tensor]):
    """PyTorch backend tests."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestRosenbrock10DBenchmark(Generic[Array], unittest.TestCase):
    """Tests for rosenbrock_10d benchmark instance."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_protocol_compliance(self) -> None:
        """Test that rosenbrock_10d satisfies BenchmarkProtocol."""
        benchmark = rosenbrock_10d(self._bkd)
        self.assertIsInstance(benchmark, BenchmarkProtocol)

    def test_name(self) -> None:
        """Test benchmark name."""
        benchmark = rosenbrock_10d(self._bkd)
        self.assertEqual(benchmark.name(), "rosenbrock_10d")

    def test_function_nvars(self) -> None:
        """Test function nvars."""
        benchmark = rosenbrock_10d(self._bkd)
        self.assertEqual(benchmark.function().nvars(), 10)

    def test_domain_nvars(self) -> None:
        """Test domain nvars."""
        benchmark = rosenbrock_10d(self._bkd)
        self.assertEqual(benchmark.domain().nvars(), 10)

    def test_function_at_minimizer(self) -> None:
        """Test function value at minimizer is 0."""
        benchmark = rosenbrock_10d(self._bkd)
        func = benchmark.function()
        minimizer = self._bkd.ones((10, 1))
        value = func(minimizer)
        self._bkd.assert_allclose(value, self._bkd.zeros((1, 1)), atol=1e-14)


class TestRosenbrock10DBenchmarkNumpy(TestRosenbrock10DBenchmark[NDArray[Any]]):
    """NumPy backend tests."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestRosenbrock10DBenchmarkTorch(TestRosenbrock10DBenchmark[torch.Tensor]):
    """PyTorch backend tests."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestBenchmarkRegistryOptimization(unittest.TestCase):
    """Test registry for optimization benchmarks."""

    def test_rosenbrock_2d_registered(self) -> None:
        """Test rosenbrock_2d is registered."""
        from pyapprox.typing.benchmarks.instances import optimization  # noqa: F401

        bkd = NumpyBkd()
        benchmark = BenchmarkRegistry.get("rosenbrock_2d", bkd)
        self.assertEqual(benchmark.name(), "rosenbrock_2d")

    def test_rosenbrock_10d_registered(self) -> None:
        """Test rosenbrock_10d is registered."""
        from pyapprox.typing.benchmarks.instances import optimization  # noqa: F401

        bkd = NumpyBkd()
        benchmark = BenchmarkRegistry.get("rosenbrock_10d", bkd)
        self.assertEqual(benchmark.name(), "rosenbrock_10d")

    def test_benchmarks_in_optimization_category(self) -> None:
        """Test benchmarks are in optimization category."""
        from pyapprox.typing.benchmarks.instances import optimization  # noqa: F401

        opt_benchmarks = BenchmarkRegistry.list_category("optimization")
        self.assertIn("rosenbrock_2d", opt_benchmarks)
        self.assertIn("rosenbrock_10d", opt_benchmarks)


if __name__ == "__main__":
    unittest.main()
