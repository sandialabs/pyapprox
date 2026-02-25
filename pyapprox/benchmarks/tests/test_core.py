"""Tests for benchmark core infrastructure."""

import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.test_utils import load_tests

from pyapprox.benchmarks.protocols import (
    DomainProtocol,
    GroundTruthProtocol,
    BenchmarkProtocol,
)
from pyapprox.benchmarks.ground_truth import (
    SensitivityGroundTruth,
    OptimizationGroundTruth,
    QuadratureGroundTruth,
)
from pyapprox.benchmarks.benchmark import (
    BoxDomain,
    Benchmark,
    BenchmarkWithPrior,
)
from pyapprox.benchmarks.registry import BenchmarkRegistry


class TestGroundTruth(unittest.TestCase):
    """Tests for ground truth dataclasses."""

    def test_sensitivity_ground_truth_available(self) -> None:
        """Test that available() returns correct properties."""
        gt = SensitivityGroundTruth(
            mean=1.0,
            variance=2.0,
        )
        available = gt.available()
        self.assertIn("mean", available)
        self.assertIn("variance", available)
        self.assertNotIn("main_effects", available)

    def test_sensitivity_ground_truth_get(self) -> None:
        """Test that get() returns correct values."""
        gt = SensitivityGroundTruth(
            mean=1.5,
            variance=2.5,
        )
        self.assertEqual(gt.get("mean"), 1.5)
        self.assertEqual(gt.get("variance"), 2.5)

    def test_sensitivity_ground_truth_get_missing(self) -> None:
        """Test that get() raises for missing properties."""
        gt = SensitivityGroundTruth(mean=1.0)
        with self.assertRaises(ValueError) as ctx:
            gt.get("variance")
        self.assertIn("not available", str(ctx.exception))

    def test_optimization_ground_truth(self) -> None:
        """Test optimization ground truth."""
        import numpy as np

        gt = OptimizationGroundTruth(
            global_minimum=0.0,
            global_minimizers=np.array([[1.0], [1.0]]),
        )
        self.assertEqual(gt.get("global_minimum"), 0.0)
        self.assertIn("global_minimizers", gt.available())

    def test_quadrature_ground_truth(self) -> None:
        """Test quadrature ground truth."""
        gt = QuadratureGroundTruth(
            integral=3.14159,
            integral_formula="pi",
        )
        self.assertAlmostEqual(gt.get("integral"), 3.14159)
        self.assertEqual(gt.get("integral_formula"), "pi")

    def test_ground_truth_is_frozen(self) -> None:
        """Test that ground truth is immutable."""
        gt = SensitivityGroundTruth(mean=1.0)
        with self.assertRaises(Exception):
            gt.mean = 2.0  # type: ignore


class TestBoxDomain(Generic[Array], unittest.TestCase):
    """Tests for BoxDomain."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_bounds_shape(self) -> None:
        """Test that bounds have correct shape."""
        bounds = self._bkd.array([[-1.0, 1.0], [-2.0, 2.0], [-3.0, 3.0]])
        domain = BoxDomain(_bounds=bounds, _bkd=self._bkd)
        self.assertEqual(domain.bounds().shape, (3, 2))

    def test_nvars(self) -> None:
        """Test that nvars returns correct count."""
        bounds = self._bkd.array([[-1.0, 1.0], [-2.0, 2.0]])
        domain = BoxDomain(_bounds=bounds, _bkd=self._bkd)
        self.assertEqual(domain.nvars(), 2)

    def test_bkd(self) -> None:
        """Test that bkd returns the backend."""
        bounds = self._bkd.array([[-1.0, 1.0]])
        domain = BoxDomain(_bounds=bounds, _bkd=self._bkd)
        self.assertIs(domain.bkd(), self._bkd)

    def test_protocol_compliance(self) -> None:
        """Test that BoxDomain satisfies DomainProtocol."""
        bounds = self._bkd.array([[-1.0, 1.0]])
        domain = BoxDomain(_bounds=bounds, _bkd=self._bkd)
        self.assertIsInstance(domain, DomainProtocol)


class TestBoxDomainNumpy(TestBoxDomain[NDArray[Any]]):
    """NumPy backend tests for BoxDomain."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestBoxDomainTorch(TestBoxDomain[torch.Tensor]):
    """PyTorch backend tests for BoxDomain."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestBenchmarkRegistry(unittest.TestCase):
    """Tests for BenchmarkRegistry."""

    def setUp(self) -> None:
        # Save the current state
        self._saved_benchmarks = dict(BenchmarkRegistry._benchmarks)
        self._saved_categories = {
            k: list(v) for k, v in BenchmarkRegistry._categories.items()
        }
        self._saved_descriptions = dict(BenchmarkRegistry._descriptions)
        # Clear for isolated testing
        BenchmarkRegistry.clear()

    def tearDown(self) -> None:
        # Restore the saved state
        BenchmarkRegistry.clear()
        BenchmarkRegistry._benchmarks.update(self._saved_benchmarks)
        for k, v in self._saved_categories.items():
            BenchmarkRegistry._categories[k] = v
        BenchmarkRegistry._descriptions.update(self._saved_descriptions)

    def test_register_and_get(self) -> None:
        """Test registering and getting a benchmark."""

        @BenchmarkRegistry.register("test_bench", category="test")
        def _factory(bkd: Backend[Any]) -> str:
            return "test_benchmark_instance"

        bkd = NumpyBkd()
        result = BenchmarkRegistry.get("test_bench", bkd)
        self.assertEqual(result, "test_benchmark_instance")

    def test_list_all(self) -> None:
        """Test listing all benchmarks."""

        @BenchmarkRegistry.register("bench1", category="cat1")
        def _f1(bkd: Backend[Any]) -> str:
            return "b1"

        @BenchmarkRegistry.register("bench2", category="cat2")
        def _f2(bkd: Backend[Any]) -> str:
            return "b2"

        all_benchmarks = BenchmarkRegistry.list_all()
        self.assertIn("bench1", all_benchmarks)
        self.assertIn("bench2", all_benchmarks)

    def test_list_category(self) -> None:
        """Test listing benchmarks by category."""

        @BenchmarkRegistry.register("sens1", category="sensitivity")
        def _f1(bkd: Backend[Any]) -> str:
            return "s1"

        @BenchmarkRegistry.register("sens2", category="sensitivity")
        def _f2(bkd: Backend[Any]) -> str:
            return "s2"

        @BenchmarkRegistry.register("opt1", category="optimization")
        def _f3(bkd: Backend[Any]) -> str:
            return "o1"

        sens = BenchmarkRegistry.list_category("sensitivity")
        self.assertEqual(len(sens), 2)
        self.assertIn("sens1", sens)
        self.assertIn("sens2", sens)

        opt = BenchmarkRegistry.list_category("optimization")
        self.assertEqual(len(opt), 1)
        self.assertIn("opt1", opt)

    def test_categories(self) -> None:
        """Test listing all categories."""

        @BenchmarkRegistry.register("b1", category="cat1")
        def _f1(bkd: Backend[Any]) -> str:
            return "b1"

        @BenchmarkRegistry.register("b2", category="cat2")
        def _f2(bkd: Backend[Any]) -> str:
            return "b2"

        cats = BenchmarkRegistry.categories()
        self.assertIn("cat1", cats)
        self.assertIn("cat2", cats)

    def test_get_unknown_raises(self) -> None:
        """Test that getting unknown benchmark raises KeyError."""
        bkd = NumpyBkd()
        with self.assertRaises(KeyError):
            BenchmarkRegistry.get("nonexistent", bkd)

    def test_description(self) -> None:
        """Test benchmark description."""

        @BenchmarkRegistry.register(
            "described", category="test", description="A test benchmark"
        )
        def _f(bkd: Backend[Any]) -> str:
            return "d"

        desc = BenchmarkRegistry.description("described")
        self.assertEqual(desc, "A test benchmark")


if __name__ == "__main__":
    unittest.main()
