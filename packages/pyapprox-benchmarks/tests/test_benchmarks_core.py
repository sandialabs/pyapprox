"""Tests for benchmark core infrastructure."""

import pytest

from pyapprox_benchmarks.benchmark import (
    BoxDomain,
)
from pyapprox_benchmarks.ground_truth import (
    OptimizationGroundTruth,
    QuadratureGroundTruth,
    SensitivityGroundTruth,
)
from pyapprox_benchmarks.protocols import (
    DomainProtocol,
)
from pyapprox_benchmarks.registry import BenchmarkRegistry
from pyapprox.util.backends.numpy import NumpyBkd


class TestGroundTruth:
    """Tests for ground truth dataclasses."""

    def test_sensitivity_ground_truth_available(self) -> None:
        """Test that available() returns correct properties."""
        gt = SensitivityGroundTruth(
            mean=1.0,
            variance=2.0,
        )
        available = gt.available()
        assert "mean" in available
        assert "variance" in available
        assert "main_effects" not in available

    def test_sensitivity_ground_truth_get(self) -> None:
        """Test that get() returns correct values."""
        gt = SensitivityGroundTruth(
            mean=1.5,
            variance=2.5,
        )
        assert gt.get("mean") == 1.5
        assert gt.get("variance") == 2.5

    def test_sensitivity_ground_truth_get_missing(self) -> None:
        """Test that get() raises for missing properties."""
        gt = SensitivityGroundTruth(mean=1.0)
        with pytest.raises(ValueError) as ctx:
            gt.get("variance")
        assert "not available" in str(ctx.value)

    def test_optimization_ground_truth(self) -> None:
        """Test optimization ground truth."""
        import numpy as np

        gt = OptimizationGroundTruth(
            global_minimum=0.0,
            global_minimizers=np.array([[1.0], [1.0]]),
        )
        assert gt.get("global_minimum") == 0.0
        assert "global_minimizers" in gt.available()

    def test_quadrature_ground_truth(self) -> None:
        """Test quadrature ground truth."""
        gt = QuadratureGroundTruth(
            integral=3.14159,
            integral_formula="pi",
        )
        assert gt.get("integral") == pytest.approx(3.14159)
        assert gt.get("integral_formula") == "pi"

    def test_ground_truth_is_frozen(self) -> None:
        """Test that ground truth is immutable."""
        gt = SensitivityGroundTruth(mean=1.0)
        with pytest.raises(Exception):
            gt.mean = 2.0  # type: ignore


class TestBoxDomain:
    """Tests for BoxDomain."""

    def test_bounds_shape(self, bkd) -> None:
        """Test that bounds have correct shape."""
        bounds = bkd.array([[-1.0, 1.0], [-2.0, 2.0], [-3.0, 3.0]])
        domain = BoxDomain(_bounds=bounds, _bkd=bkd)
        assert domain.bounds().shape == (3, 2)

    def test_nvars(self, bkd) -> None:
        """Test that nvars returns correct count."""
        bounds = bkd.array([[-1.0, 1.0], [-2.0, 2.0]])
        domain = BoxDomain(_bounds=bounds, _bkd=bkd)
        assert domain.nvars() == 2

    def test_bkd(self, bkd) -> None:
        """Test that bkd returns the backend."""
        bounds = bkd.array([[-1.0, 1.0]])
        domain = BoxDomain(_bounds=bounds, _bkd=bkd)
        assert domain.bkd() is bkd

    def test_protocol_compliance(self, bkd) -> None:
        """Test that BoxDomain satisfies DomainProtocol."""
        bounds = bkd.array([[-1.0, 1.0]])
        domain = BoxDomain(_bounds=bounds, _bkd=bkd)
        assert isinstance(domain, DomainProtocol)


class TestBenchmarkRegistry:
    """Tests for BenchmarkRegistry."""

    @pytest.fixture(autouse=True)
    def _save_restore_registry(self):
        # Save the current state
        saved_benchmarks = dict(BenchmarkRegistry._benchmarks)
        saved_categories = {
            k: list(v) for k, v in BenchmarkRegistry._categories.items()
        }
        saved_descriptions = dict(BenchmarkRegistry._descriptions)
        # Clear for isolated testing
        BenchmarkRegistry.clear()
        yield
        # Restore the saved state
        BenchmarkRegistry.clear()
        BenchmarkRegistry._benchmarks.update(saved_benchmarks)
        for k, v in saved_categories.items():
            BenchmarkRegistry._categories[k] = v
        BenchmarkRegistry._descriptions.update(saved_descriptions)

    def test_register_and_get(self) -> None:
        """Test registering and getting a benchmark."""
        from typing import Any

        @BenchmarkRegistry.register("test_bench", category="test")
        def _factory(bkd: Any) -> str:
            return "test_benchmark_instance"

        bkd = NumpyBkd()
        result = BenchmarkRegistry.get("test_bench", bkd)
        assert result == "test_benchmark_instance"

    def test_list_all(self) -> None:
        """Test listing all benchmarks."""
        from typing import Any

        @BenchmarkRegistry.register("bench1", category="cat1")
        def _f1(bkd: Any) -> str:
            return "b1"

        @BenchmarkRegistry.register("bench2", category="cat2")
        def _f2(bkd: Any) -> str:
            return "b2"

        all_benchmarks = BenchmarkRegistry.list_all()
        assert "bench1" in all_benchmarks
        assert "bench2" in all_benchmarks

    def test_list_category(self) -> None:
        """Test listing benchmarks by category."""
        from typing import Any

        @BenchmarkRegistry.register("sens1", category="sensitivity")
        def _f1(bkd: Any) -> str:
            return "s1"

        @BenchmarkRegistry.register("sens2", category="sensitivity")
        def _f2(bkd: Any) -> str:
            return "s2"

        @BenchmarkRegistry.register("opt1", category="optimization")
        def _f3(bkd: Any) -> str:
            return "o1"

        sens = BenchmarkRegistry.list_category("sensitivity")
        assert len(sens) == 2
        assert "sens1" in sens
        assert "sens2" in sens

        opt = BenchmarkRegistry.list_category("optimization")
        assert len(opt) == 1
        assert "opt1" in opt

    def test_categories(self) -> None:
        """Test listing all categories."""
        from typing import Any

        @BenchmarkRegistry.register("b1", category="cat1")
        def _f1(bkd: Any) -> str:
            return "b1"

        @BenchmarkRegistry.register("b2", category="cat2")
        def _f2(bkd: Any) -> str:
            return "b2"

        cats = BenchmarkRegistry.categories()
        assert "cat1" in cats
        assert "cat2" in cats

    def test_get_unknown_raises(self) -> None:
        """Test that getting unknown benchmark raises KeyError."""
        bkd = NumpyBkd()
        with pytest.raises(KeyError):
            BenchmarkRegistry.get("nonexistent", bkd)

    def test_description(self) -> None:
        """Test benchmark description."""
        from typing import Any

        @BenchmarkRegistry.register(
            "described", category="test", description="A test benchmark"
        )
        def _f(bkd: Any) -> str:
            return "d"

        desc = BenchmarkRegistry.description("described")
        assert desc == "A test benchmark"
