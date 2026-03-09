"""Tests for optimization benchmark instances."""

from pyapprox.benchmarks.instances.optimization import (
    rosenbrock_2d,
    rosenbrock_10d,
)
from pyapprox.benchmarks.protocols import BenchmarkProtocol
from pyapprox.benchmarks.registry import BenchmarkRegistry
from pyapprox.util.backends.numpy import NumpyBkd

#TODO: this test class should be where function is defined not at this level which is for integration tests.

class TestRosenbrock2DBenchmark:
    """Tests for rosenbrock_2d benchmark instance."""

    def test_protocol_compliance(self, bkd) -> None:
        """Test that rosenbrock_2d satisfies BenchmarkProtocol."""
        benchmark = rosenbrock_2d(bkd)
        assert isinstance(benchmark, BenchmarkProtocol)

    def test_name(self, bkd) -> None:
        """Test benchmark name."""
        benchmark = rosenbrock_2d(bkd)
        assert benchmark.name() == "rosenbrock_2d"

    def test_function_nvars(self, bkd) -> None:
        """Test function nvars."""
        benchmark = rosenbrock_2d(bkd)
        assert benchmark.function().nvars() == 2

    def test_function_nqoi(self, bkd) -> None:
        """Test function nqoi."""
        benchmark = rosenbrock_2d(bkd)
        assert benchmark.function().nqoi() == 1

    def test_domain_nvars(self, bkd) -> None:
        """Test domain nvars."""
        benchmark = rosenbrock_2d(bkd)
        assert benchmark.domain().nvars() == 2

    def test_domain_bounds(self, bkd) -> None:
        """Test domain bounds."""
        benchmark = rosenbrock_2d(bkd)
        bounds = benchmark.domain().bounds()
        expected = bkd.array([[-5.0, 10.0], [-5.0, 10.0]])
        bkd.assert_allclose(bounds, expected, atol=1e-14)

    def test_ground_truth_global_minimum(self, bkd) -> None:
        """Test ground truth global minimum is 0."""
        benchmark = rosenbrock_2d(bkd)
        gt = benchmark.ground_truth()
        bkd.assert_allclose(
            bkd.asarray([gt.get("global_minimum")]),
            bkd.asarray([0.0]),
            atol=1e-14,
        )

    def test_ground_truth_global_minimizer(self, bkd) -> None:
        """Test ground truth global minimizer is (1, 1)."""
        benchmark = rosenbrock_2d(bkd)
        gt = benchmark.ground_truth()
        minimizer = gt.get("global_minimizers")
        expected = bkd.asarray([[1.0], [1.0]])
        bkd.assert_allclose(
            bkd.asarray(minimizer),
            expected,
            atol=1e-14,
        )

    def test_function_at_minimizer(self, bkd) -> None:
        """Test function value at minimizer is 0."""
        benchmark = rosenbrock_2d(bkd)
        func = benchmark.function()
        minimizer = bkd.array([[1.0], [1.0]])
        value = func(minimizer)
        bkd.assert_allclose(value, bkd.zeros((1, 1)), atol=1e-14)


class TestRosenbrock10DBenchmark:
    """Tests for rosenbrock_10d benchmark instance."""

    def test_protocol_compliance(self, bkd) -> None:
        """Test that rosenbrock_10d satisfies BenchmarkProtocol."""
        benchmark = rosenbrock_10d(bkd)
        assert isinstance(benchmark, BenchmarkProtocol)

    def test_name(self, bkd) -> None:
        """Test benchmark name."""
        benchmark = rosenbrock_10d(bkd)
        assert benchmark.name() == "rosenbrock_10d"

    def test_function_nvars(self, bkd) -> None:
        """Test function nvars."""
        benchmark = rosenbrock_10d(bkd)
        assert benchmark.function().nvars() == 10

    def test_domain_nvars(self, bkd) -> None:
        """Test domain nvars."""
        benchmark = rosenbrock_10d(bkd)
        assert benchmark.domain().nvars() == 10

    def test_function_at_minimizer(self, bkd) -> None:
        """Test function value at minimizer is 0."""
        benchmark = rosenbrock_10d(bkd)
        func = benchmark.function()
        minimizer = bkd.ones((10, 1))
        value = func(minimizer)
        bkd.assert_allclose(value, bkd.zeros((1, 1)), atol=1e-14)


class TestBenchmarkRegistryOptimization:
    """Test registry for optimization benchmarks."""

    def test_rosenbrock_2d_registered(self) -> None:
        """Test rosenbrock_2d is registered."""
        from pyapprox.benchmarks.instances import optimization  # noqa: F401

        bkd = NumpyBkd()
        benchmark = BenchmarkRegistry.get("rosenbrock_2d", bkd)
        assert benchmark.name() == "rosenbrock_2d"

    def test_rosenbrock_10d_registered(self) -> None:
        """Test rosenbrock_10d is registered."""
        from pyapprox.benchmarks.instances import optimization  # noqa: F401

        bkd = NumpyBkd()
        benchmark = BenchmarkRegistry.get("rosenbrock_10d", bkd)
        assert benchmark.name() == "rosenbrock_10d"

    def test_benchmarks_in_analytic_category(self) -> None:
        """Test optimization benchmarks are in analytic category."""
        from pyapprox.benchmarks.instances import analytic  # noqa: F401

        analytic_benchmarks = BenchmarkRegistry.list_category("analytic")
        assert "rosenbrock_2d" in analytic_benchmarks
        assert "rosenbrock_10d" in analytic_benchmarks
        assert "branin_2d" in analytic_benchmarks
