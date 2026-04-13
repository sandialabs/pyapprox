"""Tests for optimization benchmark instances."""

from pyapprox_benchmarks.optimization import RosenbrockBenchmark


class TestRosenbrockBenchmark2D:
    """Tests for RosenbrockBenchmark with nvars=2."""

    def test_problem_name(self, bkd) -> None:
        benchmark = RosenbrockBenchmark(bkd, nvars=2)
        assert benchmark.problem().name() == "rosenbrock_2d"

    def test_function_nvars(self, bkd) -> None:
        benchmark = RosenbrockBenchmark(bkd, nvars=2)
        assert benchmark.problem().function().nvars() == 2

    def test_function_nqoi(self, bkd) -> None:
        benchmark = RosenbrockBenchmark(bkd, nvars=2)
        assert benchmark.problem().function().nqoi() == 1

    def test_domain_nvars(self, bkd) -> None:
        benchmark = RosenbrockBenchmark(bkd, nvars=2)
        assert benchmark.problem().domain().nvars() == 2

    def test_domain_bounds(self, bkd) -> None:
        benchmark = RosenbrockBenchmark(bkd, nvars=2)
        bounds = benchmark.problem().domain().bounds()
        expected = bkd.array([[-5.0, 10.0], [-5.0, 10.0]])
        bkd.assert_allclose(bounds, expected, atol=1e-14)

    def test_global_minimum(self, bkd) -> None:
        benchmark = RosenbrockBenchmark(bkd, nvars=2)
        bkd.assert_allclose(
            bkd.asarray([benchmark.global_minimum()]),
            bkd.asarray([0.0]),
            atol=1e-14,
        )

    def test_global_minimizer(self, bkd) -> None:
        benchmark = RosenbrockBenchmark(bkd, nvars=2)
        minimizer = benchmark.global_minimizers()
        expected = bkd.asarray([[1.0], [1.0]])
        bkd.assert_allclose(bkd.asarray(minimizer), expected, atol=1e-14)

    def test_function_at_minimizer(self, bkd) -> None:
        benchmark = RosenbrockBenchmark(bkd, nvars=2)
        func = benchmark.problem().function()
        minimizer = bkd.array([[1.0], [1.0]])
        value = func(minimizer)
        bkd.assert_allclose(value, bkd.zeros((1, 1)), atol=1e-14)


class TestRosenbrockBenchmark10D:
    """Tests for RosenbrockBenchmark with nvars=10."""

    def test_problem_name(self, bkd) -> None:
        benchmark = RosenbrockBenchmark(bkd, nvars=10)
        assert benchmark.problem().name() == "rosenbrock_10d"

    def test_function_nvars(self, bkd) -> None:
        benchmark = RosenbrockBenchmark(bkd, nvars=10)
        assert benchmark.problem().function().nvars() == 10

    def test_domain_nvars(self, bkd) -> None:
        benchmark = RosenbrockBenchmark(bkd, nvars=10)
        assert benchmark.problem().domain().nvars() == 10

    def test_function_at_minimizer(self, bkd) -> None:
        benchmark = RosenbrockBenchmark(bkd, nvars=10)
        func = benchmark.problem().function()
        minimizer = bkd.ones((10, 1))
        value = func(minimizer)
        bkd.assert_allclose(value, bkd.zeros((1, 1)), atol=1e-14)
