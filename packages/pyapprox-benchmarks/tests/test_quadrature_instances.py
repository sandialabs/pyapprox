"""Tests for quadrature benchmark classes."""

from pyapprox_benchmarks.quadrature import (
    GenzCornerPeakBenchmark,
    GenzGaussianPeakBenchmark,
    GenzOscillatoryBenchmark,
    GenzProductPeakBenchmark,
)


class TestGenzOscillatory2D:
    """Tests for GenzOscillatoryBenchmark with nvars=2."""

    def test_problem_name(self, bkd) -> None:
        benchmark = GenzOscillatoryBenchmark(bkd, nvars=2)
        assert benchmark.problem().name() == "genz_oscillatory_2d"

    def test_domain_nvars(self, bkd) -> None:
        benchmark = GenzOscillatoryBenchmark(bkd, nvars=2)
        assert benchmark.problem().domain().nvars() == 2

    def test_domain_bounds(self, bkd) -> None:
        benchmark = GenzOscillatoryBenchmark(bkd, nvars=2)
        bounds = benchmark.problem().domain().bounds()
        expected = bkd.array([[0.0, 1.0], [0.0, 1.0]])
        bkd.assert_allclose(bounds, expected, rtol=1e-12)

    def test_function_nvars(self, bkd) -> None:
        benchmark = GenzOscillatoryBenchmark(bkd, nvars=2)
        assert benchmark.problem().function().nvars() == 2

    def test_function_nqoi(self, bkd) -> None:
        benchmark = GenzOscillatoryBenchmark(bkd, nvars=2)
        assert benchmark.problem().function().nqoi() == 1

    def test_integral(self, bkd) -> None:
        benchmark = GenzOscillatoryBenchmark(bkd, nvars=2)
        assert abs(benchmark.integral()) < float("inf")

    def test_function_evaluation(self, bkd) -> None:
        benchmark = GenzOscillatoryBenchmark(bkd, nvars=2)
        func = benchmark.problem().function()
        sample = bkd.array([[0.5], [0.5]])
        result = func(sample)
        assert result.shape == (1, 1)


class TestGenzProductPeak2D:
    """Tests for GenzProductPeakBenchmark with nvars=2."""

    def test_problem_name(self, bkd) -> None:
        benchmark = GenzProductPeakBenchmark(bkd, nvars=2)
        assert benchmark.problem().name() == "genz_product_peak_2d"

    def test_domain_nvars(self, bkd) -> None:
        benchmark = GenzProductPeakBenchmark(bkd, nvars=2)
        assert benchmark.problem().domain().nvars() == 2

    def test_integral(self, bkd) -> None:
        benchmark = GenzProductPeakBenchmark(bkd, nvars=2)
        assert abs(benchmark.integral()) < float("inf")


class TestGenzCornerPeak2D:
    """Tests for GenzCornerPeakBenchmark with nvars=2."""

    def test_problem_name(self, bkd) -> None:
        benchmark = GenzCornerPeakBenchmark(bkd, nvars=2)
        assert benchmark.problem().name() == "genz_corner_peak_2d"

    def test_domain_nvars(self, bkd) -> None:
        benchmark = GenzCornerPeakBenchmark(bkd, nvars=2)
        assert benchmark.problem().domain().nvars() == 2

    def test_integral(self, bkd) -> None:
        benchmark = GenzCornerPeakBenchmark(bkd, nvars=2)
        assert abs(benchmark.integral()) < float("inf")


class TestGenzGaussianPeak2D:
    """Tests for GenzGaussianPeakBenchmark with nvars=2."""

    def test_problem_name(self, bkd) -> None:
        benchmark = GenzGaussianPeakBenchmark(bkd, nvars=2)
        assert benchmark.problem().name() == "genz_gaussian_peak_2d"

    def test_domain_nvars(self, bkd) -> None:
        benchmark = GenzGaussianPeakBenchmark(bkd, nvars=2)
        assert benchmark.problem().domain().nvars() == 2

    def test_integral(self, bkd) -> None:
        benchmark = GenzGaussianPeakBenchmark(bkd, nvars=2)
        assert abs(benchmark.integral()) < float("inf")


class TestGenzOscillatory5D:
    """Tests for GenzOscillatoryBenchmark with nvars=5."""

    def test_problem_name(self, bkd) -> None:
        benchmark = GenzOscillatoryBenchmark(bkd, nvars=5)
        assert benchmark.problem().name() == "genz_oscillatory_5d"

    def test_domain_nvars(self, bkd) -> None:
        benchmark = GenzOscillatoryBenchmark(bkd, nvars=5)
        assert benchmark.problem().domain().nvars() == 5

    def test_function_nvars(self, bkd) -> None:
        benchmark = GenzOscillatoryBenchmark(bkd, nvars=5)
        assert benchmark.problem().function().nvars() == 5

    def test_integral(self, bkd) -> None:
        benchmark = GenzOscillatoryBenchmark(bkd, nvars=5)
        assert abs(benchmark.integral()) < float("inf")


class TestGenzGaussianPeak5D:
    """Tests for GenzGaussianPeakBenchmark with nvars=5."""

    def test_problem_name(self, bkd) -> None:
        benchmark = GenzGaussianPeakBenchmark(bkd, nvars=5)
        assert benchmark.problem().name() == "genz_gaussian_peak_5d"

    def test_domain_nvars(self, bkd) -> None:
        benchmark = GenzGaussianPeakBenchmark(bkd, nvars=5)
        assert benchmark.problem().domain().nvars() == 5

    def test_integral(self, bkd) -> None:
        benchmark = GenzGaussianPeakBenchmark(bkd, nvars=5)
        assert abs(benchmark.integral()) < float("inf")
