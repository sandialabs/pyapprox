"""Tests for quadrature benchmark instances."""

# TODO: this test class should be where function is defined
# not at this level which is for integration tests.

from pyapprox_benchmarks.instances.quadrature import (
    genz_corner_peak_2d,
    genz_gaussian_peak_2d,
    genz_gaussian_peak_5d,
    genz_oscillatory_2d,
    genz_oscillatory_5d,
    genz_product_peak_2d,
)
from pyapprox_benchmarks.registry import BenchmarkRegistry


class TestGenzOscillatory2D:
    """Tests for genz_oscillatory_2d benchmark instance."""

    def test_name(self, bkd) -> None:
        """Test benchmark name."""
        benchmark = genz_oscillatory_2d(bkd)
        assert benchmark.name() == "genz_oscillatory_2d"

    def test_domain_nvars(self, bkd) -> None:
        """Test domain has 2 variables."""
        benchmark = genz_oscillatory_2d(bkd)
        assert benchmark.domain().nvars() == 2

    def test_domain_bounds(self, bkd) -> None:
        """Test domain bounds are [0, 1]^2."""
        benchmark = genz_oscillatory_2d(bkd)
        bounds = benchmark.domain().bounds()
        expected = bkd.array([[0.0, 1.0], [0.0, 1.0]])
        bkd.assert_allclose(bounds, expected, rtol=1e-12)

    def test_function_nvars(self, bkd) -> None:
        """Test function has 2 input variables."""
        benchmark = genz_oscillatory_2d(bkd)
        assert benchmark.function().nvars() == 2

    def test_function_nqoi(self, bkd) -> None:
        """Test function has 1 output."""
        benchmark = genz_oscillatory_2d(bkd)
        assert benchmark.function().nqoi() == 1

    def test_ground_truth_integral(self, bkd) -> None:
        """Test ground truth has integral."""
        benchmark = genz_oscillatory_2d(bkd)
        gt = benchmark.ground_truth()
        assert "integral" in gt.available()
        # Integral should be a finite number
        assert abs(gt.integral) < float("inf")

    def test_function_evaluation(self, bkd) -> None:
        """Test function can be evaluated."""
        benchmark = genz_oscillatory_2d(bkd)
        func = benchmark.function()
        sample = bkd.array([[0.5], [0.5]])
        result = func(sample)
        assert result.shape == (1, 1)


class TestGenzProductPeak2D:
    """Tests for genz_product_peak_2d benchmark instance."""

    def test_name(self, bkd) -> None:
        """Test benchmark name."""
        benchmark = genz_product_peak_2d(bkd)
        assert benchmark.name() == "genz_product_peak_2d"

    def test_domain_nvars(self, bkd) -> None:
        """Test domain has 2 variables."""
        benchmark = genz_product_peak_2d(bkd)
        assert benchmark.domain().nvars() == 2

    def test_ground_truth_integral(self, bkd) -> None:
        """Test ground truth has integral."""
        benchmark = genz_product_peak_2d(bkd)
        gt = benchmark.ground_truth()
        assert "integral" in gt.available()


class TestGenzCornerPeak2D:
    """Tests for genz_corner_peak_2d benchmark instance."""

    def test_name(self, bkd) -> None:
        """Test benchmark name."""
        benchmark = genz_corner_peak_2d(bkd)
        assert benchmark.name() == "genz_corner_peak_2d"

    def test_domain_nvars(self, bkd) -> None:
        """Test domain has 2 variables."""
        benchmark = genz_corner_peak_2d(bkd)
        assert benchmark.domain().nvars() == 2

    def test_ground_truth_integral(self, bkd) -> None:
        """Test ground truth has integral."""
        benchmark = genz_corner_peak_2d(bkd)
        gt = benchmark.ground_truth()
        assert "integral" in gt.available()


class TestGenzGaussianPeak2D:
    """Tests for genz_gaussian_peak_2d benchmark instance."""

    def test_name(self, bkd) -> None:
        """Test benchmark name."""
        benchmark = genz_gaussian_peak_2d(bkd)
        assert benchmark.name() == "genz_gaussian_peak_2d"

    def test_domain_nvars(self, bkd) -> None:
        """Test domain has 2 variables."""
        benchmark = genz_gaussian_peak_2d(bkd)
        assert benchmark.domain().nvars() == 2

    def test_ground_truth_integral(self, bkd) -> None:
        """Test ground truth has integral."""
        benchmark = genz_gaussian_peak_2d(bkd)
        gt = benchmark.ground_truth()
        assert "integral" in gt.available()


class TestGenzOscillatory5D:
    """Tests for genz_oscillatory_5d benchmark instance."""

    def test_name(self, bkd) -> None:
        """Test benchmark name."""
        benchmark = genz_oscillatory_5d(bkd)
        assert benchmark.name() == "genz_oscillatory_5d"

    def test_domain_nvars(self, bkd) -> None:
        """Test domain has 5 variables."""
        benchmark = genz_oscillatory_5d(bkd)
        assert benchmark.domain().nvars() == 5

    def test_function_nvars(self, bkd) -> None:
        """Test function has 5 input variables."""
        benchmark = genz_oscillatory_5d(bkd)
        assert benchmark.function().nvars() == 5

    def test_ground_truth_integral(self, bkd) -> None:
        """Test ground truth has integral."""
        benchmark = genz_oscillatory_5d(bkd)
        gt = benchmark.ground_truth()
        assert "integral" in gt.available()


class TestGenzGaussianPeak5D:
    """Tests for genz_gaussian_peak_5d benchmark instance."""

    def test_name(self, bkd) -> None:
        """Test benchmark name."""
        benchmark = genz_gaussian_peak_5d(bkd)
        assert benchmark.name() == "genz_gaussian_peak_5d"

    def test_domain_nvars(self, bkd) -> None:
        """Test domain has 5 variables."""
        benchmark = genz_gaussian_peak_5d(bkd)
        assert benchmark.domain().nvars() == 5

    def test_ground_truth_integral(self, bkd) -> None:
        """Test ground truth has integral."""
        benchmark = genz_gaussian_peak_5d(bkd)
        gt = benchmark.ground_truth()
        assert "integral" in gt.available()


class TestBenchmarkRegistryQuadrature:
    """Tests for BenchmarkRegistry quadrature category."""

    def test_oscillatory_2d_registered(self) -> None:
        """Test genz_oscillatory_2d is registered."""
        assert "genz_oscillatory_2d" in BenchmarkRegistry.list_all()

    def test_product_peak_2d_registered(self) -> None:
        """Test genz_product_peak_2d is registered."""
        assert "genz_product_peak_2d" in BenchmarkRegistry.list_all()

    def test_corner_peak_2d_registered(self) -> None:
        """Test genz_corner_peak_2d is registered."""
        assert "genz_corner_peak_2d" in BenchmarkRegistry.list_all()

    def test_gaussian_peak_2d_registered(self) -> None:
        """Test genz_gaussian_peak_2d is registered."""
        assert "genz_gaussian_peak_2d" in BenchmarkRegistry.list_all()

    def test_analytic_category(self) -> None:
        """Test Genz benchmarks are in analytic category."""
        analytic = BenchmarkRegistry.list_category("analytic")
        assert "genz_oscillatory_2d" in analytic
        assert "genz_product_peak_2d" in analytic
        assert "genz_corner_peak_2d" in analytic
        assert "genz_gaussian_peak_2d" in analytic
