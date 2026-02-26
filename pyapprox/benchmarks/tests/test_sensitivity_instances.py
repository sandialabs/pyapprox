"""Tests for sensitivity benchmark instances."""

import math

from pyapprox.benchmarks.instances.sensitivity import ishigami_3d
from pyapprox.benchmarks.protocols import BenchmarkWithPriorProtocol
from pyapprox.benchmarks.registry import BenchmarkRegistry
from pyapprox.util.backends.numpy import NumpyBkd


class TestIshigami3DBenchmark:
    """Tests for ishigami_3d benchmark instance."""

    def test_protocol_compliance(self, bkd) -> None:
        """Test that ishigami_3d satisfies BenchmarkWithPriorProtocol."""
        benchmark = ishigami_3d(bkd)
        assert isinstance(benchmark, BenchmarkWithPriorProtocol)

    def test_name(self, bkd) -> None:
        """Test benchmark name."""
        benchmark = ishigami_3d(bkd)
        assert benchmark.name() == "ishigami_3d"

    def test_function_nvars(self, bkd) -> None:
        """Test function nvars."""
        benchmark = ishigami_3d(bkd)
        assert benchmark.function().nvars() == 3

    def test_function_nqoi(self, bkd) -> None:
        """Test function nqoi."""
        benchmark = ishigami_3d(bkd)
        assert benchmark.function().nqoi() == 1

    def test_domain_nvars(self, bkd) -> None:
        """Test domain nvars."""
        benchmark = ishigami_3d(bkd)
        assert benchmark.domain().nvars() == 3

    def test_domain_bounds(self, bkd) -> None:
        """Test domain bounds are [-pi, pi]^3."""
        benchmark = ishigami_3d(bkd)
        bounds = benchmark.domain().bounds()
        pi = math.pi
        expected = bkd.array([[-pi, pi], [-pi, pi], [-pi, pi]])
        bkd.assert_allclose(bounds, expected, atol=1e-14)

    def test_ground_truth_mean(self, bkd) -> None:
        """Test ground truth mean (a/2 = 3.5)."""
        benchmark = ishigami_3d(bkd)
        gt = benchmark.ground_truth()
        bkd.assert_allclose(
            bkd.asarray([gt.get("mean")]),
            bkd.asarray([3.5]),
            atol=1e-14,
        )

    def test_ground_truth_variance(self, bkd) -> None:
        """Test ground truth variance."""
        benchmark = ishigami_3d(bkd)
        gt = benchmark.ground_truth()
        a, b = 7.0, 0.1
        pi = math.pi
        expected_var = a**2 / 8 + b * pi**4 / 5 + b**2 * pi**8 / 18 + 0.5
        bkd.assert_allclose(
            bkd.asarray([gt.get("variance")]),
            bkd.asarray([expected_var]),
            rtol=1e-12,
        )

    def test_ground_truth_main_effects_sum(self, bkd) -> None:
        """Test main effects sum (should not exceed 1)."""
        benchmark = ishigami_3d(bkd)
        gt = benchmark.ground_truth()
        main_effects = gt.get("main_effects")
        # main_effects has shape (nvars, 1)
        assert main_effects.shape == (3, 1)
        # Sum of main effects should be less than 1 due to interactions
        main_sum = float(bkd.sum(main_effects))
        assert main_sum < 1.0
        assert main_sum > 0.0

    def test_ground_truth_total_effects_sum(self, bkd) -> None:
        """Test total effects sum (can exceed 1 due to interactions)."""
        benchmark = ishigami_3d(bkd)
        gt = benchmark.ground_truth()
        total_effects = gt.get("total_effects")
        # total_effects has shape (nvars, 1)
        assert total_effects.shape == (3, 1)
        # Total effects sum can exceed 1
        total_sum = float(bkd.sum(total_effects))
        assert total_sum > 1.0

    def test_ground_truth_sobol_indices(self, bkd) -> None:
        """Test Sobol indices are available."""
        benchmark = ishigami_3d(bkd)
        gt = benchmark.ground_truth()
        sobol = gt.get("sobol_indices")
        assert (0,) in sobol
        assert (1,) in sobol
        assert (2,) in sobol
        assert (0, 2) in sobol
        # S2 should be 0
        bkd.assert_allclose(
            bkd.asarray([sobol[(2,)]]),
            bkd.asarray([0.0]),
            atol=1e-14,
        )

    def test_prior_nvars(self, bkd) -> None:
        """Test prior has correct nvars."""
        benchmark = ishigami_3d(bkd)
        prior = benchmark.prior()
        assert prior.nvars() == 3

    def test_prior_samples_in_domain(self, bkd) -> None:
        """Test that samples from prior are in domain."""
        benchmark = ishigami_3d(bkd)
        prior = benchmark.prior()
        bounds = benchmark.domain().bounds()

        samples = prior.rvs(100)
        assert samples.shape == (3, 100)

        # Check all samples are within bounds
        for i in range(3):
            assert bkd.all_bool(samples[i, :] >= bounds[i, 0])
            assert bkd.all_bool(samples[i, :] <= bounds[i, 1])

    def test_function_evaluation_at_prior_samples(self, bkd) -> None:
        """Test function can be evaluated at prior samples."""
        benchmark = ishigami_3d(bkd)
        prior = benchmark.prior()
        func = benchmark.function()

        samples = prior.rvs(10)
        values = func(samples)
        assert values.shape == (1, 10)


class TestBenchmarkRegistrySensitivity:
    """Test registry for sensitivity benchmarks."""

    def test_ishigami_registered(self) -> None:
        """Test ishigami_3d is registered."""
        # Import to trigger registration
        from pyapprox.benchmarks.instances import sensitivity  # noqa: F401

        bkd = NumpyBkd()
        benchmark = BenchmarkRegistry.get("ishigami_3d", bkd)
        assert benchmark.name() == "ishigami_3d"

    def test_ishigami_in_analytic_category(self) -> None:
        """Test ishigami_3d is in analytic category."""
        from pyapprox.benchmarks.instances import analytic  # noqa: F401

        assert "ishigami_3d" in BenchmarkRegistry.list_category("analytic")
