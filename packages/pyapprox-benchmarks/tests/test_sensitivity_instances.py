"""Tests for sensitivity benchmark instances."""

import math

from pyapprox_benchmarks.sensitivity import IshigamiBenchmark


class TestIshigamiBenchmark:
    """Tests for IshigamiBenchmark."""

    def test_problem_name(self, bkd) -> None:
        benchmark = IshigamiBenchmark(bkd)
        assert benchmark.problem().name() == "ishigami"

    def test_function_nvars(self, bkd) -> None:
        benchmark = IshigamiBenchmark(bkd)
        assert benchmark.problem().function().nvars() == 3

    def test_function_nqoi(self, bkd) -> None:
        benchmark = IshigamiBenchmark(bkd)
        assert benchmark.problem().function().nqoi() == 1

    def test_domain_nvars(self, bkd) -> None:
        benchmark = IshigamiBenchmark(bkd)
        assert benchmark.domain().nvars() == 3

    def test_domain_bounds(self, bkd) -> None:
        benchmark = IshigamiBenchmark(bkd)
        bounds = benchmark.domain().bounds()
        pi = math.pi
        expected = bkd.array([[-pi, pi], [-pi, pi], [-pi, pi]])
        bkd.assert_allclose(bounds, expected, atol=1e-14)

    def test_mean(self, bkd) -> None:
        """Test mean (a/2 = 3.5)."""
        benchmark = IshigamiBenchmark(bkd)
        bkd.assert_allclose(
            bkd.asarray([benchmark.mean()]),
            bkd.asarray([3.5]),
            atol=1e-14,
        )

    def test_variance(self, bkd) -> None:
        benchmark = IshigamiBenchmark(bkd)
        a, b = 7.0, 0.1
        pi = math.pi
        expected_var = a**2 / 8 + b * pi**4 / 5 + b**2 * pi**8 / 18 + 0.5
        bkd.assert_allclose(
            bkd.asarray([benchmark.variance()]),
            bkd.asarray([expected_var]),
            rtol=1e-12,
        )

    def test_main_effects_sum(self, bkd) -> None:
        benchmark = IshigamiBenchmark(bkd)
        main_effects = benchmark.main_effects()
        assert main_effects.shape == (3, 1)
        main_sum = float(bkd.sum(main_effects))
        assert main_sum < 1.0
        assert main_sum > 0.0

    def test_total_effects_sum(self, bkd) -> None:
        benchmark = IshigamiBenchmark(bkd)
        total_effects = benchmark.total_effects()
        assert total_effects.shape == (3, 1)
        total_sum = float(bkd.sum(total_effects))
        assert total_sum > 1.0

    def test_sobol_indices(self, bkd) -> None:
        benchmark = IshigamiBenchmark(bkd)
        sobol = benchmark.sobol_indices()
        assert (0,) in sobol
        assert (1,) in sobol
        assert (2,) in sobol
        assert (0, 2) in sobol
        bkd.assert_allclose(
            bkd.asarray([sobol[(2,)]]),
            bkd.asarray([0.0]),
            atol=1e-14,
        )

    def test_prior_nvars(self, bkd) -> None:
        benchmark = IshigamiBenchmark(bkd)
        prior = benchmark.problem().prior()
        assert prior.nvars() == 3

    def test_prior_samples_in_domain(self, bkd) -> None:
        benchmark = IshigamiBenchmark(bkd)
        prior = benchmark.problem().prior()
        bounds = benchmark.domain().bounds()

        samples = prior.rvs(100)
        assert samples.shape == (3, 100)

        for i in range(3):
            assert bkd.all_bool(samples[i, :] >= bounds[i, 0])
            assert bkd.all_bool(samples[i, :] <= bounds[i, 1])

    def test_function_evaluation_at_prior_samples(self, bkd) -> None:
        benchmark = IshigamiBenchmark(bkd)
        prior = benchmark.problem().prior()
        func = benchmark.problem().function()

        samples = prior.rvs(10)
        values = func(samples)
        assert values.shape == (1, 10)
