"""Tests for sensitivity benchmark instances."""

import math
import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.test_utils import load_tests

from pyapprox.typing.benchmarks.protocols import BenchmarkWithPriorProtocol
from pyapprox.typing.benchmarks.registry import BenchmarkRegistry
from pyapprox.typing.benchmarks.instances.sensitivity import ishigami_3d


class TestIshigami3DBenchmark(Generic[Array], unittest.TestCase):
    """Tests for ishigami_3d benchmark instance."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_protocol_compliance(self) -> None:
        """Test that ishigami_3d satisfies BenchmarkWithPriorProtocol."""
        benchmark = ishigami_3d(self._bkd)
        self.assertIsInstance(benchmark, BenchmarkWithPriorProtocol)

    def test_name(self) -> None:
        """Test benchmark name."""
        benchmark = ishigami_3d(self._bkd)
        self.assertEqual(benchmark.name(), "ishigami_3d")

    def test_function_nvars(self) -> None:
        """Test function nvars."""
        benchmark = ishigami_3d(self._bkd)
        self.assertEqual(benchmark.function().nvars(), 3)

    def test_function_nqoi(self) -> None:
        """Test function nqoi."""
        benchmark = ishigami_3d(self._bkd)
        self.assertEqual(benchmark.function().nqoi(), 1)

    def test_domain_nvars(self) -> None:
        """Test domain nvars."""
        benchmark = ishigami_3d(self._bkd)
        self.assertEqual(benchmark.domain().nvars(), 3)

    def test_domain_bounds(self) -> None:
        """Test domain bounds are [-pi, pi]^3."""
        benchmark = ishigami_3d(self._bkd)
        bounds = benchmark.domain().bounds()
        pi = math.pi
        expected = self._bkd.array([[-pi, pi], [-pi, pi], [-pi, pi]])
        self._bkd.assert_allclose(bounds, expected, atol=1e-14)

    def test_ground_truth_mean(self) -> None:
        """Test ground truth mean (a/2 = 3.5)."""
        benchmark = ishigami_3d(self._bkd)
        gt = benchmark.ground_truth()
        self._bkd.assert_allclose(
            self._bkd.asarray([gt.get("mean")]),
            self._bkd.asarray([3.5]),
            atol=1e-14,
        )

    def test_ground_truth_variance(self) -> None:
        """Test ground truth variance."""
        benchmark = ishigami_3d(self._bkd)
        gt = benchmark.ground_truth()
        a, b = 7.0, 0.1
        pi = math.pi
        expected_var = a**2 / 8 + b * pi**4 / 5 + b**2 * pi**8 / 18 + 0.5
        self._bkd.assert_allclose(
            self._bkd.asarray([gt.get("variance")]),
            self._bkd.asarray([expected_var]),
            rtol=1e-12,
        )

    def test_ground_truth_main_effects_sum(self) -> None:
        """Test main effects sum (should not exceed 1)."""
        benchmark = ishigami_3d(self._bkd)
        gt = benchmark.ground_truth()
        main_effects = gt.get("main_effects")
        # main_effects has shape (nvars, 1)
        self.assertEqual(main_effects.shape, (3, 1))
        # Sum of main effects should be less than 1 due to interactions
        main_sum = float(self._bkd.sum(main_effects))
        self.assertLess(main_sum, 1.0)
        self.assertGreater(main_sum, 0.0)

    def test_ground_truth_total_effects_sum(self) -> None:
        """Test total effects sum (can exceed 1 due to interactions)."""
        benchmark = ishigami_3d(self._bkd)
        gt = benchmark.ground_truth()
        total_effects = gt.get("total_effects")
        # total_effects has shape (nvars, 1)
        self.assertEqual(total_effects.shape, (3, 1))
        # Total effects sum can exceed 1
        total_sum = float(self._bkd.sum(total_effects))
        self.assertGreater(total_sum, 1.0)

    def test_ground_truth_sobol_indices(self) -> None:
        """Test Sobol indices are available."""
        benchmark = ishigami_3d(self._bkd)
        gt = benchmark.ground_truth()
        sobol = gt.get("sobol_indices")
        self.assertIn((0,), sobol)
        self.assertIn((1,), sobol)
        self.assertIn((2,), sobol)
        self.assertIn((0, 2), sobol)
        # S2 should be 0
        self._bkd.assert_allclose(
            self._bkd.asarray([sobol[(2,)]]),
            self._bkd.asarray([0.0]),
            atol=1e-14,
        )

    def test_prior_nvars(self) -> None:
        """Test prior has correct nvars."""
        benchmark = ishigami_3d(self._bkd)
        prior = benchmark.prior()
        self.assertEqual(prior.nvars(), 3)

    def test_prior_samples_in_domain(self) -> None:
        """Test that samples from prior are in domain."""
        benchmark = ishigami_3d(self._bkd)
        prior = benchmark.prior()
        bounds = benchmark.domain().bounds()

        samples = prior.rvs(100)
        self.assertEqual(samples.shape, (3, 100))

        # Check all samples are within bounds
        for i in range(3):
            self.assertTrue(
                self._bkd.all_bool(samples[i, :] >= bounds[i, 0])
            )
            self.assertTrue(
                self._bkd.all_bool(samples[i, :] <= bounds[i, 1])
            )

    def test_function_evaluation_at_prior_samples(self) -> None:
        """Test function can be evaluated at prior samples."""
        benchmark = ishigami_3d(self._bkd)
        prior = benchmark.prior()
        func = benchmark.function()

        samples = prior.rvs(10)
        values = func(samples)
        self.assertEqual(values.shape, (1, 10))


class TestIshigami3DBenchmarkNumpy(TestIshigami3DBenchmark[NDArray[Any]]):
    """NumPy backend tests."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestIshigami3DBenchmarkTorch(TestIshigami3DBenchmark[torch.Tensor]):
    """PyTorch backend tests."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestBenchmarkRegistrySensitivity(unittest.TestCase):
    """Test registry for sensitivity benchmarks."""

    def test_ishigami_registered(self) -> None:
        """Test ishigami_3d is registered."""
        # Import to trigger registration
        from pyapprox.typing.benchmarks.instances import sensitivity  # noqa: F401

        bkd = NumpyBkd()
        benchmark = BenchmarkRegistry.get("ishigami_3d", bkd)
        self.assertEqual(benchmark.name(), "ishigami_3d")

    def test_ishigami_in_analytic_category(self) -> None:
        """Test ishigami_3d is in analytic category."""
        from pyapprox.typing.benchmarks.instances import analytic  # noqa: F401

        self.assertIn(
            "ishigami_3d",
            BenchmarkRegistry.list_category("analytic")
        )


if __name__ == "__main__":
    unittest.main()
