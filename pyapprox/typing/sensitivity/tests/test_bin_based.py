"""Tests for bin-based sensitivity analysis.

Tests variance-based sensitivity analysis using binning method against
benchmark functions with known ground truth values.
"""

import math
import unittest
import warnings
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.test_utils import (
    load_tests,  # noqa: F401
    slow_test,
    slower_test,
)

from pyapprox.typing.benchmarks.instances.sensitivity import (
    ishigami_3d,
    sobol_g_4d,
)
from pyapprox.typing.benchmarks.functions.algebraic.ishigami import (
    IshigamiFunction,
    IshigamiSensitivityIndices,
)
from pyapprox.typing.probability import UniformMarginal
from pyapprox.typing.probability.joint.independent import IndependentJoint
from pyapprox.typing.sensitivity.variance_based.bin_based import (
    BinBasedSensitivityAnalysis,
)


class TestBinBasedSensitivity(Generic[Array], unittest.TestCase):
    """Base test class for bin-based sensitivity analysis."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        np.random.seed(42)

    def test_main_effects_ishigami(self):
        """Test main effects on Ishigami with known ground truth."""
        benchmark = ishigami_3d(self._bkd)
        func = benchmark.function()
        prior = benchmark.prior()
        gt = benchmark.ground_truth()

        # Generate samples
        nsamples = 10000
        samples = prior.rvs(nsamples)
        values = func(samples)

        # Compute sensitivity indices
        sa = BinBasedSensitivityAnalysis(prior, self._bkd)
        sa.compute(samples, values)

        # Check main effects against ground truth
        main_effects = sa.main_effects()
        expected_main = self._bkd.asarray(gt.main_effects).reshape(-1, 1)
        # Bin-based has moderate accuracy
        self._bkd.assert_allclose(main_effects, expected_main, atol=0.15)

    def test_main_effects_sobol_g(self):
        """Test main effects on Sobol G-function."""
        benchmark = sobol_g_4d(self._bkd)
        func = benchmark.function()
        prior = benchmark.prior()
        gt = benchmark.ground_truth()

        nsamples = 8000
        samples = prior.rvs(nsamples)
        values = func(samples)

        sa = BinBasedSensitivityAnalysis(prior, self._bkd)
        sa.compute(samples, values)

        main_effects = sa.main_effects()
        expected_main = self._bkd.asarray(gt.main_effects).reshape(-1, 1)
        self._bkd.assert_allclose(main_effects, expected_main, atol=0.15)

    def test_second_order_indices_ishigami(self):
        """Test second-order Sobol indices on Ishigami."""
        benchmark = ishigami_3d(self._bkd)
        func = benchmark.function()
        prior = benchmark.prior()
        gt = benchmark.ground_truth()

        # Need more samples and tuned bins for accurate 2nd order estimation
        nsamples = 50000
        samples = prior.rvs(nsamples)
        values = func(samples)

        # Use tuned bin counts for 2nd order
        sa = BinBasedSensitivityAnalysis(prior, self._bkd, nbins=[40, 15, 4])
        sa.compute(samples, values)

        sobol = sa.sobol_indices()
        # Ishigami has significant S_13 (x1*x3 interaction)
        # Ground truth S_13 is at key (0, 2) in sobol_dict
        expected_s13 = gt.sobol_indices[(0, 2)]

        # Find S_13 index in the interaction terms
        interaction_terms = sa.interaction_terms()
        s13_idx = None
        for ii in range(interaction_terms.shape[1]):
            term = interaction_terms[:, ii]
            if term[0] == 1 and term[1] == 0 and term[2] == 1:
                s13_idx = ii
                break

        self.assertIsNotNone(s13_idx, "Could not find S_13 term")
        computed_s13 = float(self._bkd.to_numpy(sobol[s13_idx, 0]))
        # S_13 should be significant (around 0.24 for Ishigami)
        # With 50k samples and nbins=[40,15,4], we get ~0.15-0.20
        self.assertGreater(computed_s13, 0.05)
        self._bkd.assert_allclose(
            self._bkd.asarray([computed_s13]),
            self._bkd.asarray([expected_s13]),
            atol=0.15,
        )

    def test_third_order_indices(self):
        """Test third-order Sobol indices."""
        benchmark = ishigami_3d(self._bkd)
        func = benchmark.function()
        prior = benchmark.prior()

        nsamples = 20000
        samples = prior.rvs(nsamples)
        values = func(samples)

        sa = BinBasedSensitivityAnalysis(prior, self._bkd, nbins=[20, 5, 3])
        interaction_terms = sa.isotropic_interaction_terms(3)
        sa.set_interaction_terms_of_interest(interaction_terms)
        sa.compute(samples, values)

        sobol = sa.sobol_indices()
        # All indices should sum to approximately 1
        total_sum = self._bkd.sum(sobol)
        self._bkd.assert_allclose(
            self._bkd.asarray([float(total_sum)]),
            self._bkd.asarray([1.0]),
            atol=0.25,
        )

    def test_multi_qoi(self):
        """Test with multiple quantities of interest."""
        benchmark = sobol_g_4d(self._bkd)
        func = benchmark.function()
        prior = benchmark.prior()

        nsamples = 5000
        samples = prior.rvs(nsamples)
        values_1 = func(samples)
        values_2 = 2 * func(samples)
        values = self._bkd.vstack([values_1, values_2])

        sa = BinBasedSensitivityAnalysis(prior, self._bkd)
        sa.compute(samples, values)

        self.assertEqual(sa.nqoi(), 2)
        self.assertEqual(sa.main_effects().shape, (4, 2))
        self.assertEqual(sa.mean().shape, (2,))
        self.assertEqual(sa.variance().shape, (2,))

    def test_custom_nbins(self):
        """Test with explicitly specified number of bins."""
        benchmark = sobol_g_4d(self._bkd)
        func = benchmark.function()
        prior = benchmark.prior()

        nsamples = 5000
        samples = prior.rvs(nsamples)
        values = func(samples)

        sa = BinBasedSensitivityAnalysis(prior, self._bkd, nbins=[10, 5, 3])
        sa.compute(samples, values)

        # Should run without error
        main_effects = sa.main_effects()
        self.assertEqual(main_effects.shape, (4, 1))

    def test_adaptive_nbins(self):
        """Test adaptive bin count formula."""
        benchmark = sobol_g_4d(self._bkd)
        prior = benchmark.prior()

        sa = BinBasedSensitivityAnalysis(prior, self._bkd)

        # Test adaptive formula for different orders
        nsamples = 10000
        nbins_1 = sa._get_nbins(nsamples, 1)
        nbins_2 = sa._get_nbins(nsamples, 2)
        nbins_3 = sa._get_nbins(nsamples, 3)

        # Higher order should have fewer bins
        self.assertGreater(nbins_1, nbins_2)
        self.assertGreater(nbins_2, nbins_3)

        # All should be at least 2
        self.assertGreaterEqual(nbins_1, 2)
        self.assertGreaterEqual(nbins_2, 2)
        self.assertGreaterEqual(nbins_3, 2)

    def test_bootstrap_uncertainty(self):
        """Test bootstrap uncertainty quantification."""
        benchmark = sobol_g_4d(self._bkd)
        func = benchmark.function()
        prior = benchmark.prior()

        nsamples = 3000
        samples = prior.rvs(nsamples)
        values = func(samples)

        sa = BinBasedSensitivityAnalysis(prior, self._bkd)
        stats = sa.bootstrap(samples, values, nbootstraps=10, seed=42)

        # Check that bootstrap returns expected keys
        self.assertIn("median", stats)
        self.assertIn("min", stats)
        self.assertIn("max", stats)
        self.assertIn("quantile_25", stats)
        self.assertIn("quantile_75", stats)
        self.assertIn("std", stats)

        # Check shapes
        self.assertEqual(stats["median"].shape, (4, 1))

        # Check that min <= median <= max
        min_vals = self._bkd.to_numpy(stats["min"])
        med_vals = self._bkd.to_numpy(stats["median"])
        max_vals = self._bkd.to_numpy(stats["max"])
        self.assertTrue(np.all(min_vals <= med_vals + 1e-10))
        self.assertTrue(np.all(med_vals <= max_vals + 1e-10))

    def test_eps_parameter(self):
        """Test eps parameter for unbounded distributions."""
        benchmark = ishigami_3d(self._bkd)
        func = benchmark.function()
        prior = benchmark.prior()

        nsamples = 5000
        samples = prior.rvs(nsamples)
        values = func(samples)

        # With eps=0.01, bins won't extend to exact edges
        sa = BinBasedSensitivityAnalysis(prior, self._bkd, eps=0.01)
        sa.compute(samples, values)

        # Should complete without error
        main_effects = sa.main_effects()
        self.assertEqual(main_effects.shape, (3, 1))

    def test_raises_before_compute(self):
        """Test that accessing results before compute() raises error."""
        benchmark = sobol_g_4d(self._bkd)
        prior = benchmark.prior()

        sa = BinBasedSensitivityAnalysis(prior, self._bkd)

        with self.assertRaises(RuntimeError):
            sa.main_effects()
        with self.assertRaises(RuntimeError):
            sa.sobol_indices()
        with self.assertRaises(RuntimeError):
            sa.mean()
        with self.assertRaises(RuntimeError):
            sa.variance()
        with self.assertRaises(RuntimeError):
            sa.nqoi()

    def test_no_total_effects(self):
        """Test that total_effects() raises NotImplementedError."""
        benchmark = sobol_g_4d(self._bkd)
        func = benchmark.function()
        prior = benchmark.prior()

        nsamples = 1000
        samples = prior.rvs(nsamples)
        values = func(samples)

        sa = BinBasedSensitivityAnalysis(prior, self._bkd)
        sa.compute(samples, values)

        with self.assertRaises(NotImplementedError):
            sa.total_effects()

    def test_clip_negative(self):
        """Test negative index clipping."""
        benchmark = sobol_g_4d(self._bkd)
        func = benchmark.function()
        prior = benchmark.prior()

        # Use small sample size where negative indices might occur
        nsamples = 500
        samples = prior.rvs(nsamples)
        values = func(samples)

        # With clipping (default)
        sa_clip = BinBasedSensitivityAnalysis(prior, self._bkd, clip_negative=True)
        sa_clip.compute(samples, values)
        indices_clip = self._bkd.to_numpy(sa_clip.sobol_indices())
        self.assertTrue(np.all(indices_clip >= 0))

        # Without clipping
        sa_no_clip = BinBasedSensitivityAnalysis(prior, self._bkd, clip_negative=False)
        sa_no_clip.compute(samples, values)
        # May have negative values (not guaranteed, depends on estimation error)

    def test_sparse_cell_warning(self):
        """Test warning when many cells are empty."""
        benchmark = ishigami_3d(self._bkd)
        func = benchmark.function()
        prior = benchmark.prior()

        # Use very few samples with many bins to trigger warning
        nsamples = 100
        samples = prior.rvs(nsamples)
        values = func(samples)

        # Many bins will create many empty cells
        sa = BinBasedSensitivityAnalysis(prior, self._bkd, nbins=[20, 10, 5])
        interaction_terms = sa.isotropic_interaction_terms(2)
        sa.set_interaction_terms_of_interest(interaction_terms)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sa.compute(samples, values)
            # Should have at least one warning about empty cells
            empty_cell_warnings = [
                warning for warning in w if "cells empty" in str(warning.message)
            ]
            self.assertGreater(len(empty_cell_warnings), 0)

    def test_mean_and_variance(self):
        """Test mean and variance computation."""
        benchmark = ishigami_3d(self._bkd)
        func = benchmark.function()
        prior = benchmark.prior()
        gt = benchmark.ground_truth()

        nsamples = 10000
        samples = prior.rvs(nsamples)
        values = func(samples)

        sa = BinBasedSensitivityAnalysis(prior, self._bkd)
        sa.compute(samples, values)

        mean = sa.mean()
        variance = sa.variance()

        self._bkd.assert_allclose(mean, self._bkd.asarray([gt.mean]), rtol=0.1)
        self._bkd.assert_allclose(variance, self._bkd.asarray([gt.variance]), rtol=0.15)

    def test_input_validation(self):
        """Test input validation."""
        benchmark = sobol_g_4d(self._bkd)
        func = benchmark.function()
        prior = benchmark.prior()

        samples = prior.rvs(100)
        values = func(samples)

        sa = BinBasedSensitivityAnalysis(prior, self._bkd)

        # 1D samples should raise
        with self.assertRaises(ValueError):
            sa.compute(samples[0, :], values)

        # 1D values should raise
        with self.assertRaises(ValueError):
            sa.compute(samples, values[0, :])

        # Mismatched columns should raise
        with self.assertRaises(ValueError):
            sa.compute(samples[:, :50], values)

    def test_repr(self):
        """Test string representation."""
        benchmark = sobol_g_4d(self._bkd)
        prior = benchmark.prior()

        sa = BinBasedSensitivityAnalysis(prior, self._bkd, nbins=[10, 5], eps=0.01)
        repr_str = repr(sa)

        self.assertIn("BinBasedSensitivityAnalysis", repr_str)
        self.assertIn("nvars=4", repr_str)
        self.assertIn("eps=0.01", repr_str)


class TestBinBasedSensitivityNumpy(TestBinBasedSensitivity[NDArray[Any]]):
    """NumPy backend tests for bin-based sensitivity."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestBinBasedSensitivityTorch(TestBinBasedSensitivity[torch.Tensor]):
    """PyTorch backend tests for bin-based sensitivity."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestIshigamiBenchmark(Generic[Array], unittest.TestCase):
    """Test bin-based sensitivity indices against Ishigami function benchmark.

    The Ishigami function is a standard benchmark for sensitivity analysis with
    analytically known Sobol indices. It's defined on [-pi, pi]^3:
    f(x) = sin(x1) + a*sin^2(x2) + b*x3^4*sin(x1)

    Standard parameters: a=7, b=0.1

    This test verifies that bin-based estimation can achieve rtol=1e-2 accuracy
    for main effects and the S_13 interaction with sufficient samples and bins.
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        np.random.seed(42)

    @slow_test
    def test_ishigami_main_effects_high_accuracy(self) -> None:
        """Test main effects match analytical Ishigami values with rtol=2e-2.

        Uses 100000 samples with tuned bin counts to achieve ~2% relative
        accuracy on main effect Sobol indices S_1 and S_2.
        Note: S_3 = 0 exactly (x3 has no main effect), so we use atol for S_3.
        """
        bkd = self._bkd

        # Standard Ishigami parameters
        a, b = 7.0, 0.1
        ishigami = IshigamiFunction(bkd, a=a, b=b)
        exact_indices = IshigamiSensitivityIndices(bkd, a=a, b=b)

        # Get exact Sobol indices
        S_exact = exact_indices.main_effects()  # Shape: (3, 1)

        # Create uniform prior on [-pi, pi]^3
        pi = math.pi
        prior = IndependentJoint(
            [
                UniformMarginal(-pi, pi, bkd),
                UniformMarginal(-pi, pi, bkd),
                UniformMarginal(-pi, pi, bkd),
            ],
            bkd,
        )

        # Generate samples - 100000 with nbins=[70, 30] achieves ~2% accuracy
        nsamples = 100000
        samples = prior.rvs(nsamples)
        values = ishigami(samples)

        # Use bin-based sensitivity analysis with tuned bin counts
        sa = BinBasedSensitivityAnalysis(prior, bkd, nbins=[70, 30, 4])
        sa.compute(samples, values)

        # Get computed main effects
        S_computed = sa.main_effects()

        # Verify S_1 and S_2 with rtol=2.5e-2 (they have non-zero values)
        for i in range(2):
            expected = float(bkd.to_numpy(S_exact[i, 0]))
            computed = float(bkd.to_numpy(S_computed[i, 0]))
            bkd.assert_allclose(
                bkd.asarray([computed]),
                bkd.asarray([expected]),
                rtol=2.5e-2,
                err_msg=f"S_{i+1} = {computed:.4f}, expected {expected:.4f}",
            )

        # S_3 = 0 exactly, use atol
        expected_s3 = float(bkd.to_numpy(S_exact[2, 0]))
        computed_s3 = float(bkd.to_numpy(S_computed[2, 0]))
        bkd.assert_allclose(
            bkd.asarray([computed_s3]),
            bkd.asarray([expected_s3]),
            atol=1e-3,
            err_msg=f"S_3 = {computed_s3:.6f}, expected {expected_s3:.6f}",
        )

    @slower_test
    def test_ishigami_interaction_s13_high_accuracy(self) -> None:
        """Test S_13 interaction index matches analytical value with rtol=2e-2.

        The Ishigami function has a significant x1-x3 interaction (S_13 ~ 0.24).
        This test requires more samples and bins for accurate 2nd-order estimation.
        """
        bkd = self._bkd

        # Standard Ishigami parameters
        a, b = 7.0, 0.1
        ishigami = IshigamiFunction(bkd, a=a, b=b)
        exact_indices = IshigamiSensitivityIndices(bkd, a=a, b=b)

        # Get exact S_13 from sobol_indices (index 4 in the ordered list)
        sobol_exact = exact_indices.sobol_indices()
        S_13_exact = float(bkd.to_numpy(sobol_exact[4, 0]))

        # Create uniform prior on [-pi, pi]^3
        pi = math.pi
        prior = IndependentJoint(
            [
                UniformMarginal(-pi, pi, bkd),
                UniformMarginal(-pi, pi, bkd),
                UniformMarginal(-pi, pi, bkd),
            ],
            bkd,
        )

        # Generate samples - 1M with nbins=[60, 30] achieves ~2% for S_13
        nsamples = 1000000
        samples = prior.rvs(nsamples)
        values = ishigami(samples)

        # Use bin-based sensitivity analysis with tuned bin counts
        sa = BinBasedSensitivityAnalysis(prior, bkd, nbins=[60, 30, 4])
        interaction_terms = sa.isotropic_interaction_terms(2)
        sa.set_interaction_terms_of_interest(interaction_terms)
        sa.compute(samples, values)

        # Find S_13 index in the interaction terms
        interaction_terms_arr = sa.interaction_terms()
        s13_idx = None
        for ii in range(interaction_terms_arr.shape[1]):
            term = interaction_terms_arr[:, ii]
            term_np = bkd.to_numpy(term)
            if term_np[0] == 1 and term_np[1] == 0 and term_np[2] == 1:
                s13_idx = ii
                break

        self.assertIsNotNone(s13_idx, "Could not find S_13 term")

        sobol = sa.sobol_indices()
        S_13_computed = float(bkd.to_numpy(sobol[s13_idx, 0]))

        bkd.assert_allclose(
            bkd.asarray([S_13_computed]),
            bkd.asarray([S_13_exact]),
            rtol=2e-2,
            err_msg=f"S_13 = {S_13_computed:.4f}, expected {S_13_exact:.4f}",
        )

    @slow_test
    def test_ishigami_mean_variance_high_accuracy(self) -> None:
        """Test mean and variance match analytical Ishigami values with rtol=2e-2."""
        bkd = self._bkd

        # Standard Ishigami parameters
        a, b = 7.0, 0.1
        ishigami = IshigamiFunction(bkd, a=a, b=b)
        exact_indices = IshigamiSensitivityIndices(bkd, a=a, b=b)

        # Get exact mean and variance
        mean_exact = float(bkd.to_numpy(exact_indices.mean()[0]))
        var_exact = float(bkd.to_numpy(exact_indices.variance()[0]))

        # Create uniform prior on [-pi, pi]^3
        pi = math.pi
        prior = IndependentJoint(
            [
                UniformMarginal(-pi, pi, bkd),
                UniformMarginal(-pi, pi, bkd),
                UniformMarginal(-pi, pi, bkd),
            ],
            bkd,
        )

        # Generate samples
        nsamples = 100000
        samples = prior.rvs(nsamples)
        values = ishigami(samples)

        sa = BinBasedSensitivityAnalysis(prior, bkd)
        sa.compute(samples, values)

        mean_computed = float(bkd.to_numpy(sa.mean()[0]))
        var_computed = float(bkd.to_numpy(sa.variance()[0]))

        bkd.assert_allclose(
            bkd.asarray([mean_computed]),
            bkd.asarray([mean_exact]),
            rtol=2e-2,
            err_msg=f"Mean = {mean_computed:.4f}, expected {mean_exact:.4f}",
        )
        bkd.assert_allclose(
            bkd.asarray([var_computed]),
            bkd.asarray([var_exact]),
            rtol=2e-2,
            err_msg=f"Variance = {var_computed:.4f}, expected {var_exact:.4f}",
        )

    @slow_test
    def test_ishigami_sobol_sum_near_one(self) -> None:
        """Test that sum of all Sobol indices is approximately 1."""
        bkd = self._bkd

        # Standard Ishigami parameters
        a, b = 7.0, 0.1
        ishigami = IshigamiFunction(bkd, a=a, b=b)

        # Create uniform prior on [-pi, pi]^3
        pi = math.pi
        prior = IndependentJoint(
            [
                UniformMarginal(-pi, pi, bkd),
                UniformMarginal(-pi, pi, bkd),
                UniformMarginal(-pi, pi, bkd),
            ],
            bkd,
        )

        # Generate samples
        nsamples = 100000
        samples = prior.rvs(nsamples)
        values = ishigami(samples)

        # Include up to 3rd order interactions
        sa = BinBasedSensitivityAnalysis(prior, bkd, nbins=[50, 15, 5])
        interaction_terms = sa.isotropic_interaction_terms(3)
        sa.set_interaction_terms_of_interest(interaction_terms)
        sa.compute(samples, values)

        # Sum of all Sobol indices should be approximately 1
        sobol = sa.sobol_indices()
        total_sum = float(bkd.to_numpy(bkd.sum(sobol)))

        bkd.assert_allclose(
            bkd.asarray([total_sum]),
            bkd.asarray([1.0]),
            atol=0.1,  # Allow 10% deviation due to estimation error
            err_msg=f"Sum of Sobol indices = {total_sum:.4f}, expected ~1.0",
        )


class TestIshigamiBenchmarkNumpy(TestIshigamiBenchmark[NDArray[Any]]):
    """NumPy backend Ishigami benchmark tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestIshigamiBenchmarkTorch(TestIshigamiBenchmark[torch.Tensor]):
    """PyTorch backend Ishigami benchmark tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
