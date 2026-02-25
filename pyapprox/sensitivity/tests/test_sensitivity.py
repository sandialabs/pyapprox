"""Tests for sensitivity analysis module.

Tests variance-based sensitivity analysis implementations against
benchmark functions with known ground truth values.
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.benchmarks.instances.sensitivity import (
    ishigami_3d,
    sobol_g_4d,
)
from pyapprox.sensitivity import (
    MorrisSensitivityAnalysis,
    SobolSequenceSensitivityAnalysis,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401


class TestSampleBasedSensitivity(Generic[Array], unittest.TestCase):
    """Base test class for sample-based sensitivity analysis."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        # Use fixed seed for reproducibility
        np.random.seed(42)

    def test_sobol_sequence_ishigami(self):
        """Test Sobol sequence sensitivity on Ishigami function."""
        benchmark = ishigami_3d(self._bkd)
        func = benchmark.function()
        prior = benchmark.prior()
        gt = benchmark.ground_truth()

        # Create sensitivity analysis with Sobol sequence
        sa = SobolSequenceSensitivityAnalysis(prior, self._bkd, seed=42)

        # Generate samples and evaluate
        nsamples = 4096
        samples = sa.generate_samples(nsamples)
        values = func(samples)

        # Compute sensitivity indices
        sa.compute(values)

        # Check mean and variance
        mean = sa.mean()
        variance = sa.variance()

        # Relaxed tolerances for Monte Carlo estimation
        self._bkd.assert_allclose(mean, self._bkd.asarray([gt.mean]), rtol=0.1)
        self._bkd.assert_allclose(variance, self._bkd.asarray([gt.variance]), rtol=0.15)

        # Check main effects
        main_effects = sa.main_effects()
        expected_main = self._bkd.asarray(gt.main_effects).reshape(-1, 1)
        # Main effects should be reasonably close to ground truth
        # Note: sample-based estimates have significant variance
        self._bkd.assert_allclose(main_effects, expected_main, atol=0.1)

    def test_sobol_sequence_sobol_g(self):
        """Test Sobol sequence sensitivity on Sobol G function."""
        benchmark = sobol_g_4d(self._bkd)
        func = benchmark.function()
        prior = benchmark.prior()
        gt = benchmark.ground_truth()

        # Create sensitivity analysis
        sa = SobolSequenceSensitivityAnalysis(prior, self._bkd, seed=42)

        # Generate samples and evaluate
        nsamples = 2048
        samples = sa.generate_samples(nsamples)
        values = func(samples)

        # Compute sensitivity indices
        sa.compute(values)

        # Check main effects (all equal for equal importance)
        main_effects = sa.main_effects()
        expected_main = self._bkd.asarray(gt.main_effects).reshape(-1, 1)
        self._bkd.assert_allclose(main_effects, expected_main, atol=0.15)

    def test_sample_based_nqoi(self):
        """Test multi-output sensitivity analysis."""
        benchmark = sobol_g_4d(self._bkd)
        func = benchmark.function()
        prior = benchmark.prior()

        sa = SobolSequenceSensitivityAnalysis(prior, self._bkd, seed=42)
        nsamples = 1024
        samples = sa.generate_samples(nsamples)

        # Create multi-output by duplicating
        values_1 = func(samples)
        values_2 = 2 * func(samples)
        values = self._bkd.vstack([values_1, values_2])

        sa.compute(values)

        self.assertEqual(sa.nqoi(), 2)
        self.assertEqual(sa.main_effects().shape, (4, 2))
        self.assertEqual(sa.total_effects().shape, (4, 2))


class TestMorrisScreening(Generic[Array], unittest.TestCase):
    """Base test class for Morris screening."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        np.random.seed(42)

    def test_morris_ishigami(self):
        """Test Morris screening on Ishigami function."""
        benchmark = ishigami_3d(self._bkd)
        func = benchmark.function()
        prior = benchmark.prior()
        # Create Morris analysis
        morris = MorrisSensitivityAnalysis(prior, nlevels=4, bkd=self._bkd)

        # Generate trajectories
        ntrajectories = 20
        samples = morris.generate_samples(ntrajectories)

        # Evaluate function
        values = func(samples)

        # Compute Morris indices
        morris.compute(values)

        # Get results
        mu_star = morris.mu_star()
        sigma = morris.sigma()

        # Check shapes
        self.assertEqual(mu_star.shape, (3, 1))
        self.assertEqual(sigma.shape, (3, 1))

        # For Ishigami, x1 and x2 should be more important than x3
        # x3 only appears in interaction with x1
        mu_star_np = self._bkd.to_numpy(mu_star).flatten()
        self.assertTrue(mu_star_np[0] > mu_star_np[2])

    def test_morris_sobol_g(self):
        """Test Morris screening on Sobol G function."""
        benchmark = sobol_g_4d(self._bkd)
        func = benchmark.function()
        prior = benchmark.prior()

        morris = MorrisSensitivityAnalysis(prior, nlevels=4, bkd=self._bkd)

        ntrajectories = 15
        samples = morris.generate_samples(ntrajectories)
        values = func(samples)

        morris.compute(values)

        mu_star = morris.mu_star()
        _ = morris.sigma()

        # For equal importance Sobol G, all mu_star should be similar
        mu_star_np = self._bkd.to_numpy(mu_star).flatten()
        # Check relative spread is not too large
        cv = np.std(mu_star_np) / np.mean(mu_star_np)
        self.assertTrue(cv < 0.5)  # Coefficient of variation < 50%

    def test_morris_trajectory_selection(self):
        """Test Morris with trajectory downselection."""
        benchmark = sobol_g_4d(self._bkd)
        prior = benchmark.prior()

        morris = MorrisSensitivityAnalysis(prior, nlevels=4, bkd=self._bkd)

        # Request 5 trajectories, select from 10 candidates
        # Using small numbers to keep test fast (C(10,5) = 252 combinations)
        ntrajectories = 5
        ncandidate = 10
        samples = morris.generate_samples(ntrajectories, ncandidate)

        # Check shape: nvars * (nvars+1) * ntrajectories
        nvars = prior.nvars()
        expected_nsamples = ntrajectories * (nvars + 1)
        self.assertEqual(samples.shape, (nvars, expected_nsamples))

    def test_morris_nlevels_validation(self):
        """Test that nlevels must be even."""
        benchmark = sobol_g_4d(self._bkd)
        prior = benchmark.prior()

        with self.assertRaises(ValueError):
            MorrisSensitivityAnalysis(prior, nlevels=3, bkd=self._bkd)


class TestSampleBasedSensitivityNumpy(TestSampleBasedSensitivity[NDArray[Any]]):
    """NumPy backend tests for sample-based sensitivity."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestSampleBasedSensitivityTorch(TestSampleBasedSensitivity[torch.Tensor]):
    """PyTorch backend tests for sample-based sensitivity."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestMorrisScreeningNumpy(TestMorrisScreening[NDArray[Any]]):
    """NumPy backend tests for Morris screening."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestMorrisScreeningTorch(TestMorrisScreening[torch.Tensor]):
    """PyTorch backend tests for Morris screening."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
