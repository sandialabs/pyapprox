"""Tests for sensitivity analysis module.

Tests variance-based sensitivity analysis implementations against
benchmark functions with known ground truth values.
"""

import numpy as np
import pytest

from pyapprox_benchmarks.sensitivity import IshigamiBenchmark, SobolGBenchmark
from pyapprox.sensitivity import (
    MorrisSensitivityAnalysis,
    SobolSequenceSensitivityAnalysis,
)


class TestSampleBasedSensitivity:
    """Base test class for sample-based sensitivity analysis."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def test_sobol_sequence_ishigami(self, bkd):
        """Test Sobol sequence sensitivity on Ishigami function."""
        benchmark = IshigamiBenchmark(bkd)
        func = benchmark.problem().function()
        prior = benchmark.problem().prior()

        # Create sensitivity analysis with Sobol sequence
        sa = SobolSequenceSensitivityAnalysis(prior, bkd, seed=42)

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
        bkd.assert_allclose(mean, bkd.asarray([benchmark.mean()]), rtol=0.1)
        bkd.assert_allclose(
            variance, bkd.asarray([benchmark.variance()]), rtol=0.15,
        )

        # Check main effects
        main_effects = sa.main_effects()
        expected_main = benchmark.main_effects()
        # Main effects should be reasonably close to ground truth
        # Note: sample-based estimates have significant variance
        bkd.assert_allclose(main_effects, expected_main, atol=0.1)

    def test_sobol_sequence_sobol_g(self, bkd):
        """Test Sobol sequence sensitivity on Sobol G function."""
        benchmark = SobolGBenchmark(bkd, a=[0.0, 0.0, 0.0, 0.0])
        func = benchmark.problem().function()
        prior = benchmark.problem().prior()

        # Create sensitivity analysis
        sa = SobolSequenceSensitivityAnalysis(prior, bkd, seed=42)

        # Generate samples and evaluate
        nsamples = 2048
        samples = sa.generate_samples(nsamples)
        values = func(samples)

        # Compute sensitivity indices
        sa.compute(values)

        # Check main effects (all equal for equal importance)
        main_effects = sa.main_effects()
        expected_main = benchmark.main_effects()
        bkd.assert_allclose(main_effects, expected_main, atol=0.15)

    def test_sample_based_nqoi(self, bkd):
        """Test multi-output sensitivity analysis."""
        benchmark = SobolGBenchmark(bkd, a=[0.0, 0.0, 0.0, 0.0])
        func = benchmark.problem().function()
        prior = benchmark.problem().prior()

        sa = SobolSequenceSensitivityAnalysis(prior, bkd, seed=42)
        nsamples = 1024
        samples = sa.generate_samples(nsamples)

        # Create multi-output by duplicating
        values_1 = func(samples)
        values_2 = 2 * func(samples)
        values = bkd.vstack([values_1, values_2])

        sa.compute(values)

        assert sa.nqoi() == 2
        assert sa.main_effects().shape == (4, 2)
        assert sa.total_effects().shape == (4, 2)


class TestMorrisScreening:
    """Base test class for Morris screening."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def test_morris_ishigami(self, bkd):
        """Test Morris screening on Ishigami function."""
        benchmark = IshigamiBenchmark(bkd)
        func = benchmark.problem().function()
        prior = benchmark.problem().prior()
        # Create Morris analysis
        morris = MorrisSensitivityAnalysis(prior, nlevels=4, bkd=bkd)

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
        assert mu_star.shape == (3, 1)
        assert sigma.shape == (3, 1)

        # For Ishigami, x1 and x2 should be more important than x3
        # x3 only appears in interaction with x1
        mu_star_np = bkd.to_numpy(mu_star).flatten()
        assert mu_star_np[0] > mu_star_np[2]

    def test_morris_sobol_g(self, bkd):
        """Test Morris screening on Sobol G function."""
        benchmark = SobolGBenchmark(bkd, a=[0.0, 0.0, 0.0, 0.0])
        func = benchmark.problem().function()
        prior = benchmark.problem().prior()

        morris = MorrisSensitivityAnalysis(prior, nlevels=4, bkd=bkd)

        ntrajectories = 15
        samples = morris.generate_samples(ntrajectories)
        values = func(samples)

        morris.compute(values)

        mu_star = morris.mu_star()
        _ = morris.sigma()

        # For equal importance Sobol G, all mu_star should be similar
        mu_star_np = bkd.to_numpy(mu_star).flatten()
        # Check relative spread is not too large
        cv = np.std(mu_star_np) / np.mean(mu_star_np)
        assert cv < 0.5  # Coefficient of variation < 50%

    def test_morris_trajectory_selection(self, bkd):
        """Test Morris with trajectory downselection."""
        benchmark = SobolGBenchmark(bkd, a=[0.0, 0.0, 0.0, 0.0])
        prior = benchmark.problem().prior()

        morris = MorrisSensitivityAnalysis(prior, nlevels=4, bkd=bkd)

        # Request 5 trajectories, select from 10 candidates
        # Using small numbers to keep test fast (C(10,5) = 252 combinations)
        ntrajectories = 5
        ncandidate = 10
        samples = morris.generate_samples(ntrajectories, ncandidate)

        # Check shape: nvars * (nvars+1) * ntrajectories
        nvars = prior.nvars()
        expected_nsamples = ntrajectories * (nvars + 1)
        assert samples.shape == (nvars, expected_nsamples)

    def test_morris_nlevels_validation(self, bkd):
        """Test that nlevels must be even."""
        benchmark = SobolGBenchmark(bkd, a=[0.0, 0.0, 0.0, 0.0])
        prior = benchmark.problem().prior()

        with pytest.raises(ValueError):
            MorrisSensitivityAnalysis(prior, nlevels=3, bkd=bkd)
