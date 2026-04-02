"""
Tests for quadrature samplers.

Tests cover:
- Sample shape correctness
- Weight normalization
- Reproducibility with seeds/reset
- Halton sequence properties
- Gaussian quadrature accuracy
- OED sampler joint/prior sampling
"""

from __future__ import annotations

# TODO: should we split up into smaller tests files.
import numpy as np
import pytest

from pyapprox.expdesign.quadrature import (
    GaussianQuadratureSampler,
    HaltonSampler,
    MonteCarloSampler,
    OEDQuadratureSampler,
    SobolSampler,
)
from pyapprox.probability.joint import IndependentJoint
from pyapprox.probability.univariate import (
    GaussianMarginal,
    UniformMarginal,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend


class TestMonteCarloSampler:
    """Base test class for Monte Carlo sampler."""

    def _create_gaussian_distribution(
        self, bkd: Backend[Array], nvars: int = 3,
    ) -> IndependentJoint[Array]:
        """Create a standard normal joint distribution."""
        marginals = [GaussianMarginal(0.0, 1.0, bkd) for _ in range(nvars)]
        return IndependentJoint(marginals, bkd)

    def _create_uniform_distribution(
        self, bkd: Backend[Array], nvars: int = 3,
    ) -> IndependentJoint[Array]:
        """Create a uniform [0, 1] joint distribution."""
        marginals = [UniformMarginal(0.0, 1.0, bkd) for _ in range(nvars)]
        return IndependentJoint(marginals, bkd)

    def test_weights_sum_to_one(self, bkd: Backend[Array]) -> None:
        """Test that weights sum to 1."""
        distribution = self._create_gaussian_distribution(bkd)
        sampler = MonteCarloSampler(distribution, bkd)
        _, weights = sampler.sample(100)

        weight_sum = bkd.sum(weights)
        expected = bkd.asarray(1.0)
        bkd.assert_allclose(weight_sum, expected, rtol=1e-10)

    def test_uniform_weights(self, bkd):
        """Test that all weights are equal."""
        distribution = self._create_gaussian_distribution(bkd)
        sampler = MonteCarloSampler(distribution, bkd)
        nsamples = 50
        _, weights = sampler.sample(nsamples)

        expected_weight = bkd.asarray(1.0 / nsamples)
        bkd.assert_allclose(weights[0], expected_weight, rtol=1e-10)
        bkd.assert_allclose(weights[-1], expected_weight, rtol=1e-10)

    def test_uniform_samples_in_range(self, bkd):
        """Test uniform distribution samples are in [0, 1]."""
        distribution = self._create_uniform_distribution(bkd)
        sampler = MonteCarloSampler(distribution, bkd)
        samples, _ = sampler.sample(1000)

        samples_np = bkd.to_numpy(samples)
        assert np.all(samples_np >= 0.0)
        assert np.all(samples_np <= 1.0)


class TestHaltonSampler:
    """Base test class for Halton sampler."""

    def test_weights_sum_to_one(self, bkd: Backend[Array]) -> None:
        """Test that weights sum to 1."""
        sampler = HaltonSampler(3, bkd, seed=42)
        _, weights = sampler.sample(100)

        weight_sum = bkd.sum(weights)
        expected = bkd.asarray(1.0)
        bkd.assert_allclose(weight_sum, expected, rtol=1e-10)

    def test_deterministic_with_seed(self, bkd):
        """Test that same seed gives same samples."""
        sampler1 = HaltonSampler(3, bkd, seed=42)
        sampler2 = HaltonSampler(3, bkd, seed=42)

        samples1, _ = sampler1.sample(50)
        samples2, _ = sampler2.sample(50)

        bkd.assert_allclose(samples1, samples2, rtol=1e-12)

    def test_deterministic_without_scramble(self, bkd):
        """Test that unscrambled Halton is deterministic."""
        sampler1 = HaltonSampler(3, bkd, start_index=0, scramble=False)
        sampler2 = HaltonSampler(3, bkd, start_index=0, scramble=False)

        samples1, _ = sampler1.sample(50)
        samples2, _ = sampler2.sample(50)

        bkd.assert_allclose(samples1, samples2, rtol=1e-12)

    def test_reset_restarts_sequence(self, bkd):
        """Test that reset restarts the sequence."""
        sampler = HaltonSampler(3, bkd, start_index=0, seed=42)

        samples1, _ = sampler.sample(50)
        sampler.reset()
        samples2, _ = sampler.sample(50)

        bkd.assert_allclose(samples1, samples2, rtol=1e-12)

    def test_sequential_samples_different(self, bkd):
        """Test that sequential batches give different samples."""
        sampler = HaltonSampler(3, bkd, start_index=0, seed=42)

        samples1, _ = sampler.sample(50)
        samples2, _ = sampler.sample(50)

        # The two batches should be different (continuing sequence)
        samples1_np = bkd.to_numpy(samples1)
        samples2_np = bkd.to_numpy(samples2)
        assert not np.allclose(samples1_np, samples2_np)

    def test_uniform_samples_without_transform(self, bkd):
        """Test uniform samples are in [0, 1] when not transforming."""
        sampler = HaltonSampler(
            3, bkd, start_index=1, transform_to_normal=False, seed=42
        )
        samples, _ = sampler.sample(100)

        samples_np = bkd.to_numpy(samples)
        assert np.all(samples_np >= 0.0)
        assert np.all(samples_np <= 1.0)

    def test_normal_transform(self, bkd):
        """Test that transformed samples have approximately normal stats."""
        sampler = HaltonSampler(
            3, bkd, start_index=1, transform_to_normal=True, seed=42
        )
        samples, _ = sampler.sample(1000)

        samples_np = bkd.to_numpy(samples)
        # Check mean is close to 0
        assert np.abs(np.mean(samples_np)) < 0.1
        # Check std is close to 1
        assert np.abs(np.std(samples_np) - 1.0) < 0.1

    def test_start_index_skips_samples(self, bkd):
        """Test that start_index skips the initial samples."""
        # Get samples starting from 0
        sampler1 = HaltonSampler(
            3,
            bkd,
            start_index=0,
            scramble=False,
            transform_to_normal=False,
        )
        samples_full, _ = sampler1.sample(10)

        # Get samples starting from 2
        sampler2 = HaltonSampler(
            3,
            bkd,
            start_index=2,
            scramble=False,
            transform_to_normal=False,
        )
        samples_skip, _ = sampler2.sample(8)

        # Samples starting from 2 should match samples[2:] from full
        bkd.assert_allclose(samples_full[:, 2:], samples_skip, rtol=1e-12)


class TestSobolSampler:
    """Base test class for Sobol sampler."""

    def test_weights_sum_to_one(self, bkd: Backend[Array]) -> None:
        """Test that weights sum to 1."""
        sampler = SobolSampler(3, bkd, seed=42)
        _, weights = sampler.sample(100)

        weight_sum = bkd.sum(weights)
        expected = bkd.asarray(1.0)
        bkd.assert_allclose(weight_sum, expected, rtol=1e-10)

    def test_deterministic_with_seed(self, bkd):
        """Test that same seed gives same samples."""
        sampler1 = SobolSampler(3, bkd, seed=42)
        sampler2 = SobolSampler(3, bkd, seed=42)

        samples1, _ = sampler1.sample(50)
        samples2, _ = sampler2.sample(50)

        bkd.assert_allclose(samples1, samples2, rtol=1e-12)

    def test_deterministic_without_scramble(self, bkd):
        """Test that unscrambled Sobol is deterministic."""
        sampler1 = SobolSampler(3, bkd, start_index=0, scramble=False)
        sampler2 = SobolSampler(3, bkd, start_index=0, scramble=False)

        samples1, _ = sampler1.sample(50)
        samples2, _ = sampler2.sample(50)

        bkd.assert_allclose(samples1, samples2, rtol=1e-12)

    def test_reset_restarts_sequence(self, bkd):
        """Test that reset restarts the sequence."""
        sampler = SobolSampler(3, bkd, start_index=0, seed=42)

        samples1, _ = sampler.sample(50)
        sampler.reset()
        samples2, _ = sampler.sample(50)

        bkd.assert_allclose(samples1, samples2, rtol=1e-12)

    def test_sequential_samples_different(self, bkd):
        """Test that sequential batches give different samples."""
        sampler = SobolSampler(3, bkd, start_index=0, seed=42)

        samples1, _ = sampler.sample(50)
        samples2, _ = sampler.sample(50)

        # The two batches should be different (continuing sequence)
        samples1_np = bkd.to_numpy(samples1)
        samples2_np = bkd.to_numpy(samples2)
        assert not np.allclose(samples1_np, samples2_np)

    def test_uniform_samples_without_transform(self, bkd):
        """Test uniform samples are in [0, 1] when not transforming."""
        sampler = SobolSampler(
            3, bkd, start_index=1, transform_to_normal=False, seed=42
        )
        samples, _ = sampler.sample(100)

        samples_np = bkd.to_numpy(samples)
        assert np.all(samples_np >= 0.0)
        assert np.all(samples_np <= 1.0)

    def test_normal_transform(self, bkd):
        """Test that transformed samples have approximately normal stats."""
        sampler = SobolSampler(
            3, bkd, start_index=1, transform_to_normal=True, seed=42
        )
        samples, _ = sampler.sample(1000)

        samples_np = bkd.to_numpy(samples)
        # Check mean is close to 0
        assert np.abs(np.mean(samples_np)) < 0.1
        # Check std is close to 1
        assert np.abs(np.std(samples_np) - 1.0) < 0.1

    def test_start_index_skips_samples(self, bkd):
        """Test that start_index skips the initial samples."""
        # Get samples starting from 0
        sampler1 = SobolSampler(
            3,
            bkd,
            start_index=0,
            scramble=False,
            transform_to_normal=False,
        )
        samples_full, _ = sampler1.sample(10)

        # Get samples starting from 2
        sampler2 = SobolSampler(
            3,
            bkd,
            start_index=2,
            scramble=False,
            transform_to_normal=False,
        )
        samples_skip, _ = sampler2.sample(8)

        # Samples starting from 2 should match samples[2:] from full
        bkd.assert_allclose(samples_full[:, 2:], samples_skip, rtol=1e-12)

    def test_sobol_known_values(self, bkd):
        """Test Sobol sequence against known values."""
        # First few Sobol points in 2D (unscrambled)
        sampler = SobolSampler(
            2, bkd, start_index=0, scramble=False, transform_to_normal=False
        )
        samples, _ = sampler.sample(4)
        bkd.to_numpy(samples)

        # Known Sobol sequence values (first 4 points in 2D)
        # [0, 0], [0.5, 0.5], [0.75, 0.25], [0.25, 0.75]
        expected = bkd.asarray(
            [
                [0.0, 0.5, 0.75, 0.25],
                [0.0, 0.5, 0.25, 0.75],
            ]
        )
        bkd.assert_allclose(samples, expected, rtol=1e-10)


class TestGaussianQuadratureSampler:
    """Base test class for Gaussian quadrature sampler."""

    def test_npoints_method(self, bkd):
        """Test npoints() returns correct value."""
        nvars = 3
        npoints_1d = 4
        sampler = GaussianQuadratureSampler(nvars, bkd, npoints_1d)

        assert sampler.npoints() == npoints_1d**nvars

    def test_weights_positive(self, bkd):
        """Test that all weights are positive."""
        sampler = GaussianQuadratureSampler(2, bkd, 5)
        _, weights = sampler.sample(0)

        weights_np = bkd.to_numpy(weights)
        assert np.all(weights_np > 0)

    def test_weights_sum_approximately_one(self, bkd):
        """Test weights sum to approximately 1 (depends on normalization)."""
        sampler = GaussianQuadratureSampler(2, bkd, 5)
        _, weights = sampler.sample(0)

        # For probabilist's Hermite, weights should sum to 1
        weight_sum = bkd.sum(weights)
        expected = bkd.asarray(1.0)
        bkd.assert_allclose(weight_sum, expected, rtol=1e-6)

    def test_deterministic(self, bkd):
        """Test that quadrature is deterministic."""
        sampler1 = GaussianQuadratureSampler(2, bkd, 5)
        sampler2 = GaussianQuadratureSampler(2, bkd, 5)

        samples1, weights1 = sampler1.sample(0)
        samples2, weights2 = sampler2.sample(0)

        bkd.assert_allclose(samples1, samples2, rtol=1e-12)
        bkd.assert_allclose(weights1, weights2, rtol=1e-12)

    def test_integrates_polynomial_exactly(self, bkd):
        """Test that Gauss quadrature integrates polynomials exactly."""
        # Gauss-Hermite with n points integrates polynomials up to degree 2n-1
        nvars = 1
        npoints = 5  # Can integrate up to degree 9
        sampler = GaussianQuadratureSampler(nvars, bkd, npoints)
        samples, weights = sampler.sample(0)

        # Integrate x^2 * exp(-x^2/2) / sqrt(2*pi)
        # Expected value of X^2 for standard normal is 1
        x_squared = samples[0, :] ** 2
        integral = bkd.sum(weights * x_squared)

        expected = bkd.asarray(1.0)
        bkd.assert_allclose(integral, expected, rtol=1e-10)

    def test_integrates_multivariate_polynomial(self, bkd):
        """Test integration of multivariate polynomial."""
        nvars = 2
        npoints = 5
        sampler = GaussianQuadratureSampler(nvars, bkd, npoints)
        samples, weights = sampler.sample(0)

        # Integrate (x1^2 + x2^2) over standard 2D normal
        # E[X1^2 + X2^2] = 1 + 1 = 2
        x1_sq = samples[0, :] ** 2
        x2_sq = samples[1, :] ** 2
        integrand = x1_sq + x2_sq
        integral = bkd.sum(weights * integrand)

        expected = bkd.asarray(2.0)
        bkd.assert_allclose(integral, expected, rtol=1e-10)


class TestOEDQuadratureSampler:
    """Tests for OED quadrature sampler with single joint sampler."""

    def _make_std_normal_dist(self, ndim, bkd):
        """Build a standard normal IndependentJoint distribution."""
        marginals = [GaussianMarginal(0.0, 1.0, bkd) for _ in range(ndim)]
        return IndependentJoint(marginals, bkd)

    def test_nvars_prior_method(self, bkd):
        """Test nvars_prior() returns correct value."""
        nvars_prior = 3
        nobs = 5
        ndim = nvars_prior + nobs
        dist = self._make_std_normal_dist(ndim, bkd)
        joint_sampler = HaltonSampler(
            ndim, bkd, seed=42, distribution=dist
        )
        oed_sampler = OEDQuadratureSampler(joint_sampler, nvars_prior, bkd)

        assert oed_sampler.nvars_prior() == nvars_prior
        assert oed_sampler.nobs() == nobs

    def test_reset(self, bkd):
        """Test that reset gives reproducible results with QMC samplers."""
        nvars_prior = 3
        nobs = 5
        ndim = nvars_prior + nobs
        dist = self._make_std_normal_dist(ndim, bkd)
        joint_sampler = HaltonSampler(
            ndim, bkd, seed=42, distribution=dist
        )
        oed_sampler = OEDQuadratureSampler(joint_sampler, nvars_prior, bkd)

        prior1, latent1, _ = oed_sampler.sample_joint(50)
        oed_sampler.reset()
        prior2, latent2, _ = oed_sampler.sample_joint(50)

        bkd.assert_allclose(prior1, prior2, rtol=1e-12)
        bkd.assert_allclose(latent1, latent2, rtol=1e-12)

    def test_sample_joint_shapes(self, bkd):
        """Test sample_joint returns correct shapes."""
        nvars_prior = 3
        nobs = 5
        nsamples = 100
        ndim = nvars_prior + nobs
        dist = self._make_std_normal_dist(ndim, bkd)
        joint_sampler = HaltonSampler(
            ndim, bkd, seed=42, distribution=dist
        )
        oed_sampler = OEDQuadratureSampler(joint_sampler, nvars_prior, bkd)

        prior_samples, latent_samples, weights = oed_sampler.sample_joint(
            nsamples
        )

        assert prior_samples.shape == (nvars_prior, nsamples)
        assert latent_samples.shape == (nobs, nsamples)
        assert weights.shape == (nsamples,)

    def test_sample_prior_shapes(self, bkd):
        """Test sample_prior returns correct shapes."""
        nvars_prior = 3
        nobs = 5
        nsamples = 100
        ndim = nvars_prior + nobs
        dist = self._make_std_normal_dist(ndim, bkd)
        joint_sampler = HaltonSampler(
            ndim, bkd, seed=42, distribution=dist
        )
        oed_sampler = OEDQuadratureSampler(joint_sampler, nvars_prior, bkd)

        prior_samples, weights = oed_sampler.sample_prior(nsamples)

        assert prior_samples.shape == (nvars_prior, nsamples)
        assert weights.shape == (nsamples,)

    def test_invalid_nparams_raises(self, bkd):
        """Test that nparams >= joint_sampler.nvars() raises."""
        joint_sampler = HaltonSampler(5, bkd, seed=42)
        with pytest.raises(ValueError):
            OEDQuadratureSampler(joint_sampler, 5, bkd)

    def test_from_problem(self, bkd):
        """Test from_problem classmethod with distribution-based factory."""
        from pyapprox.expdesign.benchmarks.functions import (
            build_linear_gaussian_inference_problem,
        )

        problem = build_linear_gaussian_inference_problem(
            nobs=5, degree=2, noise_std=1.0, prior_std=1.0, bkd=bkd
        )
        oed_sampler = OEDQuadratureSampler.from_problem(
            problem,
            lambda dist, b: HaltonSampler(
                dist.nvars(), b, seed=42, distribution=dist
            ),
            bkd,
        )

        assert oed_sampler.nvars_prior() == problem.nparams()
        assert oed_sampler.nobs() == problem.nobs()

        prior, latent, weights = oed_sampler.sample_joint(50)
        assert prior.shape == (problem.nparams(), 50)
        assert latent.shape == (problem.nobs(), 50)


class TestQuadratureStrategies:
    """Test quadrature strategy dispatch and registry."""

    def _make_gaussian_joint(self, nvars, bkd):
        """Helper: create IndependentJoint with standard Gaussian marginals."""
        marginals = [GaussianMarginal(0.0, 1.0, bkd) for _ in range(nvars)]
        return IndependentJoint(marginals, bkd)

    def test_halton_strategy_matches_sampler(self):
        """Test HaltonStrategy.sample() matches direct HaltonSampler."""
        from pyapprox.expdesign.quadrature.strategies import (
            HaltonStrategy,
        )

        bkd = NumpyBkd()
        dist = self._make_gaussian_joint(nvars=2, bkd=bkd)

        # Via strategy
        strategy = HaltonStrategy()
        samples_s, weights_s = strategy.sample(dist, 100, bkd, seed=42)

        # Via sampler directly
        sampler = HaltonSampler(2, bkd, distribution=dist, seed=42)
        samples_d, weights_d = sampler.sample(100)

        bkd.assert_allclose(samples_s, samples_d)
        bkd.assert_allclose(weights_s, weights_d)

    def test_sobol_strategy_matches_sampler(self):
        """Test SobolStrategy.sample() matches direct SobolSampler."""
        from pyapprox.expdesign.quadrature.strategies import (
            SobolStrategy,
        )

        bkd = NumpyBkd()
        dist = self._make_gaussian_joint(nvars=2, bkd=bkd)

        # Via strategy
        strategy = SobolStrategy()
        samples_s, weights_s = strategy.sample(dist, 100, bkd, seed=42)

        # Via sampler directly
        sampler = SobolSampler(2, bkd, distribution=dist, seed=42)
        samples_d, weights_d = sampler.sample(100)

        bkd.assert_allclose(samples_s, samples_d)
        bkd.assert_allclose(weights_s, weights_d)

    def test_get_sampler_unknown_raises(self):
        """Test get_sampler with unknown name raises ValueError."""
        from pyapprox.expdesign.quadrature.strategies import (
            get_sampler,
        )

        with pytest.raises(ValueError):
            get_sampler("nonexistent")

    def test_list_samplers(self):
        """Test list_samplers returns all 4 registered names."""
        from pyapprox.expdesign.quadrature.strategies import (
            list_samplers,
        )

        names = list_samplers()
        assert set(names) == {"gauss", "mc", "halton", "sobol"}
