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

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401

from pyapprox.typing.expdesign.quadrature import (
    MonteCarloSampler,
    HaltonSampler,
    SobolSampler,
    GaussianQuadratureSampler,
    OEDQuadratureSampler,
)
from pyapprox.typing.probability.joint import IndependentJoint
from pyapprox.typing.probability.univariate import (
    GaussianMarginal,
    UniformMarginal,
)


class TestMonteCarloSampler(Generic[Array], unittest.TestCase):
    """Base test class for Monte Carlo sampler."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        self._nvars = 3

    def _create_gaussian_distribution(self):
        """Create a standard normal joint distribution."""
        marginals = [GaussianMarginal(0.0, 1.0, self._bkd) for _ in range(self._nvars)]
        return IndependentJoint(marginals, self._bkd)

    def _create_uniform_distribution(self):
        """Create a uniform [0, 1] joint distribution."""
        marginals = [UniformMarginal(0.0, 1.0, self._bkd) for _ in range(self._nvars)]
        return IndependentJoint(marginals, self._bkd)

    def test_weights_sum_to_one(self):
        """Test that weights sum to 1."""
        distribution = self._create_gaussian_distribution()
        sampler = MonteCarloSampler(distribution, self._bkd)
        _, weights = sampler.sample(100)

        weight_sum = self._bkd.sum(weights)
        expected = self._bkd.asarray(1.0)
        self._bkd.assert_allclose(weight_sum, expected, rtol=1e-10)

    def test_uniform_weights(self):
        """Test that all weights are equal."""
        distribution = self._create_gaussian_distribution()
        sampler = MonteCarloSampler(distribution, self._bkd)
        nsamples = 50
        _, weights = sampler.sample(nsamples)

        expected_weight = self._bkd.asarray(1.0 / nsamples)
        self._bkd.assert_allclose(weights[0], expected_weight, rtol=1e-10)
        self._bkd.assert_allclose(weights[-1], expected_weight, rtol=1e-10)

    def test_uniform_samples_in_range(self):
        """Test uniform distribution samples are in [0, 1]."""
        distribution = self._create_uniform_distribution()
        sampler = MonteCarloSampler(distribution, self._bkd)
        samples, _ = sampler.sample(1000)

        samples_np = self._bkd.to_numpy(samples)
        self.assertTrue(np.all(samples_np >= 0.0))
        self.assertTrue(np.all(samples_np <= 1.0))



class TestMonteCarloSamplerNumpy(TestMonteCarloSampler[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestMonteCarloSamplerTorch(TestMonteCarloSampler[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestHaltonSampler(Generic[Array], unittest.TestCase):
    """Base test class for Halton sampler."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        self._nvars = 3

    def test_weights_sum_to_one(self):
        """Test that weights sum to 1."""
        sampler = HaltonSampler(self._nvars, self._bkd, seed=42)
        _, weights = sampler.sample(100)

        weight_sum = self._bkd.sum(weights)
        expected = self._bkd.asarray(1.0)
        self._bkd.assert_allclose(weight_sum, expected, rtol=1e-10)

    def test_deterministic_with_seed(self):
        """Test that same seed gives same samples."""
        sampler1 = HaltonSampler(self._nvars, self._bkd, seed=42)
        sampler2 = HaltonSampler(self._nvars, self._bkd, seed=42)

        samples1, _ = sampler1.sample(50)
        samples2, _ = sampler2.sample(50)

        self._bkd.assert_allclose(samples1, samples2, rtol=1e-12)

    def test_deterministic_without_scramble(self):
        """Test that unscrambled Halton is deterministic."""
        sampler1 = HaltonSampler(
            self._nvars, self._bkd, start_index=0, scramble=False
        )
        sampler2 = HaltonSampler(
            self._nvars, self._bkd, start_index=0, scramble=False
        )

        samples1, _ = sampler1.sample(50)
        samples2, _ = sampler2.sample(50)

        self._bkd.assert_allclose(samples1, samples2, rtol=1e-12)

    def test_reset_restarts_sequence(self):
        """Test that reset restarts the sequence."""
        sampler = HaltonSampler(self._nvars, self._bkd, start_index=0, seed=42)

        samples1, _ = sampler.sample(50)
        sampler.reset()
        samples2, _ = sampler.sample(50)

        self._bkd.assert_allclose(samples1, samples2, rtol=1e-12)

    def test_sequential_samples_different(self):
        """Test that sequential batches give different samples."""
        sampler = HaltonSampler(self._nvars, self._bkd, start_index=0, seed=42)

        samples1, _ = sampler.sample(50)
        samples2, _ = sampler.sample(50)

        # The two batches should be different (continuing sequence)
        samples1_np = self._bkd.to_numpy(samples1)
        samples2_np = self._bkd.to_numpy(samples2)
        self.assertFalse(np.allclose(samples1_np, samples2_np))

    def test_uniform_samples_without_transform(self):
        """Test uniform samples are in [0, 1] when not transforming."""
        sampler = HaltonSampler(
            self._nvars, self._bkd, start_index=1, transform_to_normal=False, seed=42
        )
        samples, _ = sampler.sample(100)

        samples_np = self._bkd.to_numpy(samples)
        self.assertTrue(np.all(samples_np >= 0.0))
        self.assertTrue(np.all(samples_np <= 1.0))

    def test_normal_transform(self):
        """Test that transformed samples have approximately normal stats."""
        sampler = HaltonSampler(
            self._nvars, self._bkd, start_index=1, transform_to_normal=True, seed=42
        )
        samples, _ = sampler.sample(1000)

        samples_np = self._bkd.to_numpy(samples)
        # Check mean is close to 0
        self.assertTrue(np.abs(np.mean(samples_np)) < 0.1)
        # Check std is close to 1
        self.assertTrue(np.abs(np.std(samples_np) - 1.0) < 0.1)

    def test_start_index_skips_samples(self):
        """Test that start_index skips the initial samples."""
        # Get samples starting from 0
        sampler1 = HaltonSampler(
            self._nvars, self._bkd, start_index=0, scramble=False,
            transform_to_normal=False
        )
        samples_full, _ = sampler1.sample(10)

        # Get samples starting from 2
        sampler2 = HaltonSampler(
            self._nvars, self._bkd, start_index=2, scramble=False,
            transform_to_normal=False
        )
        samples_skip, _ = sampler2.sample(8)

        # Samples starting from 2 should match samples[2:] from full
        self._bkd.assert_allclose(samples_full[:, 2:], samples_skip, rtol=1e-12)


class TestHaltonSamplerNumpy(TestHaltonSampler[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestHaltonSamplerTorch(TestHaltonSampler[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestSobolSampler(Generic[Array], unittest.TestCase):
    """Base test class for Sobol sampler."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        self._nvars = 3

    def test_weights_sum_to_one(self):
        """Test that weights sum to 1."""
        sampler = SobolSampler(self._nvars, self._bkd, seed=42)
        _, weights = sampler.sample(100)

        weight_sum = self._bkd.sum(weights)
        expected = self._bkd.asarray(1.0)
        self._bkd.assert_allclose(weight_sum, expected, rtol=1e-10)

    def test_deterministic_with_seed(self):
        """Test that same seed gives same samples."""
        sampler1 = SobolSampler(self._nvars, self._bkd, seed=42)
        sampler2 = SobolSampler(self._nvars, self._bkd, seed=42)

        samples1, _ = sampler1.sample(50)
        samples2, _ = sampler2.sample(50)

        self._bkd.assert_allclose(samples1, samples2, rtol=1e-12)

    def test_deterministic_without_scramble(self):
        """Test that unscrambled Sobol is deterministic."""
        sampler1 = SobolSampler(
            self._nvars, self._bkd, start_index=0, scramble=False
        )
        sampler2 = SobolSampler(
            self._nvars, self._bkd, start_index=0, scramble=False
        )

        samples1, _ = sampler1.sample(50)
        samples2, _ = sampler2.sample(50)

        self._bkd.assert_allclose(samples1, samples2, rtol=1e-12)

    def test_reset_restarts_sequence(self):
        """Test that reset restarts the sequence."""
        sampler = SobolSampler(self._nvars, self._bkd, start_index=0, seed=42)

        samples1, _ = sampler.sample(50)
        sampler.reset()
        samples2, _ = sampler.sample(50)

        self._bkd.assert_allclose(samples1, samples2, rtol=1e-12)

    def test_sequential_samples_different(self):
        """Test that sequential batches give different samples."""
        sampler = SobolSampler(self._nvars, self._bkd, start_index=0, seed=42)

        samples1, _ = sampler.sample(50)
        samples2, _ = sampler.sample(50)

        # The two batches should be different (continuing sequence)
        samples1_np = self._bkd.to_numpy(samples1)
        samples2_np = self._bkd.to_numpy(samples2)
        self.assertFalse(np.allclose(samples1_np, samples2_np))

    def test_uniform_samples_without_transform(self):
        """Test uniform samples are in [0, 1] when not transforming."""
        sampler = SobolSampler(
            self._nvars, self._bkd, start_index=1, transform_to_normal=False, seed=42
        )
        samples, _ = sampler.sample(100)

        samples_np = self._bkd.to_numpy(samples)
        self.assertTrue(np.all(samples_np >= 0.0))
        self.assertTrue(np.all(samples_np <= 1.0))

    def test_normal_transform(self):
        """Test that transformed samples have approximately normal stats."""
        sampler = SobolSampler(
            self._nvars, self._bkd, start_index=1, transform_to_normal=True, seed=42
        )
        samples, _ = sampler.sample(1000)

        samples_np = self._bkd.to_numpy(samples)
        # Check mean is close to 0
        self.assertTrue(np.abs(np.mean(samples_np)) < 0.1)
        # Check std is close to 1
        self.assertTrue(np.abs(np.std(samples_np) - 1.0) < 0.1)

    def test_start_index_skips_samples(self):
        """Test that start_index skips the initial samples."""
        # Get samples starting from 0
        sampler1 = SobolSampler(
            self._nvars, self._bkd, start_index=0, scramble=False,
            transform_to_normal=False
        )
        samples_full, _ = sampler1.sample(10)

        # Get samples starting from 2
        sampler2 = SobolSampler(
            self._nvars, self._bkd, start_index=2, scramble=False,
            transform_to_normal=False
        )
        samples_skip, _ = sampler2.sample(8)

        # Samples starting from 2 should match samples[2:] from full
        self._bkd.assert_allclose(samples_full[:, 2:], samples_skip, rtol=1e-12)

    def test_sobol_known_values(self):
        """Test Sobol sequence against known values."""
        # First few Sobol points in 2D (unscrambled)
        sampler = SobolSampler(
            2, self._bkd, start_index=0, scramble=False, transform_to_normal=False
        )
        samples, _ = sampler.sample(4)
        samples_np = self._bkd.to_numpy(samples)

        # Known Sobol sequence values (first 4 points in 2D)
        # [0, 0], [0.5, 0.5], [0.75, 0.25], [0.25, 0.75]
        expected = self._bkd.asarray([
            [0.0, 0.5, 0.75, 0.25],
            [0.0, 0.5, 0.25, 0.75],
        ])
        self._bkd.assert_allclose(samples, expected, rtol=1e-10)


class TestSobolSamplerNumpy(TestSobolSampler[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestSobolSamplerTorch(TestSobolSampler[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestGaussianQuadratureSampler(Generic[Array], unittest.TestCase):
    """Base test class for Gaussian quadrature sampler."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def test_npoints_method(self):
        """Test npoints() returns correct value."""
        nvars = 3
        npoints_1d = 4
        sampler = GaussianQuadratureSampler(nvars, self._bkd, npoints_1d)

        self.assertEqual(sampler.npoints(), npoints_1d**nvars)

    def test_weights_positive(self):
        """Test that all weights are positive."""
        sampler = GaussianQuadratureSampler(2, self._bkd, 5)
        _, weights = sampler.sample(0)

        weights_np = self._bkd.to_numpy(weights)
        self.assertTrue(np.all(weights_np > 0))

    def test_weights_sum_approximately_one(self):
        """Test weights sum to approximately 1 (depends on normalization)."""
        sampler = GaussianQuadratureSampler(2, self._bkd, 5)
        _, weights = sampler.sample(0)

        # For probabilist's Hermite, weights should sum to 1
        weight_sum = self._bkd.sum(weights)
        expected = self._bkd.asarray(1.0)
        self._bkd.assert_allclose(weight_sum, expected, rtol=1e-6)

    def test_deterministic(self):
        """Test that quadrature is deterministic."""
        sampler1 = GaussianQuadratureSampler(2, self._bkd, 5)
        sampler2 = GaussianQuadratureSampler(2, self._bkd, 5)

        samples1, weights1 = sampler1.sample(0)
        samples2, weights2 = sampler2.sample(0)

        self._bkd.assert_allclose(samples1, samples2, rtol=1e-12)
        self._bkd.assert_allclose(weights1, weights2, rtol=1e-12)

    def test_integrates_polynomial_exactly(self):
        """Test that Gauss quadrature integrates polynomials exactly."""
        # Gauss-Hermite with n points integrates polynomials up to degree 2n-1
        nvars = 1
        npoints = 5  # Can integrate up to degree 9
        sampler = GaussianQuadratureSampler(nvars, self._bkd, npoints)
        samples, weights = sampler.sample(0)

        # Integrate x^2 * exp(-x^2/2) / sqrt(2*pi)
        # Expected value of X^2 for standard normal is 1
        x_squared = samples[0, :] ** 2
        integral = self._bkd.sum(weights * x_squared)

        expected = self._bkd.asarray(1.0)
        self._bkd.assert_allclose(integral, expected, rtol=1e-10)

    def test_integrates_multivariate_polynomial(self):
        """Test integration of multivariate polynomial."""
        nvars = 2
        npoints = 5
        sampler = GaussianQuadratureSampler(nvars, self._bkd, npoints)
        samples, weights = sampler.sample(0)

        # Integrate (x1^2 + x2^2) over standard 2D normal
        # E[X1^2 + X2^2] = 1 + 1 = 2
        x1_sq = samples[0, :] ** 2
        x2_sq = samples[1, :] ** 2
        integrand = x1_sq + x2_sq
        integral = self._bkd.sum(weights * integrand)

        expected = self._bkd.asarray(2.0)
        self._bkd.assert_allclose(integral, expected, rtol=1e-10)


class TestGaussianQuadratureSamplerNumpy(TestGaussianQuadratureSampler[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGaussianQuadratureSamplerTorch(TestGaussianQuadratureSampler[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestOEDQuadratureSampler(Generic[Array], unittest.TestCase):
    """Base test class for OED quadrature sampler."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        self._nvars_prior = 3
        self._nobs = 5

    def _create_prior_distribution(self):
        """Create a standard normal prior distribution."""
        marginals = [
            GaussianMarginal(0.0, 1.0, self._bkd)
            for _ in range(self._nvars_prior)
        ]
        return IndependentJoint(marginals, self._bkd)

    def test_nvars_prior_method(self):
        """Test nvars_prior() returns correct value."""
        prior_dist = self._create_prior_distribution()
        prior_sampler = MonteCarloSampler(prior_dist, self._bkd)
        oed_sampler = OEDQuadratureSampler(
            prior_sampler, self._nobs, self._bkd
        )

        self.assertEqual(oed_sampler.nvars_prior(), self._nvars_prior)

    def test_reset(self):
        """Test that reset gives reproducible results with QMC samplers."""
        # Use Halton samplers which support deterministic reset
        prior_sampler = HaltonSampler(
            self._nvars_prior, self._bkd, seed=42, transform_to_normal=True
        )
        noise_sampler = HaltonSampler(
            self._nobs, self._bkd, seed=123, transform_to_normal=True
        )
        oed_sampler = OEDQuadratureSampler(
            prior_sampler, self._nobs, self._bkd, noise_sampler
        )

        prior1, latent1, _ = oed_sampler.sample_joint(50)
        oed_sampler.reset()
        prior2, latent2, _ = oed_sampler.sample_joint(50)

        self._bkd.assert_allclose(prior1, prior2, rtol=1e-12)
        self._bkd.assert_allclose(latent1, latent2, rtol=1e-12)

    def test_custom_noise_sampler(self):
        """Test with custom noise sampler."""
        prior_dist = self._create_prior_distribution()
        prior_sampler = MonteCarloSampler(prior_dist, self._bkd)
        # Use Halton for noise
        noise_sampler = HaltonSampler(self._nobs, self._bkd, start_index=0)
        oed_sampler = OEDQuadratureSampler(
            prior_sampler, self._nobs, self._bkd, noise_sampler
        )

        _, latent_samples, _ = oed_sampler.sample_joint(100)

        # Should have correct shape
        self.assertEqual(latent_samples.shape, (self._nobs, 100))

    def test_with_gaussian_prior_sampler(self):
        """Test with Gaussian quadrature for prior."""
        nvars = 2
        npoints_1d = 3
        prior_sampler = GaussianQuadratureSampler(nvars, self._bkd, npoints_1d)
        oed_sampler = OEDQuadratureSampler(
            prior_sampler, self._nobs, self._bkd
        )

        # nsamples is ignored for Gaussian quadrature
        prior_samples, weights = oed_sampler.sample_prior(0)

        expected_npoints = npoints_1d**nvars
        self.assertEqual(prior_samples.shape, (nvars, expected_npoints))
        self.assertEqual(weights.shape, (expected_npoints,))


class TestOEDQuadratureSamplerNumpy(TestOEDQuadratureSampler[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestOEDQuadratureSamplerTorch(TestOEDQuadratureSampler[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
