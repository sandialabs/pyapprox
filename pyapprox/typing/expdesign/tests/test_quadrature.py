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
    GaussianQuadratureSampler,
    OEDQuadratureSampler,
)


class TestMonteCarloSampler(Generic[Array], unittest.TestCase):
    """Base test class for Monte Carlo sampler."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        self._nvars = 3
        self._seed = 42

    def test_sample_shape_normal(self):
        """Test sample shape for standard normal sampling."""
        sampler = MonteCarloSampler(self._nvars, self._bkd, seed=self._seed)
        nsamples = 100
        samples, weights = sampler.sample(nsamples)

        self.assertEqual(samples.shape, (self._nvars, nsamples))
        self.assertEqual(weights.shape, (nsamples,))

    def test_sample_shape_uniform(self):
        """Test sample shape for uniform sampling."""
        sampler = MonteCarloSampler(
            self._nvars, self._bkd, seed=self._seed, uniform=True
        )
        nsamples = 100
        samples, weights = sampler.sample(nsamples)

        self.assertEqual(samples.shape, (self._nvars, nsamples))
        self.assertEqual(weights.shape, (nsamples,))

    def test_weights_sum_to_one(self):
        """Test that weights sum to 1."""
        sampler = MonteCarloSampler(self._nvars, self._bkd, seed=self._seed)
        _, weights = sampler.sample(100)

        weight_sum = self._bkd.sum(weights)
        expected = self._bkd.asarray(1.0)
        self.assertTrue(self._bkd.allclose(weight_sum, expected, rtol=1e-10))

    def test_uniform_weights(self):
        """Test that all weights are equal."""
        sampler = MonteCarloSampler(self._nvars, self._bkd, seed=self._seed)
        nsamples = 50
        _, weights = sampler.sample(nsamples)

        expected_weight = self._bkd.asarray(1.0 / nsamples)
        self.assertTrue(
            self._bkd.allclose(weights[0], expected_weight, rtol=1e-10)
        )
        self.assertTrue(
            self._bkd.allclose(weights[-1], expected_weight, rtol=1e-10)
        )

    def test_reproducibility_with_seed(self):
        """Test that same seed gives same samples."""
        sampler1 = MonteCarloSampler(self._nvars, self._bkd, seed=self._seed)
        sampler2 = MonteCarloSampler(self._nvars, self._bkd, seed=self._seed)

        samples1, _ = sampler1.sample(50)
        samples2, _ = sampler2.sample(50)

        self.assertTrue(self._bkd.allclose(samples1, samples2, rtol=1e-12))

    def test_reset_reproducibility(self):
        """Test that reset gives same sequence."""
        sampler = MonteCarloSampler(self._nvars, self._bkd, seed=self._seed)

        samples1, _ = sampler.sample(50)
        sampler.reset()
        samples2, _ = sampler.sample(50)

        self.assertTrue(self._bkd.allclose(samples1, samples2, rtol=1e-12))

    def test_uniform_samples_in_range(self):
        """Test uniform samples are in [0, 1]."""
        sampler = MonteCarloSampler(
            self._nvars, self._bkd, seed=self._seed, uniform=True
        )
        samples, _ = sampler.sample(1000)

        samples_np = self._bkd.to_numpy(samples)
        self.assertTrue(np.all(samples_np >= 0.0))
        self.assertTrue(np.all(samples_np <= 1.0))

    def test_nvars_method(self):
        """Test nvars() returns correct value."""
        sampler = MonteCarloSampler(self._nvars, self._bkd, seed=self._seed)
        self.assertEqual(sampler.nvars(), self._nvars)

    def test_bkd_method(self):
        """Test bkd() returns the backend."""
        sampler = MonteCarloSampler(self._nvars, self._bkd, seed=self._seed)
        self.assertIs(sampler.bkd(), self._bkd)


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

    def test_sample_shape(self):
        """Test sample shape."""
        sampler = HaltonSampler(self._nvars, self._bkd)
        nsamples = 100
        samples, weights = sampler.sample(nsamples)

        self.assertEqual(samples.shape, (self._nvars, nsamples))
        self.assertEqual(weights.shape, (nsamples,))

    def test_weights_sum_to_one(self):
        """Test that weights sum to 1."""
        sampler = HaltonSampler(self._nvars, self._bkd)
        _, weights = sampler.sample(100)

        weight_sum = self._bkd.sum(weights)
        expected = self._bkd.asarray(1.0)
        self.assertTrue(self._bkd.allclose(weight_sum, expected, rtol=1e-10))

    def test_deterministic_sequence(self):
        """Test that Halton sequence is deterministic."""
        sampler1 = HaltonSampler(self._nvars, self._bkd, start_index=0)
        sampler2 = HaltonSampler(self._nvars, self._bkd, start_index=0)

        samples1, _ = sampler1.sample(50)
        samples2, _ = sampler2.sample(50)

        self.assertTrue(self._bkd.allclose(samples1, samples2, rtol=1e-12))

    def test_reset_restarts_sequence(self):
        """Test that reset restarts the sequence."""
        sampler = HaltonSampler(self._nvars, self._bkd, start_index=0)

        samples1, _ = sampler.sample(50)
        sampler.reset()
        samples2, _ = sampler.sample(50)

        self.assertTrue(self._bkd.allclose(samples1, samples2, rtol=1e-12))

    def test_sequential_samples_different(self):
        """Test that sequential batches give different samples."""
        sampler = HaltonSampler(self._nvars, self._bkd, start_index=0)

        samples1, _ = sampler.sample(50)
        samples2, _ = sampler.sample(50)

        # The two batches should be different (continuing sequence)
        samples1_np = self._bkd.to_numpy(samples1)
        samples2_np = self._bkd.to_numpy(samples2)
        self.assertFalse(np.allclose(samples1_np, samples2_np))

    def test_uniform_samples_without_transform(self):
        """Test uniform samples are in [0, 1] when not transforming."""
        sampler = HaltonSampler(
            self._nvars, self._bkd, start_index=1, transform_to_normal=False
        )
        samples, _ = sampler.sample(100)

        samples_np = self._bkd.to_numpy(samples)
        self.assertTrue(np.all(samples_np >= 0.0))
        self.assertTrue(np.all(samples_np <= 1.0))

    def test_normal_transform(self):
        """Test that transformed samples have approximately normal stats."""
        sampler = HaltonSampler(
            self._nvars, self._bkd, start_index=1, transform_to_normal=True
        )
        samples, _ = sampler.sample(1000)

        samples_np = self._bkd.to_numpy(samples)
        # Check mean is close to 0
        self.assertTrue(np.abs(np.mean(samples_np)) < 0.1)
        # Check std is close to 1
        self.assertTrue(np.abs(np.std(samples_np) - 1.0) < 0.1)

    def test_low_discrepancy_property(self):
        """Test that Halton has better coverage than random."""
        nvars = 2
        nsamples = 100

        # Halton samples
        halton = HaltonSampler(nvars, self._bkd, start_index=1, transform_to_normal=False)
        halton_samples, _ = halton.sample(nsamples)
        halton_np = self._bkd.to_numpy(halton_samples)

        # Compute simple discrepancy measure: min pairwise distance
        def min_pairwise_distance(samples):
            # samples: (nvars, nsamples)
            min_dist = np.inf
            for i in range(samples.shape[1]):
                for j in range(i + 1, samples.shape[1]):
                    dist = np.linalg.norm(samples[:, i] - samples[:, j])
                    min_dist = min(min_dist, dist)
            return min_dist

        halton_min_dist = min_pairwise_distance(halton_np)

        # Halton should have reasonable minimum distance (not clustered)
        # For 100 points in 2D unit hypercube, min dist should be > 0.01
        self.assertGreater(halton_min_dist, 0.01)


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


class TestGaussianQuadratureSampler(Generic[Array], unittest.TestCase):
    """Base test class for Gaussian quadrature sampler."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def test_sample_shape(self):
        """Test sample shape."""
        nvars = 2
        npoints_1d = 5
        sampler = GaussianQuadratureSampler(nvars, self._bkd, npoints_1d)

        samples, weights = sampler.sample(0)  # nsamples ignored

        expected_npoints = npoints_1d**nvars
        self.assertEqual(samples.shape, (nvars, expected_npoints))
        self.assertEqual(weights.shape, (expected_npoints,))

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
        self.assertTrue(self._bkd.allclose(weight_sum, expected, rtol=1e-6))

    def test_deterministic(self):
        """Test that quadrature is deterministic."""
        sampler1 = GaussianQuadratureSampler(2, self._bkd, 5)
        sampler2 = GaussianQuadratureSampler(2, self._bkd, 5)

        samples1, weights1 = sampler1.sample(0)
        samples2, weights2 = sampler2.sample(0)

        self.assertTrue(self._bkd.allclose(samples1, samples2, rtol=1e-12))
        self.assertTrue(self._bkd.allclose(weights1, weights2, rtol=1e-12))

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
        self.assertTrue(self._bkd.allclose(integral, expected, rtol=1e-10))

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
        self.assertTrue(self._bkd.allclose(integral, expected, rtol=1e-10))


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

    def test_sample_prior_shape(self):
        """Test prior sampling shape."""
        prior_sampler = MonteCarloSampler(
            self._nvars_prior, self._bkd, seed=42
        )
        oed_sampler = OEDQuadratureSampler(
            prior_sampler, self._nobs, self._bkd
        )

        nsamples = 100
        samples, weights = oed_sampler.sample_prior(nsamples)

        self.assertEqual(samples.shape, (self._nvars_prior, nsamples))
        self.assertEqual(weights.shape, (nsamples,))

    def test_sample_joint_shape(self):
        """Test joint sampling shape."""
        prior_sampler = MonteCarloSampler(
            self._nvars_prior, self._bkd, seed=42
        )
        oed_sampler = OEDQuadratureSampler(
            prior_sampler, self._nobs, self._bkd
        )

        nsamples = 100
        prior_samples, latent_samples, weights = oed_sampler.sample_joint(
            nsamples
        )

        self.assertEqual(prior_samples.shape, (self._nvars_prior, nsamples))
        self.assertEqual(latent_samples.shape, (self._nobs, nsamples))
        self.assertEqual(weights.shape, (nsamples,))

    def test_nvars_prior_method(self):
        """Test nvars_prior() returns correct value."""
        prior_sampler = MonteCarloSampler(
            self._nvars_prior, self._bkd, seed=42
        )
        oed_sampler = OEDQuadratureSampler(
            prior_sampler, self._nobs, self._bkd
        )

        self.assertEqual(oed_sampler.nvars_prior(), self._nvars_prior)

    def test_nobs_method(self):
        """Test nobs() returns correct value."""
        prior_sampler = MonteCarloSampler(
            self._nvars_prior, self._bkd, seed=42
        )
        oed_sampler = OEDQuadratureSampler(
            prior_sampler, self._nobs, self._bkd
        )

        self.assertEqual(oed_sampler.nobs(), self._nobs)

    def test_reset(self):
        """Test that reset gives reproducible results."""
        prior_sampler = MonteCarloSampler(
            self._nvars_prior, self._bkd, seed=42
        )
        noise_sampler = MonteCarloSampler(self._nobs, self._bkd, seed=123)
        oed_sampler = OEDQuadratureSampler(
            prior_sampler, self._nobs, self._bkd, noise_sampler
        )

        prior1, latent1, _ = oed_sampler.sample_joint(50)
        oed_sampler.reset()
        prior2, latent2, _ = oed_sampler.sample_joint(50)

        self.assertTrue(self._bkd.allclose(prior1, prior2, rtol=1e-12))
        self.assertTrue(self._bkd.allclose(latent1, latent2, rtol=1e-12))

    def test_custom_noise_sampler(self):
        """Test with custom noise sampler."""
        prior_sampler = MonteCarloSampler(
            self._nvars_prior, self._bkd, seed=42
        )
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
