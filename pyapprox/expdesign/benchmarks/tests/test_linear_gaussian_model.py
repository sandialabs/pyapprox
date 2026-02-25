"""
Tests for LinearGaussianOEDModel.

Dual-backend (NumPy + Torch) tests verifying:
1. Shape validation: constructor rejects mismatched shapes
2. Accessor methods return correct shapes
3. Isotropic equivalence: model.exact_eig matches old formula
4. Non-isotropic EIG: non-uniform prior/noise gives finite positive EIG
5. Data generation: correct shapes, seed reproducibility
6. Cholesky backward compat: isotropic generate_parameter_samples matches
   prior_std * randn exactly (same seeds)
7. Forward model consistency: y == A @ theta
8. d_optimal_objective == -exact_eig
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.test_utils import load_tests  # noqa: F401

from pyapprox.expdesign.benchmarks.linear_gaussian_model import (
    LinearGaussianOEDModel,
)
from pyapprox.expdesign.benchmarks.linear_gaussian import (
    LinearGaussianOEDBenchmark,
)


class TestLinearGaussianOEDModel(Generic[Array], unittest.TestCase):
    """Tests for LinearGaussianOEDModel."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def _make_isotropic_model(
        self, nobs: int = 5, nparams: int = 3,
        noise_std: float = 0.5, prior_std: float = 0.5,
    ) -> LinearGaussianOEDModel[Array]:
        """Build an isotropic model for testing."""
        bkd = self._bkd
        np.random.seed(123)
        A = bkd.asarray(np.random.randn(nobs, nparams))
        prior_mean = bkd.zeros((nparams, 1))
        prior_cov = bkd.eye(nparams) * prior_std ** 2
        noise_cov = bkd.eye(nobs) * noise_std ** 2
        locations = bkd.linspace(0.0, 1.0, nobs)
        return LinearGaussianOEDModel(
            A, prior_mean, prior_cov, noise_cov, bkd, locations,
        )

    def _make_nonisotropic_model(
        self, nobs: int = 4, nparams: int = 3,
    ) -> LinearGaussianOEDModel[Array]:
        """Build a non-isotropic model for testing."""
        bkd = self._bkd
        np.random.seed(456)
        A = bkd.asarray(np.random.randn(nobs, nparams))
        prior_mean = bkd.zeros((nparams, 1))
        # Diagonal but non-uniform prior
        prior_diag = np.array([1.0, 0.5, 0.25])
        prior_cov = bkd.diag(bkd.asarray(prior_diag))
        # Diagonal but non-uniform noise
        noise_diag = np.array([0.1, 0.2, 0.3, 0.4])
        noise_cov = bkd.diag(bkd.asarray(noise_diag))
        return LinearGaussianOEDModel(
            A, prior_mean, prior_cov, noise_cov, bkd,
        )

    # ==========================================================================
    # Shape validation tests
    # ==========================================================================

    def test_constructor_rejects_wrong_prior_mean_shape(self):
        """Constructor raises ValueError for mismatched prior_mean."""
        bkd = self._bkd
        A = bkd.asarray(np.eye(3, 2))
        prior_mean = bkd.zeros((3, 1))  # wrong: nparams=2, not 3
        prior_cov = bkd.eye(2)
        noise_cov = bkd.eye(3)
        with self.assertRaises(ValueError):
            LinearGaussianOEDModel(A, prior_mean, prior_cov, noise_cov, bkd)

    def test_constructor_rejects_wrong_prior_cov_shape(self):
        """Constructor raises ValueError for mismatched prior_covariance."""
        bkd = self._bkd
        A = bkd.asarray(np.eye(3, 2))
        prior_mean = bkd.zeros((2, 1))
        prior_cov = bkd.eye(3)  # wrong: should be (2, 2)
        noise_cov = bkd.eye(3)
        with self.assertRaises(ValueError):
            LinearGaussianOEDModel(A, prior_mean, prior_cov, noise_cov, bkd)

    def test_constructor_rejects_wrong_noise_cov_shape(self):
        """Constructor raises ValueError for mismatched noise_covariance."""
        bkd = self._bkd
        A = bkd.asarray(np.eye(3, 2))
        prior_mean = bkd.zeros((2, 1))
        prior_cov = bkd.eye(2)
        noise_cov = bkd.eye(2)  # wrong: should be (3, 3)
        with self.assertRaises(ValueError):
            LinearGaussianOEDModel(A, prior_mean, prior_cov, noise_cov, bkd)

    def test_constructor_rejects_wrong_locations_shape(self):
        """Constructor raises ValueError for mismatched design_locations."""
        bkd = self._bkd
        A = bkd.asarray(np.eye(3, 2))
        prior_mean = bkd.zeros((2, 1))
        prior_cov = bkd.eye(2)
        noise_cov = bkd.eye(3)
        locations = bkd.linspace(0.0, 1.0, 5)  # wrong: should be 3
        with self.assertRaises(ValueError):
            LinearGaussianOEDModel(
                A, prior_mean, prior_cov, noise_cov, bkd, locations,
            )

    # ==========================================================================
    # Accessor tests
    # ==========================================================================

    def test_accessor_shapes(self):
        """Test all accessors return correct shapes."""
        nobs, nparams = 5, 3
        model = self._make_isotropic_model(nobs, nparams)

        self.assertEqual(model.nobs(), nobs)
        self.assertEqual(model.nparams(), nparams)
        self.assertEqual(model.design_matrix().shape, (nobs, nparams))
        self.assertIsNotNone(model.design_locations())
        self.assertEqual(model.design_locations().shape, (nobs,))
        self.assertEqual(model.prior_mean().shape, (nparams, 1))
        self.assertEqual(model.prior_covariance().shape, (nparams, nparams))
        self.assertEqual(model.noise_covariance().shape, (nobs, nobs))
        self.assertEqual(model.noise_variances().shape, (nobs,))

    def test_no_locations(self):
        """Test model works without design_locations."""
        bkd = self._bkd
        A = bkd.asarray(np.eye(3, 2))
        prior_mean = bkd.zeros((2, 1))
        prior_cov = bkd.eye(2)
        noise_cov = bkd.eye(3)
        model = LinearGaussianOEDModel(
            A, prior_mean, prior_cov, noise_cov, bkd,
        )
        self.assertIsNone(model.design_locations())

    # ==========================================================================
    # Isotropic equivalence: model.exact_eig matches old formula
    # ==========================================================================

    def test_isotropic_eig_matches_old_formula(self):
        """Verify model.exact_eig matches 0.5*log(det(I + A^T diag(w) A * s^2/n^2))."""
        bkd = self._bkd
        nobs, nparams = 5, 3
        noise_std, prior_std = 0.5, 0.5
        ratio = prior_std ** 2 / noise_std ** 2

        np.random.seed(99)
        A_np = np.random.randn(nobs, nparams)
        A = bkd.asarray(A_np)
        prior_mean = bkd.zeros((nparams, 1))
        prior_cov = bkd.eye(nparams) * prior_std ** 2
        noise_cov = bkd.eye(nobs) * noise_std ** 2

        model = LinearGaussianOEDModel(
            A, prior_mean, prior_cov, noise_cov, bkd,
        )

        # Uniform weights
        weights = bkd.ones((nobs, 1)) / nobs

        # Old formula
        w = bkd.reshape(weights, (nobs,))
        AtWA = bkd.dot(A.T, w[:, None] * A)
        Y = bkd.eye(nparams) + AtWA * ratio
        _, logdet = bkd.slogdet(Y)
        expected_eig = 0.5 * float(bkd.to_numpy(logdet))

        actual_eig = model.exact_eig(weights)

        self._bkd.assert_allclose(
            bkd.asarray([actual_eig]),
            bkd.asarray([expected_eig]),
            rtol=1e-10,
        )

    def test_isotropic_eig_matches_benchmark(self):
        """Verify model.exact_eig matches LinearGaussianOEDBenchmark.exact_eig."""
        bkd = self._bkd
        nobs, degree = 5, 2
        noise_std, prior_std = 0.5, 0.5

        benchmark = LinearGaussianOEDBenchmark(
            nobs, degree, noise_std, prior_std, bkd,
        )
        model = benchmark.model()

        weights = bkd.ones((nobs, 1)) / nobs
        eig_benchmark = benchmark.exact_eig(weights)
        eig_model = model.exact_eig(weights)

        self._bkd.assert_allclose(
            bkd.asarray([eig_model]),
            bkd.asarray([eig_benchmark]),
            rtol=1e-12,
        )

    def test_isotropic_eig_random_weights(self):
        """Test isotropic equivalence with random Dirichlet weights."""
        bkd = self._bkd
        nobs, nparams = 6, 3
        noise_std, prior_std = 0.3, 1.0
        ratio = prior_std ** 2 / noise_std ** 2

        np.random.seed(77)
        A_np = np.random.randn(nobs, nparams)
        A = bkd.asarray(A_np)
        prior_mean = bkd.zeros((nparams, 1))
        prior_cov = bkd.eye(nparams) * prior_std ** 2
        noise_cov = bkd.eye(nobs) * noise_std ** 2

        model = LinearGaussianOEDModel(
            A, prior_mean, prior_cov, noise_cov, bkd,
        )

        np.random.seed(88)
        w_np = np.random.dirichlet(np.ones(nobs))[:, None]
        weights = bkd.asarray(w_np)

        # Old formula
        w = bkd.reshape(weights, (nobs,))
        AtWA = bkd.dot(A.T, w[:, None] * A)
        Y = bkd.eye(nparams) + AtWA * ratio
        _, logdet = bkd.slogdet(Y)
        expected_eig = 0.5 * float(bkd.to_numpy(logdet))

        actual_eig = model.exact_eig(weights)

        self._bkd.assert_allclose(
            bkd.asarray([actual_eig]),
            bkd.asarray([expected_eig]),
            rtol=1e-10,
        )

    # ==========================================================================
    # Non-isotropic EIG
    # ==========================================================================

    def test_nonisotropic_eig_positive(self):
        """Non-isotropic model gives finite positive EIG."""
        model = self._make_nonisotropic_model()
        nobs = model.nobs()
        weights = self._bkd.ones((nobs, 1)) / nobs
        eig = model.exact_eig(weights)
        self.assertTrue(np.isfinite(eig))
        self.assertGreater(eig, 0.0)

    def test_eig_zero_weights_returns_zero(self):
        """All-zero weights give zero EIG."""
        model = self._make_isotropic_model()
        nobs = model.nobs()
        weights = self._bkd.zeros((nobs, 1))
        eig = model.exact_eig(weights)
        self._bkd.assert_allclose(
            self._bkd.asarray([eig]),
            self._bkd.asarray([0.0]),
            atol=1e-15,
        )

    def test_eig_partial_weights(self):
        """Weights with some zeros give valid EIG."""
        model = self._make_isotropic_model(nobs=5)
        np.random.seed(42)
        w_np = np.zeros((5, 1))
        w_np[0, 0] = 0.5
        w_np[2, 0] = 0.5
        weights = self._bkd.asarray(w_np)
        eig = model.exact_eig(weights)
        self.assertTrue(np.isfinite(eig))
        self.assertGreater(eig, 0.0)

    # ==========================================================================
    # Data generation
    # ==========================================================================

    def test_generate_parameter_samples_shape(self):
        """Parameter samples have correct shape."""
        model = self._make_isotropic_model(nobs=5, nparams=3)
        samples = model.generate_parameter_samples(10, seed=42)
        self.assertEqual(samples.shape, (3, 10))

    def test_generate_observation_data_shapes(self):
        """Observation data has correct shapes."""
        model = self._make_isotropic_model(nobs=5, nparams=3)
        theta, y = model.generate_observation_data(10, seed=42)
        self.assertEqual(theta.shape, (3, 10))
        self.assertEqual(y.shape, (5, 10))

    def test_generate_noisy_observations_shapes(self):
        """Noisy observations have correct shapes."""
        model = self._make_isotropic_model(nobs=5, nparams=3)
        theta, y_clean, y_noisy = model.generate_noisy_observations(10)
        self.assertEqual(theta.shape, (3, 10))
        self.assertEqual(y_clean.shape, (5, 10))
        self.assertEqual(y_noisy.shape, (5, 10))

    def test_generate_latent_samples_shape(self):
        """Latent samples have correct shape."""
        model = self._make_isotropic_model(nobs=5, nparams=3)
        latent = model.generate_latent_samples(10, seed=42)
        self.assertEqual(latent.shape, (5, 10))

    def test_data_generation_reproducible(self):
        """Same seed produces same data."""
        model = self._make_isotropic_model()
        theta1, y1 = model.generate_observation_data(10, seed=42)
        theta2, y2 = model.generate_observation_data(10, seed=42)
        self._bkd.assert_allclose(theta1, theta2, rtol=1e-12)
        self._bkd.assert_allclose(y1, y2, rtol=1e-12)

    # ==========================================================================
    # Cholesky backward compatibility
    # ==========================================================================

    def test_isotropic_parameter_samples_match_direct(self):
        """For isotropic prior, L_prior @ z = prior_std * z exactly.

        Cholesky of s^2 * I = s * I, so L @ z = s * z. This verifies
        the refactored code produces identical samples to the old
        `prior_std * randn()` approach.
        """
        bkd = self._bkd
        nobs, nparams = 5, 3
        prior_std = 0.7
        nsamples = 20
        seed = 42

        # Build isotropic model
        np.random.seed(123)
        A = bkd.asarray(np.random.randn(nobs, nparams))
        prior_mean = bkd.zeros((nparams, 1))
        prior_cov = bkd.eye(nparams) * prior_std ** 2
        noise_cov = bkd.eye(nobs) * 0.25

        model = LinearGaussianOEDModel(
            A, prior_mean, prior_cov, noise_cov, bkd,
        )

        # Model approach: L_prior @ z + prior_mean
        theta_model = model.generate_parameter_samples(nsamples, seed)

        # Old approach: prior_std * randn(...)
        np.random.seed(seed)
        theta_old = bkd.asarray(prior_std * np.random.randn(nparams, nsamples))

        self._bkd.assert_allclose(theta_model, theta_old, rtol=1e-12)

    def test_isotropic_noisy_observations_match_direct(self):
        """For isotropic noise, L_noise @ z = noise_std * z exactly."""
        bkd = self._bkd
        nobs, nparams = 4, 2
        noise_std = 0.3
        prior_std = 0.5
        nsamples = 15
        seed = 42

        np.random.seed(111)
        A = bkd.asarray(np.random.randn(nobs, nparams))
        prior_mean = bkd.zeros((nparams, 1))
        prior_cov = bkd.eye(nparams) * prior_std ** 2
        noise_cov = bkd.eye(nobs) * noise_std ** 2

        model = LinearGaussianOEDModel(
            A, prior_mean, prior_cov, noise_cov, bkd,
        )

        _, _, y_noisy_model = model.generate_noisy_observations(nsamples, seed)

        # Reproduce old approach manually
        np.random.seed(seed)
        theta_np = prior_std * np.random.randn(nparams, nsamples)
        theta = bkd.asarray(theta_np)
        y_clean = bkd.dot(A, theta)
        np.random.seed(seed + 1000)
        noise_np = noise_std * np.random.randn(nobs, nsamples)
        y_noisy_old = y_clean + bkd.asarray(noise_np)

        self._bkd.assert_allclose(y_noisy_model, y_noisy_old, rtol=1e-12)

    # ==========================================================================
    # Forward model consistency
    # ==========================================================================

    def test_forward_model_consistency(self):
        """y = A @ theta for noiseless observations."""
        model = self._make_isotropic_model()
        theta, y = model.generate_observation_data(10, seed=42)
        y_expected = self._bkd.dot(model.design_matrix(), theta)
        self._bkd.assert_allclose(y, y_expected, rtol=1e-12)

    def test_noisy_observations_differ_from_clean(self):
        """Noisy observations differ from noiseless."""
        model = self._make_isotropic_model()
        _, y_clean, y_noisy = model.generate_noisy_observations(10, seed=42)
        diff = self._bkd.to_numpy(y_noisy - y_clean)
        self.assertGreater(np.abs(diff).max(), 0.0)

    # ==========================================================================
    # D-optimal objective
    # ==========================================================================

    def test_d_optimal_is_negative_eig(self):
        """D-optimal objective = -exact_eig."""
        model = self._make_isotropic_model()
        nobs = model.nobs()
        weights = self._bkd.ones((nobs, 1)) / nobs
        eig = model.exact_eig(weights)
        d_opt = model.d_optimal_objective(weights)
        self._bkd.assert_allclose(
            self._bkd.asarray([d_opt]),
            self._bkd.asarray([-eig]),
            rtol=1e-12,
        )


class TestLinearGaussianOEDModelNumpy(
    TestLinearGaussianOEDModel[NDArray[Any]]
):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestLinearGaussianOEDModelTorch(
    TestLinearGaussianOEDModel[torch.Tensor]
):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
