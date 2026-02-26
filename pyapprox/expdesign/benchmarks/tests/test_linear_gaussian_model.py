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

import numpy as np
import pytest

from pyapprox.expdesign.benchmarks.linear_gaussian import (
    LinearGaussianOEDBenchmark,
)
from pyapprox.expdesign.benchmarks.linear_gaussian_model import (
    LinearGaussianOEDModel,
)


class TestLinearGaussianOEDModel:
    """Tests for LinearGaussianOEDModel."""

    def _make_isotropic_model(
        self,
        bkd,
        nobs: int = 5,
        nparams: int = 3,
        noise_std: float = 0.5,
        prior_std: float = 0.5,
    ):
        """Build an isotropic model for testing."""
        np.random.seed(123)
        A = bkd.asarray(np.random.randn(nobs, nparams))
        prior_mean = bkd.zeros((nparams, 1))
        prior_cov = bkd.eye(nparams) * prior_std**2
        noise_cov = bkd.eye(nobs) * noise_std**2
        locations = bkd.linspace(0.0, 1.0, nobs)
        return LinearGaussianOEDModel(
            A,
            prior_mean,
            prior_cov,
            noise_cov,
            bkd,
            locations,
        )

    def _make_nonisotropic_model(
        self,
        bkd,
        nobs: int = 4,
        nparams: int = 3,
    ):
        """Build a non-isotropic model for testing."""
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
            A,
            prior_mean,
            prior_cov,
            noise_cov,
            bkd,
        )

    # ==========================================================================
    # Shape validation tests
    # ==========================================================================

    def test_constructor_rejects_wrong_prior_mean_shape(self, bkd):
        """Constructor raises ValueError for mismatched prior_mean."""
        A = bkd.asarray(np.eye(3, 2))
        prior_mean = bkd.zeros((3, 1))  # wrong: nparams=2, not 3
        prior_cov = bkd.eye(2)
        noise_cov = bkd.eye(3)
        with pytest.raises(ValueError):
            LinearGaussianOEDModel(A, prior_mean, prior_cov, noise_cov, bkd)

    def test_constructor_rejects_wrong_prior_cov_shape(self, bkd):
        """Constructor raises ValueError for mismatched prior_covariance."""
        A = bkd.asarray(np.eye(3, 2))
        prior_mean = bkd.zeros((2, 1))
        prior_cov = bkd.eye(3)  # wrong: should be (2, 2)
        noise_cov = bkd.eye(3)
        with pytest.raises(ValueError):
            LinearGaussianOEDModel(A, prior_mean, prior_cov, noise_cov, bkd)

    def test_constructor_rejects_wrong_noise_cov_shape(self, bkd):
        """Constructor raises ValueError for mismatched noise_covariance."""
        A = bkd.asarray(np.eye(3, 2))
        prior_mean = bkd.zeros((2, 1))
        prior_cov = bkd.eye(2)
        noise_cov = bkd.eye(2)  # wrong: should be (3, 3)
        with pytest.raises(ValueError):
            LinearGaussianOEDModel(A, prior_mean, prior_cov, noise_cov, bkd)

    def test_constructor_rejects_wrong_locations_shape(self, bkd):
        """Constructor raises ValueError for mismatched design_locations."""
        A = bkd.asarray(np.eye(3, 2))
        prior_mean = bkd.zeros((2, 1))
        prior_cov = bkd.eye(2)
        noise_cov = bkd.eye(3)
        locations = bkd.linspace(0.0, 1.0, 5)  # wrong: should be 3
        with pytest.raises(ValueError):
            LinearGaussianOEDModel(
                A,
                prior_mean,
                prior_cov,
                noise_cov,
                bkd,
                locations,
            )

    # ==========================================================================
    # Accessor tests
    # ==========================================================================

    def test_accessor_shapes(self, bkd):
        """Test all accessors return correct shapes."""
        nobs, nparams = 5, 3
        model = self._make_isotropic_model(bkd, nobs, nparams)

        assert model.nobs() == nobs
        assert model.nparams() == nparams
        assert model.design_matrix().shape == (nobs, nparams)
        assert model.design_locations() is not None
        assert model.design_locations().shape == (nobs,)
        assert model.prior_mean().shape == (nparams, 1)
        assert model.prior_covariance().shape == (nparams, nparams)
        assert model.noise_covariance().shape == (nobs, nobs)
        assert model.noise_variances().shape == (nobs,)

    def test_no_locations(self, bkd):
        """Test model works without design_locations."""
        A = bkd.asarray(np.eye(3, 2))
        prior_mean = bkd.zeros((2, 1))
        prior_cov = bkd.eye(2)
        noise_cov = bkd.eye(3)
        model = LinearGaussianOEDModel(
            A,
            prior_mean,
            prior_cov,
            noise_cov,
            bkd,
        )
        assert model.design_locations() is None

    # ==========================================================================
    # Isotropic equivalence: model.exact_eig matches old formula
    # ==========================================================================

    def test_isotropic_eig_matches_old_formula(self, bkd):
        """Verify model.exact_eig matches 0.5*log(det(I + A^T diag(w) A * s^2/n^2))."""
        nobs, nparams = 5, 3
        noise_std, prior_std = 0.5, 0.5
        ratio = prior_std**2 / noise_std**2

        np.random.seed(99)
        A_np = np.random.randn(nobs, nparams)
        A = bkd.asarray(A_np)
        prior_mean = bkd.zeros((nparams, 1))
        prior_cov = bkd.eye(nparams) * prior_std**2
        noise_cov = bkd.eye(nobs) * noise_std**2

        model = LinearGaussianOEDModel(
            A,
            prior_mean,
            prior_cov,
            noise_cov,
            bkd,
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

        bkd.assert_allclose(
            bkd.asarray([actual_eig]),
            bkd.asarray([expected_eig]),
            rtol=1e-10,
        )

    def test_isotropic_eig_matches_benchmark(self, bkd):
        """Verify model.exact_eig matches LinearGaussianOEDBenchmark.exact_eig."""
        nobs, degree = 5, 2
        noise_std, prior_std = 0.5, 0.5

        benchmark = LinearGaussianOEDBenchmark(
            nobs,
            degree,
            noise_std,
            prior_std,
            bkd,
        )
        model = benchmark.model()

        weights = bkd.ones((nobs, 1)) / nobs
        eig_benchmark = benchmark.exact_eig(weights)
        eig_model = model.exact_eig(weights)

        bkd.assert_allclose(
            bkd.asarray([eig_model]),
            bkd.asarray([eig_benchmark]),
            rtol=1e-12,
        )

    def test_isotropic_eig_random_weights(self, bkd):
        """Test isotropic equivalence with random Dirichlet weights."""
        nobs, nparams = 6, 3
        noise_std, prior_std = 0.3, 1.0
        ratio = prior_std**2 / noise_std**2

        np.random.seed(77)
        A_np = np.random.randn(nobs, nparams)
        A = bkd.asarray(A_np)
        prior_mean = bkd.zeros((nparams, 1))
        prior_cov = bkd.eye(nparams) * prior_std**2
        noise_cov = bkd.eye(nobs) * noise_std**2

        model = LinearGaussianOEDModel(
            A,
            prior_mean,
            prior_cov,
            noise_cov,
            bkd,
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

        bkd.assert_allclose(
            bkd.asarray([actual_eig]),
            bkd.asarray([expected_eig]),
            rtol=1e-10,
        )

    # ==========================================================================
    # Non-isotropic EIG
    # ==========================================================================

    def test_nonisotropic_eig_positive(self, bkd):
        """Non-isotropic model gives finite positive EIG."""
        model = self._make_nonisotropic_model(bkd)
        nobs = model.nobs()
        weights = bkd.ones((nobs, 1)) / nobs
        eig = model.exact_eig(weights)
        assert np.isfinite(eig)
        assert eig > 0.0

    def test_eig_zero_weights_returns_zero(self, bkd):
        """All-zero weights give zero EIG."""
        model = self._make_isotropic_model(bkd)
        nobs = model.nobs()
        weights = bkd.zeros((nobs, 1))
        eig = model.exact_eig(weights)
        bkd.assert_allclose(
            bkd.asarray([eig]),
            bkd.asarray([0.0]),
            atol=1e-15,
        )

    def test_eig_partial_weights(self, bkd):
        """Weights with some zeros give valid EIG."""
        model = self._make_isotropic_model(bkd, nobs=5)
        np.random.seed(42)
        w_np = np.zeros((5, 1))
        w_np[0, 0] = 0.5
        w_np[2, 0] = 0.5
        weights = bkd.asarray(w_np)
        eig = model.exact_eig(weights)
        assert np.isfinite(eig)
        assert eig > 0.0

    # ==========================================================================
    # Data generation
    # ==========================================================================

    def test_generate_parameter_samples_shape(self, bkd):
        """Parameter samples have correct shape."""
        model = self._make_isotropic_model(bkd, nobs=5, nparams=3)
        samples = model.generate_parameter_samples(10, seed=42)
        assert samples.shape == (3, 10)

    def test_generate_observation_data_shapes(self, bkd):
        """Observation data has correct shapes."""
        model = self._make_isotropic_model(bkd, nobs=5, nparams=3)
        theta, y = model.generate_observation_data(10, seed=42)
        assert theta.shape == (3, 10)
        assert y.shape == (5, 10)

    def test_generate_noisy_observations_shapes(self, bkd):
        """Noisy observations have correct shapes."""
        model = self._make_isotropic_model(bkd, nobs=5, nparams=3)
        theta, y_clean, y_noisy = model.generate_noisy_observations(10)
        assert theta.shape == (3, 10)
        assert y_clean.shape == (5, 10)
        assert y_noisy.shape == (5, 10)

    def test_generate_latent_samples_shape(self, bkd):
        """Latent samples have correct shape."""
        model = self._make_isotropic_model(bkd, nobs=5, nparams=3)
        latent = model.generate_latent_samples(10, seed=42)
        assert latent.shape == (5, 10)

    def test_data_generation_reproducible(self, bkd):
        """Same seed produces same data."""
        model = self._make_isotropic_model(bkd)
        theta1, y1 = model.generate_observation_data(10, seed=42)
        theta2, y2 = model.generate_observation_data(10, seed=42)
        bkd.assert_allclose(theta1, theta2, rtol=1e-12)
        bkd.assert_allclose(y1, y2, rtol=1e-12)

    # ==========================================================================
    # Cholesky backward compatibility
    # ==========================================================================

    def test_isotropic_parameter_samples_match_direct(self, bkd):
        """For isotropic prior, L_prior @ z = prior_std * z exactly.

        Cholesky of s^2 * I = s * I, so L @ z = s * z. This verifies
        the refactored code produces identical samples to the old
        `prior_std * randn()` approach.
        """
        nobs, nparams = 5, 3
        prior_std = 0.7
        nsamples = 20
        seed = 42

        # Build isotropic model
        np.random.seed(123)
        A = bkd.asarray(np.random.randn(nobs, nparams))
        prior_mean = bkd.zeros((nparams, 1))
        prior_cov = bkd.eye(nparams) * prior_std**2
        noise_cov = bkd.eye(nobs) * 0.25

        model = LinearGaussianOEDModel(
            A,
            prior_mean,
            prior_cov,
            noise_cov,
            bkd,
        )

        # Model approach: L_prior @ z + prior_mean
        theta_model = model.generate_parameter_samples(nsamples, seed)

        # Old approach: prior_std * randn(...)
        np.random.seed(seed)
        theta_old = bkd.asarray(prior_std * np.random.randn(nparams, nsamples))

        bkd.assert_allclose(theta_model, theta_old, rtol=1e-12)

    def test_isotropic_noisy_observations_match_direct(self, bkd):
        """For isotropic noise, L_noise @ z = noise_std * z exactly."""
        nobs, nparams = 4, 2
        noise_std = 0.3
        prior_std = 0.5
        nsamples = 15
        seed = 42

        np.random.seed(111)
        A = bkd.asarray(np.random.randn(nobs, nparams))
        prior_mean = bkd.zeros((nparams, 1))
        prior_cov = bkd.eye(nparams) * prior_std**2
        noise_cov = bkd.eye(nobs) * noise_std**2

        model = LinearGaussianOEDModel(
            A,
            prior_mean,
            prior_cov,
            noise_cov,
            bkd,
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

        bkd.assert_allclose(y_noisy_model, y_noisy_old, rtol=1e-12)

    # ==========================================================================
    # Forward model consistency
    # ==========================================================================

    def test_forward_model_consistency(self, bkd):
        """y = A @ theta for noiseless observations."""
        model = self._make_isotropic_model(bkd)
        theta, y = model.generate_observation_data(10, seed=42)
        y_expected = bkd.dot(model.design_matrix(), theta)
        bkd.assert_allclose(y, y_expected, rtol=1e-12)

    def test_noisy_observations_differ_from_clean(self, bkd):
        """Noisy observations differ from noiseless."""
        model = self._make_isotropic_model(bkd)
        _, y_clean, y_noisy = model.generate_noisy_observations(10, seed=42)
        diff = bkd.to_numpy(y_noisy - y_clean)
        assert np.abs(diff).max() > 0.0

    # ==========================================================================
    # D-optimal objective
    # ==========================================================================

    def test_d_optimal_is_negative_eig(self, bkd):
        """D-optimal objective = -exact_eig."""
        model = self._make_isotropic_model(bkd)
        nobs = model.nobs()
        weights = bkd.ones((nobs, 1)) / nobs
        eig = model.exact_eig(weights)
        d_opt = model.d_optimal_objective(weights)
        bkd.assert_allclose(
            bkd.asarray([d_opt]),
            bkd.asarray([-eig]),
            rtol=1e-12,
        )
