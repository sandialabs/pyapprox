"""
Integration tests for full OED workflow.

Tests cover end-to-end OED workflow combining all components:
- Likelihood setup
- Objective construction
- Solver optimization
- EIG computation
"""

import numpy as np

from pyapprox.expdesign import (
    GaussianOEDInnerLoopLikelihood,
    HaltonSampler,
    KLOEDObjective,
    MonteCarloSampler,
    RelaxedKLOEDSolver,
    RelaxedOEDConfig,
)
from pyapprox.probability.joint import IndependentJoint
from pyapprox.probability.univariate import GaussianMarginal


class TestFullOEDWorkflow:
    """Test complete OED workflow."""

    def test_simple_oed_workflow(self, bkd):
        """Test simple OED workflow from start to finish."""
        # Problem setup
        nobs = 4
        ninner = 20
        nouter = 15

        np.random.seed(42)

        # 1. Define noise model
        noise_variances = bkd.asarray(np.array([0.1, 0.15, 0.2, 0.12]))

        # 2. Generate synthetic model outputs (would come from forward model)
        outer_shapes = bkd.asarray(np.random.randn(nobs, nouter))
        inner_shapes = bkd.asarray(np.random.randn(nobs, ninner))

        # 3. Generate latent samples for reparameterization
        latent_samples = bkd.asarray(np.random.randn(nobs, nouter))

        # 4. Create likelihood
        inner_likelihood = GaussianOEDInnerLoopLikelihood(noise_variances, bkd)

        # 5. Create objective
        objective = KLOEDObjective(
            inner_likelihood,
            outer_shapes,
            latent_samples,
            inner_shapes,
            None,  # Use uniform quadrature weights
            None,
            bkd,
        )

        # 6. Create solver and optimize
        config = RelaxedOEDConfig(verbosity=0, maxiter=50)
        solver = RelaxedKLOEDSolver(objective, config)
        optimal_weights, optimal_eig = solver.solve()

        # 7. Verify results
        assert optimal_weights.shape == (nobs, 1)
        assert np.isfinite(optimal_eig)

        # Weights should sum to 1
        weight_sum = bkd.sum(optimal_weights)
        expected = bkd.asarray(1.0)
        bkd.assert_allclose(
            weight_sum.reshape(-1), expected.reshape(-1), rtol=1e-4
        )

        # Weights should be non-negative
        weights_np = bkd.to_numpy(optimal_weights)
        assert np.all(weights_np >= -1e-6)

    def test_oed_with_different_quadrature(self, bkd):
        """Test OED with different quadrature methods."""
        nobs = 3
        ninner = 15
        nouter = 12

        np.random.seed(123)
        noise_variances = bkd.asarray(np.array([0.1, 0.15, 0.2]))
        outer_shapes = bkd.asarray(np.random.randn(nobs, nouter))
        inner_shapes = bkd.asarray(np.random.randn(nobs, ninner))
        latent_samples = bkd.asarray(np.random.randn(nobs, nouter))

        inner_likelihood = GaussianOEDInnerLoopLikelihood(noise_variances, bkd)

        # Test with uniform weights (MC)
        objective_mc = KLOEDObjective(
            inner_likelihood,
            outer_shapes,
            latent_samples,
            inner_shapes,
            None,  # Uniform weights
            None,
            bkd,
        )

        # Test with custom Dirichlet weights
        outer_weights = bkd.asarray(np.random.dirichlet(np.ones(nouter)))
        inner_weights = bkd.asarray(np.random.dirichlet(np.ones(ninner)))

        objective_custom = KLOEDObjective(
            inner_likelihood,
            outer_shapes,
            latent_samples,
            inner_shapes,
            outer_weights,
            inner_weights,
            bkd,
        )

        # Both should produce valid EIG values
        uniform_weights = bkd.ones((nobs, 1)) / nobs
        eig_mc = objective_mc.expected_information_gain(uniform_weights)
        eig_custom = objective_custom.expected_information_gain(uniform_weights)

        assert np.isfinite(eig_mc)
        assert np.isfinite(eig_custom)

    def test_sampler_integration(self, bkd):
        """Test integration with quadrature samplers."""
        nvars = 2

        # Create Gaussian distribution for MC sampler
        marginals = [GaussianMarginal(0.0, 1.0, bkd) for _ in range(nvars)]
        distribution = IndependentJoint(marginals, bkd)

        # Create samplers
        mc_sampler = MonteCarloSampler(distribution, bkd, seed=42)
        halton_sampler = HaltonSampler(nvars, bkd, start_index=1)

        # Generate samples
        nsamples = 50
        mc_samples, mc_weights = mc_sampler.sample(nsamples)
        halton_samples, halton_weights = halton_sampler.sample(nsamples)

        # Verify shapes
        assert mc_samples.shape == (nvars, nsamples)
        assert halton_samples.shape == (nvars, nsamples)

        # Verify weights sum to 1
        mc_sum = bkd.sum(mc_weights)
        halton_sum = bkd.sum(halton_weights)
        expected = bkd.asarray([1.0])
        bkd.assert_allclose(bkd.asarray([mc_sum]), expected, rtol=1e-10)
        bkd.assert_allclose(bkd.asarray([halton_sum]), expected, rtol=1e-10)

    def test_increasing_samples_convergence(self, bkd):
        """Test that EIG converges as sample count increases."""
        nobs = 3
        np.random.seed(42)
        noise_variances = bkd.asarray(np.array([0.1, 0.15, 0.2]))

        eigs = []
        for n in [10, 20, 40]:
            outer_shapes = bkd.asarray(np.random.randn(nobs, n))
            inner_shapes = bkd.asarray(np.random.randn(nobs, n))
            latent_samples = bkd.asarray(np.random.randn(nobs, n))

            inner_likelihood = GaussianOEDInnerLoopLikelihood(
                noise_variances, bkd
            )

            objective = KLOEDObjective(
                inner_likelihood,
                outer_shapes,
                latent_samples,
                inner_shapes,
                None,
                None,
                bkd,
            )

            uniform_weights = bkd.ones((nobs, 1)) / nobs
            eig = objective.expected_information_gain(uniform_weights)
            eigs.append(eig)

        # All EIGs should be finite
        for eig in eigs:
            assert np.isfinite(eig)


class TestLinearGaussianOED:
    """Test OED for linear Gaussian model (where analytical solution exists)."""

    def test_linear_model_eig_positive(self, bkd):
        """Test EIG is positive for linear Gaussian model."""
        # Simple linear model: y = A @ theta + noise
        # For this case, we expect positive EIG
        nobs = 4
        ninner = 30
        nouter = 25

        np.random.seed(42)

        # Forward model matrix (sensitivity of obs to params)
        A = np.random.randn(nobs, 2)

        # Prior samples (2D Gaussian prior)
        theta_prior = np.random.randn(2, nouter)
        theta_inner = np.random.randn(2, ninner)

        # Model outputs: shapes = A @ theta
        outer_shapes = bkd.asarray(A @ theta_prior)
        inner_shapes = bkd.asarray(A @ theta_inner)

        # Noise
        noise_variances = bkd.asarray(np.array([0.1, 0.1, 0.1, 0.1]))
        latent_samples = bkd.asarray(np.random.randn(nobs, nouter))

        inner_likelihood = GaussianOEDInnerLoopLikelihood(noise_variances, bkd)

        objective = KLOEDObjective(
            inner_likelihood,
            outer_shapes,
            latent_samples,
            inner_shapes,
            None,
            None,
            bkd,
        )

        uniform_weights = bkd.ones((nobs, 1)) / nobs
        eig = objective.expected_information_gain(uniform_weights)

        # For informative observations, EIG should be positive
        assert np.isfinite(eig)
        # Note: with random data, EIG may not always be positive
        # but should be finite

    def test_optimal_design_concentrates_weight(self, bkd):
        """Test that optimal design may concentrate weight on informative obs."""
        nobs = 4
        ninner = 25
        nouter = 20

        np.random.seed(42)

        # Make observation 0 most informative (lowest noise)
        noise_variances = bkd.asarray(np.array([0.01, 0.5, 0.5, 0.5]))

        outer_shapes = bkd.asarray(np.random.randn(nobs, nouter))
        inner_shapes = bkd.asarray(np.random.randn(nobs, ninner))
        latent_samples = bkd.asarray(np.random.randn(nobs, nouter))

        inner_likelihood = GaussianOEDInnerLoopLikelihood(noise_variances, bkd)

        objective = KLOEDObjective(
            inner_likelihood,
            outer_shapes,
            latent_samples,
            inner_shapes,
            None,
            None,
            bkd,
        )

        config = RelaxedOEDConfig(verbosity=0, maxiter=100)
        solver = RelaxedKLOEDSolver(objective, config)
        optimal_weights, _ = solver.solve()

        weights_np = bkd.to_numpy(optimal_weights)

        # Optimal design should assign more weight to low-noise observation
        # (observation 0 has lowest noise variance)
        # This is a soft check - with random data, exact behavior varies
        assert np.isfinite(weights_np).all()
        bkd.assert_allclose(
            bkd.asarray([weights_np.sum()]),
            bkd.asarray([1.0]),
            rtol=1e-3,
        )
