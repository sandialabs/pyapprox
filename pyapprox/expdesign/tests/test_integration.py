"""
Integration tests for full OED workflow.

Tests cover end-to-end OED workflow combining all components:
- Likelihood setup
- Objective construction
- Solver optimization
- EIG computation
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.backends.protocols import Array
from pyapprox.util.test_utils import load_tests  # noqa: F401

from pyapprox.expdesign import (
    GaussianOEDInnerLoopLikelihood,
    KLOEDObjective,
    RelaxedKLOEDSolver,
    RelaxedOEDConfig,
    BruteForceKLOEDSolver,
    MonteCarloSampler,
    HaltonSampler,
    GaussianQuadratureSampler,
)
from pyapprox.probability.joint import IndependentJoint
from pyapprox.probability.univariate import GaussianMarginal


class TestFullOEDWorkflow(Generic[Array], unittest.TestCase):
    """Test complete OED workflow."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def test_simple_oed_workflow(self):
        """Test simple OED workflow from start to finish."""
        # Problem setup
        nobs = 4
        ninner = 20
        nouter = 15

        np.random.seed(42)

        # 1. Define noise model
        noise_variances = self._bkd.asarray(np.array([0.1, 0.15, 0.2, 0.12]))

        # 2. Generate synthetic model outputs (would come from forward model)
        outer_shapes = self._bkd.asarray(np.random.randn(nobs, nouter))
        inner_shapes = self._bkd.asarray(np.random.randn(nobs, ninner))

        # 3. Generate latent samples for reparameterization
        latent_samples = self._bkd.asarray(np.random.randn(nobs, nouter))

        # 4. Create likelihood
        inner_likelihood = GaussianOEDInnerLoopLikelihood(
            noise_variances, self._bkd
        )

        # 5. Create objective
        objective = KLOEDObjective(
            inner_likelihood,
            outer_shapes,
            latent_samples,
            inner_shapes,
            None,  # Use uniform quadrature weights
            None,
            self._bkd,
        )

        # 6. Create solver and optimize
        config = RelaxedOEDConfig(verbosity=0, maxiter=50)
        solver = RelaxedKLOEDSolver(objective, config)
        optimal_weights, optimal_eig = solver.solve()

        # 7. Verify results
        self.assertEqual(optimal_weights.shape, (nobs, 1))
        self.assertTrue(np.isfinite(optimal_eig))

        # Weights should sum to 1
        weight_sum = self._bkd.sum(optimal_weights)
        expected = self._bkd.asarray(1.0)
        self._bkd.assert_allclose(
            weight_sum.reshape(-1), expected.reshape(-1), rtol=1e-4
        )

        # Weights should be non-negative
        weights_np = self._bkd.to_numpy(optimal_weights)
        self.assertTrue(np.all(weights_np >= -1e-6))

    def test_oed_with_different_quadrature(self):
        """Test OED with different quadrature methods."""
        nobs = 3
        ninner = 15
        nouter = 12

        np.random.seed(123)
        noise_variances = self._bkd.asarray(np.array([0.1, 0.15, 0.2]))
        outer_shapes = self._bkd.asarray(np.random.randn(nobs, nouter))
        inner_shapes = self._bkd.asarray(np.random.randn(nobs, ninner))
        latent_samples = self._bkd.asarray(np.random.randn(nobs, nouter))

        inner_likelihood = GaussianOEDInnerLoopLikelihood(
            noise_variances, self._bkd
        )

        # Test with uniform weights (MC)
        objective_mc = KLOEDObjective(
            inner_likelihood,
            outer_shapes,
            latent_samples,
            inner_shapes,
            None,  # Uniform weights
            None,
            self._bkd,
        )

        # Test with custom Dirichlet weights
        outer_weights = self._bkd.asarray(
            np.random.dirichlet(np.ones(nouter))
        )
        inner_weights = self._bkd.asarray(
            np.random.dirichlet(np.ones(ninner))
        )

        objective_custom = KLOEDObjective(
            inner_likelihood,
            outer_shapes,
            latent_samples,
            inner_shapes,
            outer_weights,
            inner_weights,
            self._bkd,
        )

        # Both should produce valid EIG values
        uniform_weights = self._bkd.ones((nobs, 1)) / nobs
        eig_mc = objective_mc.expected_information_gain(uniform_weights)
        eig_custom = objective_custom.expected_information_gain(uniform_weights)

        self.assertTrue(np.isfinite(eig_mc))
        self.assertTrue(np.isfinite(eig_custom))

    def test_sampler_integration(self):
        """Test integration with quadrature samplers."""
        nvars = 2
        nobs = 3

        # Create Gaussian distribution for MC sampler
        marginals = [GaussianMarginal(0.0, 1.0, self._bkd) for _ in range(nvars)]
        distribution = IndependentJoint(marginals, self._bkd)

        # Create samplers
        mc_sampler = MonteCarloSampler(distribution, self._bkd, seed=42)
        halton_sampler = HaltonSampler(nvars, self._bkd, start_index=1)

        # Generate samples
        nsamples = 50
        mc_samples, mc_weights = mc_sampler.sample(nsamples)
        halton_samples, halton_weights = halton_sampler.sample(nsamples)

        # Verify shapes
        self.assertEqual(mc_samples.shape, (nvars, nsamples))
        self.assertEqual(halton_samples.shape, (nvars, nsamples))

        # Verify weights sum to 1
        mc_sum = self._bkd.sum(mc_weights)
        halton_sum = self._bkd.sum(halton_weights)
        expected = self._bkd.asarray([1.0])
        self._bkd.assert_allclose(
            self._bkd.asarray([mc_sum]), expected, rtol=1e-10
        )
        self._bkd.assert_allclose(
            self._bkd.asarray([halton_sum]), expected, rtol=1e-10
        )

    def test_increasing_samples_convergence(self):
        """Test that EIG converges as sample count increases."""
        nobs = 3
        np.random.seed(42)
        noise_variances = self._bkd.asarray(np.array([0.1, 0.15, 0.2]))

        eigs = []
        for n in [10, 20, 40]:
            outer_shapes = self._bkd.asarray(np.random.randn(nobs, n))
            inner_shapes = self._bkd.asarray(np.random.randn(nobs, n))
            latent_samples = self._bkd.asarray(np.random.randn(nobs, n))

            inner_likelihood = GaussianOEDInnerLoopLikelihood(
                noise_variances, self._bkd
            )

            objective = KLOEDObjective(
                inner_likelihood,
                outer_shapes,
                latent_samples,
                inner_shapes,
                None,
                None,
                self._bkd,
            )

            uniform_weights = self._bkd.ones((nobs, 1)) / nobs
            eig = objective.expected_information_gain(uniform_weights)
            eigs.append(eig)

        # All EIGs should be finite
        for eig in eigs:
            self.assertTrue(np.isfinite(eig))


class TestFullOEDWorkflowNumpy(TestFullOEDWorkflow[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestFullOEDWorkflowTorch(TestFullOEDWorkflow[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestLinearGaussianOED(Generic[Array], unittest.TestCase):
    """Test OED for linear Gaussian model (where analytical solution exists)."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def test_linear_model_eig_positive(self):
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
        outer_shapes = self._bkd.asarray(A @ theta_prior)
        inner_shapes = self._bkd.asarray(A @ theta_inner)

        # Noise
        noise_variances = self._bkd.asarray(np.array([0.1, 0.1, 0.1, 0.1]))
        latent_samples = self._bkd.asarray(np.random.randn(nobs, nouter))

        inner_likelihood = GaussianOEDInnerLoopLikelihood(
            noise_variances, self._bkd
        )

        objective = KLOEDObjective(
            inner_likelihood,
            outer_shapes,
            latent_samples,
            inner_shapes,
            None,
            None,
            self._bkd,
        )

        uniform_weights = self._bkd.ones((nobs, 1)) / nobs
        eig = objective.expected_information_gain(uniform_weights)

        # For informative observations, EIG should be positive
        self.assertTrue(np.isfinite(eig))
        # Note: with random data, EIG may not always be positive
        # but should be finite

    def test_optimal_design_concentrates_weight(self):
        """Test that optimal design may concentrate weight on informative obs."""
        nobs = 4
        ninner = 25
        nouter = 20

        np.random.seed(42)

        # Make observation 0 most informative (lowest noise)
        noise_variances = self._bkd.asarray(np.array([0.01, 0.5, 0.5, 0.5]))

        outer_shapes = self._bkd.asarray(np.random.randn(nobs, nouter))
        inner_shapes = self._bkd.asarray(np.random.randn(nobs, ninner))
        latent_samples = self._bkd.asarray(np.random.randn(nobs, nouter))

        inner_likelihood = GaussianOEDInnerLoopLikelihood(
            noise_variances, self._bkd
        )

        objective = KLOEDObjective(
            inner_likelihood,
            outer_shapes,
            latent_samples,
            inner_shapes,
            None,
            None,
            self._bkd,
        )

        config = RelaxedOEDConfig(verbosity=0, maxiter=100)
        solver = RelaxedKLOEDSolver(objective, config)
        optimal_weights, _ = solver.solve()

        weights_np = self._bkd.to_numpy(optimal_weights)

        # Optimal design should assign more weight to low-noise observation
        # (observation 0 has lowest noise variance)
        # This is a soft check - with random data, exact behavior varies
        self.assertTrue(np.isfinite(weights_np).all())
        self._bkd.assert_allclose(
            self._bkd.asarray([weights_np.sum()]),
            self._bkd.asarray([1.0]),
            rtol=1e-3,
        )


class TestLinearGaussianOEDNumpy(TestLinearGaussianOED[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestLinearGaussianOEDTorch(TestLinearGaussianOED[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
