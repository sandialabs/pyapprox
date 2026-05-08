"""
Tests for OED solvers.

Tests cover:
- Relaxed solver optimization
- Brute-force solver correctness
- Consistency between solvers
- Constraint satisfaction
"""

import numpy as np
import pytest

from pyapprox.expdesign.likelihood import GaussianOEDInnerLoopLikelihood
from pyapprox.expdesign.objective import KLOEDObjective
from pyapprox.expdesign.solver import (
    BruteForceKLOEDSolver,
    RelaxedKLOEDSolver,
    RelaxedOEDConfig,
    solve_kl_oed,
)


class TestRelaxedKLOEDSolver:
    """Base test class for relaxed solver."""

    @pytest.fixture(autouse=True)
    def _setup(self, bkd):
        self._nobs = 4
        self._ninner = 20
        self._nouter = 15

        np.random.seed(123)
        self._noise_variances = bkd.asarray(np.array([0.1, 0.15, 0.2, 0.12]))
        self._outer_shapes = bkd.asarray(
            np.random.randn(self._nobs, self._nouter)
        )
        self._inner_shapes = bkd.asarray(
            np.random.randn(self._nobs, self._ninner)
        )
        self._latent_samples = bkd.asarray(
            np.random.randn(self._nobs, self._nouter)
        )

        self._inner_likelihood = GaussianOEDInnerLoopLikelihood(
            self._noise_variances, bkd
        )

        self._objective = KLOEDObjective(
            self._inner_likelihood,
            self._outer_shapes,
            self._latent_samples,
            self._inner_shapes,
            None,
            None,
            bkd,
        )

    def test_weights_sum_to_one(self, bkd):
        """Test that optimal weights sum to 1."""
        config = RelaxedOEDConfig(verbosity=0, maxiter=50)
        solver = RelaxedKLOEDSolver(self._objective, config)
        weights, _ = solver.solve()

        weight_sum = bkd.sum(weights)
        expected = bkd.asarray(1.0)
        bkd.assert_allclose(
            weight_sum.reshape(-1), expected.reshape(-1), rtol=1e-4
        )

    def test_weights_in_bounds(self, bkd):
        """Test that weights are in [0, 1]."""
        config = RelaxedOEDConfig(verbosity=0, maxiter=50)
        solver = RelaxedKLOEDSolver(self._objective, config)
        weights, _ = solver.solve()

        weights_np = bkd.to_numpy(weights)
        assert np.all(weights_np >= -1e-6)
        assert np.all(weights_np <= 1 + 1e-6)

    def test_custom_initial_weights(self, bkd):
        """Test solver with custom initial weights."""
        config = RelaxedOEDConfig(verbosity=0, maxiter=50)
        solver = RelaxedKLOEDSolver(self._objective, config)

        # Non-uniform initial weights
        init_weights = bkd.asarray([[0.4], [0.3], [0.2], [0.1]])
        weights, eig = solver.solve(init_weights)

        assert weights.shape == (self._nobs, 1)
        assert np.isfinite(eig)

    @pytest.mark.slow_on("NumpyBkd")
    def test_solve_multistart(self, bkd):
        """Test multi-start solver EIG >= single-start EIG."""
        config = RelaxedOEDConfig(verbosity=0, maxiter=50)
        solver = RelaxedKLOEDSolver(self._objective, config)

        # Single start as reference
        weights_single, eig_single = solver.solve()

        # Multi-start should find >= the single-start EIG
        weights_ms, eig_ms = solver.solve_multistart(n_starts=3, seed=42)

        # Weights must sum to 1
        bkd.assert_allclose(
            bkd.sum(weights_ms).reshape(-1),
            bkd.asarray([1.0]),
            rtol=1e-4,
        )
        # Multi-start EIG should be >= single-start (or very close)
        assert eig_ms >= eig_single - 1e-6

    def test_solver_accessors(self, bkd):
        """Test bkd() and nobs() accessors."""
        config = RelaxedOEDConfig(verbosity=0)
        solver = RelaxedKLOEDSolver(self._objective, config)
        assert solver.nobs() == self._nobs
        assert solver.bkd() is not None


class TestBruteForceKLOEDSolver:
    """Base test class for brute-force solver."""

    @pytest.fixture(autouse=True)
    def _setup(self, bkd):
        self._nobs = 4
        self._ninner = 15
        self._nouter = 10

        np.random.seed(456)
        self._noise_variances = bkd.asarray(np.array([0.1, 0.15, 0.2, 0.12]))
        self._outer_shapes = bkd.asarray(
            np.random.randn(self._nobs, self._nouter)
        )
        self._inner_shapes = bkd.asarray(
            np.random.randn(self._nobs, self._ninner)
        )
        self._latent_samples = bkd.asarray(
            np.random.randn(self._nobs, self._nouter)
        )

        self._inner_likelihood = GaussianOEDInnerLoopLikelihood(
            self._noise_variances, bkd
        )

        self._objective = KLOEDObjective(
            self._inner_likelihood,
            self._outer_shapes,
            self._latent_samples,
            self._inner_shapes,
            None,
            None,
            bkd,
        )

    def test_brute_force_k1(self, bkd):
        """Test brute-force with k=1."""
        solver = BruteForceKLOEDSolver(self._objective)
        weights, eig, indices = solver.solve(k=1)

        assert weights.shape == (self._nobs, 1)
        assert len(indices) == 1
        assert np.isfinite(eig)

        # Weight should be 1 at selected index
        weights_np = bkd.to_numpy(weights)
        bkd.assert_allclose(
            bkd.asarray([weights_np[indices[0], 0]]),
            bkd.asarray([1.0]),
            rtol=1e-7,
        )

    def test_brute_force_k2(self, bkd):
        """Test brute-force with k=2."""
        solver = BruteForceKLOEDSolver(self._objective)
        weights, eig, indices = solver.solve(k=2)

        assert weights.shape == (self._nobs, 1)
        assert len(indices) == 2
        assert np.isfinite(eig)

        # Weights should be 1.0 at selected indices (selection indicator)
        weights_np = bkd.to_numpy(weights)
        for idx in indices:
            bkd.assert_allclose(
                bkd.asarray([weights_np[idx, 0]]),
                bkd.asarray([1.0]),
                rtol=1e-7,
            )

    def test_brute_force_k_equals_n(self, bkd):
        """Test brute-force with k=nobs (select all)."""
        solver = BruteForceKLOEDSolver(self._objective)
        weights, eig, indices = solver.solve(k=self._nobs)

        assert len(indices) == self._nobs

        # All weights should be 1.0 (all selected)
        bkd.assert_allclose(weights, bkd.ones((self._nobs, 1)), rtol=1e-7)

    def test_brute_force_invalid_k(self, bkd):
        """Test that invalid k raises error."""
        solver = BruteForceKLOEDSolver(self._objective)

        with pytest.raises(ValueError):
            solver.solve(k=0)

        with pytest.raises(ValueError):
            solver.solve(k=self._nobs + 1)

    def test_n_combinations(self, bkd):
        """Test n_combinations calculation."""
        solver = BruteForceKLOEDSolver(self._objective)

        assert solver.n_combinations(1) == 4
        assert solver.n_combinations(2) == 6
        assert solver.n_combinations(3) == 4
        assert solver.n_combinations(4) == 1

    def test_solve_all_k(self, bkd):
        """Test solve_all_k returns results for all k."""
        solver = BruteForceKLOEDSolver(self._objective)
        results = solver.solve_all_k(k_min=1, k_max=3)

        assert len(results) == 3

        for k, weights, eig, indices in results:
            assert len(indices) == k
            assert np.isfinite(eig)

    def test_solve_all_k_default_max(self, bkd):
        """Test solve_all_k with default k_max=None matches explicit."""
        solver = BruteForceKLOEDSolver(self._objective)
        results_default = solver.solve_all_k(k_min=1)
        results_explicit = solver.solve_all_k(k_min=1, k_max=self._nobs)

        assert len(results_default) == len(results_explicit)
        # EIG values should match exactly
        for (k1, _, eig1, idx1), (k2, _, eig2, idx2) in zip(
            results_default, results_explicit
        ):
            assert k1 == k2
            bkd.assert_allclose(
                bkd.asarray([eig1]), bkd.asarray([eig2])
            )
            assert idx1 == idx2

    def test_brute_force_accessors(self, bkd):
        """Test bkd() and nobs() accessors."""
        solver = BruteForceKLOEDSolver(self._objective)
        assert solver.nobs() == self._nobs
        assert solver.bkd() is not None


class TestSolverConsistency:
    """Tests comparing relaxed and brute-force solvers."""

    @pytest.fixture(autouse=True)
    def _setup(self, bkd):
        # Small problem for tractable brute-force
        self._nobs = 3
        self._ninner = 15
        self._nouter = 12

        np.random.seed(789)
        self._noise_variances = bkd.asarray(np.array([0.1, 0.15, 0.2]))
        self._outer_shapes = bkd.asarray(
            np.random.randn(self._nobs, self._nouter)
        )
        self._inner_shapes = bkd.asarray(
            np.random.randn(self._nobs, self._ninner)
        )
        self._latent_samples = bkd.asarray(
            np.random.randn(self._nobs, self._nouter)
        )

        self._inner_likelihood = GaussianOEDInnerLoopLikelihood(
            self._noise_variances, bkd
        )

        self._objective = KLOEDObjective(
            self._inner_likelihood,
            self._outer_shapes,
            self._latent_samples,
            self._inner_shapes,
            None,
            None,
            bkd,
        )


class TestSolveKLOED:
    """Test the solve_kl_oed convenience function."""

    @pytest.fixture(autouse=True)
    def _setup(self, bkd):
        self._nobs = 4
        self._ninner = 20
        self._nouter = 15

        np.random.seed(123)
        self._noise_variances = bkd.asarray(np.array([0.1, 0.15, 0.2, 0.12]))
        self._outer_shapes = bkd.asarray(
            np.random.randn(self._nobs, self._nouter)
        )
        self._inner_shapes = bkd.asarray(
            np.random.randn(self._nobs, self._ninner)
        )
        self._latent_samples = bkd.asarray(
            np.random.randn(self._nobs, self._nouter)
        )

    def test_solve_kl_oed_matches_manual(self, bkd):
        """Test solve_kl_oed matches manual objective+solver pipeline."""
        # Convenience function
        weights_conv, eig_conv = solve_kl_oed(
            self._noise_variances,
            self._outer_shapes,
            self._inner_shapes,
            self._latent_samples,
            bkd,
            config=RelaxedOEDConfig(verbosity=0, maxiter=50),
        )

        # Manual pipeline (same seed -> same data -> same uniform init)
        inner_likelihood = GaussianOEDInnerLoopLikelihood(
            self._noise_variances, bkd
        )
        objective = KLOEDObjective(
            inner_likelihood,
            self._outer_shapes,
            self._latent_samples,
            self._inner_shapes,
            None,
            None,
            bkd,
        )
        solver = RelaxedKLOEDSolver(
            objective, RelaxedOEDConfig(verbosity=0, maxiter=50)
        )
        weights_manual, eig_manual = solver.solve()

        # Both should converge to same EIG (same init = uniform)
        bkd.assert_allclose(
            bkd.asarray([eig_conv]),
            bkd.asarray([eig_manual]),
            rtol=1e-4,
        )
        bkd.assert_allclose(weights_conv, weights_manual, rtol=1e-3)
