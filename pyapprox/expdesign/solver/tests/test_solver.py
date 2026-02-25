"""
Tests for OED solvers.

Tests cover:
- Relaxed solver optimization
- Brute-force solver correctness
- Consistency between solvers
- Constraint satisfaction
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.expdesign.likelihood import GaussianOEDInnerLoopLikelihood
from pyapprox.expdesign.objective import KLOEDObjective
from pyapprox.expdesign.solver import (
    BruteForceKLOEDSolver,
    RelaxedKLOEDSolver,
    RelaxedOEDConfig,
    solve_kl_oed,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401


class TestRelaxedKLOEDSolver(Generic[Array], unittest.TestCase):
    """Base test class for relaxed solver."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        self._nobs = 4
        self._ninner = 20
        self._nouter = 15

        np.random.seed(123)
        self._noise_variances = self._bkd.asarray(np.array([0.1, 0.15, 0.2, 0.12]))
        self._outer_shapes = self._bkd.asarray(
            np.random.randn(self._nobs, self._nouter)
        )
        self._inner_shapes = self._bkd.asarray(
            np.random.randn(self._nobs, self._ninner)
        )
        self._latent_samples = self._bkd.asarray(
            np.random.randn(self._nobs, self._nouter)
        )

        self._inner_likelihood = GaussianOEDInnerLoopLikelihood(
            self._noise_variances, self._bkd
        )

        self._objective = KLOEDObjective(
            self._inner_likelihood,
            self._outer_shapes,
            self._latent_samples,
            self._inner_shapes,
            None,
            None,
            self._bkd,
        )

    def test_weights_sum_to_one(self):
        """Test that optimal weights sum to 1."""
        config = RelaxedOEDConfig(verbosity=0, maxiter=50)
        solver = RelaxedKLOEDSolver(self._objective, config)
        weights, _ = solver.solve()

        weight_sum = self._bkd.sum(weights)
        expected = self._bkd.asarray(1.0)
        self._bkd.assert_allclose(
            weight_sum.reshape(-1), expected.reshape(-1), rtol=1e-4
        )

    def test_weights_in_bounds(self):
        """Test that weights are in [0, 1]."""
        config = RelaxedOEDConfig(verbosity=0, maxiter=50)
        solver = RelaxedKLOEDSolver(self._objective, config)
        weights, _ = solver.solve()

        weights_np = self._bkd.to_numpy(weights)
        self.assertTrue(np.all(weights_np >= -1e-6))
        self.assertTrue(np.all(weights_np <= 1 + 1e-6))

    def test_custom_initial_weights(self):
        """Test solver with custom initial weights."""
        config = RelaxedOEDConfig(verbosity=0, maxiter=50)
        solver = RelaxedKLOEDSolver(self._objective, config)

        # Non-uniform initial weights
        init_weights = self._bkd.asarray([[0.4], [0.3], [0.2], [0.1]])
        weights, eig = solver.solve(init_weights)

        self.assertEqual(weights.shape, (self._nobs, 1))
        self.assertTrue(np.isfinite(eig))

    def test_solve_multistart(self):
        """Test multi-start solver EIG >= single-start EIG."""
        config = RelaxedOEDConfig(verbosity=0, maxiter=50)
        solver = RelaxedKLOEDSolver(self._objective, config)

        # Single start as reference
        weights_single, eig_single = solver.solve()

        # Multi-start should find >= the single-start EIG
        weights_ms, eig_ms = solver.solve_multistart(n_starts=3, seed=42)

        # Weights must sum to 1
        self._bkd.assert_allclose(
            self._bkd.sum(weights_ms).reshape(-1),
            self._bkd.asarray([1.0]),
            rtol=1e-4,
        )
        # Multi-start EIG should be >= single-start (or very close)
        self.assertGreaterEqual(eig_ms, eig_single - 1e-6)

    def test_solver_accessors(self):
        """Test bkd() and nobs() accessors."""
        config = RelaxedOEDConfig(verbosity=0)
        solver = RelaxedKLOEDSolver(self._objective, config)
        self.assertEqual(solver.nobs(), self._nobs)
        self.assertIsNotNone(solver.bkd())


class TestRelaxedKLOEDSolverNumpy(TestRelaxedKLOEDSolver[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestRelaxedKLOEDSolverTorch(TestRelaxedKLOEDSolver[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestBruteForceKLOEDSolver(Generic[Array], unittest.TestCase):
    """Base test class for brute-force solver."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        self._nobs = 4
        self._ninner = 15
        self._nouter = 10

        np.random.seed(456)
        self._noise_variances = self._bkd.asarray(np.array([0.1, 0.15, 0.2, 0.12]))
        self._outer_shapes = self._bkd.asarray(
            np.random.randn(self._nobs, self._nouter)
        )
        self._inner_shapes = self._bkd.asarray(
            np.random.randn(self._nobs, self._ninner)
        )
        self._latent_samples = self._bkd.asarray(
            np.random.randn(self._nobs, self._nouter)
        )

        self._inner_likelihood = GaussianOEDInnerLoopLikelihood(
            self._noise_variances, self._bkd
        )

        self._objective = KLOEDObjective(
            self._inner_likelihood,
            self._outer_shapes,
            self._latent_samples,
            self._inner_shapes,
            None,
            None,
            self._bkd,
        )

    def test_brute_force_k1(self):
        """Test brute-force with k=1."""
        solver = BruteForceKLOEDSolver(self._objective)
        weights, eig, indices = solver.solve(k=1)

        self.assertEqual(weights.shape, (self._nobs, 1))
        self.assertEqual(len(indices), 1)
        self.assertTrue(np.isfinite(eig))

        # Weight should be 1 at selected index
        weights_np = self._bkd.to_numpy(weights)
        self._bkd.assert_allclose(
            self._bkd.asarray([weights_np[indices[0], 0]]),
            self._bkd.asarray([1.0]),
            rtol=1e-7,
        )

    def test_brute_force_k2(self):
        """Test brute-force with k=2."""
        solver = BruteForceKLOEDSolver(self._objective)
        weights, eig, indices = solver.solve(k=2)

        self.assertEqual(weights.shape, (self._nobs, 1))
        self.assertEqual(len(indices), 2)
        self.assertTrue(np.isfinite(eig))

        # Weights should be 1.0 at selected indices (selection indicator)
        weights_np = self._bkd.to_numpy(weights)
        for idx in indices:
            self._bkd.assert_allclose(
                self._bkd.asarray([weights_np[idx, 0]]),
                self._bkd.asarray([1.0]),
                rtol=1e-7,
            )

    def test_brute_force_k_equals_n(self):
        """Test brute-force with k=nobs (select all)."""
        solver = BruteForceKLOEDSolver(self._objective)
        weights, eig, indices = solver.solve(k=self._nobs)

        self.assertEqual(len(indices), self._nobs)

        # All weights should be 1.0 (all selected)
        self._bkd.assert_allclose(weights, self._bkd.ones((self._nobs, 1)), rtol=1e-7)

    def test_brute_force_invalid_k(self):
        """Test that invalid k raises error."""
        solver = BruteForceKLOEDSolver(self._objective)

        with self.assertRaises(ValueError):
            solver.solve(k=0)

        with self.assertRaises(ValueError):
            solver.solve(k=self._nobs + 1)

    def test_n_combinations(self):
        """Test n_combinations calculation."""
        solver = BruteForceKLOEDSolver(self._objective)

        self.assertEqual(solver.n_combinations(1), 4)
        self.assertEqual(solver.n_combinations(2), 6)
        self.assertEqual(solver.n_combinations(3), 4)
        self.assertEqual(solver.n_combinations(4), 1)

    def test_solve_all_k(self):
        """Test solve_all_k returns results for all k."""
        solver = BruteForceKLOEDSolver(self._objective)
        results = solver.solve_all_k(k_min=1, k_max=3)

        self.assertEqual(len(results), 3)

        for k, weights, eig, indices in results:
            self.assertEqual(len(indices), k)
            self.assertTrue(np.isfinite(eig))

    def test_solve_all_k_default_max(self):
        """Test solve_all_k with default k_max=None matches explicit."""
        solver = BruteForceKLOEDSolver(self._objective)
        results_default = solver.solve_all_k(k_min=1)
        results_explicit = solver.solve_all_k(k_min=1, k_max=self._nobs)

        self.assertEqual(len(results_default), len(results_explicit))
        # EIG values should match exactly
        for (k1, _, eig1, idx1), (k2, _, eig2, idx2) in zip(
            results_default, results_explicit
        ):
            self.assertEqual(k1, k2)
            self._bkd.assert_allclose(
                self._bkd.asarray([eig1]), self._bkd.asarray([eig2])
            )
            self.assertEqual(idx1, idx2)

    def test_brute_force_accessors(self):
        """Test bkd() and nobs() accessors."""
        solver = BruteForceKLOEDSolver(self._objective)
        self.assertEqual(solver.nobs(), self._nobs)
        self.assertIsNotNone(solver.bkd())


class TestBruteForceKLOEDSolverNumpy(TestBruteForceKLOEDSolver[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestBruteForceKLOEDSolverTorch(TestBruteForceKLOEDSolver[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestSolverConsistency(Generic[Array], unittest.TestCase):
    """Tests comparing relaxed and brute-force solvers."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        # Small problem for tractable brute-force
        self._nobs = 3
        self._ninner = 15
        self._nouter = 12

        np.random.seed(789)
        self._noise_variances = self._bkd.asarray(np.array([0.1, 0.15, 0.2]))
        self._outer_shapes = self._bkd.asarray(
            np.random.randn(self._nobs, self._nouter)
        )
        self._inner_shapes = self._bkd.asarray(
            np.random.randn(self._nobs, self._ninner)
        )
        self._latent_samples = self._bkd.asarray(
            np.random.randn(self._nobs, self._nouter)
        )

        self._inner_likelihood = GaussianOEDInnerLoopLikelihood(
            self._noise_variances, self._bkd
        )

        self._objective = KLOEDObjective(
            self._inner_likelihood,
            self._outer_shapes,
            self._latent_samples,
            self._inner_shapes,
            None,
            None,
            self._bkd,
        )


class TestSolveKLOED(Generic[Array], unittest.TestCase):
    """Test the solve_kl_oed convenience function."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        self._nobs = 4
        self._ninner = 20
        self._nouter = 15

        np.random.seed(123)
        self._noise_variances = self._bkd.asarray(np.array([0.1, 0.15, 0.2, 0.12]))
        self._outer_shapes = self._bkd.asarray(
            np.random.randn(self._nobs, self._nouter)
        )
        self._inner_shapes = self._bkd.asarray(
            np.random.randn(self._nobs, self._ninner)
        )
        self._latent_samples = self._bkd.asarray(
            np.random.randn(self._nobs, self._nouter)
        )

    def test_solve_kl_oed_matches_manual(self):
        """Test solve_kl_oed matches manual objective+solver pipeline."""
        # Convenience function
        weights_conv, eig_conv = solve_kl_oed(
            self._noise_variances,
            self._outer_shapes,
            self._inner_shapes,
            self._latent_samples,
            self._bkd,
            config=RelaxedOEDConfig(verbosity=0, maxiter=50),
        )

        # Manual pipeline (same seed → same data → same uniform init)
        inner_likelihood = GaussianOEDInnerLoopLikelihood(
            self._noise_variances, self._bkd
        )
        objective = KLOEDObjective(
            inner_likelihood,
            self._outer_shapes,
            self._latent_samples,
            self._inner_shapes,
            None,
            None,
            self._bkd,
        )
        solver = RelaxedKLOEDSolver(
            objective, RelaxedOEDConfig(verbosity=0, maxiter=50)
        )
        weights_manual, eig_manual = solver.solve()

        # Both should converge to same EIG (same init = uniform)
        self._bkd.assert_allclose(
            self._bkd.asarray([eig_conv]),
            self._bkd.asarray([eig_manual]),
            rtol=1e-4,
        )
        self._bkd.assert_allclose(weights_conv, weights_manual, rtol=1e-3)


class TestSolveKLOEDNumpy(TestSolveKLOED[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestSolveKLOEDTorch(TestSolveKLOED[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
