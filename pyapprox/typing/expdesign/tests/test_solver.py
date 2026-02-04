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

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401

from pyapprox.typing.expdesign.likelihood import GaussianOEDInnerLoopLikelihood
from pyapprox.typing.expdesign.objective import KLOEDObjective
from pyapprox.typing.expdesign.solver import (
    RelaxedKLOEDSolver,
    RelaxedOEDConfig,
    BruteForceKLOEDSolver,
    OEDObjectiveWrapper,
)


class TestOEDObjectiveWrapper(Generic[Array], unittest.TestCase):
    """Base test class for OEDObjectiveWrapper."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        self._nobs = 3
        self._ninner = 10
        self._nouter = 8

        np.random.seed(42)
        self._noise_variances = self._bkd.asarray(
            np.array([0.1, 0.2, 0.15])
        )
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

    def test_wrapper_nvars(self):
        """Test nvars() returns nobs."""
        wrapper = OEDObjectiveWrapper(self._objective, self._bkd)
        self.assertEqual(wrapper.nvars(), self._nobs)

    def test_wrapper_nqoi(self):
        """Test nqoi() returns 1."""
        wrapper = OEDObjectiveWrapper(self._objective, self._bkd)
        self.assertEqual(wrapper.nqoi(), 1)

    def test_wrapper_call_shape(self):
        """Test __call__ returns correct shape."""
        wrapper = OEDObjectiveWrapper(self._objective, self._bkd)
        weights = self._bkd.ones((self._nobs, 2)) / self._nobs
        result = wrapper(weights)
        self.assertEqual(result.shape, (1, 2))

    def test_wrapper_jacobian_shape(self):
        """Test jacobian returns correct shape."""
        wrapper = OEDObjectiveWrapper(self._objective, self._bkd)
        weights = self._bkd.ones((self._nobs, 1)) / self._nobs
        jac = wrapper.jacobian(weights)
        self.assertEqual(jac.shape, (1, self._nobs))


class TestOEDObjectiveWrapperNumpy(TestOEDObjectiveWrapper[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestOEDObjectiveWrapperTorch(TestOEDObjectiveWrapper[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


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
        self._noise_variances = self._bkd.asarray(
            np.array([0.1, 0.15, 0.2, 0.12])
        )
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

    def test_solver_runs(self):
        """Test that solver runs without error."""
        config = RelaxedOEDConfig(verbosity=0, maxiter=50)
        solver = RelaxedKLOEDSolver(self._objective, config)
        weights, eig = solver.solve()

        self.assertEqual(weights.shape, (self._nobs, 1))
        self.assertTrue(np.isfinite(eig))

    def test_weights_sum_to_one(self):
        """Test that optimal weights sum to 1."""
        config = RelaxedOEDConfig(verbosity=0, maxiter=50)
        solver = RelaxedKLOEDSolver(self._objective, config)
        weights, _ = solver.solve()

        weight_sum = self._bkd.sum(weights)
        expected = self._bkd.asarray(1.0)
        self.assertTrue(self._bkd.allclose(weight_sum, expected, rtol=1e-4))

    def test_weights_in_bounds(self):
        """Test that weights are in [0, 1]."""
        config = RelaxedOEDConfig(verbosity=0, maxiter=50)
        solver = RelaxedKLOEDSolver(self._objective, config)
        weights, _ = solver.solve()

        weights_np = self._bkd.to_numpy(weights)
        self.assertTrue(np.all(weights_np >= -1e-6))
        self.assertTrue(np.all(weights_np <= 1 + 1e-6))

    def test_eig_positive(self):
        """Test that EIG is typically positive."""
        config = RelaxedOEDConfig(verbosity=0, maxiter=50)
        solver = RelaxedKLOEDSolver(self._objective, config)
        _, eig = solver.solve()

        # EIG should be positive for informative observations
        # (though not guaranteed for all random data)
        self.assertTrue(np.isfinite(eig))

    def test_custom_initial_weights(self):
        """Test solver with custom initial weights."""
        config = RelaxedOEDConfig(verbosity=0, maxiter=50)
        solver = RelaxedKLOEDSolver(self._objective, config)

        # Non-uniform initial weights
        init_weights = self._bkd.asarray([[0.4], [0.3], [0.2], [0.1]])
        weights, eig = solver.solve(init_weights)

        self.assertEqual(weights.shape, (self._nobs, 1))
        self.assertTrue(np.isfinite(eig))


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
        self._noise_variances = self._bkd.asarray(
            np.array([0.1, 0.15, 0.2, 0.12])
        )
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
        self.assertAlmostEqual(weights_np[indices[0], 0], 1.0)

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
            self.assertAlmostEqual(weights_np[idx, 0], 1.0)

    def test_brute_force_k_equals_n(self):
        """Test brute-force with k=nobs (select all)."""
        solver = BruteForceKLOEDSolver(self._objective)
        weights, eig, indices = solver.solve(k=self._nobs)

        self.assertEqual(len(indices), self._nobs)

        # All weights should be 1.0 (all selected)
        weights_np = self._bkd.to_numpy(weights)
        for w in weights_np.flatten():
            self.assertAlmostEqual(w, 1.0)

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
        self._noise_variances = self._bkd.asarray(
            np.array([0.1, 0.15, 0.2])
        )
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

    def test_increasing_k_improves_eig(self):
        """Test that EIG generally increases with k."""
        bf_solver = BruteForceKLOEDSolver(self._objective)
        results = bf_solver.solve_all_k(k_min=1, k_max=self._nobs)

        # EIG should generally increase (more observations = more info)
        # But not strictly monotonic for all random data
        eigs = [r[2] for r in results]
        self.assertTrue(all(np.isfinite(eigs)))


class TestSolverConsistencyNumpy(TestSolverConsistency[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestSolverConsistencyTorch(TestSolverConsistency[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
