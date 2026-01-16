"""Tests for linear system solvers used in basis expansion fitting.

Tests the solvers in pyapprox.typing.surrogates.affine.solvers:
- LeastSquaresSolver
- RidgeRegressionSolver
- LinearlyConstrainedLstSqSolver
- OMPSolver
- BasisPursuitSolver
- BasisPursuitDenoisingSolver
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.test_utils import load_tests

from pyapprox.typing.surrogates.affine.solvers import (
    LeastSquaresSolver,
    RidgeRegressionSolver,
    LinearlyConstrainedLstSqSolver,
    OMPSolver,
    OMPTerminationFlag,
    BasisPursuitSolver,
    BasisPursuitDenoisingSolver,
)


class TestLeastSquaresSolver(Generic[Array], unittest.TestCase):
    """Test LeastSquaresSolver."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def test_overdetermined_linear(self):
        """Test least squares on overdetermined linear system y = 2x + 1."""
        bkd = self._bkd
        solver = LeastSquaresSolver(bkd)

        # Generate data: y = 2x + 1
        x = bkd.asarray(np.linspace(0, 1, 20).reshape(-1, 1))
        basis_matrix = bkd.concatenate([bkd.ones((20, 1)), x], axis=1)
        y = 2 * x + 1

        coef = solver.solve(basis_matrix, y)
        # Intercept should be 1.0
        bkd.assert_allclose(
            bkd.reshape(coef[0, 0], (1,)), bkd.asarray([1.0]), atol=1e-10
        )
        # Slope should be 2.0
        bkd.assert_allclose(
            bkd.reshape(coef[1, 0], (1,)), bkd.asarray([2.0]), atol=1e-10
        )

    def test_quadratic_fit(self):
        """Test fitting a quadratic: y = x^2 - x + 0.5."""
        bkd = self._bkd
        solver = LeastSquaresSolver(bkd)

        # Generate data: y = x^2 - x + 0.5
        x = bkd.asarray(np.linspace(-1, 1, 30).reshape(-1, 1))
        x2 = x**2
        # Basis: [1, x, x^2]
        basis_matrix = bkd.concatenate([bkd.ones((30, 1)), x, x2], axis=1)
        y = x2 - x + 0.5

        coef = solver.solve(basis_matrix, y)
        expected = bkd.asarray([[0.5], [-1.0], [1.0]])
        bkd.assert_allclose(coef, expected, atol=1e-10)

    def test_multi_qoi(self):
        """Test least squares with multiple quantities of interest."""
        bkd = self._bkd
        solver = LeastSquaresSolver(bkd)

        x = bkd.asarray(np.linspace(0, 1, 15).reshape(-1, 1))
        basis_matrix = bkd.concatenate([bkd.ones((15, 1)), x], axis=1)
        # Two QoIs: y1 = x + 1, y2 = 2x + 3
        y1 = x + 1
        y2 = 2 * x + 3
        y = bkd.concatenate([y1, y2], axis=1)

        coef = solver.solve(basis_matrix, y)
        self.assertEqual(coef.shape, (2, 2))
        # First QoI: intercept=1, slope=1
        bkd.assert_allclose(
            coef[:, 0:1], bkd.asarray([[1.0], [1.0]]), atol=1e-10
        )
        # Second QoI: intercept=3, slope=2
        bkd.assert_allclose(
            coef[:, 1:2], bkd.asarray([[3.0], [2.0]]), atol=1e-10
        )


class TestLeastSquaresSolverNumpy(TestLeastSquaresSolver[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestLeastSquaresSolverTorch(TestLeastSquaresSolver[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestRidgeRegressionSolver(Generic[Array], unittest.TestCase):
    """Test RidgeRegressionSolver."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def test_small_regularization(self):
        """Test ridge regression with small regularization approaches LS."""
        bkd = self._bkd
        solver = RidgeRegressionSolver(bkd, alpha=1e-10)

        x = bkd.asarray(np.linspace(0, 1, 20).reshape(-1, 1))
        basis_matrix = bkd.concatenate([bkd.ones((20, 1)), x], axis=1)
        y = 2 * x + 1

        coef = solver.solve(basis_matrix, y)
        # With tiny regularization, should be close to LS solution
        bkd.assert_allclose(
            bkd.reshape(coef[0, 0], (1,)), bkd.asarray([1.0]), atol=1e-6
        )
        bkd.assert_allclose(
            bkd.reshape(coef[1, 0], (1,)), bkd.asarray([2.0]), atol=1e-6
        )

    def test_regularization_shrinks_coefficients(self):
        """Test that larger regularization shrinks coefficients."""
        bkd = self._bkd

        x = bkd.asarray(np.linspace(0, 1, 20).reshape(-1, 1))
        basis_matrix = bkd.concatenate([bkd.ones((20, 1)), x], axis=1)
        y = 2 * x + 1

        solver_small = RidgeRegressionSolver(bkd, alpha=0.01)
        solver_large = RidgeRegressionSolver(bkd, alpha=10.0)

        coef_small = solver_small.solve(basis_matrix, y)
        coef_large = solver_large.solve(basis_matrix, y)

        # Larger regularization should produce smaller coefficient norms
        norm_small = bkd.norm(coef_small)
        norm_large = bkd.norm(coef_large)
        self.assertTrue(bkd.to_numpy(norm_large) < bkd.to_numpy(norm_small))

    def test_set_regularization(self):
        """Test changing regularization parameter."""
        bkd = self._bkd
        solver = RidgeRegressionSolver(bkd, alpha=0.1)

        x = bkd.asarray(np.linspace(0, 1, 20).reshape(-1, 1))
        basis_matrix = bkd.concatenate([bkd.ones((20, 1)), x], axis=1)
        y = 2 * x + 1

        coef1 = solver.solve(basis_matrix, y)
        solver.set_regularization(1.0)
        coef2 = solver.solve(basis_matrix, y)

        # Different regularization should give different results
        # Assert coefficients differ by computing norm of difference
        diff_norm = float(bkd.to_numpy(bkd.norm(coef1 - coef2)))
        self.assertGreater(diff_norm, 0.01)


class TestRidgeRegressionSolverNumpy(TestRidgeRegressionSolver[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestRidgeRegressionSolverTorch(TestRidgeRegressionSolver[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestLinearlyConstrainedLstSqSolver(Generic[Array], unittest.TestCase):
    """Test LinearlyConstrainedLstSqSolver."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def test_single_constraint(self):
        """Test least squares with single equality constraint."""
        bkd = self._bkd

        # Fit y = ax + b with constraint that b = 0 (line through origin)
        x = bkd.asarray(np.linspace(0, 1, 20).reshape(-1, 1))
        basis_matrix = bkd.concatenate([bkd.ones((20, 1)), x], axis=1)
        # Data generated from y = 2x + 0.5, but we constrain intercept = 0
        y = 2 * x + 0.5

        # Constraint: c[0] = 0
        constraint_matrix = bkd.asarray([[1.0, 0.0]])
        constraint_vector = bkd.asarray([0.0])

        solver = LinearlyConstrainedLstSqSolver(
            bkd, constraint_matrix, constraint_vector
        )
        coef = solver.solve(basis_matrix, y)

        # Intercept should be constrained to 0
        bkd.assert_allclose(
            bkd.reshape(coef[0, 0], (1,)), bkd.asarray([0.0]), atol=1e-10
        )

    def test_multiple_constraints(self):
        """Test with multiple constraints."""
        bkd = self._bkd

        # Fit y = ax^2 + bx + c with constraints: c=1, a+b=0
        x = bkd.asarray(np.linspace(-1, 1, 30).reshape(-1, 1))
        x2 = x**2
        basis_matrix = bkd.concatenate([bkd.ones((30, 1)), x, x2], axis=1)
        # Target function
        y = x2 + 2 * x + 3

        # Constraints: c[0] = 1 (intercept=1), c[1] + c[2] = 0
        # (slope + quad = 0)
        constraint_matrix = bkd.asarray(
            [
                [1.0, 0.0, 0.0],  # c = 1
                [0.0, 1.0, 1.0],  # b + a = 0
            ]
        )
        constraint_vector = bkd.asarray([1.0, 0.0])

        solver = LinearlyConstrainedLstSqSolver(
            bkd, constraint_matrix, constraint_vector
        )
        coef = solver.solve(basis_matrix, y)

        # Check constraints are satisfied
        bkd.assert_allclose(
            bkd.reshape(coef[0, 0], (1,)), bkd.asarray([1.0]), atol=1e-10
        )
        bkd.assert_allclose(
            bkd.reshape(coef[1, 0] + coef[2, 0], (1,)),
            bkd.asarray([0.0]),
            atol=1e-10,
        )


class TestLinearlyConstrainedLstSqSolverNumpy(
    TestLinearlyConstrainedLstSqSolver[NDArray[Any]]
):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestLinearlyConstrainedLstSqSolverTorch(
    TestLinearlyConstrainedLstSqSolver[torch.Tensor]
):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestOMPSolver(Generic[Array], unittest.TestCase):
    """Test Orthogonal Matching Pursuit solver."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def test_exact_sparse_recovery(self):
        """Test OMP recovers exactly sparse solution."""
        bkd = self._bkd
        np.random.seed(42)

        # Create sparse problem: y = Φc where c has only 3 nonzeros
        nsamples, nterms = 50, 20
        true_coef = bkd.zeros((nterms, 1))
        # Set 3 nonzero coefficients
        true_coef[2, 0] = 1.5
        true_coef[7, 0] = -2.0
        true_coef[15, 0] = 0.8

        basis_matrix = bkd.asarray(np.random.randn(nsamples, nterms))
        y = bkd.dot(basis_matrix, true_coef)

        solver = OMPSolver(bkd, max_nonzeros=5, rtol=1e-10)
        coef = solver.solve(basis_matrix, y)

        # Should recover true coefficients
        bkd.assert_allclose(coef, true_coef, atol=1e-8)

    def test_termination_flag_residual(self):
        """Test termination by residual tolerance."""
        bkd = self._bkd
        np.random.seed(42)

        nsamples, nterms = 30, 10
        true_coef = bkd.zeros((nterms, 1))
        true_coef[3, 0] = 2.0

        basis_matrix = bkd.asarray(np.random.randn(nsamples, nterms))
        y = bkd.dot(basis_matrix, true_coef)

        solver = OMPSolver(bkd, max_nonzeros=5, rtol=1e-10)
        solver.solve(basis_matrix, y)

        self.assertEqual(
            solver.termination_flag, OMPTerminationFlag.RESIDUAL_TOLERANCE
        )

    def test_termination_flag_max_nonzeros(self):
        """Test termination by max nonzeros."""
        bkd = self._bkd
        np.random.seed(42)

        nsamples, nterms = 30, 10
        # Create dense target that requires many terms
        basis_matrix = bkd.asarray(np.random.randn(nsamples, nterms))
        y = bkd.sum(basis_matrix, axis=1, keepdims=True)

        solver = OMPSolver(bkd, max_nonzeros=3, rtol=1e-15)
        solver.solve(basis_matrix, y)

        self.assertEqual(
            solver.termination_flag, OMPTerminationFlag.MAX_NONZEROS
        )


class TestOMPSolverNumpy(TestOMPSolver[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestOMPSolverTorch(TestOMPSolver[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestBasisPursuitSolver(Generic[Array], unittest.TestCase):
    """Test Basis Pursuit (L1 minimization) solver."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def test_exact_sparse_recovery(self):
        """Test Basis Pursuit recovers exactly sparse solution."""
        bkd = self._bkd
        np.random.seed(42)

        # Create sparse problem: y = Φc where c has only 2 nonzeros
        # Basis Pursuit can exactly recover sparse solutions when
        # the basis matrix satisfies certain conditions
        nsamples, nterms = 40, 15
        true_coef = bkd.zeros((nterms, 1))
        true_coef[3, 0] = 1.0
        true_coef[10, 0] = -1.5

        basis_matrix = bkd.asarray(np.random.randn(nsamples, nterms))
        y = bkd.dot(basis_matrix, true_coef)

        solver = BasisPursuitSolver(bkd)
        coef = solver.solve(basis_matrix, y)

        # Should recover approximately true coefficients
        bkd.assert_allclose(coef, true_coef, atol=1e-6)

    def test_sparse_solution(self):
        """Test that solution is sparse (many near-zero coefficients)."""
        bkd = self._bkd
        np.random.seed(42)

        nsamples, nterms = 30, 10
        true_coef = bkd.zeros((nterms, 1))
        true_coef[2, 0] = 1.0

        basis_matrix = bkd.asarray(np.random.randn(nsamples, nterms))
        y = bkd.dot(basis_matrix, true_coef)

        solver = BasisPursuitSolver(bkd)
        coef = solver.solve(basis_matrix, y)

        # Count near-zero coefficients
        near_zero_count = int(bkd.sum(bkd.abs(coef) < 1e-6))
        # Most coefficients should be near zero
        self.assertGreater(near_zero_count, nterms // 2)


class TestBasisPursuitSolverNumpy(TestBasisPursuitSolver[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestBasisPursuitSolverTorch(TestBasisPursuitSolver[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestBasisPursuitDenoisingSolver(Generic[Array], unittest.TestCase):
    """Test Basis Pursuit Denoising (LASSO) solver."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def test_sparse_solution_with_noise(self):
        """Test BPDN produces sparse solution with noisy data."""
        bkd = self._bkd
        np.random.seed(42)

        nsamples, nterms = 50, 15
        true_coef = bkd.zeros((nterms, 1))
        true_coef[3, 0] = 2.0
        true_coef[8, 0] = -1.5

        basis_matrix = bkd.asarray(np.random.randn(nsamples, nterms))
        y_clean = bkd.dot(basis_matrix, true_coef)
        # Add noise
        noise = bkd.asarray(0.1 * np.random.randn(nsamples, 1))
        y = y_clean + noise

        solver = BasisPursuitDenoisingSolver(bkd, penalty=0.1)
        coef = solver.solve(basis_matrix, y)

        # Solution should be sparse (L1 penalty promotes sparsity)
        near_zero_count = int(bkd.sum(bkd.abs(coef) < 0.1))
        self.assertGreater(near_zero_count, nterms // 2)

    def test_penalty_affects_sparsity(self):
        """Test that larger penalty produces smaller coefficient norm."""
        bkd = self._bkd
        np.random.seed(42)

        nsamples, nterms = 40, 12
        basis_matrix = bkd.asarray(np.random.randn(nsamples, nterms))
        y = bkd.sum(basis_matrix, axis=1, keepdims=True)

        solver_small = BasisPursuitDenoisingSolver(bkd, penalty=0.01)
        solver_large = BasisPursuitDenoisingSolver(bkd, penalty=10.0)

        coef_small = solver_small.solve(basis_matrix, y)
        coef_large = solver_large.solve(basis_matrix, y)

        # Larger penalty should produce smaller L1 norm (more shrinkage)
        l1_small = float(bkd.to_numpy(bkd.sum(bkd.abs(coef_small))))
        l1_large = float(bkd.to_numpy(bkd.sum(bkd.abs(coef_large))))

        self.assertGreater(l1_small, l1_large)


class TestBasisPursuitDenoisingSolverNumpy(
    TestBasisPursuitDenoisingSolver[NDArray[Any]]
):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestBasisPursuitDenoisingSolverTorch(
    TestBasisPursuitDenoisingSolver[torch.Tensor]
):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
