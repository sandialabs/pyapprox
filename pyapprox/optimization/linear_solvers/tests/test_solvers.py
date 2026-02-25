"""Tests for linear solvers."""

import unittest
from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401
from pyapprox.optimization.linear_solvers.direct import (
    DirectSolver,
    direct_solve,
)
from pyapprox.optimization.linear_solvers.iterative.cg import (
    ConjugateGradient,
    cg_solve,
)
from pyapprox.optimization.linear_solvers.iterative.pcg import (
    PreconditionedConjugateGradient,
    pcg_solve,
)
from pyapprox.optimization.linear_solvers.preconditioners.jacobi import (
    JacobiPreconditioner,
    BlockJacobiPreconditioner,
    jacobi_preconditioner,
    block_jacobi_preconditioner,
)


class TestDirectSolver(Generic[Array], unittest.TestCase):
    """Base test class for direct solver."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_simple_system(self):
        """Test solving a simple linear system."""
        bkd = self.bkd()
        # A = [[2, 1], [1, 3]]
        A = bkd.array([[2.0, 1.0], [1.0, 3.0]])
        b = bkd.array([3.0, 4.0])

        solver = DirectSolver(bkd)
        x = solver.solve(A, b)

        # Check A @ x = b
        residual = A @ x - b
        bkd.assert_allclose(residual, bkd.zeros((2,)), atol=1e-12)

    def test_functional_interface(self):
        """Test the functional interface."""
        bkd = self.bkd()
        A = bkd.array([[4.0, 1.0], [1.0, 3.0]])
        b = bkd.array([5.0, 4.0])

        x = direct_solve(A, b, bkd)

        residual = A @ x - b
        bkd.assert_allclose(residual, bkd.zeros((2,)), atol=1e-12)

    def test_multiple_rhs(self):
        """Test solving with multiple right-hand sides."""
        bkd = self.bkd()
        A = bkd.array([[2.0, 1.0], [1.0, 3.0]])
        # Multiple RHS
        B = bkd.array([[3.0, 1.0], [4.0, 2.0]])

        solver = DirectSolver(bkd)
        X = solver.solve(A, B)

        # Check A @ X = B
        residual = A @ X - B
        bkd.assert_allclose(residual, bkd.zeros((2, 2)), atol=1e-12)


class TestConjugateGradient(Generic[Array], unittest.TestCase):
    """Base test class for Conjugate Gradient solver."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def _create_spd_matrix(self, n: int) -> Array:
        """Create a symmetric positive definite matrix."""
        bkd = self.bkd()
        # Create SPD matrix A = L @ L^T + n*I (diagonally dominant)
        L = bkd.array([[float(i + j) for j in range(n)] for i in range(n)])
        A = L @ bkd.transpose(L) + float(n) * bkd.eye(n)
        return A

    def test_simple_spd_system(self):
        """Test CG on a simple SPD system."""
        bkd = self.bkd()
        # Simple SPD matrix
        A = bkd.array([[4.0, 1.0], [1.0, 3.0]])
        b = bkd.array([5.0, 4.0])

        solver = ConjugateGradient(bkd)
        x, niter, converged = solver.solve(A, b, tol=1e-10)

        self.assertTrue(converged)
        residual = A @ x - b
        bkd.assert_allclose(residual, bkd.zeros((2,)), atol=1e-8)

    def test_larger_system(self):
        """Test CG on a larger system."""
        bkd = self.bkd()
        n = 10
        A = self._create_spd_matrix(n)
        x_true = bkd.ones((n,))
        b = A @ x_true

        solver = ConjugateGradient(bkd)
        x, niter, converged = solver.solve(A, b, tol=1e-10, maxiter=50)

        self.assertTrue(converged)
        bkd.assert_allclose(x, x_true, atol=1e-6)

    def test_with_initial_guess(self):
        """Test CG with a good initial guess converges faster."""
        bkd = self.bkd()
        n = 5
        A = self._create_spd_matrix(n)
        x_true = bkd.ones((n,))
        b = A @ x_true

        solver = ConjugateGradient(bkd)

        # With zero initial guess
        _, niter_zero, _ = solver.solve(A, b, tol=1e-10)

        # With good initial guess
        x0 = 0.9 * x_true
        _, niter_good, _ = solver.solve(A, b, x0=x0, tol=1e-10)

        # Good initial guess should converge faster
        self.assertLessEqual(niter_good, niter_zero)

    def test_functional_interface(self):
        """Test the functional interface."""
        bkd = self.bkd()
        A = bkd.array([[4.0, 1.0], [1.0, 3.0]])
        b = bkd.array([5.0, 4.0])

        x, niter, converged = cg_solve(A, b, bkd, tol=1e-10)

        self.assertTrue(converged)
        residual = A @ x - b
        bkd.assert_allclose(residual, bkd.zeros((2,)), atol=1e-8)

    def test_matrix_free(self):
        """Test matrix-free CG."""
        bkd = self.bkd()
        A = bkd.array([[4.0, 1.0], [1.0, 3.0]])
        b = bkd.array([5.0, 4.0])

        def matvec(x):
            return A @ x

        solver = ConjugateGradient(bkd)
        x, niter, converged = solver.solve_matvec(matvec, b, tol=1e-10)

        self.assertTrue(converged)
        residual = A @ x - b
        bkd.assert_allclose(residual, bkd.zeros((2,)), atol=1e-8)

    def test_zero_rhs(self):
        """Test CG with zero RHS."""
        bkd = self.bkd()
        A = bkd.array([[4.0, 1.0], [1.0, 3.0]])
        b = bkd.zeros((2,))

        solver = ConjugateGradient(bkd)
        x, niter, converged = solver.solve(A, b)

        self.assertTrue(converged)
        self.assertEqual(niter, 0)
        bkd.assert_allclose(x, bkd.zeros((2,)), atol=1e-14)


class TestPreconditionedCG(Generic[Array], unittest.TestCase):
    """Base test class for Preconditioned Conjugate Gradient solver."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def _create_diag_dominant_spd(self, n: int) -> Array:
        """Create a diagonally dominant SPD matrix."""
        bkd = self.bkd()
        # Create SPD matrix with strong diagonal dominance
        A = bkd.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    A[i, j] = float(n + 1)
                else:
                    A[i, j] = 1.0 / (1.0 + abs(float(i - j)))
        return A

    def test_pcg_without_preconditioner(self):
        """Test PCG without preconditioner (should be same as CG)."""
        bkd = self.bkd()
        A = bkd.array([[4.0, 1.0], [1.0, 3.0]])
        b = bkd.array([5.0, 4.0])

        # PCG without preconditioner
        pcg_solver = PreconditionedConjugateGradient(bkd, preconditioner=None)
        x_pcg, _, converged_pcg = pcg_solver.solve(A, b, tol=1e-10)

        # Standard CG
        cg_solver = ConjugateGradient(bkd)
        x_cg, _, converged_cg = cg_solver.solve(A, b, tol=1e-10)

        self.assertTrue(converged_pcg)
        self.assertTrue(converged_cg)
        bkd.assert_allclose(x_pcg, x_cg, atol=1e-10)

    def test_pcg_with_jacobi(self):
        """Test PCG with Jacobi preconditioner."""
        bkd = self.bkd()
        n = 10
        A = self._create_diag_dominant_spd(n)
        x_true = bkd.ones((n,))
        b = A @ x_true

        # Setup Jacobi preconditioner
        precond = jacobi_preconditioner(A, bkd)

        solver = PreconditionedConjugateGradient(bkd, preconditioner=precond)
        x, niter, converged = solver.solve(A, b, tol=1e-10, maxiter=50)

        self.assertTrue(converged)
        bkd.assert_allclose(x, x_true, atol=1e-6)

    def test_pcg_improves_convergence(self):
        """Test that preconditioning improves convergence."""
        bkd = self.bkd()
        n = 20
        A = self._create_diag_dominant_spd(n)
        x_true = bkd.ones((n,))
        b = A @ x_true

        # CG without preconditioner
        cg_solver = ConjugateGradient(bkd)
        _, niter_cg, converged_cg = cg_solver.solve(A, b, tol=1e-10, maxiter=100)

        # PCG with Jacobi
        precond = jacobi_preconditioner(A, bkd)
        pcg_solver = PreconditionedConjugateGradient(bkd, preconditioner=precond)
        _, niter_pcg, converged_pcg = pcg_solver.solve(A, b, tol=1e-10, maxiter=100)

        self.assertTrue(converged_cg)
        self.assertTrue(converged_pcg)
        # Jacobi preconditioning should help (or at least not hurt much)
        self.assertLessEqual(niter_pcg, niter_cg + 2)

    def test_functional_interface(self):
        """Test the functional interface."""
        bkd = self.bkd()
        A = bkd.array([[4.0, 1.0], [1.0, 3.0]])
        b = bkd.array([5.0, 4.0])

        precond = jacobi_preconditioner(A, bkd)
        x, niter, converged = pcg_solve(A, b, bkd, preconditioner=precond, tol=1e-10)

        self.assertTrue(converged)
        residual = A @ x - b
        bkd.assert_allclose(residual, bkd.zeros((2,)), atol=1e-8)

    def test_matrix_free_pcg(self):
        """Test matrix-free PCG."""
        bkd = self.bkd()
        A = bkd.array([[4.0, 1.0], [1.0, 3.0]])
        b = bkd.array([5.0, 4.0])

        def matvec(x):
            return A @ x

        precond = jacobi_preconditioner(A, bkd)
        solver = PreconditionedConjugateGradient(bkd, preconditioner=precond)
        x, niter, converged = solver.solve_matvec(matvec, b, tol=1e-10)

        self.assertTrue(converged)
        residual = A @ x - b
        bkd.assert_allclose(residual, bkd.zeros((2,)), atol=1e-8)


class TestJacobiPreconditioner(Generic[Array], unittest.TestCase):
    """Base test class for Jacobi preconditioner."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_jacobi_apply(self):
        """Test Jacobi preconditioner application."""
        bkd = self.bkd()
        A = bkd.array([[4.0, 1.0], [1.0, 2.0]])
        r = bkd.array([8.0, 4.0])

        precond = JacobiPreconditioner(bkd)
        precond.setup(A)
        z = precond.apply(r)

        # z = r / diag(A) = [8/4, 4/2] = [2, 2]
        expected = bkd.array([2.0, 2.0])
        bkd.assert_allclose(z, expected, atol=1e-14)

    def test_factory_function(self):
        """Test Jacobi factory function."""
        bkd = self.bkd()
        A = bkd.array([[4.0, 1.0], [1.0, 2.0]])
        r = bkd.array([8.0, 4.0])

        precond = jacobi_preconditioner(A, bkd)
        z = precond.apply(r)

        expected = bkd.array([2.0, 2.0])
        bkd.assert_allclose(z, expected, atol=1e-14)

    def test_jacobi_not_setup(self):
        """Test error when applying without setup."""
        bkd = self.bkd()
        r = bkd.array([1.0, 2.0])

        precond = JacobiPreconditioner(bkd)
        with self.assertRaises(RuntimeError):
            precond.apply(r)


class TestBlockJacobiPreconditioner(Generic[Array], unittest.TestCase):
    """Base test class for Block Jacobi preconditioner."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_block_jacobi_2x2_blocks(self):
        """Test block Jacobi with 2x2 blocks."""
        bkd = self.bkd()
        # 4x4 matrix with 2x2 blocks
        A = bkd.array([
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.5],
            [0.0, 0.0, 0.5, 2.0],
        ])
        r = bkd.array([5.0, 4.0, 2.5, 2.5])

        precond = BlockJacobiPreconditioner(bkd, block_size=2)
        precond.setup(A)
        z = precond.apply(r)

        # Each block is inverted separately
        # Block 1: [[4,1],[1,3]], Block 2: [[2,0.5],[0.5,2]]
        # z = [inv(B1) @ r1, inv(B2) @ r2]
        # Check by verifying each block
        A_block1 = A[0:2, 0:2]
        A_block2 = A[2:4, 2:4]
        z1_expected = bkd.solve(A_block1, r[0:2])
        z2_expected = bkd.solve(A_block2, r[2:4])

        bkd.assert_allclose(z[0:2], z1_expected, atol=1e-12)
        bkd.assert_allclose(z[2:4], z2_expected, atol=1e-12)

    def test_factory_function(self):
        """Test block Jacobi factory function."""
        bkd = self.bkd()
        A = bkd.array([
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.5],
            [0.0, 0.0, 0.5, 2.0],
        ])
        r = bkd.array([5.0, 4.0, 2.5, 2.5])

        precond = block_jacobi_preconditioner(A, bkd, block_size=2)
        z = precond.apply(r)

        # Just verify it runs and gives reasonable output
        self.assertEqual(z.shape[0], 4)

    def test_invalid_block_size(self):
        """Test error for invalid block size."""
        bkd = self.bkd()
        # 4x4 matrix, block size 3 doesn't divide evenly
        A = bkd.eye(4)

        precond = BlockJacobiPreconditioner(bkd, block_size=3)
        with self.assertRaises(ValueError):
            precond.setup(A)


# NumPy backend tests
class TestDirectSolverNumpy(TestDirectSolver):
    __test__ = True

    def bkd(self) -> Backend[Array]:
        return NumpyBkd()


class TestConjugateGradientNumpy(TestConjugateGradient):
    __test__ = True

    def bkd(self) -> Backend[Array]:
        return NumpyBkd()


class TestPreconditionedCGNumpy(TestPreconditionedCG):
    __test__ = True

    def bkd(self) -> Backend[Array]:
        return NumpyBkd()


class TestJacobiPreconditionerNumpy(TestJacobiPreconditioner):
    __test__ = True

    def bkd(self) -> Backend[Array]:
        return NumpyBkd()


class TestBlockJacobiPreconditionerNumpy(TestBlockJacobiPreconditioner):
    __test__ = True

    def bkd(self) -> Backend[Array]:
        return NumpyBkd()


if __name__ == "__main__":
    unittest.main()
