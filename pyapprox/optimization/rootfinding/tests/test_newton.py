import unittest
from typing import Generic
import numpy as np
import torch
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.optimization.rootfinding.newton import (
    NewtonSolver,
    NewtonSolverResidualProtocol,
)


class TestNewtonSolver(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        """
        Override this method in derived classes to provide the specific
        backend.
        """
        raise NotImplementedError(
            "Derived classes must implement this method."
        )

    def test_newton_solve_for_polynomial_roots(self) -> None:
        """
        Test the Newton solver for finding polynomial roots.
        """
        bkd = self.bkd()

        # Define the residual class
        class PolynomialEquation(NewtonSolverResidualProtocol[Array]):
            def __init__(self, backend: Backend[Array]) -> None:
                self._bkd = backend

            def bkd(self) -> Backend[Array]:
                return self._bkd

            def __call__(self, iterate: Array) -> Array:
                # Residual function: f(x) = x^2 - 4 = 0
                return self._bkd.array([iterate[0] ** 2 - 4])

            def linsolve(self, sol: Array, prev_residual: Array) -> Array:
                # Linear solve: delta = residual / jacobian
                jacobian = self._bkd.array([[2.0 * sol[0]]])
                return self._bkd.solve(jacobian, prev_residual)

        residual = PolynomialEquation(bkd)
        solver = NewtonSolver(residual)

        # Initial iterate: start near one of the roots (e.g., x = 2)
        init_iterate = bkd.array([1.0])

        # Solve the residual equation
        solution = solver.solve(init_iterate)

        # Assert that the solution matches the expected root
        bkd.assert_allclose(solution, bkd.array([2.0]))

    def test_newton_solve_coupled_parabola_circle_equations(self) -> None:
        """
        Test the Newton solver for solving coupled parabola-circle equations.
        """
        bkd = self.bkd()

        # Define the residual class
        class ParabolaCircleEquations(NewtonSolverResidualProtocol[Array]):
            def __init__(self, backend: Backend[Array]) -> None:
                self._bkd = backend

            def bkd(self) -> Backend[Array]:
                return self._bkd

            def __call__(self, iterate: Array) -> Array:
                x, y = iterate
                # Residual functions:
                # f1(x, y) = x^2 + y^2 - 1, f2(x, y) = x^2 - y
                return self._bkd.array([x**2 + y**2 - 1, x**2 - y])

            def linsolve(self, sol: Array, prev_residual: Array) -> Array:
                x, y = sol
                # Jacobian matrix
                jacobian = self._bkd.array([[2 * x, 2 * y], [2 * x, -1]])
                return self._bkd.solve(jacobian, prev_residual)

        residual = ParabolaCircleEquations(bkd)
        solver = NewtonSolver(residual)

        # True solution
        true_solution = bkd.array(
            [np.sqrt((-1 + np.sqrt(5)) / 2), (-1 + np.sqrt(5)) / 2]
        )

        # Initial iterate: start near the true solution
        init_iterate = true_solution + 0.1

        # Solve the residual equation
        solution = solver.solve(init_iterate)

        # Assert that the solution matches the expected values
        bkd.assert_allclose(solution, true_solution)


# Derived test class for NumPy backend
class TestNewtonSolverNumpy(TestNewtonSolver[Array]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# Derived test class for PyTorch backend
class TestNewtonSolverTorch(TestNewtonSolver[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


# Custom test loader to exclude the base class
def load_tests(
    loader: unittest.TestLoader, tests, pattern: str
) -> unittest.TestSuite:
    """
    Custom test loader to exclude the base class
    ContinuousScipyRandomVariable1D.
    """
    test_suite = unittest.TestSuite()
    for test_class in [
        TestNewtonSolverNumpy,
        TestNewtonSolverTorch,
    ]:
        test_suite.addTests(loader.loadTestsFromTestCase(test_class))
    return test_suite


# Main block to explicitly run tests using the custom loader
if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = load_tests(loader, [], None)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
