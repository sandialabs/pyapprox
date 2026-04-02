import numpy as np

from pyapprox.util.backends.protocols import Backend
from pyapprox.util.rootfinding.newton import (
    NewtonSolver,
    NewtonSolverResidualProtocol,
)


class TestNewtonSolver:

    def test_newton_solve_for_polynomial_roots(self, bkd) -> None:
        """
        Test the Newton solver for finding polynomial roots.
        """
        # Define the residual class
        class PolynomialEquation(NewtonSolverResidualProtocol):
            def __init__(self, backend: Backend) -> None:
                self._bkd = backend

            def bkd(self) -> Backend:
                return self._bkd

            def __call__(self, iterate):
                # Residual function: f(x) = x^2 - 4 = 0
                return self._bkd.array([iterate[0] ** 2 - 4])

            def linsolve(self, sol, prev_residual):
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

    def test_newton_solve_coupled_parabola_circle_equations(self, bkd) -> None:
        """
        Test the Newton solver for solving coupled parabola-circle equations.
        """
        # Define the residual class
        class ParabolaCircleEquations(NewtonSolverResidualProtocol):
            def __init__(self, backend: Backend) -> None:
                self._bkd = backend

            def bkd(self) -> Backend:
                return self._bkd

            def __call__(self, iterate):
                x, y = iterate
                # Residual functions:
                # f1(x, y) = x^2 + y^2 - 1, f2(x, y) = x^2 - y
                return self._bkd.array([x**2 + y**2 - 1, x**2 - y])

            def linsolve(self, sol, prev_residual):
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
