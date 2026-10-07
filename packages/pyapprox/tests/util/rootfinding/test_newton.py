import numpy as np
import pytest

from pyapprox.util.backends.protocols import Array, Backend
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


class _ScaledQuadratic:
    """``R(x) = scale (x^2 - 2)``: the root sqrt(2) is not representable,
    so with ``scale = 1e12`` the best residual is 1e12 * 4.4e-16 =
    4.4e-4, far above any small absolute tolerance."""

    def __init__(self, bkd: Backend[Array], scale: float) -> None:
        self._bkd = bkd
        self._scale = scale

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def __call__(self, iterate: Array, /) -> Array:
        return self._scale * (iterate**2 - 2.0)

    def linsolve(self, state: Array, prev_residual: Array) -> Array:
        return prev_residual / (self._scale * 2.0 * state)

    def is_one_step_solvable(self) -> bool:
        return False


class _Arctan:
    """``R(x) = arctan(x)``: full Newton steps diverge from |x0| > 1.39."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def __call__(self, iterate: Array, /) -> Array:
        return self._bkd.arctan(iterate)

    def linsolve(self, state: Array, prev_residual: Array) -> Array:
        return prev_residual * (1.0 + state**2)

    def is_one_step_solvable(self) -> bool:
        return False


class _Affine:
    """``R(x) = A x - b``, declared one-step solvable."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        self._A = bkd.asarray([[4.0, 1.0], [1.0, 3.0]])
        self._b = bkd.asarray([1.0, 2.0])

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def __call__(self, iterate: Array, /) -> Array:
        return self._A @ iterate - self._b

    def linsolve(self, state: Array, prev_residual: Array) -> Array:
        return self._bkd.solve(self._A, prev_residual)

    def is_one_step_solvable(self) -> bool:
        return True


class TestNewtonStoppingAndSteps:
    def test_rtol_converges_where_atol_cannot(
        self, bkd: Backend[Array]
    ) -> None:
        """With the residual scaled by 1e12, its round-off (4.4e-4)
        exceeds atol = 1e-10: only the relative test can be met."""
        solver = NewtonSolver(_ScaledQuadratic(bkd, 1e12))
        solver.set_options(maxiters=30, atol=1e-10, rtol=0.0)
        result = solver.solve_with_info(bkd.asarray([3.0]))
        assert not result.converged
        assert result.iterations == 30
        with pytest.raises(RuntimeError, match="Max iterations"):
            solver.solve(bkd.asarray([3.0]))

        solver.set_options(maxiters=30, atol=1e-10, rtol=1e-12)
        result = solver.solve_with_info(bkd.asarray([3.0]))
        assert result.converged
        bkd.assert_allclose(
            result.solution, bkd.asarray([np.sqrt(2.0)]), rtol=1e-12
        )

    def test_line_search_is_opt_in(self, bkd: Backend[Array]) -> None:
        """From x0 = 1.5 full Newton steps on arctan diverge; the line
        search converges to the root 0."""
        solver = NewtonSolver(_Arctan(bkd))
        solver.set_options(maxiters=20, atol=1e-12, rtol=0.0)
        assert not solver.solve_with_info(bkd.asarray([1.5])).converged

        solver.set_options(
            maxiters=20, atol=1e-12, rtol=0.0, line_search=True
        )
        result = solver.solve_with_info(bkd.asarray([1.5]))
        assert result.converged
        bkd.assert_allclose(
            result.solution, bkd.asarray([0.0]), atol=1e-12
        )

    def test_one_step_path(self, bkd: Backend[Array]) -> None:
        """A residual declared one-step solvable takes exactly one step,
        exact up to round-off, without re-evaluating the residual."""
        residual = _Affine(bkd)
        result = NewtonSolver(residual).solve_with_info(bkd.zeros((2,)))
        assert result.converged
        assert result.iterations == 1
        assert result.residual_norm is None
        bkd.assert_allclose(
            residual(result.solution), bkd.zeros((2,)), atol=1e-14
        )
