import unittest
from typing import Type

import numpy as np

from pyapprox.util.backends.template import Array
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin
from pyapprox.optimization.newton import (
    NewtonResidual,
    BisectionSearch,
    BisectionResidual,
)


class TestNewton:
    def get_backend(self):
        raise NotImplementedError

    def setUp(self):
        np.random.seed(1)

    def test_newton_solve_for_polynomial_roots(self) -> None:
        bkd = self.get_backend()

        class PolynomialEquation(NewtonResidual):
            def nstates(self) -> int:
                return 1  # Single state variable

            def _value(self, iterate: Array) -> Array:
                # Residual function: f(x) = x^2 - 4 = 0
                return self._bkd.array([iterate[0] ** 2 - 4])

            def _jacobian(self, iterate: Array) -> Array:
                # Compute the Jacobian of f(x) = x^2 - 4 = 0
                return 2.0 * iterate[None, :]

        residual = PolynomialEquation(bkd)

        # Initial iterate: start near one of the roots (e.g., x = 2)
        init_iterate = bkd.array([1.0])

        # Solve the residual equation
        solution = residual.solve(init_iterate)
        bkd.assert_allclose(solution, bkd.array([2.0]))

    def test_newton_solve_coupled_parabola_circle_equations(self) -> None:
        bkd = self.get_backend()

        class ParabolaCircleEquations(NewtonResidual):
            def nstates(self) -> int:
                return 2  # Single state variable

            def _value(self, iterate: Array) -> Array:
                x, y = iterate
                return self._bkd.array([x**2 + y**2 - 1, x**2 - y])

            def _jacobian(self, iterate: Array) -> Array:
                x, y = iterate
                return self._bkd.array([[2 * x, 2 * y], [2 * x, -1]])

        residual = ParabolaCircleEquations(bkd)

        # Initial iterate: start near one of the roots (e.g., x = 2)
        true_solution = bkd.array(
            [np.sqrt((-1 + np.sqrt(5)) / 2), (-1 + np.sqrt(5)) / 2]
        )
        init_iterate = true_solution + 0.1

        # Solve the residual equation
        solution = residual.solve(init_iterate)
        bkd.assert_allclose(solution, true_solution)

    def test_bisection_search(self) -> None:
        bkd = self.get_backend()

        class Residual(BisectionResidual):
            def nstates(self) -> int:
                return 3

            def _value(self, iterate: Array) -> Array:
                self._rhs = self._bkd.array([0.1, 0.3, 0.6])
                return iterate**2 - self._rhs

        bisearch = BisectionSearch()
        residual = Residual(backend=bkd)
        bisearch.set_residual(residual)
        bounds = bkd.array([[0.0, 0.5], [0.1, 1], [0.5, 1.0]])
        roots = bisearch.solve(bounds)
        assert bkd.allclose(roots, bkd.sqrt(residual._rhs))


class TestNumpyNewton(TestNewton, unittest.TestCase):
    def get_backend(self) -> Type[NumpyMixin]:
        return NumpyMixin


class TestTorchNewton(TestNewton, unittest.TestCase):
    def get_backend(self) -> Type[TorchMixin]:
        return TorchMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
