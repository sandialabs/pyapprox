import unittest
from typing import Generic, Any

import torch
import numpy as np
from numpy.typing import NDArray

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.surrogates.basis.piecewisepoly import (
    PiecewiseConstantLeft,
    PiecewiseConstantMidpoint,
    PiecewiseConstantRight,
    PiecewiseLinear,
    PiecewiseQuadratic,
    PiecewiseCubic,
)


class TestPiecewisePolynomialBasis(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        """
        Override this method in derived classes to provide the specific
        backend.
        """
        raise NotImplementedError(
            "Derived classes must implement this method."
        )

    def test_piecewise_linear(self) -> None:
        """
        Test that the piecewise linear basis interpolates a linear function
        and the quadrature rule exactly integrates the linear function.
        """
        nodes = self.bkd().asarray([0.0, 0.5, 1.0])
        xx = self.bkd().linspace(0.0, 1.0, 100)
        linear_function = lambda x: x

        # Create PiecewiseLinear instance
        piecewise_linear = PiecewiseLinear(nodes, self.bkd())

        # Evaluate basis functions
        basis_values = piecewise_linear(xx)
        interpolated_values = basis_values @ linear_function(nodes)

        # Check interpolation
        expected_values = linear_function(xx)
        self.bkd().assert_allclose(
            interpolated_values, expected_values, atol=1e-7
        )

        # Check quadrature rule
        quadrature_points, quadrature_weights = (
            piecewise_linear.quadrature_rule()
        )
        integral = quadrature_weights @ linear_function(quadrature_points)
        expected_integral = 0.5  # Integral of f(x) = x over [0, 1]
        self.assertAlmostEqual(integral, expected_integral, places=7)

    def test_piecewise_quadratic(self) -> None:
        """
        Test that the piecewise quadratic basis interpolates a quadratic function
        and the quadrature rule exactly integrates the quadratic function.
        """
        nodes = self.bkd().asarray([0.0, 0.5, 1.0])
        xx = self.bkd().linspace(0.0, 1.0, 100)
        quadratic_function = lambda x: x**2

        # Create PiecewiseQuadratic instance
        piecewise_quadratic = PiecewiseQuadratic(nodes, self.bkd())

        # Evaluate basis functions
        basis_values = piecewise_quadratic(xx)
        interpolated_values = basis_values @ quadratic_function(nodes)

        # Check interpolation
        expected_values = quadratic_function(xx)
        self.bkd().assert_allclose(
            interpolated_values, expected_values, atol=1e-7
        )

        # Check quadrature rule
        quadrature_points, quadrature_weights = (
            piecewise_quadratic.quadrature_rule()
        )
        integral = quadrature_weights @ quadratic_function(quadrature_points)
        expected_integral = 1 / 3  # Integral of f(x) = x^2 over [0, 1]
        self.assertAlmostEqual(integral, expected_integral, places=7)

    def test_piecewise_cubic(self) -> None:
        """
        Test that the piecewise cubic basis interpolates a cubic function
        and the quadrature rule exactly integrates the cubic function.
        """
        nodes = self.bkd().asarray([0.0, 0.25, 0.5, 1.0])
        xx = self.bkd().linspace(0.0, 1.0, 100)
        cubic_function = lambda x: x**3

        # Create PiecewiseCubic instance
        piecewise_cubic = PiecewiseCubic(nodes, self.bkd())

        # Evaluate basis functions
        basis_values = piecewise_cubic(xx)
        interpolated_values = basis_values @ cubic_function(nodes)

        # Check interpolation
        expected_values = cubic_function(xx)
        self.bkd().assert_allclose(
            interpolated_values, expected_values, atol=1e-7
        )

        # Check quadrature rule
        quadrature_points, quadrature_weights = (
            piecewise_cubic.quadrature_rule()
        )
        integral = quadrature_weights @ cubic_function(quadrature_points)
        expected_integral = 0.25  # Integral of f(x) = x^3 over [0, 1]
        self.assertAlmostEqual(integral, expected_integral, places=7)


class TestPiecewisePolynomialConvergence(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        """
        Override this method in derived classes to provide the specific
        backend.
        """
        raise NotImplementedError(
            "Derived classes must implement this method."
        )

    def setUp(self) -> None:
        self.target_function = lambda x: x**4  # Polynomial degree 4
        self.interval = (0.0, 1.0)  # Interval [0, 1]

    def compute_error(self, basis_class, nodes: Array, xx: Array) -> float:
        """
        Compute the interpolation error for a given basis class and nodes.

        Parameters
        ----------
        basis_class : class
            The basis class to test (e.g., PiecewiseLinear, PiecewiseQuadratic).
        nodes : Array
            The nodes used to define the basis functions.
        xx : Array
            The evaluation points.

        Returns
        -------
        float
            The interpolation error.
        """
        basis = basis_class(nodes, self.bkd())
        quad_points = basis.quadrature_rule()[0]
        basis_values = basis(xx)
        interpolated_values = basis_values @ self.target_function(quad_points)
        exact_values = self.target_function(xx)
        error = self.bkd().norm(interpolated_values - exact_values)
        return error.item()

    def test_convergence_piecewise_linear(self) -> None:
        """
        Test convergence of piecewise linear basis functions to the target
        polynomial function of degree 4.
        """
        errors = []
        node_counts = [5, 10, 20, 40]  # Increasing number of nodes
        for n in node_counts:
            nodes = self.bkd().linspace(self.interval[0], self.interval[1], n)
            xx = self.bkd().linspace(self.interval[0], self.interval[1], 1000)
            error = self.compute_error(PiecewiseLinear, nodes, xx)
            errors.append(error)

        # Check convergence rate
        for i in range(1, len(errors)):
            rate = errors[i - 1] / errors[i]
            self.assertGreater(
                rate,
                2.0,
                "PiecewiseLinear does not converge at the expected rate.",
            )

    def test_convergence_piecewise_quadratic(self) -> None:
        """
        Test convergence of piecewise quadratic basis functions to the target
        polynomial function of degree 4.
        """
        errors = []
        node_counts = [5, 11, 21, 41]  # Increasing number of nodes
        for n in node_counts:
            nodes = self.bkd().linspace(self.interval[0], self.interval[1], n)
            xx = self.bkd().linspace(self.interval[0], self.interval[1], 1000)
            error = self.compute_error(PiecewiseQuadratic, nodes, xx)
            errors.append(error)

        # Check convergence rate
        for i in range(1, len(errors)):
            rate = errors[i - 1] / errors[i]
            self.assertGreater(
                rate,
                4.0,
                "PiecewiseQuadratic does not converge at the expected rate.",
            )

    def test_convergence_piecewise_cubic(self) -> None:
        """
        Test convergence of piecewise cubic basis functions to the target
        polynomial function of degree 4.
        """
        errors = []
        node_counts = [7, 13, 25, 46]  # Increasing number of nodes
        for n in node_counts:
            nodes = self.bkd().linspace(self.interval[0], self.interval[1], n)
            xx = self.bkd().linspace(self.interval[0], self.interval[1], 1000)
            error = self.compute_error(PiecewiseCubic, nodes, xx)
            errors.append(error)

        # Check convergence rate
        for i in range(1, len(errors)):
            rate = errors[i - 1] / errors[i]
            self.assertGreater(
                rate,
                8.0,
                "PiecewiseCubic does not converge at the expected rate.",
            )

    def test_convergence_piecewise_constant_left(self) -> None:
        """
        Test convergence of piecewise left constant basis functions to the target
        polynomial function of degree 4.
        """
        errors = []
        node_counts = [5, 10, 20, 40]  # Increasing number of nodes
        for n in node_counts:
            nodes = self.bkd().linspace(self.interval[0], self.interval[1], n)
            xx = self.bkd().linspace(self.interval[0], self.interval[1], 1000)
            error = self.compute_error(PiecewiseConstantLeft, nodes, xx)
            errors.append(error)

        # Check convergence rate
        for i in range(1, len(errors)):
            rate = errors[i - 1] / errors[i]
            self.assertGreater(
                rate,
                1.0,
                "PiecewiseConstantLeft does not converge at the expected rate.",
            )

    def test_convergence_piecewise_constant_right(self) -> None:
        """
        Test convergence of piecewise right constant basis functions to the target
        polynomial function of degree 4.
        """
        errors = []
        node_counts = [5, 10, 20, 40]  # Increasing number of nodes
        for n in node_counts:
            nodes = self.bkd().linspace(self.interval[0], self.interval[1], n)
            xx = self.bkd().linspace(self.interval[0], self.interval[1], 1000)
            error = self.compute_error(PiecewiseConstantRight, nodes, xx)
            errors.append(error)

        # Check convergence rate
        for i in range(1, len(errors)):
            rate = errors[i - 1] / errors[i]
            self.assertGreater(
                rate,
                1.0,
                "PiecewiseConstantRight does not converge at the expected rate.",
            )

    def test_convergence_piecewise_constant_midpoint(self) -> None:
        """
        Test convergence of piecewise midpoint constant basis functions to the target
        polynomial function of degree 4.
        """
        errors = []
        node_counts = [5, 10, 20, 40]  # Increasing number of nodes
        for n in node_counts:
            nodes = self.bkd().linspace(self.interval[0], self.interval[1], n)
            xx = self.bkd().linspace(self.interval[0], self.interval[1], 1000)
            error = self.compute_error(PiecewiseConstantMidpoint, nodes, xx)
            errors.append(error)

        # Check convergence rate
        for i in range(1, len(errors)):
            rate = errors[i - 1] / errors[i]
            self.assertGreater(
                rate,
                1.0,
                "PiecewiseConstantMidpoint does not converge at the expected rate.",
            )


# Derived test class for NumPy backend
class TestPiecewisePolynomialBasisNumpy(
    TestPiecewisePolynomialBasis, unittest.TestCase
):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# Derived test class for PyTorch backend
class TestPiecewisePolynomialBasisTorch(
    TestPiecewisePolynomialBasis, unittest.TestCase
):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


# Derived test class for NumPy backend
class TestPiecewisePolynomialConvergenceNumpy(
    TestPiecewisePolynomialConvergence
):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# Derived test class for PyTorch backend
class TestPiecewisePolynomialConvergenceTorch(
    TestPiecewisePolynomialConvergence
):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


from pyapprox.typing.util.test_utils import load_tests


# Main block to explicitly run tests using the custom loader
if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = load_tests(loader, [], None)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
