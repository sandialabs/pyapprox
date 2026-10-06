"""Tests for Chebyshev derivative matrix."""


import numpy as np

from pyapprox.pde.collocation.basis.chebyshev.derivative import (
    ChebyshevDerivativeMatrix1D,
)
from pyapprox.pde.collocation.basis.chebyshev.nodes import (
    ChebyshevGaussLobattoNodes1D,
)


class TestChebyshevDerivative:
    """Base test class for Chebyshev derivative matrix."""

    def test_single_node(self, bkd):
        """Test single node case returns zero matrix."""
        nodes_gen = ChebyshevGaussLobattoNodes1D(bkd)
        deriv_comp = ChebyshevDerivativeMatrix1D(bkd)

        nodes = nodes_gen.generate(1)
        D = deriv_comp.compute(nodes)

        assert D.shape == (1, 1)
        bkd.assert_allclose(D, bkd.zeros((1, 1)), atol=1e-14)

    def test_matrix_shape(self, bkd):
        """Test derivative matrix has correct shape."""
        nodes_gen = ChebyshevGaussLobattoNodes1D(bkd)
        deriv_comp = ChebyshevDerivativeMatrix1D(bkd)

        for npts in [3, 5, 10]:
            nodes = nodes_gen.generate(npts)
            D = deriv_comp.compute(nodes)
            assert D.shape == (npts, npts)

    def test_row_sum_zero(self, bkd):
        """Test that rows sum to zero (constant function property)."""
        nodes_gen = ChebyshevGaussLobattoNodes1D(bkd)
        deriv_comp = ChebyshevDerivativeMatrix1D(bkd)

        for npts in [3, 5, 10]:
            nodes = nodes_gen.generate(npts)
            D = deriv_comp.compute(nodes)
            row_sums = bkd.sum(D, axis=1)
            bkd.assert_allclose(row_sums, bkd.zeros(npts), atol=1e-12)

    def test_constant_derivative(self, bkd):
        """Test derivative of constant is zero."""
        nodes_gen = ChebyshevGaussLobattoNodes1D(bkd)
        deriv_comp = ChebyshevDerivativeMatrix1D(bkd)

        npts = 10
        nodes = nodes_gen.generate(npts)
        D = deriv_comp.compute(nodes)

        # f(x) = c, f'(x) = 0
        f = bkd.ones(npts) * 5.0
        df = D @ f
        bkd.assert_allclose(df, bkd.zeros(npts), atol=1e-12)

    def test_linear_derivative(self, bkd):
        """Test derivative of x is 1."""
        nodes_gen = ChebyshevGaussLobattoNodes1D(bkd)
        deriv_comp = ChebyshevDerivativeMatrix1D(bkd)

        npts = 10
        nodes = nodes_gen.generate(npts)
        D = deriv_comp.compute(nodes)

        # f(x) = x, f'(x) = 1
        f = nodes
        df = D @ f
        bkd.assert_allclose(df, bkd.ones(npts), atol=1e-12)

    def test_quadratic_derivative(self, bkd):
        """Test derivative of x^2 is 2x."""
        nodes_gen = ChebyshevGaussLobattoNodes1D(bkd)
        deriv_comp = ChebyshevDerivativeMatrix1D(bkd)

        npts = 10
        nodes = nodes_gen.generate(npts)
        D = deriv_comp.compute(nodes)

        # f(x) = x^2, f'(x) = 2x
        f = nodes**2
        df = D @ f
        expected = 2.0 * nodes
        bkd.assert_allclose(df, expected, atol=1e-11)

    def test_cubic_derivative(self, bkd):
        """Test derivative of x^3 is 3x^2."""
        nodes_gen = ChebyshevGaussLobattoNodes1D(bkd)
        deriv_comp = ChebyshevDerivativeMatrix1D(bkd)

        npts = 10
        nodes = nodes_gen.generate(npts)
        D = deriv_comp.compute(nodes)

        # f(x) = x^3, f'(x) = 3x^2
        f = nodes**3
        df = D @ f
        expected = 3.0 * nodes**2
        bkd.assert_allclose(df, expected, atol=1e-10)

    def test_polynomial_exactness(self, bkd):
        """Test exact differentiation for polynomials up to degree n-1."""
        nodes_gen = ChebyshevGaussLobattoNodes1D(bkd)
        deriv_comp = ChebyshevDerivativeMatrix1D(bkd)

        npts = 8
        nodes = nodes_gen.generate(npts)
        D = deriv_comp.compute(nodes)

        # For degree d polynomial (d < n), derivative should be exact
        for degree in range(npts):
            # f(x) = x^degree
            f = nodes**degree
            df_computed = D @ f

            # f'(x) = degree * x^(degree-1) for degree > 0
            if degree == 0:
                df_expected = bkd.zeros(npts)
            else:
                df_expected = degree * nodes ** (degree - 1)

            bkd.assert_allclose(df_computed, df_expected, atol=1e-9)

    def test_second_derivative(self, bkd):
        """Test second derivative via D @ D."""
        nodes_gen = ChebyshevGaussLobattoNodes1D(bkd)
        deriv_comp = ChebyshevDerivativeMatrix1D(bkd)

        npts = 10
        nodes = nodes_gen.generate(npts)
        D = deriv_comp.compute(nodes)
        D2 = D @ D

        # f(x) = x^3, f''(x) = 6x
        f = nodes**3
        d2f = D2 @ f
        expected = 6.0 * nodes
        bkd.assert_allclose(d2f, expected, atol=1e-9)

    def test_sin_derivative(self, bkd):
        """Test derivative of sin(pi*x) converges to pi*cos(pi*x)."""
        nodes_gen = ChebyshevGaussLobattoNodes1D(bkd)
        deriv_comp = ChebyshevDerivativeMatrix1D(bkd)

        # Test convergence - error should decrease with more points
        errors = []
        npts_list = [10, 20, 40]

        for npts in npts_list:
            nodes = nodes_gen.generate(npts)
            D = deriv_comp.compute(nodes)

            # f(x) = sin(pi*x), f'(x) = pi*cos(pi*x)
            f = bkd.sin(np.pi * nodes)
            df = D @ f
            expected = np.pi * bkd.cos(np.pi * nodes)
            error = float(bkd.max(bkd.abs(df - expected)))
            errors.append(error)

        # Check convergence (errors should generally decrease until machine precision)
        # At machine precision (~1e-13), errors may fluctuate
        # Check that later errors are much smaller than earlier ones
        assert errors[-1] < errors[0] * 0.01

        # Check final accuracy is very good (near machine precision)
        assert errors[-1] < 1e-10


class TestChebyshevDerivativeNumpy(TestChebyshevDerivative):
    """NumPy backend tests for Chebyshev derivative matrix."""
