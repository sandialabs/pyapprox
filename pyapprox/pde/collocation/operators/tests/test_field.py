"""Tests for field classes."""

import unittest
from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401
from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.mesh import (
    TransformedMesh1D,
)
from pyapprox.pde.collocation.operators import (
    Field,
    scalar_field,
    input_field,
    constant_field,
    zero_field,
)


class TestField(Generic[Array], unittest.TestCase):
    """Base test class for unified Field."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_field_scalar_creation(self):
        """Test scalar field creation with shape (1, npts)."""
        bkd = self.bkd()
        mesh = TransformedMesh1D(5, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        npts = basis.npts()

        values = bkd.reshape(bkd.ones((npts,)), (1, npts))
        field = Field(basis, bkd, ncomponents=1, values=values)
        self.assertEqual(field.ncomponents(), 1)
        self.assertTrue(field.is_scalar)
        self.assertEqual(field.as_flat().shape, (npts,))

    def test_field_vector_creation(self):
        """Test vector field creation with shape (ncomponents, npts)."""
        bkd = self.bkd()
        mesh = TransformedMesh1D(5, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        npts = basis.npts()

        values = bkd.ones((2, npts))
        field = Field(basis, bkd, ncomponents=2, values=values)
        self.assertEqual(field.ncomponents(), 2)
        self.assertFalse(field.is_scalar)
        self.assertEqual(field.as_flat().shape, (2 * npts,))

    def test_field_component_extraction(self):
        """Test extracting a component from a vector field."""
        bkd = self.bkd()
        mesh = TransformedMesh1D(5, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        npts = basis.npts()

        values = bkd.zeros((2, npts))
        values = bkd.copy(values)
        for i in range(npts):
            values[0, i] = 1.0
            values[1, i] = 2.0
        field = Field(basis, bkd, ncomponents=2, values=values)

        comp0 = field.component(0)
        comp1 = field.component(1)

        bkd.assert_allclose(comp0.as_flat(), bkd.ones((npts,)) * 1.0, atol=1e-14)
        bkd.assert_allclose(comp1.as_flat(), bkd.ones((npts,)) * 2.0, atol=1e-14)


class TestScalarFieldFactories(Generic[Array], unittest.TestCase):
    """Base test class for scalar field factory functions."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_scalar_field_creation(self):
        """Test scalar_field factory function."""
        bkd = self.bkd()
        mesh = TransformedMesh1D(5, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        npts = basis.npts()

        values = bkd.ones((npts,))
        field = scalar_field(basis, bkd, values)

        self.assertEqual(field.npts(), npts)
        self.assertEqual(field.ncomponents(), 1)
        self.assertTrue(field.is_scalar)
        bkd.assert_allclose(field.as_flat(), values, atol=1e-14)

    def test_constant_field(self):
        """Test constant_field factory function."""
        bkd = self.bkd()
        mesh = TransformedMesh1D(5, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        npts = basis.npts()

        field = constant_field(basis, bkd, 3.0)

        expected = bkd.full((npts,), 3.0)
        bkd.assert_allclose(field.as_flat(), expected, atol=1e-14)

    def test_zero_field(self):
        """Test zero_field factory function."""
        bkd = self.bkd()
        mesh = TransformedMesh1D(5, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        npts = basis.npts()

        field = zero_field(basis, bkd)

        bkd.assert_allclose(field.as_flat(), bkd.zeros((npts,)), atol=1e-14)

    def test_input_field_jacobian(self):
        """Test input_field creates correct diagonal Jacobian."""
        bkd = self.bkd()
        mesh = TransformedMesh1D(5, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        npts = basis.npts()

        values = bkd.linspace(0.0, 1.0, npts)
        field = input_field(basis, bkd, values, input_index=0, ninput_funs=1)

        jac = field.get_jacobian()
        expected = bkd.eye(npts)
        bkd.assert_allclose(jac, expected, atol=1e-14)

    def test_input_field_multi_input(self):
        """Test input_field with multiple inputs."""
        bkd = self.bkd()
        mesh = TransformedMesh1D(5, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        npts = basis.npts()

        values = bkd.linspace(0.0, 1.0, npts)
        field = input_field(basis, bkd, values, input_index=1, ninput_funs=2)

        jac = field.get_jacobian()
        self.assertEqual(jac.shape, (npts, 2 * npts))

        expected = bkd.zeros((npts, 2 * npts))
        for i in range(npts):
            expected[i, npts + i] = 1.0
        bkd.assert_allclose(jac, expected, atol=1e-14)


class TestFieldArithmetic(Generic[Array], unittest.TestCase):
    """Base test class for Field arithmetic operations."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_addition(self):
        """Test field addition."""
        bkd = self.bkd()
        mesh = TransformedMesh1D(5, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        npts = basis.npts()

        f1 = input_field(basis, bkd, bkd.ones((npts,)) * 2.0)
        f2 = input_field(basis, bkd, bkd.ones((npts,)) * 3.0)

        result = f1 + f2

        bkd.assert_allclose(result.as_flat(), bkd.ones((npts,)) * 5.0, atol=1e-14)

    def test_addition_scalar(self):
        """Test field + scalar."""
        bkd = self.bkd()
        mesh = TransformedMesh1D(5, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        npts = basis.npts()

        f = input_field(basis, bkd, bkd.ones((npts,)) * 2.0)
        result = f + 3.0

        bkd.assert_allclose(result.as_flat(), bkd.ones((npts,)) * 5.0, atol=1e-14)

    def test_subtraction(self):
        """Test field subtraction."""
        bkd = self.bkd()
        mesh = TransformedMesh1D(5, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        npts = basis.npts()

        f1 = input_field(basis, bkd, bkd.ones((npts,)) * 5.0)
        f2 = input_field(basis, bkd, bkd.ones((npts,)) * 3.0)

        result = f1 - f2

        bkd.assert_allclose(result.as_flat(), bkd.ones((npts,)) * 2.0, atol=1e-14)

    def test_multiplication(self):
        """Test field multiplication with product rule."""
        bkd = self.bkd()
        mesh = TransformedMesh1D(5, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        npts = basis.npts()

        f1 = input_field(basis, bkd, bkd.ones((npts,)) * 2.0)
        f2 = input_field(basis, bkd, bkd.ones((npts,)) * 3.0)

        result = f1 * f2

        bkd.assert_allclose(result.as_flat(), bkd.ones((npts,)) * 6.0, atol=1e-14)

    def test_multiplication_scalar(self):
        """Test field * scalar."""
        bkd = self.bkd()
        mesh = TransformedMesh1D(5, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        npts = basis.npts()

        f = input_field(basis, bkd, bkd.ones((npts,)) * 2.0)
        result = f * 3.0

        bkd.assert_allclose(result.as_flat(), bkd.ones((npts,)) * 6.0, atol=1e-14)

    def test_division(self):
        """Test field division."""
        bkd = self.bkd()
        mesh = TransformedMesh1D(5, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        npts = basis.npts()

        f1 = input_field(basis, bkd, bkd.ones((npts,)) * 6.0)
        f2 = input_field(basis, bkd, bkd.ones((npts,)) * 2.0)

        result = f1 / f2

        bkd.assert_allclose(result.as_flat(), bkd.ones((npts,)) * 3.0, atol=1e-14)

    def test_power(self):
        """Test field power."""
        bkd = self.bkd()
        mesh = TransformedMesh1D(5, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        npts = basis.npts()

        f = input_field(basis, bkd, bkd.ones((npts,)) * 2.0)
        result = f ** 3

        bkd.assert_allclose(result.as_flat(), bkd.ones((npts,)) * 8.0, atol=1e-14)

    def test_negation(self):
        """Test field negation."""
        bkd = self.bkd()
        mesh = TransformedMesh1D(5, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        npts = basis.npts()

        f = input_field(basis, bkd, bkd.ones((npts,)) * 3.0)
        result = -f

        bkd.assert_allclose(result.as_flat(), bkd.ones((npts,)) * -3.0, atol=1e-14)


class TestFieldJacobian(Generic[Array], unittest.TestCase):
    """Base test class for Field Jacobian tracking."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_addition_jacobian(self):
        """Test Jacobian of f + g = df + dg."""
        bkd = self.bkd()
        mesh = TransformedMesh1D(5, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        npts = basis.npts()

        f = input_field(basis, bkd, bkd.linspace(1.0, 2.0, npts))
        g = input_field(basis, bkd, bkd.linspace(2.0, 3.0, npts))

        result = f + g
        jac = result.get_jacobian()

        expected = 2.0 * bkd.eye(npts)
        bkd.assert_allclose(jac, expected, atol=1e-14)

    def test_multiplication_jacobian(self):
        """Test Jacobian of f * g = g*df + f*dg (product rule)."""
        bkd = self.bkd()
        mesh = TransformedMesh1D(5, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        npts = basis.npts()

        f = input_field(basis, bkd, bkd.ones((npts,)) * 2.0)
        g = input_field(basis, bkd, bkd.ones((npts,)) * 3.0)

        result = f * g
        jac = result.get_jacobian()

        expected = 5.0 * bkd.eye(npts)
        bkd.assert_allclose(jac, expected, atol=1e-14)

    def test_power_jacobian(self):
        """Test Jacobian of f^n = n * f^(n-1) * df."""
        bkd = self.bkd()
        mesh = TransformedMesh1D(5, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        npts = basis.npts()

        f = input_field(basis, bkd, bkd.ones((npts,)) * 2.0)
        result = f ** 3

        jac = result.get_jacobian()

        expected = 12.0 * bkd.eye(npts)
        bkd.assert_allclose(jac, expected, atol=1e-14)


class TestFieldNumpy(TestField):
    """NumPy backend tests for Field."""

    __test__ = True

    def bkd(self) -> Backend[Array]:
        return NumpyBkd()


class TestScalarFieldFactoriesNumpy(TestScalarFieldFactories):
    """NumPy backend tests for scalar field factories."""

    __test__ = True

    def bkd(self) -> Backend[Array]:
        return NumpyBkd()


class TestFieldArithmeticNumpy(TestFieldArithmetic):
    """NumPy backend tests for Field arithmetic."""

    __test__ = True

    def bkd(self) -> Backend[Array]:
        return NumpyBkd()


class TestFieldJacobianNumpy(TestFieldJacobian):
    """NumPy backend tests for Field Jacobian tracking."""

    __test__ = True

    def bkd(self) -> Backend[Array]:
        return NumpyBkd()


if __name__ == "__main__":
    unittest.main()
