"""Tests for field classes."""


from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.mesh import (
    TransformedMesh1D,
)
from pyapprox.pde.collocation.operators import (
    Field,
    constant_field,
    input_field,
    scalar_field,
    zero_field,
)


class TestField:
    """Base test class for unified Field."""

    def test_field_scalar_creation(self, bkd):
        """Test scalar field creation with shape (1, npts)."""
        mesh = TransformedMesh1D(5, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        npts = basis.npts()

        values = bkd.reshape(bkd.ones((npts,)), (1, npts))
        field = Field(basis, bkd, ncomponents=1, values=values)
        assert field.ncomponents() == 1
        assert field.is_scalar
        assert field.as_flat().shape == (npts,)

    def test_field_vector_creation(self, bkd):
        """Test vector field creation with shape (ncomponents, npts)."""
        mesh = TransformedMesh1D(5, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        npts = basis.npts()

        values = bkd.ones((2, npts))
        field = Field(basis, bkd, ncomponents=2, values=values)
        assert field.ncomponents() == 2
        assert not field.is_scalar
        assert field.as_flat().shape == (2 * npts,)

    def test_field_component_extraction(self, bkd):
        """Test extracting a component from a vector field."""
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


class TestScalarFieldFactories:
    """Base test class for scalar field factory functions."""

    def test_scalar_field_creation(self, bkd):
        """Test scalar_field factory function."""
        mesh = TransformedMesh1D(5, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        npts = basis.npts()

        values = bkd.ones((npts,))
        field = scalar_field(basis, bkd, values)

        assert field.npts() == npts
        assert field.ncomponents() == 1
        assert field.is_scalar
        bkd.assert_allclose(field.as_flat(), values, atol=1e-14)

    def test_constant_field(self, bkd):
        """Test constant_field factory function."""
        mesh = TransformedMesh1D(5, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        npts = basis.npts()

        field = constant_field(basis, bkd, 3.0)

        expected = bkd.full((npts,), 3.0)
        bkd.assert_allclose(field.as_flat(), expected, atol=1e-14)

    def test_zero_field(self, bkd):
        """Test zero_field factory function."""
        mesh = TransformedMesh1D(5, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        npts = basis.npts()

        field = zero_field(basis, bkd)

        bkd.assert_allclose(field.as_flat(), bkd.zeros((npts,)), atol=1e-14)

    def test_input_field_jacobian(self, bkd):
        """Test input_field creates correct diagonal Jacobian."""
        mesh = TransformedMesh1D(5, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        npts = basis.npts()

        values = bkd.linspace(0.0, 1.0, npts)
        field = input_field(basis, bkd, values, input_index=0, ninput_funs=1)

        jac = field.get_jacobian()
        expected = bkd.eye(npts)
        bkd.assert_allclose(jac, expected, atol=1e-14)

    def test_input_field_multi_input(self, bkd):
        """Test input_field with multiple inputs."""
        mesh = TransformedMesh1D(5, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        npts = basis.npts()

        values = bkd.linspace(0.0, 1.0, npts)
        field = input_field(basis, bkd, values, input_index=1, ninput_funs=2)

        jac = field.get_jacobian()
        assert jac.shape == (npts, 2 * npts)

        expected = bkd.zeros((npts, 2 * npts))
        for i in range(npts):
            expected[i, npts + i] = 1.0
        bkd.assert_allclose(jac, expected, atol=1e-14)


class TestFieldArithmetic:
    """Base test class for Field arithmetic operations."""

    def test_addition(self, bkd):
        """Test field addition."""
        mesh = TransformedMesh1D(5, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        npts = basis.npts()

        f1 = input_field(basis, bkd, bkd.ones((npts,)) * 2.0)
        f2 = input_field(basis, bkd, bkd.ones((npts,)) * 3.0)

        result = f1 + f2

        bkd.assert_allclose(result.as_flat(), bkd.ones((npts,)) * 5.0, atol=1e-14)

    def test_addition_scalar(self, bkd):
        """Test field + scalar."""
        mesh = TransformedMesh1D(5, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        npts = basis.npts()

        f = input_field(basis, bkd, bkd.ones((npts,)) * 2.0)
        result = f + 3.0

        bkd.assert_allclose(result.as_flat(), bkd.ones((npts,)) * 5.0, atol=1e-14)

    def test_subtraction(self, bkd):
        """Test field subtraction."""
        mesh = TransformedMesh1D(5, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        npts = basis.npts()

        f1 = input_field(basis, bkd, bkd.ones((npts,)) * 5.0)
        f2 = input_field(basis, bkd, bkd.ones((npts,)) * 3.0)

        result = f1 - f2

        bkd.assert_allclose(result.as_flat(), bkd.ones((npts,)) * 2.0, atol=1e-14)

    def test_multiplication(self, bkd):
        """Test field multiplication with product rule."""
        mesh = TransformedMesh1D(5, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        npts = basis.npts()

        f1 = input_field(basis, bkd, bkd.ones((npts,)) * 2.0)
        f2 = input_field(basis, bkd, bkd.ones((npts,)) * 3.0)

        result = f1 * f2

        bkd.assert_allclose(result.as_flat(), bkd.ones((npts,)) * 6.0, atol=1e-14)

    def test_multiplication_scalar(self, bkd):
        """Test field * scalar."""
        mesh = TransformedMesh1D(5, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        npts = basis.npts()

        f = input_field(basis, bkd, bkd.ones((npts,)) * 2.0)
        result = f * 3.0

        bkd.assert_allclose(result.as_flat(), bkd.ones((npts,)) * 6.0, atol=1e-14)

    def test_division(self, bkd):
        """Test field division."""
        mesh = TransformedMesh1D(5, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        npts = basis.npts()

        f1 = input_field(basis, bkd, bkd.ones((npts,)) * 6.0)
        f2 = input_field(basis, bkd, bkd.ones((npts,)) * 2.0)

        result = f1 / f2

        bkd.assert_allclose(result.as_flat(), bkd.ones((npts,)) * 3.0, atol=1e-14)

    def test_power(self, bkd):
        """Test field power."""
        mesh = TransformedMesh1D(5, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        npts = basis.npts()

        f = input_field(basis, bkd, bkd.ones((npts,)) * 2.0)
        result = f**3

        bkd.assert_allclose(result.as_flat(), bkd.ones((npts,)) * 8.0, atol=1e-14)

    def test_negation(self, bkd):
        """Test field negation."""
        mesh = TransformedMesh1D(5, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        npts = basis.npts()

        f = input_field(basis, bkd, bkd.ones((npts,)) * 3.0)
        result = -f

        bkd.assert_allclose(result.as_flat(), bkd.ones((npts,)) * -3.0, atol=1e-14)


class TestFieldJacobian:
    """Base test class for Field Jacobian tracking."""

    def test_addition_jacobian(self, bkd):
        """Test Jacobian of f + g = df + dg."""
        mesh = TransformedMesh1D(5, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        npts = basis.npts()

        f = input_field(basis, bkd, bkd.linspace(1.0, 2.0, npts))
        g = input_field(basis, bkd, bkd.linspace(2.0, 3.0, npts))

        result = f + g
        jac = result.get_jacobian()

        expected = 2.0 * bkd.eye(npts)
        bkd.assert_allclose(jac, expected, atol=1e-14)

    def test_multiplication_jacobian(self, bkd):
        """Test Jacobian of f * g = g*df + f*dg (product rule)."""
        mesh = TransformedMesh1D(5, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        npts = basis.npts()

        f = input_field(basis, bkd, bkd.ones((npts,)) * 2.0)
        g = input_field(basis, bkd, bkd.ones((npts,)) * 3.0)

        result = f * g
        jac = result.get_jacobian()

        expected = 5.0 * bkd.eye(npts)
        bkd.assert_allclose(jac, expected, atol=1e-14)

    def test_power_jacobian(self, bkd):
        """Test Jacobian of f^n = n * f^(n-1) * df."""
        mesh = TransformedMesh1D(5, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        npts = basis.npts()

        f = input_field(basis, bkd, bkd.ones((npts,)) * 2.0)
        result = f**3

        jac = result.get_jacobian()

        expected = 12.0 * bkd.eye(npts)
        bkd.assert_allclose(jac, expected, atol=1e-14)


class TestFieldNumpy(TestField):
    """NumPy backend tests for Field."""


class TestScalarFieldFactoriesNumpy(TestScalarFieldFactories):
    """NumPy backend tests for scalar field factories."""


class TestFieldArithmeticNumpy(TestFieldArithmetic):
    """NumPy backend tests for Field arithmetic."""


class TestFieldJacobianNumpy(TestFieldJacobian):
    """NumPy backend tests for Field Jacobian tracking."""
