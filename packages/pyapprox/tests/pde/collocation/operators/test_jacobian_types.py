"""Tests for sparse Jacobian types."""


from pyapprox.pde.collocation.operators.jacobian_types import (
    DenseJacobian,
    DiagJacobian,
    ZeroJacobian,
)


class TestJacobianTypes:
    """Base test class for Jacobian types."""

    def test_dense_jacobian_shape(self, bkd):
        """Test DenseJacobian stores correct shape."""
        shape = (5, 10)
        jac_data = bkd.zeros(shape)
        jac = DenseJacobian(bkd, shape, jac_data)
        assert jac.shape == shape
        assert jac.get_jacobian().shape == shape

    def test_dense_jacobian_negation(self, bkd):
        """Test DenseJacobian negation."""
        shape = (3, 6)
        jac_data = bkd.asarray(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
            ]
        )
        jac = DenseJacobian(bkd, shape, jac_data)
        neg_jac = -jac
        bkd.assert_allclose(neg_jac.get_jacobian(), -jac_data)

    def test_dense_jacobian_scalar_multiply(self, bkd):
        """Test DenseJacobian scalar multiplication."""
        shape = (3, 3)
        jac_data = bkd.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        jac = DenseJacobian(bkd, shape, jac_data)
        scaled = jac * 2.0
        bkd.assert_allclose(scaled.get_jacobian(), jac_data * 2.0)

    def test_dense_jacobian_add(self, bkd):
        """Test DenseJacobian addition."""
        shape = (3, 3)
        jac1_data = bkd.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
        jac2_data = bkd.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        jac1 = DenseJacobian(bkd, shape, jac1_data)
        jac2 = DenseJacobian(bkd, shape, jac2_data)
        result = jac1 + jac2
        bkd.assert_allclose(result.get_jacobian(), jac1_data + jac2_data)

    def test_diag_jacobian_shape(self, bkd):
        """Test DiagJacobian stores correct shape."""
        nmesh = 4
        ninput = 2
        shape = (nmesh, nmesh * ninput)  # (4, 8)
        # Compact storage: (nmesh, ninput)
        diag_data = bkd.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        jac = DiagJacobian(bkd, shape, diag_data)
        assert jac.shape == shape
        full_jac = jac.get_jacobian()
        assert full_jac.shape == shape

    def test_diag_jacobian_expansion(self, bkd):
        """Test DiagJacobian expands to correct dense matrix."""
        nmesh = 3
        ninput = 2
        shape = (nmesh, nmesh * ninput)
        diag_data = bkd.asarray([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        jac = DiagJacobian(bkd, shape, diag_data)
        full_jac = jac.get_jacobian()

        # Expected: block diagonal with two 3x3 diagonal matrices
        expected = bkd.asarray(
            [
                [1.0, 0.0, 0.0, 4.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0, 5.0, 0.0],
                [0.0, 0.0, 3.0, 0.0, 0.0, 6.0],
            ]
        )
        bkd.assert_allclose(full_jac, expected)

    def test_zero_jacobian(self, bkd):
        """Test ZeroJacobian returns zeros."""
        shape = (5, 10)
        jac = ZeroJacobian(bkd, shape)
        assert jac.shape == shape
        full_jac = jac.get_jacobian()
        bkd.assert_allclose(full_jac, bkd.zeros(shape))

    def test_zero_jacobian_operations(self, bkd):
        """Test ZeroJacobian preserves zero under operations."""
        shape = (3, 6)
        zero = ZeroJacobian(bkd, shape)

        # Negation
        neg_zero = -zero
        assert isinstance(neg_zero, ZeroJacobian)

        # Scalar multiply
        scaled = zero * 5.0
        assert isinstance(scaled, ZeroJacobian)

        # Division
        divided = zero / 2.0
        assert isinstance(divided, ZeroJacobian)

    def test_dense_plus_diag(self, bkd):
        """Test adding DenseJacobian and DiagJacobian."""
        nmesh = 3
        ninput = 1
        shape = (nmesh, nmesh * ninput)

        dense_data = bkd.asarray([[0.0, 1.0, 2.0], [3.0, 0.0, 4.0], [5.0, 6.0, 0.0]])
        diag_data = bkd.asarray([[1.0], [2.0], [3.0]])

        dense = DenseJacobian(bkd, shape, dense_data)
        diag = DiagJacobian(bkd, shape, diag_data)

        result = dense + diag
        expected = bkd.asarray([[1.0, 1.0, 2.0], [3.0, 2.0, 4.0], [5.0, 6.0, 3.0]])
        bkd.assert_allclose(result.get_jacobian(), expected)

    def test_dense_plus_zero(self, bkd):
        """Test adding DenseJacobian and ZeroJacobian."""
        shape = (3, 3)
        dense_data = bkd.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        dense = DenseJacobian(bkd, shape, dense_data)
        zero = ZeroJacobian(bkd, shape)

        result = dense + zero
        bkd.assert_allclose(result.get_jacobian(), dense_data)

    def test_rdot(self, bkd):
        """Test right matrix multiplication (A @ J)."""
        shape = (3, 3)
        jac_data = bkd.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
        jac = DenseJacobian(bkd, shape, jac_data)

        A = bkd.asarray([[1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
        result = jac.rdot(A)
        expected = bkd.dot(A, jac_data)
        bkd.assert_allclose(result.get_jacobian(), expected)

    def test_copy(self, bkd):
        """Test Jacobian copy is independent."""
        shape = (3, 3)
        jac_data = bkd.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        jac = DenseJacobian(bkd, shape, jac_data)
        jac_copy = jac.copy()

        # Modify original's internal data
        jac._sparse_jac[0, 0] = 100.0

        # Copy should be unaffected
        assert float(jac_copy.get_jacobian()[0, 0]) != 100.0


class TestJacobianTypesNumpy(TestJacobianTypes):
    """NumPy backend tests for Jacobian types."""
