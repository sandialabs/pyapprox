"""Dual-backend tests for BSpline1D and HierarchicalBSpline1D.

Tests run on both NumPy and PyTorch backends using the base class pattern.
"""

import pytest

from pyapprox.surrogates.affine.univariate.bspline import (
    BSpline1D,
    HierarchicalBSpline1D,
)


class TestBSpline1D:
    """Tests for BSpline1D - dual backend."""

    def test_init_default(self, bkd) -> None:
        """Test default initialization."""
        basis = BSpline1D(bkd)
        assert basis.degree() == 3  # Default cubic
        assert basis.nterms() == 4  # degree + 1

    def test_init_custom_degree(self, bkd) -> None:
        """Test initialization with custom degree."""
        basis = BSpline1D(bkd, degree=2, nterms=6)
        assert basis.degree() == 2
        assert basis.nterms() == 6

    def test_set_nterms(self, bkd) -> None:
        """Test set_nterms method."""
        basis = BSpline1D(bkd, degree=3)
        basis.set_nterms(8)
        assert basis.nterms() == 8

    def test_knots_shape(self, bkd) -> None:
        """Test that knots have correct shape."""
        basis = BSpline1D(bkd, degree=3, nterms=5)
        knots = basis.knots()
        # For degree p and n terms: n + p + 1 knots
        expected_nknots = 5 + 3 + 1
        assert knots.shape[0] == expected_nknots

    def test_call_shape(self, bkd) -> None:
        """Test output shape of __call__."""
        basis = BSpline1D(bkd, degree=3, nterms=5)
        samples = bkd.linspace(0.0, 1.0, 10)[None, :]
        values = basis(samples)

        assert values.shape[0] == 10  # nsamples
        assert values.shape[1] == 5  # nterms

    def test_partition_of_unity(self, bkd) -> None:
        """Test that B-spline basis sums to 1 on [0, 1]."""
        basis = BSpline1D(bkd, degree=3, nterms=6)
        # Avoid exact endpoints where boundary effects occur
        samples = bkd.linspace(0.01, 0.99, 20)[None, :]
        values = basis(samples)
        row_sums = bkd.sum(values, axis=1)

        expected = bkd.ones((20,))
        bkd.assert_allclose(row_sums, expected, rtol=1e-10)

    def test_positivity(self, bkd) -> None:
        """Test that B-splines are non-negative."""
        basis = BSpline1D(bkd, degree=3, nterms=6)
        samples = bkd.linspace(0.0, 1.0, 50)[None, :]
        values = basis(samples)

        min_val = bkd.to_numpy(bkd.min(values))
        assert min_val >= -1e-10

    def test_local_support(self, bkd) -> None:
        """Test that B-splines have local support."""
        basis = BSpline1D(bkd, degree=3, nterms=7)
        samples = bkd.linspace(0.0, 1.0, 100)[None, :]
        values = basis(samples)

        # Each basis function should be zero for many samples
        for i in range(7):
            col = bkd.to_numpy(values[:, i])
            zero_count = (abs(col) < 1e-10).sum()
            assert zero_count > 0

    def test_jacobian_batch_shape(self, bkd) -> None:
        """Test jacobian_batch output shape."""
        basis = BSpline1D(bkd, degree=3, nterms=5)
        samples = bkd.linspace(0.1, 0.9, 8)[None, :]
        jac = basis.jacobian_batch(samples)

        assert jac.shape[0] == 8  # nsamples
        assert jac.shape[1] == 5  # nterms

    def test_hessian_batch_shape(self, bkd) -> None:
        """Test hessian_batch output shape."""
        basis = BSpline1D(bkd, degree=3, nterms=5)
        samples = bkd.linspace(0.1, 0.9, 8)[None, :]
        hess = basis.hessian_batch(samples)

        assert hess.shape[0] == 8  # nsamples
        assert hess.shape[1] == 5  # nterms

    def test_derivative_sum_zero(self, bkd) -> None:
        """Test that sum of derivatives is zero (constant preserving)."""
        basis = BSpline1D(bkd, degree=3, nterms=6)
        samples = bkd.linspace(0.1, 0.9, 15)[None, :]
        jac = basis.jacobian_batch(samples)
        row_sums = bkd.sum(jac, axis=1)

        expected = bkd.zeros((15,))
        bkd.assert_allclose(row_sums, expected, atol=1e-8)

    def test_numerical_first_derivative(self, bkd) -> None:
        """Test first derivative against finite differences."""
        basis = BSpline1D(bkd, degree=3, nterms=5)

        x0 = 0.4
        h = 1e-7

        samples_center = bkd.asarray([[x0]])
        samples_plus = bkd.asarray([[x0 + h]])
        samples_minus = bkd.asarray([[x0 - h]])

        vals_plus = basis(samples_plus)
        vals_minus = basis(samples_minus)
        fd_deriv = (vals_plus - vals_minus) / (2 * h)

        analytic_deriv = basis.jacobian_batch(samples_center)

        bkd.assert_allclose(analytic_deriv, fd_deriv, rtol=1e-5)

    def test_numerical_second_derivative(self, bkd) -> None:
        """Test second derivative against finite differences."""
        basis = BSpline1D(bkd, degree=3, nterms=5)

        x0 = 0.4
        h = 1e-5

        samples_center = bkd.asarray([[x0]])
        samples_plus = bkd.asarray([[x0 + h]])
        samples_minus = bkd.asarray([[x0 - h]])

        d1_plus = basis.jacobian_batch(samples_plus)
        d1_minus = basis.jacobian_batch(samples_minus)
        fd_d2 = (d1_plus - d1_minus) / (2 * h)

        analytic_d2 = basis.hessian_batch(samples_center)

        bkd.assert_allclose(analytic_d2, fd_d2, rtol=1e-4)

    def test_linear_bsplines(self, bkd) -> None:
        """Test linear (degree 1) B-splines."""
        basis = BSpline1D(bkd, degree=1, nterms=3)
        samples = bkd.linspace(0.0, 1.0, 11)[None, :]
        values = basis(samples)

        # Linear B-splines should form a partition of unity
        row_sums = bkd.sum(values, axis=1)
        expected = bkd.ones((11,))
        bkd.assert_allclose(row_sums, expected, rtol=1e-10)

    def test_repr(self, bkd) -> None:
        """Test string representation."""
        basis = BSpline1D(bkd, degree=3, nterms=5)
        repr_str = repr(basis)
        assert "BSpline1D" in repr_str
        assert "degree=3" in repr_str
        assert "nterms=5" in repr_str


class TestHierarchicalBSpline1D:
    """Tests for HierarchicalBSpline1D - dual backend."""

    def test_init_default(self, bkd) -> None:
        """Test default initialization."""
        basis = HierarchicalBSpline1D(bkd)
        assert basis.degree() == 3
        assert basis.max_level() == 5
        assert basis.nterms() == 1

    def test_init_custom(self, bkd) -> None:
        """Test custom initialization."""
        basis = HierarchicalBSpline1D(bkd, degree=2, max_level=4)
        assert basis.degree() == 2
        assert basis.max_level() == 4

    def test_set_nterms(self, bkd) -> None:
        """Test set_nterms method."""
        basis = HierarchicalBSpline1D(bkd)
        basis.set_nterms(8)
        assert basis.nterms() == 8

    def test_nbasis_at_level(self, bkd) -> None:
        """Test nbasis_at_level method."""
        basis = HierarchicalBSpline1D(bkd)

        assert basis.nbasis_at_level(0) == 1
        assert basis.nbasis_at_level(1) == 1
        assert basis.nbasis_at_level(2) == 2
        assert basis.nbasis_at_level(3) == 4

    def test_total_basis_up_to_level(self, bkd) -> None:
        """Test total_basis_up_to_level method."""
        basis = HierarchicalBSpline1D(bkd)

        assert basis.total_basis_up_to_level(0) == 1
        assert basis.total_basis_up_to_level(1) == 2
        assert basis.total_basis_up_to_level(2) == 4
        assert basis.total_basis_up_to_level(3) == 8

    def test_level_index_to_flat(self, bkd) -> None:
        """Test level_index_to_flat conversion."""
        basis = HierarchicalBSpline1D(bkd)

        assert basis.level_index_to_flat(0, 0) == 0
        assert basis.level_index_to_flat(1, 0) == 1
        assert basis.level_index_to_flat(2, 0) == 2
        assert basis.level_index_to_flat(2, 1) == 3

    def test_flat_to_level_index(self, bkd) -> None:
        """Test flat_to_level_index conversion."""
        basis = HierarchicalBSpline1D(bkd)

        assert basis.flat_to_level_index(0) == (0, 0)
        assert basis.flat_to_level_index(1) == (1, 0)
        assert basis.flat_to_level_index(2) == (2, 0)
        assert basis.flat_to_level_index(3) == (2, 1)

    def test_level_index_roundtrip(self, bkd) -> None:
        """Test roundtrip between flat and (level, index)."""
        basis = HierarchicalBSpline1D(bkd)

        for flat_idx in range(16):
            level, index = basis.flat_to_level_index(flat_idx)
            recovered = basis.level_index_to_flat(level, index)
            assert flat_idx == recovered

    def test_evaluate_hierarchical_shape(self, bkd) -> None:
        """Test evaluate_hierarchical output shape."""
        basis = HierarchicalBSpline1D(bkd, degree=3)
        samples = bkd.linspace(0.0, 1.0, 10)[None, :]

        values = basis.evaluate_hierarchical(samples, level=0, index=0)
        assert values.shape[0] == 10

        values = basis.evaluate_hierarchical(samples, level=1, index=0)
        assert values.shape[0] == 10

    def test_call_shape(self, bkd) -> None:
        """Test __call__ output shape."""
        basis = HierarchicalBSpline1D(bkd)
        basis.set_nterms(4)

        samples = bkd.linspace(0.0, 1.0, 10)[None, :]
        values = basis(samples)

        assert values.shape[0] == 10  # nsamples
        assert values.shape[1] == 4  # nterms

    def test_jacobian_batch_shape(self, bkd) -> None:
        """Test jacobian_batch output shape."""
        basis = HierarchicalBSpline1D(bkd)
        basis.set_nterms(4)

        samples = bkd.linspace(0.1, 0.9, 8)[None, :]
        jac = basis.jacobian_batch(samples)

        assert jac.shape[0] == 8  # nsamples
        assert jac.shape[1] == 4  # nterms

    def test_hierarchical_derivative(self, bkd) -> None:
        """Test evaluate_hierarchical_derivative."""
        basis = HierarchicalBSpline1D(bkd, degree=3)
        samples = bkd.linspace(0.1, 0.9, 10)[None, :]

        deriv = basis.evaluate_hierarchical_derivative(samples, level=0, index=0)
        assert deriv.shape[0] == 10

    def test_level_0_covers_domain(self, bkd) -> None:
        """Test that level 0 basis covers full domain."""
        basis = HierarchicalBSpline1D(bkd, degree=3)
        samples = bkd.linspace(0.1, 0.9, 20)[None, :]

        values = basis.evaluate_hierarchical(samples, level=0, index=0)
        vals_np = bkd.to_numpy(values)

        # Level 0 should be non-zero throughout domain interior
        assert (vals_np > 0).all()

    def test_numerical_hierarchical_derivative(self, bkd) -> None:
        """Test hierarchical derivative against finite differences."""
        basis = HierarchicalBSpline1D(bkd, degree=3)

        x0 = 0.5
        h = 1e-7

        samples_center = bkd.asarray([[x0]])
        samples_plus = bkd.asarray([[x0 + h]])
        samples_minus = bkd.asarray([[x0 - h]])

        vals_plus = basis.evaluate_hierarchical(samples_plus, level=0, index=0)
        vals_minus = basis.evaluate_hierarchical(samples_minus, level=0, index=0)
        fd_deriv = (vals_plus - vals_minus) / (2 * h)

        analytic = basis.evaluate_hierarchical_derivative(
            samples_center, level=0, index=0
        )

        bkd.assert_allclose(analytic.reshape(-1), fd_deriv.reshape(-1), rtol=1e-5)

    def test_repr(self, bkd) -> None:
        """Test string representation."""
        basis = HierarchicalBSpline1D(bkd, degree=3, max_level=5)
        basis.set_nterms(4)
        repr_str = repr(basis)
        assert "HierarchicalBSpline1D" in repr_str
        assert "degree=3" in repr_str
        assert "max_level=5" in repr_str
