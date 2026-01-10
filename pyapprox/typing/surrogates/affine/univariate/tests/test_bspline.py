"""Dual-backend tests for BSpline1D and HierarchicalBSpline1D.

Tests run on both NumPy and PyTorch backends using the base class pattern.
"""

import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.test_utils import load_tests

from pyapprox.typing.surrogates.affine.univariate.bspline import (
    BSpline1D,
    HierarchicalBSpline1D,
)


class TestBSpline1D(Generic[Array], unittest.TestCase):
    """Tests for BSpline1D - dual backend."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_init_default(self) -> None:
        """Test default initialization."""
        basis = BSpline1D(self._bkd)
        self.assertEqual(basis.degree(), 3)  # Default cubic
        self.assertEqual(basis.nterms(), 4)  # degree + 1

    def test_init_custom_degree(self) -> None:
        """Test initialization with custom degree."""
        basis = BSpline1D(self._bkd, degree=2, nterms=6)
        self.assertEqual(basis.degree(), 2)
        self.assertEqual(basis.nterms(), 6)

    def test_set_nterms(self) -> None:
        """Test set_nterms method."""
        basis = BSpline1D(self._bkd, degree=3)
        basis.set_nterms(8)
        self.assertEqual(basis.nterms(), 8)

    def test_knots_shape(self) -> None:
        """Test that knots have correct shape."""
        basis = BSpline1D(self._bkd, degree=3, nterms=5)
        knots = basis.knots()
        # For degree p and n terms: n + p + 1 knots
        expected_nknots = 5 + 3 + 1
        self.assertEqual(knots.shape[0], expected_nknots)

    def test_call_shape(self) -> None:
        """Test output shape of __call__."""
        basis = BSpline1D(self._bkd, degree=3, nterms=5)
        samples = self._bkd.linspace(0.0, 1.0, 10)[None, :]
        values = basis(samples)

        self.assertEqual(values.shape[0], 10)  # nsamples
        self.assertEqual(values.shape[1], 5)  # nterms

    def test_partition_of_unity(self) -> None:
        """Test that B-spline basis sums to 1 on [0, 1]."""
        basis = BSpline1D(self._bkd, degree=3, nterms=6)
        # Avoid exact endpoints where boundary effects occur
        samples = self._bkd.linspace(0.01, 0.99, 20)[None, :]
        values = basis(samples)
        row_sums = self._bkd.sum(values, axis=1)

        expected = self._bkd.ones((20,))
        self._bkd.assert_allclose(row_sums, expected, rtol=1e-10)

    def test_positivity(self) -> None:
        """Test that B-splines are non-negative."""
        basis = BSpline1D(self._bkd, degree=3, nterms=6)
        samples = self._bkd.linspace(0.0, 1.0, 50)[None, :]
        values = basis(samples)

        min_val = self._bkd.to_numpy(self._bkd.min(values))
        self.assertGreaterEqual(min_val, -1e-10)

    def test_local_support(self) -> None:
        """Test that B-splines have local support."""
        basis = BSpline1D(self._bkd, degree=3, nterms=7)
        samples = self._bkd.linspace(0.0, 1.0, 100)[None, :]
        values = basis(samples)

        # Each basis function should be zero for many samples
        for i in range(7):
            col = self._bkd.to_numpy(values[:, i])
            zero_count = (abs(col) < 1e-10).sum()
            self.assertGreater(zero_count, 0)

    def test_jacobian_batch_shape(self) -> None:
        """Test jacobian_batch output shape."""
        basis = BSpline1D(self._bkd, degree=3, nterms=5)
        samples = self._bkd.linspace(0.1, 0.9, 8)[None, :]
        jac = basis.jacobian_batch(samples)

        self.assertEqual(jac.shape[0], 8)  # nsamples
        self.assertEqual(jac.shape[1], 5)  # nterms

    def test_hessian_batch_shape(self) -> None:
        """Test hessian_batch output shape."""
        basis = BSpline1D(self._bkd, degree=3, nterms=5)
        samples = self._bkd.linspace(0.1, 0.9, 8)[None, :]
        hess = basis.hessian_batch(samples)

        self.assertEqual(hess.shape[0], 8)  # nsamples
        self.assertEqual(hess.shape[1], 5)  # nterms

    def test_derivative_sum_zero(self) -> None:
        """Test that sum of derivatives is zero (constant preserving)."""
        basis = BSpline1D(self._bkd, degree=3, nterms=6)
        samples = self._bkd.linspace(0.1, 0.9, 15)[None, :]
        jac = basis.jacobian_batch(samples)
        row_sums = self._bkd.sum(jac, axis=1)

        expected = self._bkd.zeros((15,))
        self._bkd.assert_allclose(row_sums, expected, atol=1e-8)

    def test_numerical_first_derivative(self) -> None:
        """Test first derivative against finite differences."""
        basis = BSpline1D(self._bkd, degree=3, nterms=5)

        x0 = 0.4
        h = 1e-7

        samples_center = self._bkd.asarray([[x0]])
        samples_plus = self._bkd.asarray([[x0 + h]])
        samples_minus = self._bkd.asarray([[x0 - h]])

        vals_plus = basis(samples_plus)
        vals_minus = basis(samples_minus)
        fd_deriv = (vals_plus - vals_minus) / (2 * h)

        analytic_deriv = basis.jacobian_batch(samples_center)

        self._bkd.assert_allclose(analytic_deriv, fd_deriv, rtol=1e-5)

    def test_numerical_second_derivative(self) -> None:
        """Test second derivative against finite differences."""
        basis = BSpline1D(self._bkd, degree=3, nterms=5)

        x0 = 0.4
        h = 1e-5

        samples_center = self._bkd.asarray([[x0]])
        samples_plus = self._bkd.asarray([[x0 + h]])
        samples_minus = self._bkd.asarray([[x0 - h]])

        d1_plus = basis.jacobian_batch(samples_plus)
        d1_minus = basis.jacobian_batch(samples_minus)
        fd_d2 = (d1_plus - d1_minus) / (2 * h)

        analytic_d2 = basis.hessian_batch(samples_center)

        self._bkd.assert_allclose(analytic_d2, fd_d2, rtol=1e-4)

    def test_linear_bsplines(self) -> None:
        """Test linear (degree 1) B-splines."""
        basis = BSpline1D(self._bkd, degree=1, nterms=3)
        samples = self._bkd.linspace(0.0, 1.0, 11)[None, :]
        values = basis(samples)

        # Linear B-splines should form a partition of unity
        row_sums = self._bkd.sum(values, axis=1)
        expected = self._bkd.ones((11,))
        self._bkd.assert_allclose(row_sums, expected, rtol=1e-10)

    def test_repr(self) -> None:
        """Test string representation."""
        basis = BSpline1D(self._bkd, degree=3, nterms=5)
        repr_str = repr(basis)
        self.assertIn("BSpline1D", repr_str)
        self.assertIn("degree=3", repr_str)
        self.assertIn("nterms=5", repr_str)


class TestHierarchicalBSpline1D(Generic[Array], unittest.TestCase):
    """Tests for HierarchicalBSpline1D - dual backend."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_init_default(self) -> None:
        """Test default initialization."""
        basis = HierarchicalBSpline1D(self._bkd)
        self.assertEqual(basis.degree(), 3)
        self.assertEqual(basis.max_level(), 5)
        self.assertEqual(basis.nterms(), 1)

    def test_init_custom(self) -> None:
        """Test custom initialization."""
        basis = HierarchicalBSpline1D(self._bkd, degree=2, max_level=4)
        self.assertEqual(basis.degree(), 2)
        self.assertEqual(basis.max_level(), 4)

    def test_set_nterms(self) -> None:
        """Test set_nterms method."""
        basis = HierarchicalBSpline1D(self._bkd)
        basis.set_nterms(8)
        self.assertEqual(basis.nterms(), 8)

    def test_nbasis_at_level(self) -> None:
        """Test nbasis_at_level method."""
        basis = HierarchicalBSpline1D(self._bkd)

        self.assertEqual(basis.nbasis_at_level(0), 1)
        self.assertEqual(basis.nbasis_at_level(1), 1)
        self.assertEqual(basis.nbasis_at_level(2), 2)
        self.assertEqual(basis.nbasis_at_level(3), 4)

    def test_total_basis_up_to_level(self) -> None:
        """Test total_basis_up_to_level method."""
        basis = HierarchicalBSpline1D(self._bkd)

        self.assertEqual(basis.total_basis_up_to_level(0), 1)
        self.assertEqual(basis.total_basis_up_to_level(1), 2)
        self.assertEqual(basis.total_basis_up_to_level(2), 4)
        self.assertEqual(basis.total_basis_up_to_level(3), 8)

    def test_level_index_to_flat(self) -> None:
        """Test level_index_to_flat conversion."""
        basis = HierarchicalBSpline1D(self._bkd)

        self.assertEqual(basis.level_index_to_flat(0, 0), 0)
        self.assertEqual(basis.level_index_to_flat(1, 0), 1)
        self.assertEqual(basis.level_index_to_flat(2, 0), 2)
        self.assertEqual(basis.level_index_to_flat(2, 1), 3)

    def test_flat_to_level_index(self) -> None:
        """Test flat_to_level_index conversion."""
        basis = HierarchicalBSpline1D(self._bkd)

        self.assertEqual(basis.flat_to_level_index(0), (0, 0))
        self.assertEqual(basis.flat_to_level_index(1), (1, 0))
        self.assertEqual(basis.flat_to_level_index(2), (2, 0))
        self.assertEqual(basis.flat_to_level_index(3), (2, 1))

    def test_level_index_roundtrip(self) -> None:
        """Test roundtrip between flat and (level, index)."""
        basis = HierarchicalBSpline1D(self._bkd)

        for flat_idx in range(16):
            level, index = basis.flat_to_level_index(flat_idx)
            recovered = basis.level_index_to_flat(level, index)
            self.assertEqual(flat_idx, recovered)

    def test_evaluate_hierarchical_shape(self) -> None:
        """Test evaluate_hierarchical output shape."""
        basis = HierarchicalBSpline1D(self._bkd, degree=3)
        samples = self._bkd.linspace(0.0, 1.0, 10)[None, :]

        values = basis.evaluate_hierarchical(samples, level=0, index=0)
        self.assertEqual(values.shape[0], 10)

        values = basis.evaluate_hierarchical(samples, level=1, index=0)
        self.assertEqual(values.shape[0], 10)

    def test_call_shape(self) -> None:
        """Test __call__ output shape."""
        basis = HierarchicalBSpline1D(self._bkd)
        basis.set_nterms(4)

        samples = self._bkd.linspace(0.0, 1.0, 10)[None, :]
        values = basis(samples)

        self.assertEqual(values.shape[0], 10)  # nsamples
        self.assertEqual(values.shape[1], 4)  # nterms

    def test_jacobian_batch_shape(self) -> None:
        """Test jacobian_batch output shape."""
        basis = HierarchicalBSpline1D(self._bkd)
        basis.set_nterms(4)

        samples = self._bkd.linspace(0.1, 0.9, 8)[None, :]
        jac = basis.jacobian_batch(samples)

        self.assertEqual(jac.shape[0], 8)  # nsamples
        self.assertEqual(jac.shape[1], 4)  # nterms

    def test_hierarchical_derivative(self) -> None:
        """Test evaluate_hierarchical_derivative."""
        basis = HierarchicalBSpline1D(self._bkd, degree=3)
        samples = self._bkd.linspace(0.1, 0.9, 10)[None, :]

        deriv = basis.evaluate_hierarchical_derivative(samples, level=0, index=0)
        self.assertEqual(deriv.shape[0], 10)

    def test_level_0_covers_domain(self) -> None:
        """Test that level 0 basis covers full domain."""
        basis = HierarchicalBSpline1D(self._bkd, degree=3)
        samples = self._bkd.linspace(0.1, 0.9, 20)[None, :]

        values = basis.evaluate_hierarchical(samples, level=0, index=0)
        vals_np = self._bkd.to_numpy(values)

        # Level 0 should be non-zero throughout domain interior
        self.assertTrue((vals_np > 0).all())

    def test_numerical_hierarchical_derivative(self) -> None:
        """Test hierarchical derivative against finite differences."""
        basis = HierarchicalBSpline1D(self._bkd, degree=3)

        x0 = 0.5
        h = 1e-7

        samples_center = self._bkd.asarray([[x0]])
        samples_plus = self._bkd.asarray([[x0 + h]])
        samples_minus = self._bkd.asarray([[x0 - h]])

        vals_plus = basis.evaluate_hierarchical(samples_plus, level=0, index=0)
        vals_minus = basis.evaluate_hierarchical(samples_minus, level=0, index=0)
        fd_deriv = (vals_plus - vals_minus) / (2 * h)

        analytic = basis.evaluate_hierarchical_derivative(
            samples_center, level=0, index=0
        )

        self._bkd.assert_allclose(
            analytic.reshape(-1), fd_deriv.reshape(-1), rtol=1e-5
        )

    def test_repr(self) -> None:
        """Test string representation."""
        basis = HierarchicalBSpline1D(self._bkd, degree=3, max_level=5)
        basis.set_nterms(4)
        repr_str = repr(basis)
        self.assertIn("HierarchicalBSpline1D", repr_str)
        self.assertIn("degree=3", repr_str)
        self.assertIn("max_level=5", repr_str)


# NumPy backend tests
class TestBSpline1DNumpy(TestBSpline1D[NDArray[Any]]):
    """NumPy backend tests for BSpline1D."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestHierarchicalBSpline1DNumpy(TestHierarchicalBSpline1D[NDArray[Any]]):
    """NumPy backend tests for HierarchicalBSpline1D."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


# PyTorch backend tests
class TestBSpline1DTorch(TestBSpline1D[torch.Tensor]):
    """PyTorch backend tests for BSpline1D."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def bkd(self) -> TorchBkd:
        return TorchBkd()


class TestHierarchicalBSpline1DTorch(TestHierarchicalBSpline1D[torch.Tensor]):
    """PyTorch backend tests for HierarchicalBSpline1D."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def bkd(self) -> TorchBkd:
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
