"""Tests for TensorProductInterpolant.

Dual-backend tests for NumPy and PyTorch.
"""

import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.test_utils import load_tests
from pyapprox.typing.surrogates.tensorproduct import TensorProductInterpolant
from pyapprox.typing.surrogates.affine.univariate import (
    LagrangeBasis1D,
    LegendrePolynomial1D,
)
from pyapprox.typing.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)


class TestTensorProductInterpolant(Generic[Array], unittest.TestCase):
    """Base tests for TensorProductInterpolant."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def _make_basis(self) -> LagrangeBasis1D[Array]:
        """Create a LagrangeBasis1D using Legendre Gauss quadrature."""
        poly = LegendrePolynomial1D(self._bkd)
        poly.set_nterms(10)
        return LagrangeBasis1D(self._bkd, poly.gauss_quadrature_rule)

    def _make_interpolant(
        self, nvars: int = 2, nterms_1d: int = 3
    ) -> TensorProductInterpolant[Array]:
        """Create a TensorProductInterpolant for testing."""
        basis = self._make_basis()
        bases = [basis] * nvars
        nterms = [nterms_1d] * nvars
        return TensorProductInterpolant(self._bkd, bases, nterms)

    # Basic functionality tests

    def test_init(self) -> None:
        """Test initialization."""
        interp = self._make_interpolant(2, 3)
        self.assertEqual(interp.nvars(), 2)
        self.assertEqual(interp.nsamples(), 9)  # 3 * 3
        self.assertEqual(interp.nqoi(), 0)  # No values set

    def test_init_asymmetric(self) -> None:
        """Test initialization with different nterms per dimension."""
        basis0 = self._make_basis()
        basis1 = self._make_basis()
        interp = TensorProductInterpolant(
            self._bkd, [basis0, basis1], [3, 4]
        )
        self.assertEqual(interp.nvars(), 2)
        self.assertEqual(interp.nsamples(), 12)  # 3 * 4

    def test_init_length_mismatch(self) -> None:
        """Test that mismatched lengths raise ValueError."""
        basis = self._make_basis()
        with self.assertRaises(ValueError):
            TensorProductInterpolant(self._bkd, [basis, basis], [3])

    def test_rejects_orthogonal_poly(self) -> None:
        """Test that orthogonal polynomial bases are rejected."""
        poly = LegendrePolynomial1D(self._bkd)
        poly.set_nterms(5)
        with self.assertRaises(TypeError) as ctx:
            TensorProductInterpolant(self._bkd, [poly, poly], [3, 3])
        self.assertIn("InterpolationBasis1DProtocol", str(ctx.exception))

    def test_get_samples_shape(self) -> None:
        """Test get_samples returns correct shape."""
        interp = self._make_interpolant(3, 4)
        samples = interp.get_samples()
        self.assertEqual(samples.shape, (3, 64))  # 4^3 = 64

    def test_set_values(self) -> None:
        """Test set_values method."""
        interp = self._make_interpolant(2, 3)
        values = self._bkd.zeros((2, 9))  # (nqoi, nsamples)
        interp.set_values(values)
        self.assertEqual(interp.nqoi(), 2)

    def test_set_values_wrong_shape(self) -> None:
        """Test set_values with wrong shape raises ValueError."""
        interp = self._make_interpolant(2, 3)
        values = self._bkd.zeros((2, 5))  # Wrong: should be 9 samples
        with self.assertRaises(ValueError):
            interp.set_values(values)

    def test_call_without_values(self) -> None:
        """Test that __call__ raises error without values set."""
        interp = self._make_interpolant(2, 3)
        samples = self._bkd.asarray([[0.0], [0.0]])
        with self.assertRaises(ValueError):
            interp(samples)

    # Interpolation tests

    def test_interpolates_polynomial(self) -> None:
        """Test exact interpolation of a polynomial."""
        interp = self._make_interpolant(2, 5)  # 5 points can exactly interp degree 4
        samples = interp.get_samples()

        # f(x, y) = x^2 + y^2 (degree 2, should be exact)
        # values shape: (nqoi, nsamples) = (1, nsamples)
        values = samples[0:1, :] ** 2 + samples[1:2, :] ** 2
        interp.set_values(values)

        # Evaluate at random points
        test_pts = self._bkd.asarray([
            [0.0, 0.3, -0.5, 0.7],
            [0.0, -0.2, 0.4, 0.1],
        ])
        result = interp(test_pts)  # (nqoi, npoints)
        expected = test_pts[0:1, :] ** 2 + test_pts[1:2, :] ** 2  # (1, npoints)

        self._bkd.assert_allclose(result, expected, rtol=1e-10, atol=1e-14)

    def test_interpolates_at_nodes(self) -> None:
        """Test that interpolant reproduces values at nodes."""
        interp = self._make_interpolant(2, 4)
        samples = interp.get_samples()
        nsamples = samples.shape[1]

        # Set arbitrary values with shape (nqoi, nsamples) = (1, nsamples)
        values = self._bkd.asarray([[float(i) for i in range(nsamples)]])
        interp.set_values(values)

        # Evaluate at nodes - result shape (nqoi, nsamples)
        result = interp(samples)
        self._bkd.assert_allclose(result, values, rtol=1e-10)

    # Derivative tests

    def test_jacobian_supported(self) -> None:
        """Test jacobian_supported returns True for LagrangeBasis1D."""
        interp = self._make_interpolant()
        self.assertTrue(interp.jacobian_supported())

    def test_hessian_supported(self) -> None:
        """Test hessian_supported returns True for LagrangeBasis1D."""
        interp = self._make_interpolant()
        self.assertTrue(interp.hessian_supported())

    def test_jacobian_shape(self) -> None:
        """Test jacobian returns correct shape."""
        interp = self._make_interpolant(3, 4)
        samples = interp.get_samples()
        values = self._bkd.zeros((2, samples.shape[1]))  # (nqoi, nsamples)
        interp.set_values(values)

        sample = self._bkd.asarray([[0.0], [0.0], [0.0]])
        jac = interp.jacobian(sample)
        self.assertEqual(jac.shape, (2, 3))  # (nqoi, nvars)

    def test_hessian_shape(self) -> None:
        """Test hessian returns correct shape."""
        interp = self._make_interpolant(3, 4)
        samples = interp.get_samples()
        values = self._bkd.zeros((1, samples.shape[1]))  # (nqoi, nsamples), nqoi must be 1
        interp.set_values(values)

        sample = self._bkd.asarray([[0.0], [0.0], [0.0]])
        hess = interp.hessian(sample)
        self.assertEqual(hess.shape, (3, 3))  # (nvars, nvars)

    def test_hvp_shape(self) -> None:
        """Test hvp returns correct shape."""
        interp = self._make_interpolant(3, 4)
        samples = interp.get_samples()
        values = self._bkd.zeros((1, samples.shape[1]))  # (nqoi, nsamples), nqoi must be 1
        interp.set_values(values)

        sample = self._bkd.asarray([[0.0], [0.0], [0.0]])
        vec = self._bkd.asarray([[1.0], [0.0], [0.0]])
        hvp_result = interp.hvp(sample, vec)
        self.assertEqual(hvp_result.shape, (3, 1))  # (nvars, 1)

    def test_whvp_shape(self) -> None:
        """Test whvp returns correct shape."""
        interp = self._make_interpolant(3, 4)
        samples = interp.get_samples()
        values = self._bkd.zeros((2, samples.shape[1]))  # (nqoi, nsamples)
        interp.set_values(values)

        sample = self._bkd.asarray([[0.0], [0.0], [0.0]])
        vec = self._bkd.asarray([[1.0], [0.0], [0.0]])
        weights = self._bkd.asarray([[0.5], [0.5]])
        whvp_result = interp.whvp(sample, vec, weights)
        self.assertEqual(whvp_result.shape, (3, 1))  # (nvars, 1)

    def test_derivatives_with_checker(self) -> None:
        """Test jacobian and hessian using DerivativeChecker.

        Uses the standard DerivativeChecker from the interface module
        as per CLAUDE.md conventions.
        """
        interp = self._make_interpolant(2, 5)
        samples = interp.get_samples()

        # f(x, y) = x^2 * y (has non-trivial jacobian and hessian)
        # values shape: (nqoi, nsamples) = (1, nsamples)
        values = samples[0:1, :] ** 2 * samples[1:2, :]
        interp.set_values(values)

        # Use DerivativeChecker to validate derivatives
        checker = DerivativeChecker(interp)
        sample = self._bkd.asarray([[0.3], [-0.4]])
        errors = checker.check_derivatives(sample)

        # Check jacobian error ratio (1e-6 tolerance per CLAUDE.md)
        jac_error = checker.error_ratio(errors[0])
        self.assertLess(float(jac_error), 1e-6)

        # Check hessian error ratio (via hvp)
        hess_error = checker.error_ratio(errors[1])
        self.assertLess(float(hess_error), 1e-6)

    def test_whvp_derivatives_with_checker(self) -> None:
        """Test whvp derivatives using DerivativeChecker for multi-QoI.

        For multi-QoI functions, DerivativeChecker uses whvp with weights.
        """
        interp = self._make_interpolant(2, 5)
        samples = interp.get_samples()

        # Two QoIs - values shape: (nqoi, nsamples) = (2, nsamples)
        q1 = samples[0:1, :] ** 2  # (1, nsamples)
        q2 = samples[1:2, :] ** 2  # (1, nsamples)
        values = self._bkd.vstack([q1, q2])  # (2, nsamples)
        interp.set_values(values)

        # Use DerivativeChecker with weights for multi-QoI
        checker = DerivativeChecker(interp)
        sample = self._bkd.asarray([[0.3], [-0.4]])
        weights = self._bkd.asarray([[0.6, 0.4]])  # Shape (1, nqoi)
        errors = checker.check_derivatives(sample, weights=weights)

        # Check jacobian error ratio (1e-6 tolerance per CLAUDE.md)
        jac_error = checker.error_ratio(errors[0])
        self.assertLess(float(jac_error), 1e-6)

        # Check whvp error ratio
        whvp_error = checker.error_ratio(errors[1])
        self.assertLess(float(whvp_error), 1e-6)

    def test_hvp_matches_hessian(self) -> None:
        """Test that hvp matches H @ v from full hessian."""
        interp = self._make_interpolant(2, 5)
        samples = interp.get_samples()

        # values shape: (nqoi, nsamples) = (1, nsamples), nqoi must be 1 for hessian/hvp
        values = samples[0:1, :] ** 2 * samples[1:2, :]
        interp.set_values(values)

        sample = self._bkd.asarray([[0.3], [-0.4]])
        vec = self._bkd.asarray([[0.7], [-0.3]])

        hvp_result = interp.hvp(sample, vec)
        hess = interp.hessian(sample)
        expected = hess @ vec

        self._bkd.assert_allclose(hvp_result, expected, rtol=1e-10)

    def test_whvp_matches_weighted_hvp(self) -> None:
        """Test that whvp matches weighted sum of individual QoI hvps."""
        # Create an interpolant with 2 QoIs
        interp = self._make_interpolant(2, 5)
        samples = interp.get_samples()

        # Two QoIs - values shape: (nqoi, nsamples) = (2, nsamples)
        q1 = samples[0:1, :] ** 2  # (1, nsamples)
        q2 = samples[1:2, :] ** 2  # (1, nsamples)
        values = self._bkd.vstack([q1, q2])  # (2, nsamples)
        interp.set_values(values)

        sample = self._bkd.asarray([[0.3], [-0.4]])
        vec = self._bkd.asarray([[0.7], [-0.3]])
        weights = self._bkd.asarray([[0.6], [0.4]])

        whvp_result = interp.whvp(sample, vec, weights)

        # Create separate single-QoI interpolants to compute individual hvps
        interp0 = self._make_interpolant(2, 5)
        interp0.set_values(q1)  # (1, nsamples)
        hvp0 = interp0.hvp(sample, vec)

        interp1 = self._make_interpolant(2, 5)
        interp1.set_values(q2)  # (1, nsamples)
        hvp1 = interp1.hvp(sample, vec)

        expected = 0.6 * hvp0 + 0.4 * hvp1

        self._bkd.assert_allclose(whvp_result, expected, rtol=1e-10)

    def test_hessian_rejects_multi_qoi(self) -> None:
        """Test that hessian raises ValueError when nqoi > 1."""
        interp = self._make_interpolant(2, 4)
        samples = interp.get_samples()
        values = self._bkd.zeros((2, samples.shape[1]))  # nqoi=2
        interp.set_values(values)

        sample = self._bkd.asarray([[0.0], [0.0]])
        with self.assertRaises(ValueError) as ctx:
            interp.hessian(sample)
        self.assertIn("nqoi=1", str(ctx.exception))

    def test_hvp_rejects_multi_qoi(self) -> None:
        """Test that hvp raises ValueError when nqoi > 1."""
        interp = self._make_interpolant(2, 4)
        samples = interp.get_samples()
        values = self._bkd.zeros((2, samples.shape[1]))  # nqoi=2
        interp.set_values(values)

        sample = self._bkd.asarray([[0.0], [0.0]])
        vec = self._bkd.asarray([[1.0], [0.0]])
        with self.assertRaises(ValueError) as ctx:
            interp.hvp(sample, vec)
        self.assertIn("nqoi=1", str(ctx.exception))

    def test_repr(self) -> None:
        """Test string representation."""
        interp = self._make_interpolant(2, 3)
        repr_str = repr(interp)
        self.assertIn("TensorProductInterpolant", repr_str)
        self.assertIn("nvars=2", repr_str)


# Concrete test classes


class TestTensorProductInterpolantNumpy(
    TestTensorProductInterpolant[NDArray[Any]]
):
    """NumPy backend tests."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestTensorProductInterpolantTorch(
    TestTensorProductInterpolant[torch.Tensor]
):
    """PyTorch backend tests."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = self.bkd()

    def bkd(self) -> TorchBkd:
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
