"""Tests for TensorProductInterpolant.

Dual-backend tests for NumPy and PyTorch.
"""

import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.surrogates.affine.univariate import (
    LagrangeBasis1D,
    LegendrePolynomial1D,
)
from pyapprox.surrogates.tensorproduct import TensorProductInterpolant
from pyapprox.util.test_utils import slow_test


class TestTensorProductInterpolant:
    """Base tests for TensorProductInterpolant."""

    def _make_basis(self, bkd):
        """Create a LagrangeBasis1D using Legendre Gauss quadrature."""
        poly = LegendrePolynomial1D(bkd)
        poly.set_nterms(10)
        return LagrangeBasis1D(bkd, poly.gauss_quadrature_rule)

    def _make_interpolant(
        self, bkd, nvars: int = 2, nterms_1d: int = 3
    ):
        """Create a TensorProductInterpolant for testing."""
        basis = self._make_basis(bkd)
        bases = [basis] * nvars
        nterms = [nterms_1d] * nvars
        return TensorProductInterpolant(bkd, bases, nterms)

    # Basic functionality tests

    def test_init(self, bkd) -> None:
        """Test initialization."""
        interp = self._make_interpolant(bkd, 2, 3)
        assert interp.nvars() == 2
        assert interp.nsamples() == 9  # 3 * 3
        assert interp.nqoi() == 0  # No values set

    def test_init_asymmetric(self, bkd) -> None:
        """Test initialization with different nterms per dimension."""
        basis0 = self._make_basis(bkd)
        basis1 = self._make_basis(bkd)
        interp = TensorProductInterpolant(bkd, [basis0, basis1], [3, 4])
        assert interp.nvars() == 2
        assert interp.nsamples() == 12  # 3 * 4

    def test_init_length_mismatch(self, bkd) -> None:
        """Test that mismatched lengths raise ValueError."""
        basis = self._make_basis(bkd)
        with pytest.raises(ValueError):
            TensorProductInterpolant(bkd, [basis, basis], [3])

    def test_rejects_orthogonal_poly(self, bkd) -> None:
        """Test that orthogonal polynomial bases are rejected."""
        poly = LegendrePolynomial1D(bkd)
        poly.set_nterms(5)
        with pytest.raises(TypeError) as ctx:
            TensorProductInterpolant(bkd, [poly, poly], [3, 3])
        assert "InterpolationBasis1DProtocol" in str(ctx.value)

    def test_get_samples_shape(self, bkd) -> None:
        """Test get_samples returns correct shape."""
        interp = self._make_interpolant(bkd, 3, 4)
        samples = interp.get_samples()
        assert samples.shape == (3, 64)  # 4^3 = 64

    def test_set_values(self, bkd) -> None:
        """Test set_values method."""
        interp = self._make_interpolant(bkd, 2, 3)
        values = bkd.zeros((2, 9))  # (nqoi, nsamples)
        interp.set_values(values)
        assert interp.nqoi() == 2

    def test_set_values_wrong_shape(self, bkd) -> None:
        """Test set_values with wrong shape raises ValueError."""
        interp = self._make_interpolant(bkd, 2, 3)
        values = bkd.zeros((2, 5))  # Wrong: should be 9 samples
        with pytest.raises(ValueError):
            interp.set_values(values)

    def test_call_without_values(self, bkd) -> None:
        """Test that __call__ raises error without values set."""
        interp = self._make_interpolant(bkd, 2, 3)
        samples = bkd.asarray([[0.0], [0.0]])
        with pytest.raises(ValueError):
            interp(samples)

    # Interpolation tests

    def test_interpolates_polynomial(self, bkd) -> None:
        """Test exact interpolation of a polynomial."""
        interp = self._make_interpolant(bkd, 2, 5)  # 5 points can exactly interp degree 4
        samples = interp.get_samples()

        # f(x, y) = x^2 + y^2 (degree 2, should be exact)
        # values shape: (nqoi, nsamples) = (1, nsamples)
        values = samples[0:1, :] ** 2 + samples[1:2, :] ** 2
        interp.set_values(values)

        # Evaluate at random points
        test_pts = bkd.asarray(
            [
                [0.0, 0.3, -0.5, 0.7],
                [0.0, -0.2, 0.4, 0.1],
            ]
        )
        result = interp(test_pts)  # (nqoi, npoints)
        expected = test_pts[0:1, :] ** 2 + test_pts[1:2, :] ** 2  # (1, npoints)

        bkd.assert_allclose(result, expected, rtol=1e-10, atol=1e-14)

    @pytest.mark.slow_on("TorchBkd")
    def test_interpolates_at_nodes(self, bkd) -> None:
        """Test that interpolant reproduces values at nodes."""
        interp = self._make_interpolant(bkd, 2, 4)
        samples = interp.get_samples()
        nsamples = samples.shape[1]

        # Set arbitrary values with shape (nqoi, nsamples) = (1, nsamples)
        values = bkd.asarray([[float(i) for i in range(nsamples)]])
        interp.set_values(values)

        # Evaluate at nodes - result shape (nqoi, nsamples)
        result = interp(samples)
        bkd.assert_allclose(result, values, rtol=1e-10)

    # Derivative tests

    def test_jacobian_supported(self, bkd) -> None:
        """Test jacobian_supported returns True for LagrangeBasis1D."""
        interp = self._make_interpolant(bkd)
        assert interp.jacobian_supported()

    def test_hessian_supported(self, bkd) -> None:
        """Test hessian_supported returns True for LagrangeBasis1D."""
        interp = self._make_interpolant(bkd)
        assert interp.hessian_supported()

    def test_jacobian_shape(self, bkd) -> None:
        """Test jacobian returns correct shape."""
        interp = self._make_interpolant(bkd, 3, 4)
        samples = interp.get_samples()
        values = bkd.zeros((2, samples.shape[1]))  # (nqoi, nsamples)
        interp.set_values(values)

        sample = bkd.asarray([[0.0], [0.0], [0.0]])
        jac = interp.jacobian(sample)
        assert jac.shape == (2, 3)  # (nqoi, nvars)

    @pytest.mark.slow_on("TorchBkd")
    def test_hessian_shape(self, bkd) -> None:
        """Test hessian returns correct shape."""
        interp = self._make_interpolant(bkd, 3, 4)
        samples = interp.get_samples()
        values = bkd.zeros(
            (1, samples.shape[1])
        )  # (nqoi, nsamples), nqoi must be 1
        interp.set_values(values)

        sample = bkd.asarray([[0.0], [0.0], [0.0]])
        hess = interp.hessian(sample)
        assert hess.shape == (3, 3)  # (nvars, nvars)

    def test_hvp_shape(self, bkd) -> None:
        """Test hvp returns correct shape."""
        interp = self._make_interpolant(bkd, 3, 4)
        samples = interp.get_samples()
        values = bkd.zeros(
            (1, samples.shape[1])
        )  # (nqoi, nsamples), nqoi must be 1
        interp.set_values(values)

        sample = bkd.asarray([[0.0], [0.0], [0.0]])
        vec = bkd.asarray([[1.0], [0.0], [0.0]])
        hvp_result = interp.hvp(sample, vec)
        assert hvp_result.shape == (3, 1)  # (nvars, 1)

    def test_whvp_shape(self, bkd) -> None:
        """Test whvp returns correct shape."""
        interp = self._make_interpolant(bkd, 3, 4)
        samples = interp.get_samples()
        values = bkd.zeros((2, samples.shape[1]))  # (nqoi, nsamples)
        interp.set_values(values)

        sample = bkd.asarray([[0.0], [0.0], [0.0]])
        vec = bkd.asarray([[1.0], [0.0], [0.0]])
        weights = bkd.asarray([[0.5], [0.5]])
        whvp_result = interp.whvp(sample, vec, weights)
        assert whvp_result.shape == (3, 1)  # (nvars, 1)

    def test_derivatives_with_checker(self, bkd) -> None:
        """Test jacobian and hessian using DerivativeChecker.

        Uses the standard DerivativeChecker from the interface module
        as per CLAUDE.md conventions.
        """
        interp = self._make_interpolant(bkd, 2, 5)
        samples = interp.get_samples()

        # f(x, y) = x^2 * y (has non-trivial jacobian and hessian)
        # values shape: (nqoi, nsamples) = (1, nsamples)
        values = samples[0:1, :] ** 2 * samples[1:2, :]
        interp.set_values(values)

        # Use DerivativeChecker to validate derivatives
        checker = DerivativeChecker(interp)
        sample = bkd.asarray([[0.3], [-0.4]])
        errors = checker.check_derivatives(sample)

        # Check jacobian error ratio
        jac_error = checker.error_ratio(errors[0])
        assert float(jac_error) < 5e-6

        # Check hessian error ratio (via hvp)
        hess_error = checker.error_ratio(errors[1])
        assert float(hess_error) < 5e-6

    def test_whvp_derivatives_with_checker(self, bkd) -> None:
        """Test whvp derivatives using DerivativeChecker for multi-QoI.

        For multi-QoI functions, DerivativeChecker uses whvp with weights.
        """
        interp = self._make_interpolant(bkd, 2, 5)
        samples = interp.get_samples()

        # Two QoIs - values shape: (nqoi, nsamples) = (2, nsamples)
        q1 = samples[0:1, :] ** 2  # (1, nsamples)
        q2 = samples[1:2, :] ** 2  # (1, nsamples)
        values = bkd.vstack([q1, q2])  # (2, nsamples)
        interp.set_values(values)

        # Use DerivativeChecker with weights for multi-QoI
        checker = DerivativeChecker(interp)
        sample = bkd.asarray([[0.3], [-0.4]])
        weights = bkd.asarray([[0.6, 0.4]])  # Shape (1, nqoi)
        errors = checker.check_derivatives(sample, weights=weights)

        # Check jacobian error ratio
        jac_error = checker.error_ratio(errors[0])
        assert float(jac_error) < 5e-6

        # Check whvp error ratio
        whvp_error = checker.error_ratio(errors[1])
        assert float(whvp_error) < 5e-6

    def test_hvp_matches_hessian(self, bkd) -> None:
        """Test that hvp matches H @ v from full hessian."""
        interp = self._make_interpolant(bkd, 2, 5)
        samples = interp.get_samples()

        # values shape: (nqoi, nsamples) = (1, nsamples), nqoi must be 1 for hessian/hvp
        values = samples[0:1, :] ** 2 * samples[1:2, :]
        interp.set_values(values)

        sample = bkd.asarray([[0.3], [-0.4]])
        vec = bkd.asarray([[0.7], [-0.3]])

        hvp_result = interp.hvp(sample, vec)
        hess = interp.hessian(sample)
        expected = hess @ vec

        bkd.assert_allclose(hvp_result, expected, rtol=1e-10)

    def test_whvp_matches_weighted_hvp(self, bkd) -> None:
        """Test that whvp matches weighted sum of individual QoI hvps."""
        # Create an interpolant with 2 QoIs
        interp = self._make_interpolant(bkd, 2, 5)
        samples = interp.get_samples()

        # Two QoIs - values shape: (nqoi, nsamples) = (2, nsamples)
        q1 = samples[0:1, :] ** 2  # (1, nsamples)
        q2 = samples[1:2, :] ** 2  # (1, nsamples)
        values = bkd.vstack([q1, q2])  # (2, nsamples)
        interp.set_values(values)

        sample = bkd.asarray([[0.3], [-0.4]])
        vec = bkd.asarray([[0.7], [-0.3]])
        weights = bkd.asarray([[0.6], [0.4]])

        whvp_result = interp.whvp(sample, vec, weights)

        # Create separate single-QoI interpolants to compute individual hvps
        interp0 = self._make_interpolant(bkd, 2, 5)
        interp0.set_values(q1)  # (1, nsamples)
        hvp0 = interp0.hvp(sample, vec)

        interp1 = self._make_interpolant(bkd, 2, 5)
        interp1.set_values(q2)  # (1, nsamples)
        hvp1 = interp1.hvp(sample, vec)

        expected = 0.6 * hvp0 + 0.4 * hvp1

        bkd.assert_allclose(whvp_result, expected, rtol=1e-10)

    def test_hessian_rejects_multi_qoi(self, bkd) -> None:
        """Test that hessian raises ValueError when nqoi > 1."""
        interp = self._make_interpolant(bkd, 2, 4)
        samples = interp.get_samples()
        values = bkd.zeros((2, samples.shape[1]))  # nqoi=2
        interp.set_values(values)

        sample = bkd.asarray([[0.0], [0.0]])
        with pytest.raises(ValueError) as ctx:
            interp.hessian(sample)
        assert "nqoi=1" in str(ctx.value)

    def test_hvp_rejects_multi_qoi(self, bkd) -> None:
        """Test that hvp raises ValueError when nqoi > 1."""
        interp = self._make_interpolant(bkd, 2, 4)
        samples = interp.get_samples()
        values = bkd.zeros((2, samples.shape[1]))  # nqoi=2
        interp.set_values(values)

        sample = bkd.asarray([[0.0], [0.0]])
        vec = bkd.asarray([[1.0], [0.0]])
        with pytest.raises(ValueError) as ctx:
            interp.hvp(sample, vec)
        assert "nqoi=1" in str(ctx.value)

    def test_repr(self, bkd) -> None:
        """Test string representation."""
        interp = self._make_interpolant(bkd, 2, 3)
        repr_str = repr(interp)
        assert "TensorProductInterpolant" in repr_str
        assert "nvars=2" in repr_str
