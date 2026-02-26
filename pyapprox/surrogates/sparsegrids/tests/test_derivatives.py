"""Dual-backend tests for sparse grid derivatives using DerivativeChecker.

Tests run on both NumPy and PyTorch backends using the base class pattern.
"""

from typing import List

import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.indices import LinearGrowthRule
from pyapprox.surrogates.sparsegrids.basis_factory import (
    BasisFactoryProtocol,
    GaussLagrangeFactory,
)
from pyapprox.surrogates.sparsegrids.isotropic_fitter import (
    IsotropicSparseGridFitter,
)
from pyapprox.surrogates.sparsegrids.subspace_factory import (
    TensorProductSubspaceFactory,
)


class TestSparseGridDerivatives:
    """Tests for sparse grid derivatives using DerivativeChecker."""

    def _build_surrogate(self, nvars, level, func, bkd):
        """Build an isotropic sparse grid surrogate for the given function."""
        marginal = UniformMarginal(-1.0, 1.0, bkd)
        factories: List[BasisFactoryProtocol] = [
            GaussLagrangeFactory(marginal, bkd) for _ in range(nvars)
        ]
        growth = LinearGrowthRule(scale=1, shift=1)
        tp_factory = TensorProductSubspaceFactory(bkd, factories, growth)
        fitter = IsotropicSparseGridFitter(bkd, tp_factory, level)
        samples = fitter.get_samples()
        values = func(samples)
        result = fitter.fit(values)
        return result.surrogate

    @pytest.mark.slow_on("TorchBkd")
    def test_jacobian_linear_function(self, bkd) -> None:
        """Test Jacobian of linear function is constant."""

        def func(s):
            return bkd.reshape(s[0, :] + 2 * s[1, :], (1, -1))

        surrogate = self._build_surrogate(2, 2, func, bkd)

        test_pt = bkd.asarray([[0.3], [0.4]])
        checker = DerivativeChecker(surrogate)
        errors = checker.check_derivatives(test_pt, verbosity=0)

        # Jacobian should be [1, 2]
        jac = surrogate.jacobian(test_pt)
        expected_jac = bkd.asarray([[1.0, 2.0]])
        bkd.assert_allclose(jac, expected_jac, rtol=1e-6)

        jac_error = float(checker.error_ratio(errors[0]).item())
        assert jac_error < 1e-6

    def test_jacobian_quadratic_function(self, bkd) -> None:
        """Test Jacobian of quadratic function."""

        def func(s):
            x, y = s[0, :], s[1, :]
            return bkd.reshape(x**2 + x * y, (1, -1))

        surrogate = self._build_surrogate(2, 3, func, bkd)

        test_pt = bkd.asarray([[0.3], [0.4]])
        checker = DerivativeChecker(surrogate)
        errors = checker.check_derivatives(test_pt, verbosity=0)

        # Jacobian at (0.3, 0.4) should be [2*0.3 + 0.4, 0.3] = [1.0, 0.3]
        jac = surrogate.jacobian(test_pt)
        expected_jac = bkd.asarray([[1.0, 0.3]])
        bkd.assert_allclose(jac, expected_jac, rtol=1e-6)

        jac_error = float(checker.error_ratio(errors[0]).item())
        assert jac_error < 1e-6

    def test_jacobian_3d_function(self, bkd) -> None:
        """Test Jacobian of 3D function."""

        def func(s):
            return bkd.reshape(s[0, :] + s[1, :] + s[2, :], (1, -1))

        surrogate = self._build_surrogate(3, 2, func, bkd)

        test_pt = bkd.asarray([[0.1], [0.2], [0.3]])
        checker = DerivativeChecker(surrogate)
        errors = checker.check_derivatives(test_pt, verbosity=0)

        # Jacobian should be [1, 1, 1]
        jac = surrogate.jacobian(test_pt)
        expected_jac = bkd.asarray([[1.0, 1.0, 1.0]])
        bkd.assert_allclose(jac, expected_jac, rtol=1e-6)

        jac_error = float(checker.error_ratio(errors[0]).item())
        assert jac_error < 1e-6

    def test_derivative_checker_passes(self, bkd) -> None:
        """Test that DerivativeChecker passes for sparse grid function."""

        def func(s):
            return bkd.reshape(s[0, :] ** 2 + s[1, :] ** 2, (1, -1))

        surrogate = self._build_surrogate(2, 3, func, bkd)

        for x_val, y_val in [(0.0, 0.0), (0.3, 0.4), (-0.5, 0.2)]:
            test_pt = bkd.asarray([[x_val], [y_val]])
            checker = DerivativeChecker(surrogate)
            errors = checker.check_derivatives(test_pt, verbosity=0)

            jac_error = float(checker.error_ratio(errors[0]).item())
            assert jac_error < 1e-6

    def test_hessian_vector_product(self, bkd) -> None:
        """Test HVP computation."""

        def func(s):
            return bkd.reshape(s[0, :] ** 2 + s[1, :] ** 2, (1, -1))

        surrogate = self._build_surrogate(2, 3, func, bkd)

        test_pt = bkd.asarray([[0.3], [0.4]])
        vec = bkd.asarray([[1.0], [0.0]])

        hvp = surrogate.hvp(test_pt, vec)

        # Hessian is [[2, 0], [0, 2]]
        # HVP with [1, 0] should give [2, 0]
        expected_hvp = bkd.asarray([[2.0], [0.0]])
        bkd.assert_allclose(hvp, expected_hvp, rtol=1e-6, atol=1e-14)

    def test_weighted_hvp(self, bkd) -> None:
        """Test weighted HVP computation."""

        def func(s):
            return bkd.reshape(s[0, :] ** 2 + s[1, :] ** 2, (1, -1))

        surrogate = self._build_surrogate(2, 3, func, bkd)

        test_pt = bkd.asarray([[0.3], [0.4]])
        vec = bkd.asarray([[1.0], [1.0]])
        weights = bkd.asarray([[0.5]])

        whvp = surrogate.whvp(test_pt, vec, weights)

        # Hessian is [[2, 0], [0, 2]]
        # WHVP with [1, 1] and weight 0.5 should give 0.5 * [2, 2] = [1, 1]
        expected_whvp = bkd.asarray([[1.0], [1.0]])
        bkd.assert_allclose(whvp, expected_whvp, rtol=1e-6)

    def test_hessian_via_derivative_checker(self, bkd) -> None:
        """Test Hessian via DerivativeChecker errors[1] for nqoi=1."""

        def func(s):
            x, y = s[0, :], s[1, :]
            return bkd.reshape(x**2 + x * y + y**2, (1, -1))

        surrogate = self._build_surrogate(2, 3, func, bkd)

        test_pt = bkd.asarray([[0.3], [0.4]])
        checker = DerivativeChecker(surrogate)
        errors = checker.check_derivatives(test_pt, verbosity=0)

        assert len(errors) == 2
        hessian_error = float(checker.error_ratio(errors[1]).item())
        assert hessian_error < 1e-6

    def test_whvp_via_derivative_checker(self, bkd) -> None:
        """Test WHVP via DerivativeChecker for nqoi=2 with weights."""

        def func(s):
            x, y = s[0, :], s[1, :]
            return bkd.stack([x**2 + y, x * y], axis=0)

        surrogate = self._build_surrogate(2, 3, func, bkd)

        test_pt = bkd.asarray([[0.3], [0.4]])
        weights = bkd.asarray([[0.6, 0.4]])

        checker = DerivativeChecker(surrogate)
        errors = checker.check_derivatives(test_pt, verbosity=0, weights=weights)

        assert len(errors) == 2
        whvp_error = float(checker.error_ratio(errors[1]).item())
        assert whvp_error < 1e-6
