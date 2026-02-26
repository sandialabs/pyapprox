"""Tests for ConditionalGamma distribution.

Tests validate:
1. logpdf values against scipy
2. rvs sampling statistics
3. jacobian_wrt_x with DerivativeChecker
4. jacobian_wrt_params with DerivativeChecker
5. Torch autograd compatibility
"""

import pytest

import numpy as np
from scipy import stats

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.probability.conditional.gamma import ConditionalGamma
from pyapprox.probability.univariate import UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.indices import compute_hyperbolic_indices
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.util.backends.numpy import NumpyBkd


def _create_basis_expansion(
    bkd, nvars: int, max_level: int, nqoi: int = 1
) -> BasisExpansion:
    """Helper to create a Legendre basis expansion."""
    marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
    bases_1d = create_bases_1d(marginals, bkd)
    indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
    basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
    return BasisExpansion(basis, bkd, nqoi=nqoi)


def _create_conditional_gamma(
    bkd, nvars: int, max_level: int = 2
) -> ConditionalGamma:
    """Helper to create a ConditionalGamma with polynomial parameter functions."""
    log_shape_func = _create_basis_expansion(bkd, nvars, max_level, nqoi=1)
    log_scale_func = _create_basis_expansion(bkd, nvars, max_level, nqoi=1)

    # Set random coefficients, but ensure reasonable log values (e.g., 0 to 1)
    # This gives shape, scale in range (1, 2.7)
    np.random.seed(42)
    log_shape_func.set_coefficients(
        bkd.asarray(0.3 + 0.3 * np.random.randn(log_shape_func.nterms(), 1))
    )
    log_scale_func.set_coefficients(
        bkd.asarray(0.3 + 0.3 * np.random.randn(log_scale_func.nterms(), 1))
    )

    return ConditionalGamma(log_shape_func, log_scale_func, bkd)


class TestConditionalGamma:
    """Test ConditionalGamma distribution."""

    def test_basic_properties(self, bkd):
        """Test basic properties of ConditionalGamma."""
        cond = _create_conditional_gamma(bkd, nvars=2)

        assert cond.nvars() == 2
        assert cond.nqoi() == 1
        assert hasattr(cond, "hyp_list")
        assert hasattr(cond, "logpdf_jacobian_wrt_x")
        assert hasattr(cond, "logpdf_jacobian_wrt_params")

    def test_logpdf_shape(self, bkd):
        """Test logpdf output shape."""
        cond = _create_conditional_gamma(bkd, nvars=2)

        nsamples = 5
        x = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, nsamples)))
        y = bkd.asarray(np.random.uniform(0.5, 3.0, (1, nsamples)))  # Must be > 0

        log_probs = cond.logpdf(x, y)
        assert log_probs.shape == (1, nsamples)

    def test_logpdf_values_against_scipy(self, bkd):
        """Test logpdf values match scipy for constant parameter functions."""
        # Create constant functions (degree-0 polynomial)
        log_shape_func = _create_basis_expansion(bkd, nvars=1, max_level=0, nqoi=1)
        log_scale_func = _create_basis_expansion(bkd, nvars=1, max_level=0, nqoi=1)

        # Set constant values: shape=2.0, scale=1.5
        shape_val = 2.0
        scale_val = 1.5
        log_shape_func.set_coefficients(bkd.asarray([[np.log(shape_val)]]))
        log_scale_func.set_coefficients(bkd.asarray([[np.log(scale_val)]]))

        cond = ConditionalGamma(log_shape_func, log_scale_func, bkd)

        # Test at several points
        np.random.seed(42)
        nsamples = 10
        x = bkd.asarray(np.random.uniform(-0.9, 0.9, (1, nsamples)))
        y = bkd.asarray(np.random.gamma(shape_val, scale_val, (1, nsamples)))

        log_probs = cond.logpdf(x, y)

        # Compare with scipy
        scipy_log_probs = stats.gamma.logpdf(
            bkd.to_numpy(y[0, :]), a=shape_val, scale=scale_val
        )

        bkd.assert_allclose(log_probs[0, :], bkd.asarray(scipy_log_probs), rtol=1e-10)

    def test_rvs_shape(self, bkd):
        """Test rvs output shape."""
        cond = _create_conditional_gamma(bkd, nvars=2)

        nsamples = 10
        x = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, nsamples)))

        np.random.seed(42)
        samples = cond.rvs(x)
        assert samples.shape == (1, nsamples)
        # All samples should be > 0
        samples_np = bkd.to_numpy(samples)
        assert np.all(samples_np > 0)

    def test_rvs_statistics(self, bkd):
        """Test rvs generates samples with correct mean."""
        # Create constant functions
        log_shape_func = _create_basis_expansion(bkd, nvars=1, max_level=0, nqoi=1)
        log_scale_func = _create_basis_expansion(bkd, nvars=1, max_level=0, nqoi=1)

        shape_val = 2.0
        scale_val = 1.5
        log_shape_func.set_coefficients(bkd.asarray([[np.log(shape_val)]]))
        log_scale_func.set_coefficients(bkd.asarray([[np.log(scale_val)]]))

        cond = ConditionalGamma(log_shape_func, log_scale_func, bkd)

        # Generate many samples for all same conditioning point
        nsamples = 10000
        x = bkd.zeros((1, nsamples))

        np.random.seed(42)
        samples = cond.rvs(x)

        sample_mean = float(bkd.to_numpy(bkd.mean(samples)))
        expected_mean = shape_val * scale_val

        assert abs(sample_mean - expected_mean) < 0.1

    def test_logpdf_jacobian_wrt_x_derivative_checker(self, bkd):
        """Test logpdf_jacobian_wrt_x using DerivativeChecker."""
        cond = _create_conditional_gamma(bkd, nvars=2, max_level=2)

        # Fix a y value
        np.random.seed(42)
        y = bkd.asarray([[1.5]])  # Must be > 0

        # Wrap as function of x
        def fun(x):
            return cond.logpdf(x, y).T  # (1, nqoi=1)

        def jacobian_func(x):
            return cond.logpdf_jacobian_wrt_x(x, y)  # (1, nvars)

        function_obj = FunctionWithJacobianFromCallable(
            nqoi=1,
            nvars=cond.nvars(),
            fun=fun,
            jacobian=jacobian_func,
            bkd=bkd,
        )

        checker = DerivativeChecker(function_obj)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        jac_error = checker.error_ratio(errors[0])
        assert float(jac_error) < 1e-6

    def test_logpdf_jacobian_wrt_params_derivative_checker(self, bkd):
        """Test logpdf_jacobian_wrt_params using DerivativeChecker."""
        cond = _create_conditional_gamma(bkd, nvars=2, max_level=2)

        # Fix x and y values
        np.random.seed(42)
        x = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))
        y = bkd.asarray([[1.5]])  # Must be > 0

        nactive = cond.nparams()

        # Wrap as function of params
        def fun(params):
            cond.hyp_list().set_active_values(params[:, 0])
            # Sync both funcs from hyp_list
            cond._log_shape_func._sync_from_hyp_list()
            cond._log_scale_func._sync_from_hyp_list()
            return cond.logpdf(x, y).T  # (1, 1)

        def jacobian_func(params):
            cond.hyp_list().set_active_values(params[:, 0])
            cond._log_shape_func._sync_from_hyp_list()
            cond._log_scale_func._sync_from_hyp_list()
            jac = cond.logpdf_jacobian_wrt_params(x, y)  # (1, nactive)
            return jac  # (nqoi=1, nactive)

        function_obj = FunctionWithJacobianFromCallable(
            nqoi=1,
            nvars=nactive,
            fun=fun,
            jacobian=jacobian_func,
            bkd=bkd,
        )

        checker = DerivativeChecker(function_obj)
        params = cond.hyp_list().get_active_values()
        sample_params = bkd.reshape(params, (nactive, 1))
        errors = checker.check_derivatives(sample_params, verbosity=0)

        jac_error = checker.error_ratio(errors[0])
        assert float(jac_error) < 1e-6

    def test_validation_errors(self, bkd):
        """Test input validation raises appropriate errors."""
        cond = _create_conditional_gamma(bkd, nvars=2)

        # x wrong shape (1D)
        x_1d = bkd.asarray(np.random.randn(2))
        y = bkd.asarray([[1.5]])
        with pytest.raises(ValueError):
            cond.logpdf(x_1d, y)

        # y wrong shape (1D)
        x = bkd.asarray(np.random.randn(2, 1))
        y_1d = bkd.asarray([1.5])
        with pytest.raises(ValueError):
            cond.logpdf(x, y_1d)

        # Mismatched sample counts
        x = bkd.asarray(np.random.randn(2, 3))
        y = bkd.asarray(np.random.uniform(0.5, 2.0, (1, 5)))
        with pytest.raises(ValueError):
            cond.logpdf(x, y)

    def test_logpdf_jacobian_wrt_params_autograd(self, bkd):
        """Verify logpdf_jacobian_wrt_params matches torch autograd."""
        if not isinstance(bkd, NumpyBkd):
            import torch
            from torch.autograd.functional import jacobian as torch_jacobian

            cond = _create_conditional_gamma(bkd, nvars=2, max_level=2)

            # Fix x and y values
            np.random.seed(42)
            x = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 3)))
            y = bkd.asarray(np.random.uniform(0.5, 2.0, (1, 3)))

            # Get analytical jacobian
            analytical_jac = cond.logpdf_jacobian_wrt_params(x, y)

            # Get autograd jacobian
            def logpdf_from_params(params: torch.Tensor) -> torch.Tensor:
                cond.hyp_list().set_active_values(params)
                cond._log_shape_func._sync_from_hyp_list()
                cond._log_scale_func._sync_from_hyp_list()
                return cond.logpdf(x, y).flatten()

            params = cond.hyp_list().get_active_values()
            autograd_jac = torch_jacobian(logpdf_from_params, params)

            bkd.assert_allclose(analytical_jac, autograd_jac, rtol=1e-10)
        else:
            pytest.skip("Torch-only test")
