"""Tests for ConditionalGaussian distribution.

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
from pyapprox.probability.conditional.gaussian import ConditionalGaussian
from pyapprox.probability.univariate import UniformMarginal
from pyapprox.probability.univariate.gaussian import GaussianMarginal
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


def _create_conditional_gaussian(
    bkd, nvars: int, max_level: int = 2
) -> ConditionalGaussian:
    """Helper to create a ConditionalGaussian with polynomial parameter
    functions."""
    mean_func = _create_basis_expansion(bkd, nvars, max_level, nqoi=1)
    log_stdev_func = _create_basis_expansion(bkd, nvars, max_level, nqoi=1)

    # Set random coefficients
    np.random.seed(42)
    mean_func.set_coefficients(bkd.asarray(np.random.randn(mean_func.nterms(), 1)))
    # Ensure log_stdev is reasonable (e.g., -1 to 1)
    log_stdev_func.set_coefficients(
        bkd.asarray(0.5 * np.random.randn(log_stdev_func.nterms(), 1))
    )

    return ConditionalGaussian(mean_func, log_stdev_func, bkd)


class TestConditionalGaussian:
    """Test ConditionalGaussian distribution."""

    def test_basic_properties(self, bkd):
        """Test basic properties of ConditionalGaussian."""
        cond = _create_conditional_gaussian(bkd, nvars=2)

        assert cond.nvars() == 2
        assert cond.nqoi() == 1
        assert hasattr(cond, "hyp_list")
        assert hasattr(cond, "logpdf_jacobian_wrt_x")
        assert hasattr(cond, "logpdf_jacobian_wrt_params")

    def test_logpdf_shape(self, bkd):
        """Test logpdf output shape."""
        cond = _create_conditional_gaussian(bkd, nvars=2)

        nsamples = 5
        x = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, nsamples)))
        y = bkd.asarray(np.random.randn(1, nsamples))

        log_probs = cond.logpdf(x, y)
        assert log_probs.shape == (1, nsamples)

    def test_logpdf_values_against_scipy(self, bkd):
        """Test logpdf values match scipy for constant parameter functions."""
        # Create constant functions (degree-0 polynomial)
        mean_func = _create_basis_expansion(bkd, nvars=1, max_level=0, nqoi=1)
        log_stdev_func = _create_basis_expansion(bkd, nvars=1, max_level=0, nqoi=1)

        # Set constant values: mean=1.5, log_stdev=log(0.8)
        mean_val = 1.5
        log_stdev_val = np.log(0.8)
        mean_func.set_coefficients(bkd.asarray([[mean_val]]))
        log_stdev_func.set_coefficients(bkd.asarray([[log_stdev_val]]))

        cond = ConditionalGaussian(mean_func, log_stdev_func, bkd)

        # Test at several points
        np.random.seed(42)
        nsamples = 10
        x = bkd.asarray(np.random.uniform(-0.9, 0.9, (1, nsamples)))
        y = bkd.asarray(np.random.randn(1, nsamples) * 0.8 + 1.5)

        log_probs = cond.logpdf(x, y)

        # Compare with scipy
        scipy_log_probs = stats.norm.logpdf(
            bkd.to_numpy(y[0, :]), loc=mean_val, scale=np.exp(log_stdev_val)
        )

        bkd.assert_allclose(log_probs[0, :], bkd.asarray(scipy_log_probs), rtol=1e-10)

    def test_rvs_shape(self, bkd):
        """Test rvs output shape."""
        cond = _create_conditional_gaussian(bkd, nvars=2)

        nsamples = 10
        x = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, nsamples)))

        np.random.seed(42)
        samples = cond.rvs(x)
        assert samples.shape == (1, nsamples)

    def test_rvs_statistics(self, bkd):
        """Test rvs generates samples with correct mean and std."""
        # Create constant functions
        mean_func = _create_basis_expansion(bkd, nvars=1, max_level=0, nqoi=1)
        log_stdev_func = _create_basis_expansion(bkd, nvars=1, max_level=0, nqoi=1)

        mean_val = 2.0
        stdev_val = 0.5
        mean_func.set_coefficients(bkd.asarray([[mean_val]]))
        log_stdev_func.set_coefficients(bkd.asarray([[np.log(stdev_val)]]))

        cond = ConditionalGaussian(mean_func, log_stdev_func, bkd)

        # Generate many samples for all same conditioning point
        nsamples = 10000
        x = bkd.zeros((1, nsamples))  # All same x

        np.random.seed(42)
        samples = cond.rvs(x)

        bkd.assert_allclose(
            bkd.asarray([bkd.mean(samples)]),
            bkd.asarray([mean_val]),
            atol=0.05,
        )
        bkd.assert_allclose(
            bkd.asarray([bkd.std(samples)]),
            bkd.asarray([stdev_val]),
            atol=0.05,
        )

    def test_logpdf_jacobian_wrt_x_derivative_checker(self, bkd):
        """Test logpdf_jacobian_wrt_x using DerivativeChecker."""
        cond = _create_conditional_gaussian(bkd, nvars=2, max_level=2)

        # Fix a y value
        np.random.seed(42)
        y = bkd.asarray(np.random.randn(1, 1))

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
        assert float(jac_error) < 5e-6

    def test_logpdf_jacobian_wrt_params_derivative_checker(self, bkd):
        """Test logpdf_jacobian_wrt_params using DerivativeChecker."""
        cond = _create_conditional_gaussian(bkd, nvars=2, max_level=2)

        # Fix x and y values
        np.random.seed(42)
        x = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))
        y = bkd.asarray(np.random.randn(1, 1))

        nactive = cond.nparams()

        # Wrap as function of params
        def fun(params):
            cond.hyp_list().set_active_values(params[:, 0])
            # Sync both funcs from hyp_list
            cond._mean_func._sync_from_hyp_list()
            cond._log_stdev_func._sync_from_hyp_list()
            return cond.logpdf(x, y).T  # (1, 1)

        def jacobian_func(params):
            cond.hyp_list().set_active_values(params[:, 0])
            cond._mean_func._sync_from_hyp_list()
            cond._log_stdev_func._sync_from_hyp_list()
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
        assert float(jac_error) < 5e-6

    def test_reparameterize_shape(self, bkd):
        """Test reparameterize output shape."""
        cond = _create_conditional_gaussian(bkd, nvars=2)

        nsamples = 5
        np.random.seed(42)
        x = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, nsamples)))
        base = bkd.asarray(np.random.randn(1, nsamples))

        z = cond.reparameterize(x, base)
        assert z.shape == (1, nsamples)

    def test_reparameterize_matches_manual(self, bkd):
        """Test reparameterize matches mean + exp(log_s) * base_samples."""
        mean_func = _create_basis_expansion(bkd, nvars=1, max_level=0, nqoi=1)
        log_stdev_func = _create_basis_expansion(bkd, nvars=1, max_level=0, nqoi=1)

        mean_val = 1.5
        log_stdev_val = np.log(0.8)
        mean_func.set_coefficients(bkd.asarray([[mean_val]]))
        log_stdev_func.set_coefficients(bkd.asarray([[log_stdev_val]]))
        cond = ConditionalGaussian(mean_func, log_stdev_func, bkd)

        np.random.seed(42)
        x = bkd.asarray(np.random.uniform(-0.9, 0.9, (1, 10)))
        base = bkd.asarray(np.random.randn(1, 10))

        z = cond.reparameterize(x, base)
        expected = mean_val + np.exp(log_stdev_val) * bkd.to_numpy(base)
        bkd.assert_allclose(z, bkd.asarray(expected), rtol=1e-12)

    def test_kl_divergence_vs_analytical(self, bkd):
        """Test KL matches GaussianMarginal.kl_divergence for constant params."""
        mean_val, stdev_val = 1.5, 0.8
        mean_func = _create_basis_expansion(bkd, nvars=1, max_level=0, nqoi=1)
        log_stdev_func = _create_basis_expansion(bkd, nvars=1, max_level=0, nqoi=1)
        mean_func.set_coefficients(bkd.asarray([[mean_val]]))
        log_stdev_func.set_coefficients(bkd.asarray([[np.log(stdev_val)]]))
        cond = ConditionalGaussian(mean_func, log_stdev_func, bkd)

        prior = GaussianMarginal(0.0, 1.0, bkd)
        x = bkd.asarray([[0.0, 0.5, -0.5]])  # x doesn't matter for degree-0
        kl = cond.kl_divergence(x, prior)
        assert kl.shape == (1, 3)

        # Reference: GaussianMarginal KL
        q = GaussianMarginal(mean_val, stdev_val, bkd)
        kl_ref = q.kl_divergence(prior)
        # All samples should give same KL since params are constant
        bkd.assert_allclose(
            kl,
            bkd.reshape(bkd.asarray([float(bkd.to_numpy(kl_ref))]), (1, 1))
            * bkd.ones((1, 3)),
            rtol=1e-12,
        )

    def test_kl_self_is_zero(self, bkd):
        """KL(q || q) = 0 when params match the prior."""
        mean_val, stdev_val = 2.0, 1.5
        mean_func = _create_basis_expansion(bkd, nvars=1, max_level=0, nqoi=1)
        log_stdev_func = _create_basis_expansion(bkd, nvars=1, max_level=0, nqoi=1)
        mean_func.set_coefficients(bkd.asarray([[mean_val]]))
        log_stdev_func.set_coefficients(bkd.asarray([[np.log(stdev_val)]]))
        cond = ConditionalGaussian(mean_func, log_stdev_func, bkd)

        prior = GaussianMarginal(mean_val, stdev_val, bkd)
        x = bkd.asarray([[0.0, 0.5]])
        kl = cond.kl_divergence(x, prior)
        bkd.assert_allclose(kl, bkd.zeros((1, 2)), atol=1e-12)

    def test_base_distribution_is_standard_normal(self, bkd):
        """base_distribution returns N(0, 1)."""
        cond = _create_conditional_gaussian(bkd, nvars=1)

        base = cond.base_distribution()
        assert isinstance(base, GaussianMarginal)
        assert base.mean_value() == 0.0
        assert base.std() == 1.0

    def test_reparameterize_jacobian_wrt_params(self, bkd):
        """Test reparameterize_jacobian_wrt_params using DerivativeChecker."""
        cond = _create_conditional_gaussian(bkd, nvars=2, max_level=2)

        np.random.seed(42)
        x = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))
        base = bkd.asarray(np.random.randn(1, 1))

        nactive = cond.nparams()

        def fun(params):
            cond.hyp_list().set_active_values(params[:, 0])
            return cond.reparameterize(x, base).T  # (1, 1)

        def jacobian_func(params):
            cond.hyp_list().set_active_values(params[:, 0])
            jac = cond.reparameterize_jacobian_wrt_params(x, base)
            return jac[0]  # (1, nactive)

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
        assert float(jac_error) < 5e-6

    def test_logpdf_after_set_active_values(self, bkd):
        """logpdf uses updated values after hyp_list().set_active_values()."""
        cond = _create_conditional_gaussian(bkd, nvars=1, max_level=0)

        x = bkd.asarray([[0.0]])
        y = bkd.asarray([[1.0]])

        logp1 = cond.logpdf(x, y)

        # Change mean coefficient to shift the distribution
        old_values = cond.hyp_list().get_active_values()
        new_values = bkd.asarray([5.0, old_values[1]])
        cond.hyp_list().set_active_values(new_values)

        logp2 = cond.logpdf(x, y)

        # logpdf should change because mean changed -- verify they differ
        diff = bkd.abs(logp2 - logp1)
        assert float(bkd.to_numpy(diff[0, 0])) > 0.1

    def test_validation_errors(self, bkd):
        """Test input validation raises appropriate errors."""
        cond = _create_conditional_gaussian(bkd, nvars=2)

        # x wrong shape (1D)
        x_1d = bkd.asarray(np.random.randn(2))
        y = bkd.asarray(np.random.randn(1, 1))
        with pytest.raises(ValueError):
            cond.logpdf(x_1d, y)

        # y wrong shape (1D)
        x = bkd.asarray(np.random.randn(2, 1))
        y_1d = bkd.asarray(np.random.randn(1))
        with pytest.raises(ValueError):
            cond.logpdf(x, y_1d)

        # Mismatched sample counts
        x = bkd.asarray(np.random.randn(2, 3))
        y = bkd.asarray(np.random.randn(1, 5))
        with pytest.raises(ValueError):
            cond.logpdf(x, y)

    def test_logpdf_jacobian_wrt_params_autograd(self, bkd):
        """Verify logpdf_jacobian_wrt_params matches torch autograd."""
        if not isinstance(bkd, NumpyBkd):
            import torch
            from torch.autograd.functional import jacobian as torch_jacobian

            cond = _create_conditional_gaussian(bkd, nvars=2, max_level=2)

            # Fix x and y values
            np.random.seed(42)
            x = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 3)))
            y = bkd.asarray(np.random.randn(1, 3))

            # Get analytical jacobian
            analytical_jac = cond.logpdf_jacobian_wrt_params(x, y)

            # Get autograd jacobian
            def logpdf_from_params(params: torch.Tensor) -> torch.Tensor:
                cond.hyp_list().set_active_values(params)
                cond._mean_func._sync_from_hyp_list()
                cond._log_stdev_func._sync_from_hyp_list()
                return cond.logpdf(x, y).flatten()

            params = cond.hyp_list().get_active_values()
            autograd_jac = torch_jacobian(logpdf_from_params, params)

            bkd.assert_allclose(analytical_jac, autograd_jac, rtol=1e-10)
        else:
            pytest.skip("Torch-only test")
