"""Tests for ConditionalBeta distribution.

Tests validate:
1. logpdf values against scipy
2. rvs sampling statistics
3. jacobian_wrt_x with DerivativeChecker
4. jacobian_wrt_params with DerivativeChecker
5. Torch autograd compatibility
6. Bounds support with various [lb, ub] configurations
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray
from scipy import stats
from unittest_parametrize import ParametrizedTestCase, parametrize

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.probability.conditional.beta import ConditionalBeta
from pyapprox.probability.univariate import UniformMarginal
from pyapprox.probability.univariate.beta import BetaMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.indices import compute_hyperbolic_indices
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd

# Test configurations for bounds: (name, lb, ub)
COND_BETA_BOUNDS_CONFIGS = [
    ("canonical", 0.0, 1.0),
    ("scaled_02", 0.0, 2.0),
    ("shifted_25", 2.0, 5.0),
    ("negative", -1.0, 3.0),
    ("wide", -10.0, 10.0),
]


class TestConditionalBeta(Generic[Array], unittest.TestCase):
    """Test ConditionalBeta distribution."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def _create_basis_expansion(
        self, nvars: int, max_level: int, nqoi: int = 1
    ) -> BasisExpansion:
        """Helper to create a Legendre basis expansion."""
        bkd = self._bkd
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    def _create_conditional_beta(
        self, nvars: int, max_level: int = 2
    ) -> ConditionalBeta:
        """Helper to create a ConditionalBeta with polynomial parameter functions."""
        bkd = self._bkd
        log_alpha_func = self._create_basis_expansion(nvars, max_level, nqoi=1)
        log_beta_func = self._create_basis_expansion(nvars, max_level, nqoi=1)

        # Set random coefficients, but ensure reasonable log values (e.g., -1 to 2)
        # This gives alpha, beta in range (exp(-1), exp(2)) = (0.37, 7.4)
        np.random.seed(42)
        log_alpha_func.set_coefficients(
            bkd.asarray(0.5 + 0.5 * np.random.randn(log_alpha_func.nterms(), 1))
        )
        log_beta_func.set_coefficients(
            bkd.asarray(0.5 + 0.5 * np.random.randn(log_beta_func.nterms(), 1))
        )

        return ConditionalBeta(log_alpha_func, log_beta_func, bkd)

    def test_basic_properties(self):
        """Test basic properties of ConditionalBeta."""
        cond = self._create_conditional_beta(nvars=2)

        self.assertEqual(cond.nvars(), 2)
        self.assertEqual(cond.nqoi(), 1)
        self.assertTrue(hasattr(cond, "hyp_list"))
        self.assertTrue(hasattr(cond, "logpdf_jacobian_wrt_x"))
        self.assertTrue(hasattr(cond, "logpdf_jacobian_wrt_params"))

    def test_logpdf_shape(self):
        """Test logpdf output shape."""
        bkd = self._bkd
        cond = self._create_conditional_beta(nvars=2)

        nsamples = 5
        x = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, nsamples)))
        y = bkd.asarray(np.random.uniform(0.1, 0.9, (1, nsamples)))  # Must be in (0, 1)

        log_probs = cond.logpdf(x, y)
        self.assertEqual(log_probs.shape, (1, nsamples))

    def test_logpdf_values_against_scipy(self):
        """Test logpdf values match scipy for constant parameter functions."""
        bkd = self._bkd

        # Create constant functions (degree-0 polynomial)
        log_alpha_func = self._create_basis_expansion(nvars=1, max_level=0, nqoi=1)
        log_beta_func = self._create_basis_expansion(nvars=1, max_level=0, nqoi=1)

        # Set constant values: alpha=2.5, beta=3.0
        alpha_val = 2.5
        beta_val = 3.0
        log_alpha_func.set_coefficients(bkd.asarray([[np.log(alpha_val)]]))
        log_beta_func.set_coefficients(bkd.asarray([[np.log(beta_val)]]))

        cond = ConditionalBeta(log_alpha_func, log_beta_func, bkd)

        # Test at several points
        np.random.seed(42)
        nsamples = 10
        x = bkd.asarray(np.random.uniform(-0.9, 0.9, (1, nsamples)))
        y = bkd.asarray(np.random.uniform(0.1, 0.9, (1, nsamples)))

        log_probs = cond.logpdf(x, y)

        # Compare with scipy
        scipy_log_probs = stats.beta.logpdf(
            bkd.to_numpy(y[0, :]), a=alpha_val, b=beta_val
        )

        bkd.assert_allclose(log_probs[0, :], bkd.asarray(scipy_log_probs), rtol=1e-10)

    def test_rvs_shape(self):
        """Test rvs output shape."""
        bkd = self._bkd
        cond = self._create_conditional_beta(nvars=2)

        nsamples = 10
        x = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, nsamples)))

        np.random.seed(42)
        samples = cond.rvs(x)
        self.assertEqual(samples.shape, (1, nsamples))
        # All samples should be in (0, 1)
        samples_np = bkd.to_numpy(samples)
        self.assertTrue(np.all(samples_np > 0))
        self.assertTrue(np.all(samples_np < 1))

    def test_rvs_statistics(self):
        """Test rvs generates samples with correct mean."""
        bkd = self._bkd

        # Create constant functions
        log_alpha_func = self._create_basis_expansion(nvars=1, max_level=0, nqoi=1)
        log_beta_func = self._create_basis_expansion(nvars=1, max_level=0, nqoi=1)

        alpha_val = 2.0
        beta_val = 5.0
        log_alpha_func.set_coefficients(bkd.asarray([[np.log(alpha_val)]]))
        log_beta_func.set_coefficients(bkd.asarray([[np.log(beta_val)]]))

        cond = ConditionalBeta(log_alpha_func, log_beta_func, bkd)

        # Generate many samples for all same conditioning point
        nsamples = 10000
        x = bkd.zeros((1, nsamples))

        np.random.seed(42)
        samples = cond.rvs(x)

        bkd.assert_allclose(
            bkd.asarray([bkd.mean(samples)]),
            bkd.asarray([alpha_val / (alpha_val + beta_val)]),
            atol=0.02,
        )

    def test_logpdf_jacobian_wrt_x_derivative_checker(self):
        """Test logpdf_jacobian_wrt_x using DerivativeChecker."""
        bkd = self._bkd
        cond = self._create_conditional_beta(nvars=2, max_level=2)

        # Fix a y value
        np.random.seed(42)
        y = bkd.asarray([[0.4]])  # Must be in (0, 1)

        # Wrap as function of x
        def fun(x: Array) -> Array:
            return cond.logpdf(x, y).T  # (1, nqoi=1)

        def jacobian_func(x: Array) -> Array:
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
        self.assertLess(float(jac_error), 1e-6)

    def test_logpdf_jacobian_wrt_params_derivative_checker(self):
        """Test logpdf_jacobian_wrt_params using DerivativeChecker."""
        bkd = self._bkd
        cond = self._create_conditional_beta(nvars=2, max_level=2)

        # Fix x and y values
        np.random.seed(42)
        x = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))
        y = bkd.asarray([[0.4]])  # Must be in (0, 1)

        nactive = cond.nparams()

        # Wrap as function of params
        def fun(params: Array) -> Array:
            cond.hyp_list().set_active_values(params[:, 0])
            # Sync both funcs from hyp_list
            cond._log_alpha_func._sync_from_hyp_list()
            cond._log_beta_func._sync_from_hyp_list()
            return cond.logpdf(x, y).T  # (1, 1)

        def jacobian_func(params: Array) -> Array:
            cond.hyp_list().set_active_values(params[:, 0])
            cond._log_alpha_func._sync_from_hyp_list()
            cond._log_beta_func._sync_from_hyp_list()
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
        self.assertLess(float(jac_error), 5e-6)

    def test_reparameterize_shape(self):
        """Test reparameterize output shape."""
        bkd = self._bkd
        cond = self._create_conditional_beta(nvars=2)

        nsamples = 5
        np.random.seed(42)
        x = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, nsamples)))
        base = bkd.asarray(np.random.uniform(0.0, 1.0, (1, nsamples)))

        z = cond.reparameterize(x, base)
        self.assertEqual(z.shape, (1, nsamples))

    def test_reparameterize_in_bounds(self):
        """Test reparameterize produces samples in [lb, ub]."""
        bkd = self._bkd
        cond = self._create_conditional_beta(nvars=2)

        nsamples = 100
        np.random.seed(42)
        x = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, nsamples)))
        base = bkd.asarray(np.random.uniform(0.01, 0.99, (1, nsamples)))

        z = cond.reparameterize(x, base)
        z_np = bkd.to_numpy(z)
        self.assertTrue(np.all(z_np >= cond.lower()))
        self.assertTrue(np.all(z_np <= cond.upper()))

    def test_kl_divergence_vs_analytical(self):
        """Test KL matches BetaMarginal.kl_divergence for constant params."""
        bkd = self._bkd

        alpha_val, beta_val = 2.5, 3.0
        log_alpha_func = self._create_basis_expansion(nvars=1, max_level=0, nqoi=1)
        log_beta_func = self._create_basis_expansion(nvars=1, max_level=0, nqoi=1)
        log_alpha_func.set_coefficients(bkd.asarray([[np.log(alpha_val)]]))
        log_beta_func.set_coefficients(bkd.asarray([[np.log(beta_val)]]))
        cond = ConditionalBeta(log_alpha_func, log_beta_func, bkd)

        prior = BetaMarginal(1.0, 1.0, bkd)  # Uniform prior
        x = bkd.asarray([[0.0, 0.5, -0.5]])
        kl = cond.kl_divergence(x, prior)
        self.assertEqual(kl.shape, (1, 3))

        # Reference: BetaMarginal KL
        q = BetaMarginal(alpha_val, beta_val, bkd)
        kl_ref = q.kl_divergence(prior)
        kl_ref_val = float(bkd.to_numpy(bkd.atleast_1d(kl_ref))[0])
        bkd.assert_allclose(
            kl,
            bkd.reshape(bkd.asarray([kl_ref_val]), (1, 1)) * bkd.ones((1, 3)),
            rtol=1e-10,
        )

    def test_kl_self_is_zero(self):
        """KL(q || q) = 0 when params match the prior."""
        bkd = self._bkd

        alpha_val, beta_val = 2.0, 5.0
        log_alpha_func = self._create_basis_expansion(nvars=1, max_level=0, nqoi=1)
        log_beta_func = self._create_basis_expansion(nvars=1, max_level=0, nqoi=1)
        log_alpha_func.set_coefficients(bkd.asarray([[np.log(alpha_val)]]))
        log_beta_func.set_coefficients(bkd.asarray([[np.log(beta_val)]]))
        cond = ConditionalBeta(log_alpha_func, log_beta_func, bkd)

        prior = BetaMarginal(alpha_val, beta_val, bkd)
        x = bkd.asarray([[0.0, 0.5]])
        kl = cond.kl_divergence(x, prior)
        bkd.assert_allclose(kl, bkd.zeros((1, 2)), atol=1e-12)

    def test_base_distribution_is_uniform(self):
        """base_distribution returns U(0, 1)."""
        cond = self._create_conditional_beta(nvars=1)

        base = cond.base_distribution()
        self.assertIsInstance(base, UniformMarginal)

    def test_validation_errors(self):
        """Test input validation raises appropriate errors."""
        bkd = self._bkd
        cond = self._create_conditional_beta(nvars=2)

        # x wrong shape (1D)
        x_1d = bkd.asarray(np.random.randn(2))
        y = bkd.asarray([[0.5]])
        with self.assertRaises(ValueError):
            cond.logpdf(x_1d, y)

        # y wrong shape (1D)
        x = bkd.asarray(np.random.randn(2, 1))
        y_1d = bkd.asarray([0.5])
        with self.assertRaises(ValueError):
            cond.logpdf(x, y_1d)

        # Mismatched sample counts
        x = bkd.asarray(np.random.randn(2, 3))
        y = bkd.asarray(np.random.uniform(0.1, 0.9, (1, 5)))
        with self.assertRaises(ValueError):
            cond.logpdf(x, y)


class TestConditionalBetaNumpy(TestConditionalBeta[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestConditionalBetaTorch(TestConditionalBeta[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def test_logpdf_jacobian_wrt_params_autograd(self):
        """Verify logpdf_jacobian_wrt_params matches torch autograd."""
        from torch.autograd.functional import jacobian as torch_jacobian

        bkd = self._bkd
        cond = self._create_conditional_beta(nvars=2, max_level=2)

        # Fix x and y values
        np.random.seed(42)
        x = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 3)))
        y = bkd.asarray(np.random.uniform(0.1, 0.9, (1, 3)))

        # Get analytical jacobian
        analytical_jac = cond.logpdf_jacobian_wrt_params(x, y)  # (nsamples, nactive)

        # Get autograd jacobian
        def logpdf_from_params(params: torch.Tensor) -> torch.Tensor:
            cond.hyp_list().set_active_values(params)
            cond._log_alpha_func._sync_from_hyp_list()
            cond._log_beta_func._sync_from_hyp_list()
            return cond.logpdf(x, y).flatten()  # (nsamples,)

        params = cond.hyp_list().get_active_values()
        autograd_jac = torch_jacobian(logpdf_from_params, params)
        # autograd_jac shape: (nsamples, nactive)

        bkd.assert_allclose(analytical_jac, autograd_jac, rtol=1e-10)

    def test_reparameterize_differentiable(self):
        """Verify torch.autograd can compute gradients through reparameterize."""
        bkd = self._bkd
        cond = self._create_conditional_beta(nvars=1, max_level=0)

        np.random.seed(42)
        x = bkd.asarray(np.random.uniform(-0.9, 0.9, (1, 5)))
        base = bkd.asarray(np.random.uniform(0.1, 0.9, (1, 5)))

        def reparam_from_params(params: torch.Tensor) -> torch.Tensor:
            cond.hyp_list().set_active_values(params)
            cond._log_alpha_func._sync_from_hyp_list()
            cond._log_beta_func._sync_from_hyp_list()
            return cond.reparameterize(x, base).flatten()

        params = cond.hyp_list().get_active_values()
        from torch.autograd.functional import jacobian as torch_jacobian

        jac = torch_jacobian(reparam_from_params, params)
        # Should have non-zero gradients
        self.assertGreater(float(torch.abs(jac).max()), 1e-8)


class TestConditionalBetaBounded(
    Generic[Array], ParametrizedTestCase, unittest.TestCase
):
    """Parametrized tests for ConditionalBeta with various bounds."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def _create_basis_expansion(
        self, nvars: int, max_level: int, nqoi: int = 1
    ) -> BasisExpansion:
        """Helper to create a Legendre basis expansion."""
        bkd = self.bkd()
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    def _create_conditional_beta_with_bounds(
        self, nvars: int, lb: float, ub: float, max_level: int = 2
    ) -> ConditionalBeta:
        """Helper to create a ConditionalBeta with bounds."""
        bkd = self.bkd()
        log_alpha_func = self._create_basis_expansion(nvars, max_level, nqoi=1)
        log_beta_func = self._create_basis_expansion(nvars, max_level, nqoi=1)

        np.random.seed(42)
        log_alpha_func.set_coefficients(
            bkd.asarray(0.5 + 0.5 * np.random.randn(log_alpha_func.nterms(), 1))
        )
        log_beta_func.set_coefficients(
            bkd.asarray(0.5 + 0.5 * np.random.randn(log_beta_func.nterms(), 1))
        )

        return ConditionalBeta(log_alpha_func, log_beta_func, bkd, lb=lb, ub=ub)

    @parametrize("name,lb,ub", COND_BETA_BOUNDS_CONFIGS)
    def test_bounds_accessors(self, name: str, lb: float, ub: float) -> None:
        """Test bounds accessor methods."""
        bkd = self.bkd()
        cond = self._create_conditional_beta_with_bounds(nvars=1, lb=lb, ub=ub)
        bkd.assert_allclose(bkd.asarray([cond.lower()]), bkd.asarray([lb]), atol=1e-10)
        bkd.assert_allclose(bkd.asarray([cond.upper()]), bkd.asarray([ub]), atol=1e-10)
        lower, upper = cond.bounds()
        bkd.assert_allclose(
            bkd.asarray([lower, upper]), bkd.asarray([lb, ub]), atol=1e-10
        )

    @parametrize("name,lb,ub", COND_BETA_BOUNDS_CONFIGS)
    def test_logpdf_matches_scipy_bounded(
        self, name: str, lb: float, ub: float
    ) -> None:
        """Test logpdf matches scipy with loc/scale for constant alpha/beta."""
        bkd = self.bkd()
        scale = ub - lb

        # Create constant functions (degree-0 polynomial)
        log_alpha_func = self._create_basis_expansion(nvars=1, max_level=0, nqoi=1)
        log_beta_func = self._create_basis_expansion(nvars=1, max_level=0, nqoi=1)

        alpha_val = 2.5
        beta_val = 3.0
        log_alpha_func.set_coefficients(bkd.asarray([[np.log(alpha_val)]]))
        log_beta_func.set_coefficients(bkd.asarray([[np.log(beta_val)]]))

        cond = ConditionalBeta(log_alpha_func, log_beta_func, bkd, lb=lb, ub=ub)

        np.random.seed(42)
        nsamples = 10
        x = bkd.asarray(np.random.uniform(-0.9, 0.9, (1, nsamples)))
        # y values in [lb, ub]
        y_vals = lb + 0.1 * scale + 0.8 * scale * np.random.uniform(0, 1, nsamples)
        y = bkd.asarray(y_vals.reshape(1, -1))

        log_probs = cond.logpdf(x, y)

        # Compare with scipy (using loc/scale)
        scipy_log_probs = stats.beta.logpdf(
            bkd.to_numpy(y[0, :]), a=alpha_val, b=beta_val, loc=lb, scale=scale
        )

        bkd.assert_allclose(log_probs[0, :], bkd.asarray(scipy_log_probs), rtol=1e-10)

    @parametrize("name,lb,ub", COND_BETA_BOUNDS_CONFIGS)
    def test_rvs_in_bounds(self, name: str, lb: float, ub: float) -> None:
        """Test rvs produces samples in [lb, ub]."""
        bkd = self.bkd()
        cond = self._create_conditional_beta_with_bounds(nvars=1, lb=lb, ub=ub)

        np.random.seed(42)
        nsamples = 1000
        x = bkd.asarray(np.random.uniform(-0.9, 0.9, (1, nsamples)))
        samples = cond.rvs(x)

        samples_np = bkd.to_numpy(samples)
        self.assertTrue(np.all(samples_np >= lb))
        self.assertTrue(np.all(samples_np <= ub))

    @parametrize("name,lb,ub", COND_BETA_BOUNDS_CONFIGS)
    def test_rvs_statistics_bounded(self, name: str, lb: float, ub: float) -> None:
        """Test rvs generates samples with correct mean for bounded domain."""
        bkd = self.bkd()
        scale = ub - lb

        # Create constant functions
        log_alpha_func = self._create_basis_expansion(nvars=1, max_level=0, nqoi=1)
        log_beta_func = self._create_basis_expansion(nvars=1, max_level=0, nqoi=1)

        alpha_val = 2.0
        beta_val = 5.0
        log_alpha_func.set_coefficients(bkd.asarray([[np.log(alpha_val)]]))
        log_beta_func.set_coefficients(bkd.asarray([[np.log(beta_val)]]))

        cond = ConditionalBeta(log_alpha_func, log_beta_func, bkd, lb=lb, ub=ub)

        nsamples = 10000
        x = bkd.zeros((1, nsamples))

        np.random.seed(42)
        samples = cond.rvs(x)

        expected_mean = lb + scale * alpha_val / (alpha_val + beta_val)
        bkd.assert_allclose(
            bkd.asarray([bkd.mean(samples)]),
            bkd.asarray([expected_mean]),
            atol=0.05 * scale,
        )

    @parametrize("name,lb,ub", COND_BETA_BOUNDS_CONFIGS)
    def test_logpdf_jacobian_wrt_x_bounded(
        self, name: str, lb: float, ub: float
    ) -> None:
        """Test logpdf_jacobian_wrt_x using DerivativeChecker for bounded domain."""
        bkd = self.bkd()
        cond = self._create_conditional_beta_with_bounds(
            nvars=2, lb=lb, ub=ub, max_level=2
        )
        scale = ub - lb

        np.random.seed(42)
        # y in (lb, ub), stay away from boundaries
        y_val = lb + 0.4 * scale
        y = bkd.asarray([[y_val]])

        def fun(x: Array) -> Array:
            return cond.logpdf(x, y).T

        def jacobian_func(x: Array) -> Array:
            return cond.logpdf_jacobian_wrt_x(x, y)

        function_obj = FunctionWithJacobianFromCallable(
            nqoi=1, nvars=cond.nvars(), fun=fun, jacobian=jacobian_func, bkd=bkd
        )

        checker = DerivativeChecker(function_obj)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)
        self.assertLess(float(checker.error_ratio(errors[0])), 1e-5)

    @parametrize("name,lb,ub", COND_BETA_BOUNDS_CONFIGS)
    def test_logpdf_jacobian_wrt_params_bounded(
        self, name: str, lb: float, ub: float
    ) -> None:
        """Test logpdf_jacobian_wrt_params using DerivativeChecker for bounded."""
        bkd = self.bkd()
        cond = self._create_conditional_beta_with_bounds(
            nvars=2, lb=lb, ub=ub, max_level=2
        )
        scale = ub - lb

        np.random.seed(42)
        x = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))
        y_val = lb + 0.4 * scale
        y = bkd.asarray([[y_val]])

        nactive = cond.nparams()

        def fun(params: Array) -> Array:
            cond.hyp_list().set_active_values(params[:, 0])
            cond._log_alpha_func._sync_from_hyp_list()
            cond._log_beta_func._sync_from_hyp_list()
            return cond.logpdf(x, y).T

        def jacobian_func(params: Array) -> Array:
            cond.hyp_list().set_active_values(params[:, 0])
            cond._log_alpha_func._sync_from_hyp_list()
            cond._log_beta_func._sync_from_hyp_list()
            jac = cond.logpdf_jacobian_wrt_params(x, y)
            return jac

        function_obj = FunctionWithJacobianFromCallable(
            nqoi=1, nvars=nactive, fun=fun, jacobian=jacobian_func, bkd=bkd
        )

        checker = DerivativeChecker(function_obj)
        params = cond.hyp_list().get_active_values()
        sample_params = bkd.reshape(params, (nactive, 1))
        errors = checker.check_derivatives(sample_params, verbosity=0)
        self.assertLess(float(checker.error_ratio(errors[0])), 1e-5)


class TestConditionalBetaBoundedNumpy(TestConditionalBetaBounded[NDArray[Any]]):
    """NumPy backend parametrized tests for ConditionalBeta with bounds."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestConditionalBetaBoundedTorch(TestConditionalBetaBounded[torch.Tensor]):
    """PyTorch backend parametrized tests for ConditionalBeta with bounds."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestConditionalBetaBoundsValidation(unittest.TestCase):
    """Tests for bounds validation in ConditionalBeta."""

    def _create_simple_expansion(self, bkd: Backend) -> BasisExpansion:
        """Create a simple constant basis expansion."""
        marginals = [UniformMarginal(-1.0, 1.0, bkd)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(1, 0, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        exp = BasisExpansion(basis, bkd, nqoi=1)
        exp.set_coefficients(bkd.asarray([[0.5]]))
        return exp

    def test_invalid_bounds_lb_equals_ub(self) -> None:
        """Test that lb == ub raises ValueError."""
        bkd = NumpyBkd()
        log_alpha = self._create_simple_expansion(bkd)
        log_beta = self._create_simple_expansion(bkd)
        with self.assertRaises(ValueError):
            ConditionalBeta(log_alpha, log_beta, bkd, lb=1.0, ub=1.0)

    def test_invalid_bounds_lb_greater_ub(self) -> None:
        """Test that lb > ub raises ValueError."""
        bkd = NumpyBkd()
        log_alpha = self._create_simple_expansion(bkd)
        log_beta = self._create_simple_expansion(bkd)
        with self.assertRaises(ValueError):
            ConditionalBeta(log_alpha, log_beta, bkd, lb=2.0, ub=1.0)


if __name__ == "__main__":
    unittest.main()
