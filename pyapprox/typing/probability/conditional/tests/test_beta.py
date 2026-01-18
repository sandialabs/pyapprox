"""Tests for ConditionalBeta distribution.

Tests validate:
1. logpdf values against scipy
2. rvs sampling statistics
3. jacobian_wrt_x with DerivativeChecker
4. jacobian_wrt_params with DerivativeChecker
5. Torch autograd compatibility
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray
from scipy import stats

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.test_utils import load_tests
from pyapprox.typing.probability.conditional.beta import ConditionalBeta
from pyapprox.typing.surrogates.affine.univariate import LegendrePolynomial1D
from pyapprox.typing.surrogates.affine.indices import compute_hyperbolic_indices
from pyapprox.typing.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.typing.surrogates.affine.expansions import BasisExpansion
from pyapprox.typing.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.typing.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)


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
        bases_1d = [LegendrePolynomial1D(bkd) for _ in range(nvars)]
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
        bkd = self._bkd
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

        bkd.assert_allclose(
            log_probs[0, :], bkd.asarray(scipy_log_probs), rtol=1e-10
        )

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

        sample_mean = float(bkd.to_numpy(bkd.mean(samples)))
        expected_mean = alpha_val / (alpha_val + beta_val)

        self.assertAlmostEqual(sample_mean, expected_mean, delta=0.02)

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
        self.assertLess(float(jac_error), 1e-6)

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


if __name__ == "__main__":
    unittest.main()
