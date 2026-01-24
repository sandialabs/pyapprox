"""Tests for ConditionalIndependentJoint distribution.

Tests validate:
1. logpdf is sum of component logpdfs
2. rvs returns stacked samples
3. hyp_list combines correctly
4. jacobian_wrt_params concatenates correctly
5. Torch autograd compatibility
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.test_utils import load_tests
from pyapprox.typing.probability.conditional.gaussian import ConditionalGaussian
from pyapprox.typing.probability.conditional.joint import ConditionalIndependentJoint
from pyapprox.typing.probability.univariate import UniformMarginal
from pyapprox.typing.surrogates.affine.univariate import create_bases_1d
from pyapprox.typing.surrogates.affine.indices import compute_hyperbolic_indices
from pyapprox.typing.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.typing.surrogates.affine.expansions import BasisExpansion
from pyapprox.typing.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.typing.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)


class TestConditionalIndependentJoint(Generic[Array], unittest.TestCase):
    """Test ConditionalIndependentJoint distribution."""

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

    def _create_conditional_gaussian(
        self, nvars: int, max_level: int = 2, seed: int = 42
    ) -> ConditionalGaussian:
        """Helper to create a ConditionalGaussian."""
        bkd = self._bkd
        mean_func = self._create_basis_expansion(nvars, max_level, nqoi=1)
        log_stdev_func = self._create_basis_expansion(nvars, max_level, nqoi=1)

        np.random.seed(seed)
        mean_func.set_coefficients(bkd.asarray(np.random.randn(mean_func.nterms(), 1)))
        log_stdev_func.set_coefficients(
            bkd.asarray(0.5 * np.random.randn(log_stdev_func.nterms(), 1))
        )

        return ConditionalGaussian(mean_func, log_stdev_func, bkd)

    def _create_joint(self, nvars: int = 2, nconditionals: int = 2):
        """Helper to create a ConditionalIndependentJoint."""
        bkd = self._bkd
        conditionals = [
            self._create_conditional_gaussian(nvars, max_level=2, seed=42 + i)
            for i in range(nconditionals)
        ]
        return ConditionalIndependentJoint(conditionals, bkd)

    def test_basic_properties(self):
        """Test basic properties of ConditionalIndependentJoint."""
        bkd = self._bkd
        joint = self._create_joint(nvars=2, nconditionals=3)

        self.assertEqual(joint.nvars(), 2)
        self.assertEqual(joint.nqoi(), 3)  # 3 conditionals, each nqoi=1
        self.assertTrue(hasattr(joint, "hyp_list"))
        self.assertTrue(hasattr(joint, "logpdf_jacobian_wrt_x"))
        self.assertTrue(hasattr(joint, "logpdf_jacobian_wrt_params"))

    def test_logpdf_shape(self):
        """Test logpdf output shape."""
        bkd = self._bkd
        joint = self._create_joint(nvars=2, nconditionals=2)

        nsamples = 5
        x = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, nsamples)))
        y = bkd.asarray(np.random.randn(2, nsamples))  # nqoi=2

        log_probs = joint.logpdf(x, y)
        self.assertEqual(log_probs.shape, (1, nsamples))

    def test_logpdf_is_sum_of_components(self):
        """Test logpdf is sum of component logpdfs."""
        bkd = self._bkd

        # Create joint and get its conditionals
        cond1 = self._create_conditional_gaussian(nvars=2, max_level=2, seed=42)
        cond2 = self._create_conditional_gaussian(nvars=2, max_level=2, seed=43)
        joint = ConditionalIndependentJoint([cond1, cond2], bkd)

        # Sample x and y
        np.random.seed(42)
        nsamples = 5
        x = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, nsamples)))
        y1 = bkd.asarray(np.random.randn(1, nsamples))
        y2 = bkd.asarray(np.random.randn(1, nsamples))
        y = bkd.vstack([y1, y2])

        # Compute joint logpdf
        joint_logpdf = joint.logpdf(x, y)

        # Compute sum of component logpdfs
        logpdf1 = cond1.logpdf(x, y1)
        logpdf2 = cond2.logpdf(x, y2)
        sum_logpdf = logpdf1 + logpdf2

        bkd.assert_allclose(joint_logpdf, sum_logpdf, rtol=1e-10)

    def test_rvs_shape(self):
        """Test rvs output shape."""
        bkd = self._bkd
        joint = self._create_joint(nvars=2, nconditionals=3)

        nsamples = 10
        x = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, nsamples)))

        np.random.seed(42)
        samples = joint.rvs(x)
        self.assertEqual(samples.shape, (3, nsamples))  # nqoi=3

    def test_rvs_returns_stacked_samples(self):
        """Test rvs returns stacked samples from each conditional."""
        bkd = self._bkd

        # Create constant-parameter conditionals for reproducibility
        mean1_func = self._create_basis_expansion(nvars=1, max_level=0, nqoi=1)
        log_stdev1_func = self._create_basis_expansion(nvars=1, max_level=0, nqoi=1)
        mean1_func.set_coefficients(bkd.asarray([[0.0]]))
        log_stdev1_func.set_coefficients(bkd.asarray([[np.log(1.0)]]))
        cond1 = ConditionalGaussian(mean1_func, log_stdev1_func, bkd)

        mean2_func = self._create_basis_expansion(nvars=1, max_level=0, nqoi=1)
        log_stdev2_func = self._create_basis_expansion(nvars=1, max_level=0, nqoi=1)
        mean2_func.set_coefficients(bkd.asarray([[5.0]]))  # Different mean
        log_stdev2_func.set_coefficients(bkd.asarray([[np.log(0.5)]]))
        cond2 = ConditionalGaussian(mean2_func, log_stdev2_func, bkd)

        joint = ConditionalIndependentJoint([cond1, cond2], bkd)

        # Generate many samples
        nsamples = 5000
        x = bkd.zeros((1, nsamples))

        np.random.seed(42)
        samples = joint.rvs(x)

        # First component should have mean ~0
        sample_mean1 = float(bkd.to_numpy(bkd.mean(samples[0, :])))
        # Second component should have mean ~5
        sample_mean2 = float(bkd.to_numpy(bkd.mean(samples[1, :])))

        self.assertAlmostEqual(sample_mean1, 0.0, delta=0.1)
        self.assertAlmostEqual(sample_mean2, 5.0, delta=0.1)

    def test_hyp_list_combines_correctly(self):
        """Test hyp_list combines all component hyp_lists."""
        bkd = self._bkd
        cond1 = self._create_conditional_gaussian(nvars=2, max_level=2, seed=42)
        cond2 = self._create_conditional_gaussian(nvars=2, max_level=2, seed=43)
        joint = ConditionalIndependentJoint([cond1, cond2], bkd)

        # Total params should be sum of component params
        expected_nparams = cond1.nparams() + cond2.nparams()
        self.assertEqual(joint.nparams(), expected_nparams)

        # Values should be concatenation of component values
        joint_values = joint.hyp_list().get_values()
        cond1_values = cond1.hyp_list().get_values()
        cond2_values = cond2.hyp_list().get_values()
        expected_values = bkd.hstack([cond1_values, cond2_values])

        bkd.assert_allclose(joint_values, expected_values, rtol=1e-10)

    def test_logpdf_jacobian_wrt_x_derivative_checker(self):
        """Test logpdf_jacobian_wrt_x using DerivativeChecker."""
        bkd = self._bkd
        joint = self._create_joint(nvars=2, nconditionals=2)

        # Fix a y value
        np.random.seed(42)
        y = bkd.asarray(np.random.randn(2, 1))

        # Wrap as function of x
        def fun(x: Array) -> Array:
            return joint.logpdf(x, y).T  # (1, nqoi=1)

        def jacobian_func(x: Array) -> Array:
            return joint.logpdf_jacobian_wrt_x(x, y)  # (1, nvars)

        function_obj = FunctionWithJacobianFromCallable(
            nqoi=1,
            nvars=joint.nvars(),
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
        joint = self._create_joint(nvars=2, nconditionals=2)

        # Fix x and y values
        np.random.seed(42)
        x = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))
        y = bkd.asarray(np.random.randn(2, 1))

        nactive = joint.nparams()

        # Wrap as function of params
        def fun(params: Array) -> Array:
            joint.hyp_list().set_active_values(params[:, 0])
            # Sync all nested funcs
            for cond in joint._conditionals:
                cond._mean_func._sync_from_hyp_list()
                cond._log_stdev_func._sync_from_hyp_list()
            return joint.logpdf(x, y).T  # (1, 1)

        def jacobian_func(params: Array) -> Array:
            joint.hyp_list().set_active_values(params[:, 0])
            for cond in joint._conditionals:
                cond._mean_func._sync_from_hyp_list()
                cond._log_stdev_func._sync_from_hyp_list()
            jac = joint.logpdf_jacobian_wrt_params(x, y)  # (1, nactive)
            return jac  # (nqoi=1, nactive)

        function_obj = FunctionWithJacobianFromCallable(
            nqoi=1,
            nvars=nactive,
            fun=fun,
            jacobian=jacobian_func,
            bkd=bkd,
        )

        checker = DerivativeChecker(function_obj)
        params = joint.hyp_list().get_active_values()
        sample_params = bkd.reshape(params, (nactive, 1))
        errors = checker.check_derivatives(sample_params, verbosity=0)

        jac_error = checker.error_ratio(errors[0])
        self.assertLess(float(jac_error), 1e-6)

    def test_logpdf_jacobian_wrt_params_concatenates_correctly(self):
        """Test jacobian_wrt_params concatenates component jacobians."""
        bkd = self._bkd
        cond1 = self._create_conditional_gaussian(nvars=2, max_level=2, seed=42)
        cond2 = self._create_conditional_gaussian(nvars=2, max_level=2, seed=43)
        joint = ConditionalIndependentJoint([cond1, cond2], bkd)

        np.random.seed(42)
        x = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 3)))
        y1 = bkd.asarray(np.random.randn(1, 3))
        y2 = bkd.asarray(np.random.randn(1, 3))
        y = bkd.vstack([y1, y2])

        # Get joint jacobian
        joint_jac = joint.logpdf_jacobian_wrt_params(x, y)  # (nsamples, nparams)

        # Get component jacobians
        jac1 = cond1.logpdf_jacobian_wrt_params(x, y1)  # (nsamples, nparams1)
        jac2 = cond2.logpdf_jacobian_wrt_params(x, y2)  # (nsamples, nparams2)
        expected_jac = bkd.hstack([jac1, jac2])

        bkd.assert_allclose(joint_jac, expected_jac, rtol=1e-10)

    def test_validation_errors(self):
        """Test input validation raises appropriate errors."""
        bkd = self._bkd
        joint = self._create_joint(nvars=2, nconditionals=2)

        # x wrong shape (1D)
        x_1d = bkd.asarray(np.random.randn(2))
        y = bkd.asarray(np.random.randn(2, 1))
        with self.assertRaises(ValueError):
            joint.logpdf(x_1d, y)

        # y wrong shape (wrong nqoi)
        x = bkd.asarray(np.random.randn(2, 1))
        y_wrong = bkd.asarray(np.random.randn(3, 1))  # Should be 2
        with self.assertRaises(ValueError):
            joint.logpdf(x, y_wrong)

        # Mismatched sample counts
        x = bkd.asarray(np.random.randn(2, 3))
        y = bkd.asarray(np.random.randn(2, 5))
        with self.assertRaises(ValueError):
            joint.logpdf(x, y)


class TestConditionalIndependentJointNumpy(
    TestConditionalIndependentJoint[NDArray[Any]]
):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestConditionalIndependentJointTorch(
    TestConditionalIndependentJoint[torch.Tensor]
):
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
        joint = self._create_joint(nvars=2, nconditionals=2)

        # Fix x and y values
        np.random.seed(42)
        x = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 3)))
        y = bkd.asarray(np.random.randn(2, 3))

        # Get analytical jacobian
        analytical_jac = joint.logpdf_jacobian_wrt_params(x, y)  # (nsamples, nactive)

        # Get autograd jacobian
        def logpdf_from_params(params: torch.Tensor) -> torch.Tensor:
            joint.hyp_list().set_active_values(params)
            for cond in joint._conditionals:
                cond._mean_func._sync_from_hyp_list()
                cond._log_stdev_func._sync_from_hyp_list()
            return joint.logpdf(x, y).flatten()  # (nsamples,)

        params = joint.hyp_list().get_active_values()
        autograd_jac = torch_jacobian(logpdf_from_params, params)
        # autograd_jac shape: (nsamples, nactive)

        bkd.assert_allclose(analytical_jac, autograd_jac, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
