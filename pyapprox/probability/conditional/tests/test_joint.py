"""Tests for ConditionalIndependentJoint distribution.

Tests validate:
1. logpdf is sum of component logpdfs
2. rvs returns stacked samples
3. hyp_list combines correctly
4. jacobian_wrt_params concatenates correctly
5. Torch autograd compatibility
"""

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.probability.conditional.gaussian import ConditionalGaussian
from pyapprox.probability.conditional.joint import ConditionalIndependentJoint
from pyapprox.probability.joint.independent import IndependentJoint
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
    bkd, nvars: int, max_level: int = 2, seed: int = 42
) -> ConditionalGaussian:
    """Helper to create a ConditionalGaussian."""
    mean_func = _create_basis_expansion(bkd, nvars, max_level, nqoi=1)
    log_stdev_func = _create_basis_expansion(bkd, nvars, max_level, nqoi=1)

    np.random.seed(seed)
    mean_func.set_coefficients(bkd.asarray(np.random.randn(mean_func.nterms(), 1)))
    log_stdev_func.set_coefficients(
        bkd.asarray(0.5 * np.random.randn(log_stdev_func.nterms(), 1))
    )

    return ConditionalGaussian(mean_func, log_stdev_func, bkd)


def _create_joint(bkd, nvars: int = 2, nconditionals: int = 2):
    """Helper to create a ConditionalIndependentJoint."""
    conditionals = [
        _create_conditional_gaussian(bkd, nvars, max_level=2, seed=42 + i)
        for i in range(nconditionals)
    ]
    return ConditionalIndependentJoint(conditionals, bkd)


class TestConditionalIndependentJoint:
    """Test ConditionalIndependentJoint distribution."""

    def test_basic_properties(self, bkd):
        """Test basic properties of ConditionalIndependentJoint."""
        joint = _create_joint(bkd, nvars=2, nconditionals=3)

        assert joint.nvars() == 2
        assert joint.nqoi() == 3  # 3 conditionals, each nqoi=1
        assert hasattr(joint, "hyp_list")
        assert hasattr(joint, "logpdf_jacobian_wrt_x")
        assert hasattr(joint, "logpdf_jacobian_wrt_params")

    def test_logpdf_shape(self, bkd):
        """Test logpdf output shape."""
        joint = _create_joint(bkd, nvars=2, nconditionals=2)

        nsamples = 5
        x = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, nsamples)))
        y = bkd.asarray(np.random.randn(2, nsamples))  # nqoi=2

        log_probs = joint.logpdf(x, y)
        assert log_probs.shape == (1, nsamples)

    def test_logpdf_is_sum_of_components(self, bkd):
        """Test logpdf is sum of component logpdfs."""
        # Create joint and get its conditionals
        cond1 = _create_conditional_gaussian(bkd, nvars=2, max_level=2, seed=42)
        cond2 = _create_conditional_gaussian(bkd, nvars=2, max_level=2, seed=43)
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

    def test_rvs_shape(self, bkd):
        """Test rvs output shape."""
        joint = _create_joint(bkd, nvars=2, nconditionals=3)

        nsamples = 10
        x = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, nsamples)))

        np.random.seed(42)
        samples = joint.rvs(x)
        assert samples.shape == (3, nsamples)  # nqoi=3

    def test_reparameterize_stacks_components(self, bkd):
        """Test reparameterize returns stacked output from each conditional."""
        # Create constant-parameter conditionals for deterministic result
        mean1_func = _create_basis_expansion(bkd, nvars=1, max_level=0, nqoi=1)
        log_stdev1_func = _create_basis_expansion(bkd, nvars=1, max_level=0, nqoi=1)
        mean1_func.set_coefficients(bkd.asarray([[2.0]]))
        log_stdev1_func.set_coefficients(bkd.asarray([[np.log(0.5)]]))
        cond1 = ConditionalGaussian(mean1_func, log_stdev1_func, bkd)

        mean2_func = _create_basis_expansion(bkd, nvars=1, max_level=0, nqoi=1)
        log_stdev2_func = _create_basis_expansion(bkd, nvars=1, max_level=0, nqoi=1)
        mean2_func.set_coefficients(bkd.asarray([[5.0]]))
        log_stdev2_func.set_coefficients(bkd.asarray([[np.log(1.5)]]))
        cond2 = ConditionalGaussian(mean2_func, log_stdev2_func, bkd)

        joint = ConditionalIndependentJoint([cond1, cond2], bkd)

        x = bkd.asarray([[0.0, 0.0, 0.0]])
        base = bkd.asarray([[1.0, -1.0, 0.5], [0.0, 2.0, -0.5]])

        z = joint.reparameterize(x, base)
        assert z.shape == (2, 3)

        # Compute expected values manually: mean + exp(log_s) * base
        z1_expected = bkd.asarray(
            [[2.0 + 0.5 * 1.0, 2.0 + 0.5 * (-1.0), 2.0 + 0.5 * 0.5]]
        )
        z2_expected = bkd.asarray(
            [[5.0 + 1.5 * 0.0, 5.0 + 1.5 * 2.0, 5.0 + 1.5 * (-0.5)]]
        )
        expected = bkd.vstack([z1_expected, z2_expected])

        bkd.assert_allclose(z, expected, rtol=1e-12)

    def test_hyp_list_combines_correctly(self, bkd):
        """Test hyp_list combines all component hyp_lists."""
        cond1 = _create_conditional_gaussian(bkd, nvars=2, max_level=2, seed=42)
        cond2 = _create_conditional_gaussian(bkd, nvars=2, max_level=2, seed=43)
        joint = ConditionalIndependentJoint([cond1, cond2], bkd)

        # Total params should be sum of component params
        expected_nparams = cond1.nparams() + cond2.nparams()
        assert joint.nparams() == expected_nparams

        # Values should be concatenation of component values
        joint_values = joint.hyp_list().get_values()
        cond1_values = cond1.hyp_list().get_values()
        cond2_values = cond2.hyp_list().get_values()
        expected_values = bkd.hstack([cond1_values, cond2_values])

        bkd.assert_allclose(joint_values, expected_values, rtol=1e-10)

    def test_logpdf_jacobian_wrt_x_derivative_checker(self, bkd):
        """Test logpdf_jacobian_wrt_x using DerivativeChecker."""
        joint = _create_joint(bkd, nvars=2, nconditionals=2)

        # Fix a y value
        np.random.seed(42)
        y = bkd.asarray(np.random.randn(2, 1))

        # Wrap as function of x
        def fun(x):
            return joint.logpdf(x, y).T  # (1, nqoi=1)

        def jacobian_func(x):
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
        assert float(jac_error) < 1e-6

    def test_logpdf_jacobian_wrt_params_derivative_checker(self, bkd):
        """Test logpdf_jacobian_wrt_params using DerivativeChecker."""
        joint = _create_joint(bkd, nvars=2, nconditionals=2)

        # Fix x and y values
        np.random.seed(42)
        x = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))
        y = bkd.asarray(np.random.randn(2, 1))

        nactive = joint.nparams()

        # Wrap as function of params
        def fun(params):
            joint.hyp_list().set_active_values(params[:, 0])
            # Sync all nested funcs
            for cond in joint._conditionals:
                cond._mean_func._sync_from_hyp_list()
                cond._log_stdev_func._sync_from_hyp_list()
            return joint.logpdf(x, y).T  # (1, 1)

        def jacobian_func(params):
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
        assert float(jac_error) < 1e-6

    def test_logpdf_jacobian_wrt_params_concatenates_correctly(self, bkd):
        """Test jacobian_wrt_params concatenates component jacobians."""
        cond1 = _create_conditional_gaussian(bkd, nvars=2, max_level=2, seed=42)
        cond2 = _create_conditional_gaussian(bkd, nvars=2, max_level=2, seed=43)
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

    def test_reparameterize_shape(self, bkd):
        """Test reparameterize output shape matches (total_nqoi, N)."""
        joint = _create_joint(bkd, nvars=2, nconditionals=3)

        nsamples = 5
        np.random.seed(42)
        x = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, nsamples)))
        base = bkd.asarray(np.random.randn(3, nsamples))

        z = joint.reparameterize(x, base)
        assert z.shape == (3, nsamples)

    def test_kl_divergence_is_sum_of_components(self, bkd):
        """Test KL divergence equals sum of component KL divergences."""
        # Create constant-param conditionals
        mean1 = _create_basis_expansion(bkd, nvars=1, max_level=0, nqoi=1)
        ls1 = _create_basis_expansion(bkd, nvars=1, max_level=0, nqoi=1)
        mean1.set_coefficients(bkd.asarray([[1.5]]))
        ls1.set_coefficients(bkd.asarray([[np.log(0.8)]]))
        cond1 = ConditionalGaussian(mean1, ls1, bkd)

        mean2 = _create_basis_expansion(bkd, nvars=1, max_level=0, nqoi=1)
        ls2 = _create_basis_expansion(bkd, nvars=1, max_level=0, nqoi=1)
        mean2.set_coefficients(bkd.asarray([[2.0]]))
        ls2.set_coefficients(bkd.asarray([[np.log(1.2)]]))
        cond2 = ConditionalGaussian(mean2, ls2, bkd)

        joint = ConditionalIndependentJoint([cond1, cond2], bkd)

        prior1 = GaussianMarginal(0.0, 1.0, bkd)
        prior2 = GaussianMarginal(0.0, 1.0, bkd)
        prior_joint = IndependentJoint([prior1, prior2], bkd)

        x = bkd.asarray([[0.0, 0.5]])
        kl_joint = joint.kl_divergence(x, prior_joint)

        kl1 = cond1.kl_divergence(x, prior1)
        kl2 = cond2.kl_divergence(x, prior2)
        expected = kl1 + kl2

        bkd.assert_allclose(kl_joint, expected, rtol=1e-12)

    def test_base_distribution_is_independent_joint(self, bkd):
        """Test base_distribution returns IndependentJoint."""
        joint = _create_joint(bkd, nvars=2, nconditionals=2)

        base = joint.base_distribution()
        assert isinstance(base, IndependentJoint)
        # Should have 2 marginals (one per conditional)
        assert len(base.marginals()) == 2

    def test_validation_errors(self, bkd):
        """Test input validation raises appropriate errors."""
        joint = _create_joint(bkd, nvars=2, nconditionals=2)

        # x wrong shape (1D)
        x_1d = bkd.asarray(np.random.randn(2))
        y = bkd.asarray(np.random.randn(2, 1))
        with pytest.raises(ValueError):
            joint.logpdf(x_1d, y)

        # y wrong shape (wrong nqoi)
        x = bkd.asarray(np.random.randn(2, 1))
        y_wrong = bkd.asarray(np.random.randn(3, 1))  # Should be 2
        with pytest.raises(ValueError):
            joint.logpdf(x, y_wrong)

        # Mismatched sample counts
        x = bkd.asarray(np.random.randn(2, 3))
        y = bkd.asarray(np.random.randn(2, 5))
        with pytest.raises(ValueError):
            joint.logpdf(x, y)

    def test_logpdf_jacobian_wrt_params_autograd(self, bkd):
        """Verify logpdf_jacobian_wrt_params matches torch autograd."""
        if not isinstance(bkd, NumpyBkd):
            import torch
            from torch.autograd.functional import jacobian as torch_jacobian

            joint = _create_joint(bkd, nvars=2, nconditionals=2)

            # Fix x and y values
            np.random.seed(42)
            x = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 3)))
            y = bkd.asarray(np.random.randn(2, 3))

            # Get analytical jacobian
            analytical_jac = joint.logpdf_jacobian_wrt_params(x, y)

            # Get autograd jacobian
            def logpdf_from_params(params: torch.Tensor) -> torch.Tensor:
                joint.hyp_list().set_active_values(params)
                for cond in joint._conditionals:
                    cond._mean_func._sync_from_hyp_list()
                    cond._log_stdev_func._sync_from_hyp_list()
                return joint.logpdf(x, y).flatten()

            params = joint.hyp_list().get_active_values()
            autograd_jac = torch_jacobian(logpdf_from_params, params)

            bkd.assert_allclose(analytical_jac, autograd_jac, rtol=1e-10)
        else:
            pytest.skip("Torch-only test")
