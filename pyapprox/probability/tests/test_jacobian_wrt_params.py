"""
Tests for logpdf_jacobian_wrt_params in distributions.

Uses parametrized tests to validate analytical Jacobians against:
1. DerivativeChecker (finite difference validation)
2. PyTorch autograd
"""

import unittest
from typing import Any, Callable

import numpy as np
import torch
from torch.autograd.functional import jacobian as torch_jacobian
from unittest_parametrize import ParametrizedTestCase, parametrize

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests
from pyapprox.probability.univariate import (
    GaussianMarginal,
    UniformMarginal,
    BetaMarginal,
    GammaMarginal,
)
from pyapprox.probability.joint import IndependentJoint
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)


def create_gaussian(bkd: Backend[Array]) -> GaussianMarginal:
    return GaussianMarginal(1.0, 2.0, bkd)


def create_uniform(bkd: Backend[Array]) -> UniformMarginal:
    return UniformMarginal(0.0, 2.0, bkd)


def create_beta(bkd: Backend[Array]) -> BetaMarginal:
    return BetaMarginal(2.0, 5.0, bkd)


def create_gamma(bkd: Backend[Array]) -> GammaMarginal:
    return GammaMarginal(2.0, 1.5, bkd)


def create_independent_joint(bkd: Backend[Array]) -> IndependentJoint:
    marginals = [
        GaussianMarginal(0.0, 1.0, bkd),
        UniformMarginal(0.0, 1.0, bkd),
    ]
    return IndependentJoint(marginals, bkd)


DISTRIBUTIONS = [
    ("Gaussian", create_gaussian, 2),
    ("Uniform", create_uniform, 2),
    ("Beta", create_beta, 2),
    ("Gamma", create_gamma, 2),
    ("IndependentJoint", create_independent_joint, 4),
]


class TestParamJacobianDerivativeChecker(ParametrizedTestCase):
    """Test logpdf_jacobian_wrt_params using DerivativeChecker (NumPy)."""

    @parametrize(
        "name,factory,expected_nparams",
        DISTRIBUTIONS,
    )
    def test_nparams(
        self, name: str, factory: Callable, expected_nparams: int
    ) -> None:
        """Test nparams returns expected value."""
        bkd = NumpyBkd()
        dist = factory(bkd)
        self.assertEqual(dist.nparams(), expected_nparams)

    @parametrize(
        "name,factory,expected_nparams",
        DISTRIBUTIONS,
    )
    def test_jacobian_derivative_checker(
        self, name: str, factory: Callable, expected_nparams: int
    ) -> None:
        """Test jacobian using DerivativeChecker."""
        bkd = NumpyBkd()
        np.random.seed(42)
        dist = factory(bkd)

        # Generate a sample using rvs
        samples = dist.rvs(1)

        # Create wrapper function: params -> logpdf
        # DerivativeChecker expects function with __call__ and jacobian
        # Use set_active_values since all params are active by default
        def fun(params: Array) -> Array:
            # params shape: (nparams, 1)
            dist.hyp_list().set_active_values(params[:, 0])
            logpdf = dist.logpdf(samples)  # (1, 1)
            return logpdf.T  # (nsamples=1, nqoi=1)

        def jacobian(params: Array) -> Array:
            # params shape: (nparams, 1)
            dist.hyp_list().set_active_values(params[:, 0])
            jac = dist.logpdf_jacobian_wrt_params(samples)  # (1, nparams)
            return jac  # (nqoi=1, nparams)

        function_obj = FunctionWithJacobianFromCallable(
            nqoi=1,
            nvars=expected_nparams,
            fun=fun,
            jacobian=jacobian,
            bkd=bkd,
        )

        checker = DerivativeChecker(function_obj)
        params = dist.hyp_list().get_values()
        sample = bkd.reshape(params, (expected_nparams, 1))
        errors = checker.check_derivatives(sample, verbosity=0)
        ratio = float(bkd.to_numpy(checker.error_ratio(errors[0])))
        self.assertLessEqual(ratio, 1e-6)


class TestParamJacobianAutograd(ParametrizedTestCase):
    """Test logpdf_jacobian_wrt_params against PyTorch autograd."""

    @parametrize(
        "name,factory,expected_nparams",
        DISTRIBUTIONS,
    )
    def test_jacobian_autograd(
        self, name: str, factory: Callable, expected_nparams: int
    ) -> None:
        """Verify analytical jacobian matches torch autograd."""
        torch.set_default_dtype(torch.float64)
        bkd = TorchBkd()
        np.random.seed(42)
        dist = factory(bkd)

        # Generate samples using rvs
        samples = dist.rvs(5)

        # Get analytical jacobian
        analytical_jac = dist.logpdf_jacobian_wrt_params(samples)

        # Get autograd jacobian
        def logpdf_from_params(params: torch.Tensor) -> torch.Tensor:
            dist.hyp_list().set_active_values(params)
            return dist.logpdf(samples).flatten()

        params = dist.hyp_list().get_active_values()
        autograd_jac = torch_jacobian(logpdf_from_params, params)

        bkd.assert_allclose(analytical_jac, autograd_jac, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
