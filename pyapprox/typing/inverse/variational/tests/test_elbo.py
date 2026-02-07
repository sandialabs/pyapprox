"""
Tests for ELBOObjective and make_single_problem_elbo.
"""

import unittest
from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray
import torch

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.probability.gaussian.diagonal import (
    DiagonalMultivariateGaussian,
)
from pyapprox.typing.interface.functions.protocols.function import (
    FunctionProtocol,
)
from pyapprox.typing.inverse.variational.gaussian_family import (
    GaussianVariationalFamily,
)
from pyapprox.typing.inverse.variational.elbo import (
    ELBOObjective,
    make_single_problem_elbo,
)
from pyapprox.typing.inverse.variational.amortization import (
    ConstantAmortization,
)


class TestELBOBase(Generic[Array], unittest.TestCase):
    """Base test class for ELBOObjective."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self._nvars = 2

    def _make_simple_elbo(self) -> ELBOObjective:
        """Create a simple ELBO for testing.

        Uses a Gaussian variational family with a Gaussian prior
        and a simple log-likelihood.
        """
        bkd = self._bkd
        family = GaussianVariationalFamily(self._nvars, bkd)
        prior = DiagonalMultivariateGaussian(
            bkd.zeros((self._nvars, 1)),
            bkd.ones((self._nvars,)),
            bkd,
        )

        def log_likelihood_fn(z: Array) -> Array:
            # Simple: log p(obs | z) = -0.5 * ||z - obs||^2
            obs = bkd.ones((self._nvars, 1))
            diff = z - obs
            return bkd.reshape(
                -0.5 * bkd.sum(diff ** 2, axis=0), (1, z.shape[1])
            )

        np.random.seed(42)
        nsamples = 50
        base_samples = bkd.asarray(
            np.random.normal(0, 1, (self._nvars, nsamples))
        )
        weights = bkd.full((1, nsamples), 1.0 / nsamples)

        return make_single_problem_elbo(
            family, log_likelihood_fn, prior, base_samples, weights, bkd,
        )

    def test_elbo_returns_correct_shape(self) -> None:
        elbo = self._make_simple_elbo()
        params = self._bkd.zeros((elbo.nvars(), 1))
        result = elbo(params)
        self.assertEqual(result.shape, (1, 1))

    def test_elbo_satisfies_function_protocol(self) -> None:
        elbo = self._make_simple_elbo()
        self.assertIsInstance(elbo, FunctionProtocol)

    def test_elbo_nqoi(self) -> None:
        elbo = self._make_simple_elbo()
        self.assertEqual(elbo.nqoi(), 1)

    def test_elbo_nvars(self) -> None:
        elbo = self._make_simple_elbo()
        # 2 means + 2 log-stdevs = 4 params for ConstantAmortization
        self.assertEqual(elbo.nvars(), 4)

    def test_elbo_shared_vs_per_sample_equivalence(self) -> None:
        """Verify ConstantAmortization per-sample path matches shared path.

        make_single_problem_elbo uses ConstantAmortization which broadcasts
        the same params to all samples. The ELBO value should match what
        we'd get calling the family directly with params=None.
        """
        bkd = self._bkd
        family = GaussianVariationalFamily(self._nvars, bkd)
        prior = DiagonalMultivariateGaussian(
            bkd.zeros((self._nvars, 1)),
            bkd.ones((self._nvars,)),
            bkd,
        )

        def log_likelihood_fn(z: Array) -> Array:
            obs = bkd.ones((self._nvars, 1))
            diff = z - obs
            return bkd.reshape(
                -0.5 * bkd.sum(diff ** 2, axis=0), (1, z.shape[1])
            )

        np.random.seed(42)
        nsamples = 50
        base_samples = bkd.asarray(
            np.random.normal(0, 1, (self._nvars, nsamples))
        )
        weights = bkd.full((1, nsamples), 1.0 / nsamples)

        # ELBO via make_single_problem_elbo
        elbo = make_single_problem_elbo(
            family, log_likelihood_fn, prior, base_samples, weights, bkd,
        )
        params = family.hyp_list().get_active_values()
        elbo_val = elbo(bkd.reshape(params, (len(params), 1)))

        # Manual ELBO with shared params (params=None)
        z = family.reparameterize(base_samples)
        log_lik = log_likelihood_fn(z)
        kl = family.kl_divergence(prior)
        manual_elbo = bkd.sum(weights * log_lik) - kl
        manual_neg_elbo = bkd.reshape(-manual_elbo, (1, 1))

        bkd.assert_allclose(elbo_val, manual_neg_elbo, rtol=1e-10)


class TestELBONumpy(TestELBOBase[NDArray[Any]], unittest.TestCase):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()

    def test_elbo_no_jacobian_numpy(self) -> None:
        elbo = self._make_simple_elbo()
        self.assertFalse(hasattr(elbo, 'jacobian'))


class TestELBOTorch(TestELBOBase[torch.Tensor], unittest.TestCase):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()

    def test_elbo_jacobian_shape(self) -> None:
        elbo = self._make_simple_elbo()
        self.assertTrue(hasattr(elbo, 'jacobian'))
        params = self._bkd.zeros((elbo.nvars(), 1))
        jac = elbo.jacobian(params)
        self.assertEqual(jac.shape, (1, elbo.nvars()))

    def test_elbo_gradient_derivative_checker(self) -> None:
        from pyapprox.typing.interface.functions.derivative_checks.derivative_checker import (
            DerivativeChecker,
        )
        elbo = self._make_simple_elbo()
        checker = DerivativeChecker(elbo)
        sample = self._bkd.zeros((elbo.nvars(), 1))
        errors = checker.check_derivatives(sample, verbosity=0)
        ratio = checker.error_ratio(errors[0])
        self.assertLessEqual(
            float(self._bkd.flatten(ratio)[0]), 1e-5
        )


from pyapprox.typing.util.test_utils import load_tests  # noqa: F401
