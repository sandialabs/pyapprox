"""
Tests for BetaVariationalFamily.
"""

import unittest
from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray
import torch

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.probability.univariate.beta import BetaMarginal
from pyapprox.typing.probability.joint.independent import IndependentJoint
from pyapprox.typing.inverse.variational.protocols import (
    VariationalFamilyProtocol,
)
from pyapprox.typing.inverse.variational.beta_family import (
    BetaVariationalFamily,
)


class TestBetaVariationalFamilyBase(Generic[Array], unittest.TestCase):
    """Base test class for BetaVariationalFamily."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_satisfies_protocol(self) -> None:
        family = BetaVariationalFamily(1, self._bkd)
        self.assertIsInstance(family, VariationalFamilyProtocol)

    def test_hyp_list_shape_1d(self) -> None:
        family = BetaVariationalFamily(1, self._bkd)
        # 1 log_alpha + 1 log_beta = 2 params
        self.assertEqual(family.hyp_list().nparams(), 2)
        self.assertEqual(family.hyp_list().nactive_params(), 2)

    def test_hyp_list_shape_3d(self) -> None:
        family = BetaVariationalFamily(3, self._bkd)
        # 3 log_alphas + 3 log_betas = 6 params
        self.assertEqual(family.hyp_list().nparams(), 6)
        self.assertEqual(family.hyp_list().nactive_params(), 6)

    def test_nvars(self) -> None:
        family = BetaVariationalFamily(3, self._bkd)
        self.assertEqual(family.nvars(), 3)

    def test_reparameterize_shape(self) -> None:
        family = BetaVariationalFamily(1, self._bkd)
        np.random.seed(42)
        base_samples = self._bkd.asarray(
            np.random.uniform(0.01, 0.99, (1, 100))
        )
        z = family.reparameterize(base_samples)
        self.assertEqual(z.shape, (1, 100))

    def test_reparameterize_shape_multidim(self) -> None:
        family = BetaVariationalFamily(3, self._bkd)
        np.random.seed(42)
        base_samples = self._bkd.asarray(
            np.random.uniform(0.01, 0.99, (3, 50))
        )
        z = family.reparameterize(base_samples)
        self.assertEqual(z.shape, (3, 50))

    def test_reparameterize_in_bounds(self) -> None:
        family = BetaVariationalFamily(1, self._bkd)
        np.random.seed(42)
        base_samples = self._bkd.asarray(
            np.random.uniform(0.01, 0.99, (1, 100))
        )
        z = family.reparameterize(base_samples)
        z_np = self._bkd.to_numpy(z)
        self.assertTrue(np.all(z_np >= 0.0))
        self.assertTrue(np.all(z_np <= 1.0))

    def test_logpdf_shape(self) -> None:
        family = BetaVariationalFamily(1, self._bkd)
        samples = self._bkd.asarray([[0.2, 0.5, 0.8]])
        logp = family.logpdf(samples)
        self.assertEqual(logp.shape, (1, 3))

    def test_logpdf_matches_beta_marginal_1d(self) -> None:
        """Compare logpdf with BetaMarginal for 1D case."""
        alpha, beta_val = 3.0, 5.0
        family = BetaVariationalFamily(
            1, self._bkd, alpha_init=[alpha], beta_init=[beta_val]
        )
        samples = self._bkd.asarray([[0.1, 0.3, 0.5, 0.7, 0.9]])
        logp_family = family.logpdf(samples)

        marginal = BetaMarginal(alpha, beta_val, self._bkd)
        logp_ref = marginal.logpdf(samples)

        self._bkd.assert_allclose(logp_family, logp_ref, rtol=1e-10)

    def test_logpdf_matches_independent_joint(self) -> None:
        """Compare logpdf with IndependentJoint of BetaMarginals for 2D."""
        alphas = [2.0, 4.0]
        betas = [3.0, 6.0]
        family = BetaVariationalFamily(
            2, self._bkd, alpha_init=alphas, beta_init=betas
        )
        samples = self._bkd.asarray([[0.2, 0.4, 0.6], [0.3, 0.5, 0.7]])
        logp_family = family.logpdf(samples)

        marginals = [
            BetaMarginal(alphas[i], betas[i], self._bkd) for i in range(2)
        ]
        joint = IndependentJoint(marginals, self._bkd)
        logp_ref = joint.logpdf(samples)

        self._bkd.assert_allclose(logp_family, logp_ref, rtol=1e-10)

    def test_kl_divergence_analytical_beta_beta(self) -> None:
        """Test analytical KL(Beta(a1,b1) || Beta(a2,b2)) for 1D."""
        bkd = self._bkd
        alpha1, beta1 = 3.0, 5.0
        alpha2, beta2 = 2.0, 4.0
        family = BetaVariationalFamily(
            1, bkd, alpha_init=[alpha1], beta_init=[beta1]
        )
        prior = BetaVariationalFamily(
            1, bkd, alpha_init=[alpha2], beta_init=[beta2]
        )

        kl = family.kl_divergence(prior)

        # Manual KL computation using scipy
        from scipy.special import gammaln, digamma
        kl_expected = (
            gammaln(alpha2) + gammaln(beta2) - gammaln(alpha2 + beta2)
            - gammaln(alpha1) - gammaln(beta1) + gammaln(alpha1 + beta1)
            + (alpha1 - alpha2) * digamma(alpha1)
            + (beta1 - beta2) * digamma(beta1)
            + (alpha2 - alpha1 + beta2 - beta1) * digamma(alpha1 + beta1)
        )

        bkd.assert_allclose(
            bkd.asarray([float(kl)]),
            bkd.asarray([kl_expected]),
            rtol=1e-10,
        )

    def test_kl_divergence_self_is_zero(self) -> None:
        """KL(q || q) should be zero."""
        bkd = self._bkd
        family = BetaVariationalFamily(
            2, bkd, alpha_init=[3.0, 5.0], beta_init=[2.0, 4.0]
        )
        prior = BetaVariationalFamily(
            2, bkd, alpha_init=[3.0, 5.0], beta_init=[2.0, 4.0]
        )

        kl = family.kl_divergence(prior)
        bkd.assert_allclose(
            bkd.asarray([float(kl)]),
            bkd.asarray([0.0]),
            atol=1e-10,
        )

    def test_kl_divergence_multidim(self) -> None:
        """Test KL for multi-dimensional Beta (sum of marginal KLs)."""
        bkd = self._bkd
        alphas_q = [3.0, 5.0]
        betas_q = [2.0, 4.0]
        alphas_p = [2.0, 3.0]
        betas_p = [3.0, 5.0]
        family = BetaVariationalFamily(
            2, bkd, alpha_init=alphas_q, beta_init=betas_q
        )
        prior = BetaVariationalFamily(
            2, bkd, alpha_init=alphas_p, beta_init=betas_p
        )

        kl = family.kl_divergence(prior)

        # KL = sum of marginal KLs
        from scipy.special import gammaln, digamma
        kl_expected = 0.0
        for i in range(2):
            a1, b1 = alphas_q[i], betas_q[i]
            a2, b2 = alphas_p[i], betas_p[i]
            kl_expected += (
                gammaln(a2) + gammaln(b2) - gammaln(a2 + b2)
                - gammaln(a1) - gammaln(b1) + gammaln(a1 + b1)
                + (a1 - a2) * digamma(a1)
                + (b1 - b2) * digamma(b1)
                + (a2 - a1 + b2 - b1) * digamma(a1 + b1)
            )

        bkd.assert_allclose(
            bkd.asarray([float(kl)]),
            bkd.asarray([kl_expected]),
            rtol=1e-10,
        )

    def test_kl_divergence_non_beta_raises(self) -> None:
        """KL with non-Beta prior should raise NotImplementedError."""
        from pyapprox.typing.probability.gaussian.diagonal import (
            DiagonalMultivariateGaussian,
        )
        bkd = self._bkd
        family = BetaVariationalFamily(1, bkd)
        prior = DiagonalMultivariateGaussian(
            bkd.zeros((1, 1)), bkd.ones((1,)), bkd
        )
        with self.assertRaises(NotImplementedError):
            family.kl_divergence(prior)

    def test_base_distribution(self) -> None:
        family = BetaVariationalFamily(3, self._bkd)
        base_dist = family.base_distribution()
        self.assertIsInstance(base_dist, IndependentJoint)
        self.assertEqual(base_dist.nvars(), 3)
        # Samples should be in [0, 1]
        np.random.seed(42)
        samples = base_dist.rvs(100)
        samples_np = self._bkd.to_numpy(samples)
        self.assertTrue(np.all(samples_np >= 0.0))
        self.assertTrue(np.all(samples_np <= 1.0))

    def test_set_params_updates(self) -> None:
        """Changing hyp_list values should change reparameterize output."""
        family = BetaVariationalFamily(1, self._bkd)
        np.random.seed(42)
        base_samples = self._bkd.asarray(
            np.random.uniform(0.01, 0.99, (1, 50))
        )
        z1 = family.reparameterize(base_samples)

        # Set alpha=10.0, beta=1.0 → distribution skewed toward 1
        new_values = self._bkd.asarray(
            [float(np.log(10.0)), float(np.log(1.0))]
        )
        family.hyp_list().set_active_values(new_values)

        z2 = family.reparameterize(base_samples)

        # Mean should shift toward 1 for Beta(10,1)
        z1_mean = float(self._bkd.to_numpy(self._bkd.mean(z1)))
        z2_mean = float(self._bkd.to_numpy(self._bkd.mean(z2)))
        self.assertGreater(z2_mean, z1_mean)

    def test_repr(self) -> None:
        family = BetaVariationalFamily(2, self._bkd)
        r = repr(family)
        self.assertIn("BetaVariationalFamily", r)
        self.assertIn("nvars=2", r)


class TestBetaVariationalFamilyNumpy(
    TestBetaVariationalFamilyBase[NDArray[Any]], unittest.TestCase
):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestBetaVariationalFamilyTorch(
    TestBetaVariationalFamilyBase[torch.Tensor], unittest.TestCase
):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


from pyapprox.typing.util.test_utils import load_tests  # noqa: F401
