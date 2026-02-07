"""
Tests for IndependentMarginalVariationalFamily.
"""

import unittest
from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray
import torch

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.probability.univariate.gaussian import GaussianMarginal
from pyapprox.typing.probability.univariate.beta import BetaMarginal
from pyapprox.typing.probability.joint.independent import IndependentJoint
from pyapprox.typing.inverse.variational.protocols import (
    VariationalFamilyProtocol,
)
from pyapprox.typing.inverse.variational.independent_family import (
    IndependentMarginalVariationalFamily,
)


class TestIndependentFamilyBase(Generic[Array], unittest.TestCase):
    """Base test class for IndependentMarginalVariationalFamily."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_satisfies_protocol(self) -> None:
        marginals = [BetaMarginal(2.0, 3.0, self._bkd)]
        family = IndependentMarginalVariationalFamily(marginals, self._bkd)
        self.assertIsInstance(family, VariationalFamilyProtocol)

    def test_hyp_list_shape_beta(self) -> None:
        marginals = [
            BetaMarginal(2.0, 3.0, self._bkd),
            BetaMarginal(4.0, 5.0, self._bkd),
        ]
        family = IndependentMarginalVariationalFamily(marginals, self._bkd)
        # 2 params per Beta marginal (log_alpha, log_beta) x 2 = 4
        self.assertEqual(family.hyp_list().nparams(), 4)
        self.assertEqual(family.hyp_list().nactive_params(), 4)

    def test_hyp_list_shape_gaussian(self) -> None:
        marginals = [
            GaussianMarginal(0.0, 1.0, self._bkd),
            GaussianMarginal(1.0, 2.0, self._bkd),
            GaussianMarginal(-1.0, 0.5, self._bkd),
        ]
        family = IndependentMarginalVariationalFamily(marginals, self._bkd)
        # 2 params per Gaussian marginal (mean, log_stdev) x 3 = 6
        self.assertEqual(family.hyp_list().nparams(), 6)

    def test_hyp_list_shape_mixed(self) -> None:
        marginals = [
            BetaMarginal(2.0, 3.0, self._bkd),
            GaussianMarginal(0.0, 1.0, self._bkd),
        ]
        family = IndependentMarginalVariationalFamily(marginals, self._bkd)
        # Beta: 2, Gaussian: 2 = 4
        self.assertEqual(family.hyp_list().nparams(), 4)

    def test_nvars(self) -> None:
        marginals = [
            BetaMarginal(2.0, 3.0, self._bkd),
            BetaMarginal(4.0, 5.0, self._bkd),
        ]
        family = IndependentMarginalVariationalFamily(marginals, self._bkd)
        self.assertEqual(family.nvars(), 2)

    def test_reparameterize_shape_beta(self) -> None:
        marginals = [
            BetaMarginal(2.0, 3.0, self._bkd),
            BetaMarginal(4.0, 5.0, self._bkd),
        ]
        family = IndependentMarginalVariationalFamily(marginals, self._bkd)
        np.random.seed(42)
        base_samples = self._bkd.asarray(
            np.random.uniform(0.01, 0.99, (2, 50))
        )
        z = family.reparameterize(base_samples)
        self.assertEqual(z.shape, (2, 50))

    def test_reparameterize_shape_gaussian(self) -> None:
        marginals = [
            GaussianMarginal(0.0, 1.0, self._bkd),
            GaussianMarginal(1.0, 2.0, self._bkd),
        ]
        family = IndependentMarginalVariationalFamily(marginals, self._bkd)
        np.random.seed(42)
        base_samples = self._bkd.asarray(
            np.random.randn(2, 50)
        )
        z = family.reparameterize(base_samples)
        self.assertEqual(z.shape, (2, 50))

    def test_reparameterize_in_bounds_beta(self) -> None:
        marginals = [
            BetaMarginal(2.0, 3.0, self._bkd),
            BetaMarginal(4.0, 5.0, self._bkd),
        ]
        family = IndependentMarginalVariationalFamily(marginals, self._bkd)
        np.random.seed(42)
        base_samples = self._bkd.asarray(
            np.random.uniform(0.01, 0.99, (2, 100))
        )
        z = family.reparameterize(base_samples)
        z_np = self._bkd.to_numpy(z)
        self.assertTrue(np.all(z_np >= 0.0))
        self.assertTrue(np.all(z_np <= 1.0))

    def test_logpdf_shape(self) -> None:
        marginals = [
            BetaMarginal(2.0, 3.0, self._bkd),
            BetaMarginal(4.0, 5.0, self._bkd),
        ]
        family = IndependentMarginalVariationalFamily(marginals, self._bkd)
        samples = self._bkd.asarray([[0.2, 0.4, 0.6], [0.3, 0.5, 0.7]])
        logp = family.logpdf(samples)
        self.assertEqual(logp.shape, (1, 3))

    def test_logpdf_matches_independent_joint_beta(self) -> None:
        marginals = [
            BetaMarginal(2.0, 3.0, self._bkd),
            BetaMarginal(4.0, 5.0, self._bkd),
        ]
        family = IndependentMarginalVariationalFamily(marginals, self._bkd)
        samples = self._bkd.asarray([[0.2, 0.4, 0.6], [0.3, 0.5, 0.7]])
        logp_family = family.logpdf(samples)
        logp_ref = IndependentJoint(marginals, self._bkd).logpdf(samples)
        self._bkd.assert_allclose(logp_family, logp_ref, rtol=1e-10)

    def test_logpdf_matches_independent_joint_gaussian(self) -> None:
        marginals = [
            GaussianMarginal(0.0, 1.0, self._bkd),
            GaussianMarginal(1.0, 2.0, self._bkd),
        ]
        family = IndependentMarginalVariationalFamily(marginals, self._bkd)
        samples = self._bkd.asarray([[-1.0, 0.0, 1.0], [0.5, 1.5, 2.5]])
        logp_family = family.logpdf(samples)
        logp_ref = IndependentJoint(marginals, self._bkd).logpdf(samples)
        self._bkd.assert_allclose(logp_family, logp_ref, rtol=1e-10)

    def test_kl_divergence_raises_non_independent_joint(self) -> None:
        """KL with non-IndependentJoint prior raises NotImplementedError."""
        marginals = [BetaMarginal(2.0, 3.0, self._bkd)]
        family = IndependentMarginalVariationalFamily(marginals, self._bkd)
        # Pass another family (not IndependentJoint) as prior
        prior = IndependentMarginalVariationalFamily(
            [BetaMarginal(1.0, 1.0, self._bkd)], self._bkd
        )
        with self.assertRaises(NotImplementedError):
            family.kl_divergence(prior)

    def test_kl_divergence_gaussian(self) -> None:
        """KL between two IndependentMarginal Gaussians matches DiagGauss KL."""
        from pyapprox.typing.probability.gaussian.diagonal import (
            DiagonalMultivariateGaussian,
        )
        bkd = self._bkd
        q_marginals = [
            GaussianMarginal(1.0, 0.5, bkd),
            GaussianMarginal(-1.0, 2.0, bkd),
        ]
        family = IndependentMarginalVariationalFamily(q_marginals, bkd)
        prior = IndependentJoint(
            [GaussianMarginal(0.0, 1.0, bkd),
             GaussianMarginal(0.0, 1.0, bkd)],
            bkd,
        )
        kl = family.kl_divergence(prior)

        # Reference: DiagonalMultivariateGaussian KL
        q_mv = DiagonalMultivariateGaussian(
            bkd.asarray([[1.0], [-1.0]]), bkd.asarray([0.25, 4.0]), bkd,
        )
        p_mv = DiagonalMultivariateGaussian(
            bkd.zeros((2, 1)), bkd.ones((2,)), bkd,
        )
        kl_ref = q_mv.kl_divergence(p_mv)
        bkd.assert_allclose(
            bkd.atleast_1d(kl), bkd.atleast_1d(bkd.asarray(kl_ref)),
            rtol=1e-12,
        )

    def test_kl_divergence_beta(self) -> None:
        """KL between two IndependentMarginal Betas matches manual KL."""
        from scipy.special import gammaln, digamma
        bkd = self._bkd
        a1, b1 = 3.0, 5.0
        a2, b2 = 2.0, 4.0
        family = IndependentMarginalVariationalFamily(
            [BetaMarginal(a1, b1, bkd)], bkd,
        )
        prior = IndependentJoint(
            [BetaMarginal(a2, b2, bkd)], bkd,
        )
        kl = family.kl_divergence(prior)
        kl_expected = (
            gammaln(a2) + gammaln(b2) - gammaln(a2 + b2)
            - gammaln(a1) - gammaln(b1) + gammaln(a1 + b1)
            + (a1 - a2) * digamma(a1)
            + (b1 - b2) * digamma(b1)
            + (a2 - a1 + b2 - b1) * digamma(a1 + b1)
        )
        bkd.assert_allclose(
            bkd.atleast_1d(kl), bkd.asarray([kl_expected]), rtol=1e-10,
        )

    def test_kl_divergence_self_is_zero(self) -> None:
        """KL(q || q) = 0."""
        bkd = self._bkd
        family = IndependentMarginalVariationalFamily(
            [GaussianMarginal(1.0, 2.0, bkd)], bkd,
        )
        prior = IndependentJoint(
            [GaussianMarginal(1.0, 2.0, bkd)], bkd,
        )
        kl = family.kl_divergence(prior)
        bkd.assert_allclose(bkd.atleast_1d(kl), bkd.asarray([0.0]), atol=1e-12)

    def test_base_distribution_heterogeneous(self) -> None:
        """Gaussian + Beta base = N(0,1) and U(0,1) rows."""
        marginals = [
            GaussianMarginal(0.0, 1.0, self._bkd),
            BetaMarginal(2.0, 3.0, self._bkd),
        ]
        family = IndependentMarginalVariationalFamily(marginals, self._bkd)
        base_dist = family.base_distribution()
        self.assertIsInstance(base_dist, IndependentJoint)
        self.assertEqual(base_dist.nvars(), 2)
        # Draw samples and verify shapes
        np.random.seed(42)
        samples = base_dist.rvs(100)
        self.assertEqual(samples.shape, (2, 100))
        samples_np = self._bkd.to_numpy(samples)
        # Row 1 (Beta base = Uniform(0,1)) should be in [0, 1]
        self.assertTrue(np.all(samples_np[1] >= 0.0))
        self.assertTrue(np.all(samples_np[1] <= 1.0))

    def test_set_params_updates(self) -> None:
        """Changing hyp_list values changes logpdf output."""
        marginals = [GaussianMarginal(0.0, 1.0, self._bkd)]
        family = IndependentMarginalVariationalFamily(marginals, self._bkd)
        samples = self._bkd.asarray([[0.0, 1.0, 2.0]])
        logp1 = family.logpdf(samples)

        # Shift mean to 2.0 → logpdf at x=2 should increase
        new_values = self._bkd.asarray(
            [2.0, float(np.log(1.0))]
        )
        family.hyp_list().set_active_values(new_values)
        logp2 = family.logpdf(samples)

        # logpdf at x=2 should be higher with mean=2 vs mean=0
        logp1_np = self._bkd.to_numpy(logp1)
        logp2_np = self._bkd.to_numpy(logp2)
        self.assertGreater(float(logp2_np[0, 2]), float(logp1_np[0, 2]))

    def test_repr(self) -> None:
        marginals = [
            BetaMarginal(2.0, 3.0, self._bkd),
            GaussianMarginal(0.0, 1.0, self._bkd),
        ]
        family = IndependentMarginalVariationalFamily(marginals, self._bkd)
        r = repr(family)
        self.assertIn("IndependentMarginalVariationalFamily", r)
        self.assertIn("nvars=2", r)

    def test_empty_marginals_raises(self) -> None:
        with self.assertRaises(ValueError):
            IndependentMarginalVariationalFamily([], self._bkd)


class TestIndependentFamilyNumpy(
    TestIndependentFamilyBase[NDArray[Any]], unittest.TestCase
):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestIndependentFamilyTorch(
    TestIndependentFamilyBase[torch.Tensor], unittest.TestCase
):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


from pyapprox.typing.util.test_utils import load_tests  # noqa: F401
