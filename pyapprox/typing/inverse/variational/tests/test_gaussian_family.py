"""
Tests for GaussianVariationalFamily.
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
from pyapprox.typing.inverse.variational.protocols import (
    VariationalFamilyProtocol,
)
from pyapprox.typing.inverse.variational.gaussian_family import (
    GaussianVariationalFamily,
)


class TestGaussianVariationalFamilyBase(Generic[Array], unittest.TestCase):
    """Base test class for GaussianVariationalFamily."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self._nvars = 3

    def test_satisfies_protocol(self) -> None:
        family = GaussianVariationalFamily(self._nvars, self._bkd)
        self.assertIsInstance(family, VariationalFamilyProtocol)

    def test_hyp_list_shape(self) -> None:
        family = GaussianVariationalFamily(self._nvars, self._bkd)
        # 3 means + 3 log-stdevs = 6 params
        self.assertEqual(family.hyp_list().nparams(), 6)
        self.assertEqual(family.hyp_list().nactive_params(), 6)

    def test_nvars(self) -> None:
        family = GaussianVariationalFamily(self._nvars, self._bkd)
        self.assertEqual(family.nvars(), 3)

    def test_reparameterize_shape(self) -> None:
        family = GaussianVariationalFamily(self._nvars, self._bkd)
        np.random.seed(42)
        base_samples = self._bkd.asarray(
            np.random.normal(0, 1, (3, 10))
        )
        z = family.reparameterize(base_samples)
        self.assertEqual(z.shape, (3, 10))

    def test_reparameterize_statistics(self) -> None:
        mean_vals = [1.0, 2.0, 3.0]
        stdev_vals = [0.5, 0.5, 0.5]
        family = GaussianVariationalFamily(
            self._nvars, self._bkd,
            mean_init=mean_vals, stdev_init=stdev_vals,
        )
        np.random.seed(42)
        base_samples = self._bkd.asarray(
            np.random.normal(0, 1, (3, 10000))
        )
        z = family.reparameterize(base_samples)
        z_np = self._bkd.to_numpy(z)
        for i in range(3):
            self._bkd.assert_allclose(
                self._bkd.asarray([np.mean(z_np[i])]),
                self._bkd.asarray([mean_vals[i]]),
                rtol=0.05,
            )
            self._bkd.assert_allclose(
                self._bkd.asarray([np.std(z_np[i])]),
                self._bkd.asarray([stdev_vals[i]]),
                rtol=0.05,
            )

    def test_logpdf_shape(self) -> None:
        family = GaussianVariationalFamily(self._nvars, self._bkd)
        samples = self._bkd.asarray(
            np.random.normal(0, 1, (3, 5))
        )
        logp = family.logpdf(samples)
        self.assertEqual(logp.shape, (1, 5))

    def test_logpdf_matches_diagonal_gaussian(self) -> None:
        mean_vals = [1.0, 2.0, 3.0]
        stdev_vals = [0.5, 1.0, 1.5]
        family = GaussianVariationalFamily(
            self._nvars, self._bkd,
            mean_init=mean_vals, stdev_init=stdev_vals,
        )
        np.random.seed(42)
        samples = self._bkd.asarray(
            np.random.normal(0, 1, (3, 10))
        )
        logp_family = family.logpdf(samples)

        # Compare with DiagonalMultivariateGaussian
        mean_arr = self._bkd.asarray([[1.0], [2.0], [3.0]])
        var_arr = self._bkd.asarray([0.25, 1.0, 2.25])
        dist = DiagonalMultivariateGaussian(mean_arr, var_arr, self._bkd)
        logp_ref = dist.logpdf(samples)

        self._bkd.assert_allclose(logp_family, logp_ref, rtol=1e-12)

    def test_kl_divergence_analytical(self) -> None:
        mean_vals = [1.0, 2.0, 3.0]
        stdev_vals = [0.5, 1.0, 1.5]
        family = GaussianVariationalFamily(
            self._nvars, self._bkd,
            mean_init=mean_vals, stdev_init=stdev_vals,
        )

        # Prior: standard normal
        prior_mean = self._bkd.zeros((3, 1))
        prior_var = self._bkd.ones((3,))
        prior = DiagonalMultivariateGaussian(prior_mean, prior_var, self._bkd)

        kl = family.kl_divergence(prior)

        # Compare with DiagonalMultivariateGaussian.kl_divergence
        q_mean = self._bkd.asarray([[1.0], [2.0], [3.0]])
        q_var = self._bkd.asarray([0.25, 1.0, 2.25])
        q_dist = DiagonalMultivariateGaussian(q_mean, q_var, self._bkd)
        kl_ref = q_dist.kl_divergence(prior)

        self._bkd.assert_allclose(
            self._bkd.asarray([float(kl)]),
            self._bkd.asarray([float(kl_ref)]),
            rtol=1e-12,
        )

    def test_kl_divergence_self_is_zero(self) -> None:
        mean_vals = [1.0, 2.0, 3.0]
        stdev_vals = [0.5, 1.0, 1.5]
        family = GaussianVariationalFamily(
            self._nvars, self._bkd,
            mean_init=mean_vals, stdev_init=stdev_vals,
        )

        # Prior with same params
        prior_mean = self._bkd.asarray([[1.0], [2.0], [3.0]])
        prior_var = self._bkd.asarray([0.25, 1.0, 2.25])
        prior = DiagonalMultivariateGaussian(
            prior_mean, prior_var, self._bkd
        )

        kl = family.kl_divergence(prior)
        self._bkd.assert_allclose(
            self._bkd.asarray([float(kl)]),
            self._bkd.asarray([0.0]),
            atol=1e-10,
        )

    def test_set_params_updates(self) -> None:
        family = GaussianVariationalFamily(self._nvars, self._bkd)
        np.random.seed(42)
        base_samples = self._bkd.asarray(
            np.random.normal(0, 1, (3, 5))
        )

        z1 = family.reparameterize(base_samples)

        # Change means
        new_values = self._bkd.asarray([10.0, 20.0, 30.0, 0.0, 0.0, 0.0])
        family.hyp_list().set_active_values(new_values)

        z2 = family.reparameterize(base_samples)

        # z2 should have much larger values than z1
        z1_mean = float(self._bkd.to_numpy(self._bkd.mean(z1)))
        z2_mean = float(self._bkd.to_numpy(self._bkd.mean(z2)))
        self.assertGreater(z2_mean, z1_mean + 5.0)

    def test_reparameterize_per_sample_params(self) -> None:
        family = GaussianVariationalFamily(self._nvars, self._bkd)
        np.random.seed(42)
        base_samples = self._bkd.asarray(
            np.random.normal(0, 1, (3, 10))
        )

        # Per-sample params: (6, 10) — means then log-stdevs
        # Make sample 0 have mean [10, 20, 30], sample 1 have mean [0, 0, 0]
        params = self._bkd.zeros((6, 10))
        # Set means for column 0
        params_np = self._bkd.to_numpy(params).copy()
        params_np[0, 0] = 10.0
        params_np[1, 0] = 20.0
        params_np[2, 0] = 30.0
        params = self._bkd.asarray(params_np)

        z = family.reparameterize(base_samples, params)
        z_np = self._bkd.to_numpy(z)

        # Sample 0 should be near [10, 20, 30] + noise
        self.assertGreater(z_np[0, 0], 5.0)
        self.assertGreater(z_np[1, 0], 15.0)
        self.assertGreater(z_np[2, 0], 25.0)

        # Sample 1 should be near [0, 0, 0] + noise
        self.assertLess(abs(z_np[0, 1]), 5.0)

    def test_logpdf_per_sample_params(self) -> None:
        family = GaussianVariationalFamily(self._nvars, self._bkd)
        np.random.seed(42)
        samples = self._bkd.asarray(
            np.random.normal(0, 1, (3, 5))
        )

        # Per-sample params: (6, 5)
        # All zeros = standard normal
        params = self._bkd.zeros((6, 5))
        logp = family.logpdf(samples, params)
        self.assertEqual(logp.shape, (1, 5))

        # Compare with shared-params standard normal
        logp_shared = family.logpdf(samples)
        self._bkd.assert_allclose(logp, logp_shared, rtol=1e-12)

    def test_base_distribution(self) -> None:
        family = GaussianVariationalFamily(self._nvars, self._bkd)
        base_dist = family.base_distribution()
        self.assertIsInstance(base_dist, DiagonalMultivariateGaussian)
        self.assertEqual(base_dist.nvars(), self._nvars)

    def test_custom_init(self) -> None:
        family = GaussianVariationalFamily(
            self._nvars, self._bkd,
            mean_init=[1.0, 2.0, 3.0],
            stdev_init=[0.1, 0.2, 0.3],
        )
        vals = self._bkd.to_numpy(family.hyp_list().get_active_values())
        # Active values are in unconstrained space
        # For mean: unconstrained = value
        # For log-stdev: unconstrained = log(value)
        self._bkd.assert_allclose(
            self._bkd.asarray(vals[:3]),
            self._bkd.asarray([1.0, 2.0, 3.0]),
            rtol=1e-12,
        )


class TestGaussianVariationalFamilyNumpy(
    TestGaussianVariationalFamilyBase[NDArray[Any]], unittest.TestCase
):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGaussianVariationalFamilyTorch(
    TestGaussianVariationalFamilyBase[torch.Tensor], unittest.TestCase
):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


from pyapprox.typing.util.test_utils import load_tests  # noqa: F401
