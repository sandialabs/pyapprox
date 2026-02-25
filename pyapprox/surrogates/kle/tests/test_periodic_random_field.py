"""Tests for PeriodicReiszGaussianRandomField."""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.surrogates.kle.periodic_random_field import (
    PeriodicReiszGaussianRandomField,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd


class TestPeriodicRandomField(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        np.random.seed(42)
        self._bkd = self.bkd()

    def test_periodic_field_shape(self) -> None:
        """Test output shape for given inputs."""
        bkd = self._bkd
        sigma, tau, gamma = 1.0, 1.0, 2.0
        neigs = 3
        bounds = bkd.array([0.0, 1.0])
        npts = 20

        field = PeriodicReiszGaussianRandomField(
            sigma,
            tau,
            gamma,
            neigs,
            bounds,
            bkd,
        )
        domain_samples = bkd.linspace(0.0, 1.0, npts)[None, :]
        field.set_domain_samples(domain_samples)

        nsamples = 5
        samples = bkd.asarray(np.random.randn(2 * neigs, nsamples))
        result = field.values(samples)
        self.assertEqual(result.shape, (npts, nsamples))

    def test_periodic_field_zero_samples(self) -> None:
        """Test that zero random samples produce zero field."""
        bkd = self._bkd
        sigma, tau, gamma = 1.0, 1.0, 2.0
        neigs = 3
        bounds = bkd.array([0.0, 1.0])
        npts = 15

        field = PeriodicReiszGaussianRandomField(
            sigma,
            tau,
            gamma,
            neigs,
            bounds,
            bkd,
        )
        domain_samples = bkd.linspace(0.0, 1.0, npts)[None, :]
        field.set_domain_samples(domain_samples)

        zero_samples = bkd.zeros((2 * neigs, 3))
        result = field.values(zero_samples)
        bkd.assert_allclose(result, bkd.zeros((npts, 3)), atol=1e-14)

    def test_periodic_field_rvs(self) -> None:
        """Test rvs produces correct shape and different realizations."""
        bkd = self._bkd
        sigma, tau, gamma = 1.0, 1.0, 2.0
        neigs = 3
        bounds = bkd.array([0.0, 1.0])
        npts = 20

        field = PeriodicReiszGaussianRandomField(
            sigma,
            tau,
            gamma,
            neigs,
            bounds,
            bkd,
        )
        domain_samples = bkd.linspace(0.0, 1.0, npts)[None, :]
        field.set_domain_samples(domain_samples)

        nsamples = 10
        result = field.rvs(nsamples)
        self.assertEqual(result.shape, (npts, nsamples))

        # Different realizations should generally differ
        self.assertFalse(bkd.all_bool(result[:, 0] == result[:, 1]))

    def test_periodic_field_eigenvalue_decay(self) -> None:
        """Test that eigenvalues decay with increasing k."""
        bkd = self._bkd
        sigma, tau, gamma = 1.0, 1.0, 3.0
        neigs = 10
        bounds = bkd.array([0.0, 1.0])

        field = PeriodicReiszGaussianRandomField(
            sigma,
            tau,
            gamma,
            neigs,
            bounds,
            bkd,
        )

        eigs = field._eigs[:, 0]  # shape (neigs,)
        # Eigenvalues should be positive and decreasing
        self.assertTrue(bkd.all_bool(eigs > 0))
        for i in range(neigs - 1):
            self.assertGreater(
                float(eigs[i]),
                float(eigs[i + 1]),
            )

    def test_periodic_field_nterms(self) -> None:
        """Test nterms equals 2*neigs."""
        bkd = self._bkd
        neigs = 5
        bounds = bkd.array([0.0, 1.0])
        field = PeriodicReiszGaussianRandomField(
            1.0,
            1.0,
            2.0,
            neigs,
            bounds,
            bkd,
        )
        self.assertEqual(field.nterms(), 2 * neigs)

    def test_periodic_field_wrong_samples_shape(self) -> None:
        """Test error on wrong sample shape."""
        bkd = self._bkd
        neigs = 3
        bounds = bkd.array([0.0, 1.0])
        field = PeriodicReiszGaussianRandomField(
            1.0,
            1.0,
            2.0,
            neigs,
            bounds,
            bkd,
        )
        domain_samples = bkd.linspace(0.0, 1.0, 10)[None, :]
        field.set_domain_samples(domain_samples)

        # Wrong number of rows
        with self.assertRaises(ValueError):
            field.values(bkd.zeros((neigs, 2)))  # should be 2*neigs


class TestPeriodicRandomFieldNumpy(TestPeriodicRandomField[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestPeriodicRandomFieldTorch(TestPeriodicRandomField[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def bkd(self) -> TorchBkd:
        return TorchBkd()


from pyapprox.util.test_utils import load_tests  # noqa: F401

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = load_tests(loader, [], None)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
