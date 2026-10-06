"""Tests for PeriodicReiszGaussianRandomField."""

import numpy as np
import pytest

from pyapprox.surrogates.kle.periodic_random_field import (
    PeriodicReiszGaussianRandomField,
)


class TestPeriodicRandomField:

    def test_periodic_field_shape(self, bkd) -> None:
        """Test output shape for given inputs."""
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
        assert result.shape == (npts, nsamples)

    def test_periodic_field_zero_samples(self, bkd) -> None:
        """Test that zero random samples produce zero field."""
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

    def test_periodic_field_rvs(self, bkd) -> None:
        """Test rvs produces correct shape and different realizations."""
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
        assert result.shape == (npts, nsamples)

        # Different realizations should generally differ
        assert not bkd.all_bool(result[:, 0] == result[:, 1])

    def test_periodic_field_eigenvalue_decay(self, bkd) -> None:
        """Test that eigenvalues decay with increasing k."""
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
        assert bkd.all_bool(eigs > 0)
        for i in range(neigs - 1):
            assert float(eigs[i]) > float(eigs[i + 1])

    def test_periodic_field_nterms(self, bkd) -> None:
        """Test nterms equals 2*neigs."""
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
        assert field.nterms() == 2 * neigs

    def test_periodic_field_wrong_samples_shape(self, bkd) -> None:
        """Test error on wrong sample shape."""
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
        with pytest.raises(ValueError):
            field.values(bkd.zeros((neigs, 2)))  # should be 2*neigs
