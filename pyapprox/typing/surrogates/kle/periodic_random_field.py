"""Periodic Riesz Gaussian random field using spectral representation.

Implements the Riesz kernel spectral decomposition for generating
periodic Gaussian random field realizations.
"""
from typing import Generic

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.affine.univariate.trigonometric import (
    TrigonometricPolynomial1D,
)
from pyapprox.typing.surrogates.affine.expansions.trigonometric import (
    TrigonometricExpansion,
)


class PeriodicReiszGaussianRandomField(Generic[Array]):
    """Periodic random field using Riesz kernel spectral representation.

    Eigenvalues: sqrt(2) * |sigma| * ((2*pi*k)^2 + tau^2)^(-gamma/2)
    Basis: trigonometric (cos/sin) on the domain.

    Parameters
    ----------
    sigma : float
        Amplitude parameter.
    tau : float
        Decay offset parameter.
    gamma : float
        Spectral decay rate.
    neigs : int
        Number of eigenfrequencies.
    bounds : Array
        Domain bounds [a, b].
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        sigma: float,
        tau: float,
        gamma: float,
        neigs: int,
        bounds: Array,
        bkd: Backend[Array],
    ):
        self._bkd = bkd
        self._sigma = sigma
        self._tau = tau
        self._gamma = gamma
        self._bounds = bounds

        self._neigs = None
        self._eigs = None
        self._trig_exp = None

        self.set_neigs(neigs)

    def set_neigs(self, neigs: int) -> None:
        """Set the number of eigenfrequencies and recompute eigenvalues.

        Parameters
        ----------
        neigs : int
            Number of eigenfrequencies.
        """
        bkd = self._bkd
        self._neigs = neigs
        k = bkd.arange(1, neigs + 1)
        self._eigs = (
            np.sqrt(2)
            * (
                abs(self._sigma)
                * ((2 * np.pi * k) ** 2 + self._tau ** 2)
                ** (-self._gamma / 2)
            )
        )[:, None]  # shape (neigs, 1)

        nterms = neigs * 2 + 1
        trig_basis = TrigonometricPolynomial1D(self._bounds, bkd)
        trig_basis.set_nterms(nterms)
        self._trig_exp = TrigonometricExpansion(trig_basis, bkd)

    def set_domain_samples(self, domain_samples: Array) -> None:
        """Set the spatial points at which the field is evaluated.

        Parameters
        ----------
        domain_samples : Array, shape (1, npts)
            Spatial evaluation points.
        """
        self._domain_samples = domain_samples
        self._nvars_spatial = domain_samples.shape[1]

    def nterms(self) -> int:
        """Return the number of random variables (2 * neigs)."""
        return self._trig_exp.nterms() - 1

    def nvars(self) -> int:
        """Return the number of spatial evaluation points."""
        if not hasattr(self, "_domain_samples"):
            raise RuntimeError("Must call set_domain_samples first.")
        return self._nvars_spatial

    def values(self, samples: Array) -> Array:
        """Evaluate the random field at given random samples.

        Parameters
        ----------
        samples : Array, shape (2*neigs, nsamples)
            Random coefficients. First neigs rows are alpha (cosine),
            last neigs rows are beta (sine).

        Returns
        -------
        Array, shape (npts, nsamples)
            Field values at the spatial points for each sample.
        """
        bkd = self._bkd
        if (
            samples.shape[0] != self._trig_exp.nterms() - 1
            or samples.ndim != 2
        ):
            raise ValueError(
                f"samples has wrong shape {samples.shape}, "
                f"expected ({self._trig_exp.nterms() - 1}, nsamples)"
            )
        if not hasattr(self, "_domain_samples"):
            raise RuntimeError("Must call set_domain_samples first.")

        nhalf = samples.shape[0] // 2
        alpha = self._eigs * samples[:nhalf]
        beta = self._eigs * samples[nhalf:]

        trig_coefs = bkd.vstack(
            (bkd.zeros((1, samples.shape[1])), alpha, beta)
        )  # shape (nterms, nsamples)

        self._trig_exp.set_coefficients(trig_coefs)

        # Shift domain samples by half the domain length
        shift = (self._bounds[1] - self._bounds[0]) / 2.0
        shifted_samples = self._domain_samples - shift
        return self._trig_exp(shifted_samples).T  # (npts, nsamples)

    def rvs(self, nsamples: int) -> Array:
        """Generate random field realizations.

        Parameters
        ----------
        nsamples : int
            Number of realizations.

        Returns
        -------
        Array, shape (npts, nsamples)
            Random field realizations.
        """
        return self.values(
            self._bkd.asarray(
                np.random.normal(0, 1, (2 * self._neigs, nsamples))
            )
        )
