"""
Copula-based joint distribution.

Composes a copula with marginal distributions to form a joint distribution:
    f(x) = c(F_1(x_1), ..., F_d(x_d)) * prod_i f_i(x_i)
    log f(x) = log c(u) + sum_i log f_i(x_i)
where u_i = F_i(x_i) is the probability integral transform.
"""

from typing import Generic, List, Sequence

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.hyperparameter import HyperParameterList
from pyapprox.typing.probability.copula.protocols import CopulaProtocol
from pyapprox.typing.probability.protocols import MarginalProtocol


class CopulaDistribution(Generic[Array]):
    """
    Joint distribution from a copula and marginals.

    The joint density is:
        f(x) = c(F_1(x_1), ..., F_d(x_d)) * prod_i f_i(x_i)

    Parameters
    ----------
    copula : CopulaProtocol[Array]
        The copula modeling the dependence structure.
    marginals : List[MarginalProtocol[Array]]
        The marginal distributions.
    bkd : Backend[Array]
        Computational backend.

    Raises
    ------
    ValueError
        If copula dimension doesn't match number of marginals.
    """

    def __init__(
        self,
        copula: CopulaProtocol[Array],
        marginals: List[MarginalProtocol[Array]],
        bkd: Backend[Array],
    ):
        if copula.nvars() != len(marginals):
            raise ValueError(
                f"Copula has {copula.nvars()} variables but "
                f"{len(marginals)} marginals were provided"
            )
        self._copula = copula
        self._marginals = marginals
        self._bkd = bkd
        self._nvars = len(marginals)

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables."""
        return self._nvars

    def nparams(self) -> int:
        """Return the total number of parameters (copula + all marginals)."""
        total = self._copula.nparams()
        for m in self._marginals:
            if hasattr(m, "nparams"):
                total += m.nparams()
        return total

    def copula(self) -> CopulaProtocol[Array]:
        """Return the copula."""
        return self._copula

    def marginals(self) -> List[MarginalProtocol[Array]]:
        """Return the marginal distributions."""
        return self._marginals

    def hyp_list(self) -> HyperParameterList:
        """
        Return concatenated hyperparameter list (copula + marginals).

        Returns
        -------
        HyperParameterList
            Combined hyperparameter list.
        """
        combined = self._copula.hyp_list()
        for m in self._marginals:
            if hasattr(m, "hyp_list"):
                combined = combined + m.hyp_list()
        return combined

    def _validate_input(self, samples: Array) -> None:
        """Validate that input is 2D with shape (nvars, nsamples)."""
        if samples.ndim != 2:
            raise ValueError(
                f"Expected 2D array with shape (nvars, nsamples), "
                f"got {samples.ndim}D"
            )
        if samples.shape[0] != self._nvars:
            raise ValueError(
                f"Expected {self._nvars} variables, got {samples.shape[0]}"
            )

    def logpdf(self, samples: Array) -> Array:
        """
        Evaluate the log joint density.

        log f(x) = log c(u_1, ..., u_d) + sum_i log f_i(x_i)
        where u_i = F_i(x_i)

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (nvars, nsamples) - must be 2D

        Returns
        -------
        Array
            Log joint density values. Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D or has wrong first dimension
        """
        self._validate_input(samples)
        nsamples = samples.shape[1]

        # Apply PIT: u_i = F_i(x_i)
        u = self._bkd.zeros((self._nvars, nsamples))
        marginal_logpdf_sum = self._bkd.zeros((1, nsamples))

        for i, marginal in enumerate(self._marginals):
            row_2d = self._bkd.reshape(samples[i], (1, -1))
            u[i] = marginal.cdf(row_2d)[0]
            marginal_logpdf_sum = marginal_logpdf_sum + marginal.logpdf(row_2d)

        # Copula log-density
        copula_logpdf = self._copula.logpdf(u)

        return copula_logpdf + marginal_logpdf_sum

    def sample(self, nsamples: int) -> Array:
        """
        Generate random samples from the joint distribution.

        Parameters
        ----------
        nsamples : int
            Number of samples to generate.

        Returns
        -------
        Array
            Random samples. Shape: (nvars, nsamples)
        """
        # Sample from copula
        u = self._copula.sample(nsamples)

        # Apply inverse PIT: x_i = F_i^{-1}(u_i)
        x = self._bkd.zeros((self._nvars, nsamples))
        for i, marginal in enumerate(self._marginals):
            u_row = self._bkd.reshape(u[i], (1, -1))
            x[i] = marginal.invcdf(u_row)[0]

        return x

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"CopulaDistribution(nvars={self._nvars}, "
            f"copula={self._copula!r})"
        )
