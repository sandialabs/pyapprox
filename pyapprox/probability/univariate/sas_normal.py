"""
Sinh-arcsinh normal (SAS) univariate distribution.

The sinh-arcsinh transform (Jones & Pewsey 2009) applied to a standard
normal produces a flexible four-parameter family with controllable
skewness and tail weight:

    z = xi + eta * sinh((arcsinh(n) + epsilon) / delta)

where n ~ N(0,1) and:
- xi: location
- eta > 0: scale
- epsilon: skewness (0 = symmetric)
- delta > 0: tail weight (1 = Gaussian tails)

At epsilon=0, delta=1 this reduces exactly to N(xi, eta^2).
"""

# TODO: Create conditional SAS normal marginal for use with amortized
# VI (optional if need arises)

import math
from typing import Any, Generic

from pyapprox.probability.protocols.distribution import MarginalProtocol
import numpy as np

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import (
    HyperParameter,
    HyperParameterList,
    LogHyperParameter,
)


class SASNormalMarginal(Generic[Array]):
    """
    Sinh-arcsinh normal distribution.

    Implements the MarginalProtocol interface with analytical formulas for
    PDF, CDF, inverse CDF, and reparameterization.

    Parameters
    ----------
    xi : float
        Location parameter.
    eta : float
        Scale parameter (positive).
    epsilon : float
        Skewness parameter (0 = symmetric).
    delta : float
        Tail weight parameter (positive, 1 = Gaussian tails).
    bkd : Backend[Array]
        The backend to use for computations.
    """

    def __init__(
        self,
        xi: float,
        eta: float,
        epsilon: float,
        delta: float,
        bkd: Backend[Array],
    ):
        self._bkd = bkd

        self._xi_hyp = HyperParameter(
            name="xi",
            nparams=1,
            values=xi,
            bounds=(-1e10, 1e10),
            bkd=bkd,
        )
        self._eta_hyp = LogHyperParameter(
            name="eta",
            nparams=1,
            user_values=eta,
            user_bounds=(1e-10, 1e10),
            bkd=bkd,
        )
        self._epsilon_hyp = HyperParameter(
            name="epsilon",
            nparams=1,
            values=epsilon,
            bounds=(-1e10, 1e10),
            bkd=bkd,
        )
        self._delta_hyp = LogHyperParameter(
            name="delta",
            nparams=1,
            user_values=delta,
            user_bounds=(1e-10, 1e10),
            bkd=bkd,
        )
        self._hyp_list = HyperParameterList(
            [self._xi_hyp, self._eta_hyp, self._epsilon_hyp, self._delta_hyp]
        )
        self._log_2pi = math.log(2.0 * math.pi)

    def _validate_input(self, samples: Array) -> Array:
        """Validate that input is 2D with shape (1, nsamples)."""
        if samples.ndim != 2:
            raise ValueError(
                f"Expected 2D array with shape (1, nsamples), got {samples.ndim}D"
            )
        if samples.shape[0] != 1:
            raise ValueError(
                f"Univariate distribution expects shape (1, nsamples), "
                f"got {samples.shape}"
            )
        return samples[0]

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def hyp_list(self) -> HyperParameterList[Array]:
        """Return the hyperparameter list for parameter optimization."""
        return self._hyp_list

    def nparams(self) -> int:
        """Return the number of distribution parameters."""
        return self._hyp_list.nparams()

    def nvars(self) -> int:
        """Return the number of variables (always 1 for univariate)."""
        return 1

    def _get_xi(self) -> Array:
        """Get xi as array (preserves autograd graph)."""
        return self._xi_hyp.get_values()[0]

    def _get_eta(self) -> Array:
        """Get eta as array (preserves autograd graph)."""
        return self._eta_hyp.exp_values()[0]

    def _get_epsilon(self) -> Array:
        """Get epsilon as array (preserves autograd graph)."""
        return self._epsilon_hyp.get_values()[0]

    def _get_delta(self) -> Array:
        """Get delta as array (preserves autograd graph)."""
        return self._delta_hyp.exp_values()[0]

    def _sas_forward(self, n: Array) -> Array:
        """Apply SAS transform: n -> z = xi + eta * sinh((arcsinh(n) + eps) / delta)."""
        xi = self._get_xi()
        eta = self._get_eta()
        eps = self._get_epsilon()
        delta = self._get_delta()
        return xi + eta * self._bkd.sinh(
            (self._bkd.arcsinh(n) + eps) / delta
        )

    def _sas_inverse(self, s_1d: Array) -> Array:
        """Apply inverse SAS: s -> x = sinh(delta * arcsinh((s - xi) / eta) - eps).

        Returns the standard normal argument x such that Phi(x) = F_SAS(s).
        """
        xi = self._get_xi()
        eta = self._get_eta()
        eps = self._get_epsilon()
        delta = self._get_delta()
        t = (s_1d - xi) / eta
        return self._bkd.sinh(delta * self._bkd.arcsinh(t) - eps)

    def logpdf(self, samples: Array) -> Array:
        """
        Evaluate the log probability density function.

        log f(s) = log phi(x) + log cosh(delta * arcsinh(t) - eps)
                   + log delta - log eta - 0.5 * log(1 + t^2)

        where t = (s - xi)/eta, x = sinh(delta * arcsinh(t) - eps),
        and phi is the standard normal density.

        Parameters
        ----------
        samples : Array
            Shape: (1, nsamples)

        Returns
        -------
        Array
            Shape: (1, nsamples)
        """
        s_1d = self._validate_input(samples)
        xi = self._get_xi()
        eta = self._get_eta()
        eps = self._get_epsilon()
        delta = self._get_delta()

        t = (s_1d - xi) / eta
        arg = delta * self._bkd.arcsinh(t) - eps
        x = self._bkd.sinh(arg)

        # log phi(x) = -0.5 * log(2*pi) - 0.5 * x^2
        log_phi = -0.5 * self._log_2pi - 0.5 * x**2
        log_cosh = self._bkd.log(self._bkd.cosh(arg))
        log_delta = self._bkd.log(delta)
        log_eta = self._bkd.log(eta)
        log_jacobian = -0.5 * self._bkd.log(1.0 + t**2)

        result = log_phi + log_cosh + log_delta - log_eta + log_jacobian
        return self._bkd.reshape(result, (1, -1))

    def pdf(self, samples: Array) -> Array:
        """
        Evaluate the probability density function.

        Parameters
        ----------
        samples : Array
            Shape: (1, nsamples)

        Returns
        -------
        Array
            Shape: (1, nsamples)
        """
        return self._bkd.exp(self.logpdf(samples))

    def __call__(self, samples: Array) -> Array:
        """Evaluate the PDF (alias for pdf())."""
        return self.pdf(samples)

    def cdf(self, samples: Array) -> Array:
        """
        Evaluate the cumulative distribution function.

        F_SAS(s) = Phi(x) where x = sinh(delta * arcsinh((s-xi)/eta) - eps).

        Parameters
        ----------
        samples : Array
            Shape: (1, nsamples)

        Returns
        -------
        Array
            Shape: (1, nsamples)
        """
        s_1d = self._validate_input(samples)
        x = self._sas_inverse(s_1d)
        result = 0.5 * (1.0 + self._bkd.erf(x / math.sqrt(2.0)))
        return self._bkd.reshape(result, (1, -1))

    def invcdf(self, probs: Array) -> Array:
        """
        Evaluate the inverse CDF (quantile function).

        F^{-1}(u) = xi + eta * sinh((arcsinh(Phi^{-1}(u)) + eps) / delta)

        Parameters
        ----------
        probs : Array
            Shape: (1, nsamples)

        Returns
        -------
        Array
            Shape: (1, nsamples)
        """
        p_1d = self._validate_input(probs)
        # Phi^{-1}(u) = sqrt(2) * erfinv(2*u - 1)
        n = math.sqrt(2.0) * self._bkd.erfinv(2.0 * p_1d - 1.0)
        result = self._sas_forward(n)
        return self._bkd.reshape(result, (1, -1))

    ppf = invcdf

    def reparameterize(self, base_samples: Array) -> Array:
        """Transform standard normal base samples to SAS samples.

        z = xi + eta * sinh((arcsinh(n) + eps) / delta)

        Parameters
        ----------
        base_samples : Array
            Samples from N(0,1), shape ``(1, nsamples)``.

        Returns
        -------
        Array
            Samples from this distribution, shape ``(1, nsamples)``.
        """
        n_1d = self._validate_input(base_samples)
        result = self._sas_forward(n_1d)
        return self._bkd.reshape(result, (1, -1))

    def base_distribution(self) -> "SASNormalMarginal[Array]":
        """Return the base distribution for reparameterization (standard normal)."""
        from pyapprox.probability.univariate.gaussian import GaussianMarginal

        return GaussianMarginal(0.0, 1.0, self._bkd)

    def is_bounded(self) -> bool:
        """SAS distribution has unbounded support."""
        return False

    def rvs(self, nsamples: int) -> Array:
        """
        Generate random samples from the distribution.

        Parameters
        ----------
        nsamples : int
            Number of samples to generate.

        Returns
        -------
        Array
            Random samples. Shape: (1, nsamples)
        """
        n = self._bkd.asarray(np.random.normal(0, 1, nsamples))
        n_2d = self._bkd.reshape(n, (1, nsamples))
        return self.reparameterize(n_2d)

    def __eq__(self, other: Any) -> bool:
        """Check equality with another SASNormalMarginal."""
        if not isinstance(other, SASNormalMarginal):
            return False
        bkd = self._bkd
        return bool(
            bkd.allclose(
                self._hyp_list.get_values(),
                other._hyp_list.get_values(),
                atol=1e-10,
            )
        )

    def __repr__(self) -> str:
        """Return string representation."""
        xi = self._bkd.to_float(self._xi_hyp.get_values()[0])
        eta = self._bkd.to_float(self._eta_hyp.exp_values()[0])
        eps = self._bkd.to_float(self._epsilon_hyp.get_values()[0])
        delta = self._bkd.to_float(self._delta_hyp.exp_values()[0])
        return (
            f"SASNormalMarginal(xi={xi}, eta={eta}, "
            f"epsilon={eps}, delta={delta})"
        )
