"""
Protocols for bivariate copula models.

Defines the interface for bivariate copulas including h-functions
(conditional CDFs) needed for vine copula construction.
"""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import HyperParameterList


@runtime_checkable
class BivariateCopulaProtocol(Protocol, Generic[Array]):
    """
    Protocol for bivariate copula models with h-functions.

    A bivariate copula on [0,1]^2 with methods for density evaluation,
    conditional CDFs (h-functions), and sampling. The h-function
    h(u1|u2) = dC(u1,u2)/du2 is the conditional CDF of U1 given U2,
    essential for vine copula construction.

    Methods
    -------
    bkd()
        Get the computational backend.
    nparams()
        Number of free parameters.
    hyp_list()
        Return the hyperparameter list for parameter optimization.
    logpdf(u)
        Evaluate log copula density.
    h_function(u1, u2)
        Conditional CDF: h(u1|u2) = dC(u1,u2)/du2.
    h_inverse(v, u2)
        Inverse of h-function: u1 = h^{-1}(v, u2).
    sample(nsamples)
        Draw samples from the copula.
    kendall_tau()
        Compute Kendall's tau from the copula parameter.
    """

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        ...

    def nparams(self) -> int:
        """
        Return the number of free parameters.

        Returns
        -------
        int
            Number of parameters.
        """
        ...

    def hyp_list(self) -> HyperParameterList[Array]:
        """
        Return the hyperparameter list for parameter optimization.

        Returns
        -------
        HyperParameterList[Array]
            Hyperparameter list containing copula parameters.
        """
        ...

    def logpdf(self, u: Array) -> Array:
        """
        Evaluate the log copula density.

        Parameters
        ----------
        u : Array
            Points in [0,1]^2. Shape: (2, nsamples)

        Returns
        -------
        Array
            Log copula density values. Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D or first dimension is not 2.
        """
        ...

    def h_function(self, u1: Array, u2: Array) -> Array:
        """
        Conditional CDF: h(u1|u2) = dC(u1,u2)/du2.

        Parameters
        ----------
        u1 : Array
            First variable values. Shape: (1, nsamples)
        u2 : Array
            Conditioning variable values. Shape: (1, nsamples)

        Returns
        -------
        Array
            Conditional CDF values. Shape: (1, nsamples)
        """
        ...

    def h_inverse(self, v: Array, u2: Array) -> Array:
        """
        Inverse of h-function: find u1 such that h(u1, u2) = v.

        Parameters
        ----------
        v : Array
            Target values. Shape: (1, nsamples)
        u2 : Array
            Conditioning variable values. Shape: (1, nsamples)

        Returns
        -------
        Array
            Recovered u1 values. Shape: (1, nsamples)
        """
        ...

    def sample(self, nsamples: int) -> Array:
        """
        Draw samples from the copula.

        Parameters
        ----------
        nsamples : int
            Number of samples to generate.

        Returns
        -------
        Array
            Samples in [0,1]^2. Shape: (2, nsamples)
        """
        ...

    def kendall_tau(self) -> Array:
        """
        Compute Kendall's tau from the copula parameter.

        Returns
        -------
        Array
            Kendall's tau as a scalar Array.
        """
        ...
