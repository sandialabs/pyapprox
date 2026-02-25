"""
Protocols for copula models.

Defines the interface for copulas at different capability levels.

Protocol Hierarchy
------------------
CopulaProtocol
    Base protocol with logpdf and sampling on [0,1]^d.
CopulaWithKLProtocol
    Adds KL divergence computation.
"""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import HyperParameterList


@runtime_checkable
class CopulaProtocol(Protocol, Generic[Array]):
    """
    Base protocol for copula models.

    A copula is a multivariate distribution on [0,1]^d with uniform
    marginals. It captures the dependence structure between variables
    independently of their marginal distributions.

    Methods
    -------
    bkd()
        Get the computational backend.
    nvars()
        Number of variables (dimension of the copula).
    nparams()
        Number of free parameters.
    hyp_list()
        Return the hyperparameter list for parameter optimization.
    logpdf(u)
        Evaluate log copula density.
    sample(nsamples)
        Draw samples from the copula.
    """

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        ...

    def nvars(self) -> int:
        """
        Return the number of variables.

        Returns
        -------
        int
            Dimension of the copula.
        """
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

    def hyp_list(self) -> HyperParameterList:
        """
        Return the hyperparameter list for parameter optimization.

        Returns
        -------
        HyperParameterList
            Hyperparameter list containing copula parameters.
        """
        ...

    def logpdf(self, u: Array) -> Array:
        """
        Evaluate the log copula density.

        Parameters
        ----------
        u : Array
            Points in [0,1]^d. Shape: (nvars, nsamples) - must be 2D

        Returns
        -------
        Array
            Log copula density values. Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D or has wrong first dimension
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
            Samples in [0,1]^d. Shape: (nvars, nsamples)
        """
        ...


@runtime_checkable
class CopulaWithKLProtocol(Protocol, Generic[Array]):
    """
    Copula with KL divergence support.

    Extends CopulaProtocol with analytical KL divergence computation.
    """

    def bkd(self) -> Backend[Array]: ...

    def nvars(self) -> int: ...

    def nparams(self) -> int: ...

    def hyp_list(self) -> HyperParameterList: ...

    def logpdf(self, u: Array) -> Array: ...

    def sample(self, nsamples: int) -> Array: ...

    def kl_divergence(self, other: "CopulaWithKLProtocol[Array]") -> Array:
        """
        Compute KL divergence KL(self || other).

        Parameters
        ----------
        other : CopulaWithKLProtocol
            The other copula.

        Returns
        -------
        Array
            KL divergence value (scalar Array, preserves autograd).
        """
        ...
