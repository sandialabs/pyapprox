"""
Protocols for correlation matrix parameterization.

Defines the interface for different strategies of representing and
computing with correlation matrices in copula models.
"""

from typing import Protocol, Generic, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import HyperParameterList


@runtime_checkable
class CorrelationParameterizationProtocol(Protocol, Generic[Array]):
    """
    Strategy for representing and operating on a correlation matrix.

    A correlation matrix Sigma is symmetric positive definite with
    unit diagonal. Different parameterizations offer different
    trade-offs between expressiveness and computational efficiency.

    Methods
    -------
    bkd()
        Get the computational backend.
    nvars()
        Dimension of the correlation matrix.
    nparams()
        Number of free parameters.
    hyp_list()
        Return the hyperparameter list for optimization.
    correlation_matrix()
        Materialize the full correlation matrix.
    log_det()
        Compute log|Sigma|.
    quad_form(z)
        Compute z^T (Sigma^{-1} - I) z for each sample.
    sample_transform(eps)
        Map standard normal samples to correlated samples.
    """

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        ...

    def nvars(self) -> int:
        """
        Return the dimension of the correlation matrix.

        Returns
        -------
        int
            Dimension d of the d x d correlation matrix.
        """
        ...

    def nparams(self) -> int:
        """
        Return the number of free parameters.

        Returns
        -------
        int
            Number of free parameters in this parameterization.
        """
        ...

    def hyp_list(self) -> HyperParameterList:
        """
        Return the hyperparameter list for optimization.

        Returns
        -------
        HyperParameterList
            Hyperparameter list containing correlation parameters.
        """
        ...

    def correlation_matrix(self) -> Array:
        """
        Materialize the full correlation matrix.

        Returns
        -------
        Array
            Correlation matrix Sigma. Shape: (nvars, nvars)
            Symmetric positive definite with unit diagonal.
        """
        ...

    def log_det(self) -> Array:
        """
        Compute log determinant of the correlation matrix.

        Returns
        -------
        Array
            log|Sigma| as a scalar Array (preserves autograd graph).
        """
        ...

    def quad_form(self, z: Array) -> Array:
        """
        Compute z^T (Sigma^{-1} - I) z for each sample column.

        This is the key quantity for the Gaussian copula log-density:
            log c(u) = -0.5 * log|Sigma| - 0.5 * quad_form(z)

        Parameters
        ----------
        z : Array
            Standard normal samples. Shape: (nvars, nsamples) - must be 2D

        Returns
        -------
        Array
            Quadratic form values. Shape: (nsamples,)

        Raises
        ------
        ValueError
            If input is not 2D or has wrong first dimension
        """
        ...

    def sample_transform(self, eps: Array) -> Array:
        """
        Map independent standard normal samples to correlated samples.

        Computes z = L @ eps where Sigma = L L^T.

        Parameters
        ----------
        eps : Array
            Independent standard normal samples.
            Shape: (nvars, nsamples) - must be 2D

        Returns
        -------
        Array
            Correlated samples z ~ N(0, Sigma).
            Shape: (nvars, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D or has wrong first dimension
        """
        ...
