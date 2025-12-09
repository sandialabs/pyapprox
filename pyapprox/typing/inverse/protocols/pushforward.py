"""
Protocols for Gaussian pushforward.

The pushforward of a Gaussian through a linear transformation:
    y = A @ x + b, where x ~ N(m, C)
gives:
    y ~ N(A @ m + b, A @ C @ A.T)

This is used to propagate posterior uncertainty through prediction models.
"""

from typing import Protocol, Generic, Any, runtime_checkable

from pyapprox.typing.util.backends.protocols import Array, Backend


@runtime_checkable
class GaussianPushforwardProtocol(Protocol, Generic[Array]):
    """
    Protocol for Gaussian pushforward through linear model.

    Given a Gaussian distribution on parameters and a linear prediction
    model, computes the resulting Gaussian distribution on predictions.
    """

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        ...

    def nvars(self) -> int:
        """
        Return the number of input variables.

        Returns
        -------
        int
            Number of input parameters.
        """
        ...

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest (outputs).

        Returns
        -------
        int
            Number of predicted outputs.
        """
        ...

    def mean(self) -> Array:
        """
        Return the pushforward mean.

        For y = A @ x + b with x ~ N(m, C):
            E[y] = A @ m + b

        Returns
        -------
        Array
            Pushforward mean. Shape: (nqoi, 1)
        """
        ...

    def covariance(self) -> Array:
        """
        Return the pushforward covariance.

        For y = A @ x + b with x ~ N(m, C):
            Cov[y] = A @ C @ A.T

        Returns
        -------
        Array
            Pushforward covariance. Shape: (nqoi, nqoi)
        """
        ...

    def pushforward_variable(self) -> Any:
        """
        Return the pushforward as a Gaussian distribution object.

        Returns
        -------
        Any
            Gaussian distribution representing the pushforward.
        """
        ...
