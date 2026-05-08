from typing import Generic, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array_co


@runtime_checkable
class OptimizerResultProtocol(Generic[Array_co], Protocol):
    """
    Protocol for the return class of optimizers.

    Defines the required attributes and methods for optimizer results, including
    the minimum value, success status, solution vector (optima), termination message,
    and access to the raw result.
    """

    def fun(self) -> float:
        """
        Get the minimum value of the objective function.

        Returns
        -------
        float
            Minimum value of the objective function.
        """
        ...

    def success(self) -> bool:
        """
        Get the success status of the optimization.

        Returns
        -------
        bool
            True if the optimization was successful, False otherwise.
        """
        ...

    def optima(self) -> Array_co:
        """
        Get the solution vector (optimal values of decision variables).

        Returns
        -------
        Optional[Array]
            Solution vector if available, otherwise None.
        """
        ...
