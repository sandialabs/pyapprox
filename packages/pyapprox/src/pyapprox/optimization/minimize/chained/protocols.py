from typing import Generic, Protocol, runtime_checkable

from pyapprox.optimization.minimize.scipy.scipy_result import (
    ScipyOptimizerResultWrapper,
)
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class OptimizerProtocol(Protocol, Generic[Array]):
    """
    This protocol defines the required methods and attributes for an optimizer
    to be compatible with the ChainedOptimizer.
    """

    def minimize(self, init_guess: Array) -> ScipyOptimizerResultWrapper[Array]:
        """
        Perform the optimization.

        Parameters
        ----------
        init_guess : Optional[Array], optional
            Initial guess for the optimization variables. Defaults to None.

        Returns
        -------
        ScipyOptimizerResultWrapper
            Wrapped optimization result.
        """
        ...

    def bkd(self) -> Backend[Array]:
        """
        Get the backend used for computations.

        Returns
        -------
        Backend[Array]
            Backend used for computations.
        """
        ...
