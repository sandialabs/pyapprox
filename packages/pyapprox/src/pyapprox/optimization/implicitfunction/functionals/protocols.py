from typing import Generic, Literal, Protocol, runtime_checkable

from pyapprox.optimization.implicitfunction.functionals.derivatives import (
    SteadyFunctionalDerivatives,
)
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class SteadyFunctionalProtocol(Protocol, Generic[Array]):
    """
    Protocol for steady functionals: quantities of interest Q(u, p).

    Their derivatives, which adjoint-based gradients need, are optional
    and come from ``derivatives()``.
    """

    def bkd(self) -> Backend[Array]:
        """
        Return the backend used for computations.

        Returns
        -------
        Backend
            Backend used for computations.
        """
        ...

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest (QoI).

        Returns
        -------
        int
            Number of quantities of interest.
        """
        ...

    def nstates(self) -> int:
        """
        Return the number of states in the system.

        Returns
        -------
        int
            Number of states.
        """
        ...

    def nparams(self) -> int:
        """
        Return the number of variables in the system.

        Returns
        -------
        int
            Number of variables.
        """
        ...

    def nunique_params(self) -> int:
        """
        Return the number of unique parameters in the functional.

        Returns
        -------
        int
            Number of unique variables.
        """
        ...

    def time_dependence(self) -> Literal["steady"]:
        """Return the kind of state this functional reads.

        One steady state, shape ``(nstates, 1)``.
        """
        ...

    def derivatives(self) -> SteadyFunctionalDerivatives[Array]:
        """Return the optional derivatives this functional provides."""
        ...

    def __call__(self, state: Array, param: Array) -> Array:
        """
        Evaluate the functional for the given state and parameters.

        Parameters
        ----------
        state : Array
            State of the system.
        param : Array
            Parameters of the system.

        Returns
        -------
        Array
            Values of the functional.
        """
        ...
