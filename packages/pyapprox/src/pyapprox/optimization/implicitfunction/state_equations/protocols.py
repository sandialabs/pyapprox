from typing import Generic, Protocol, runtime_checkable

from pyapprox.optimization.implicitfunction.state_equations.derivatives import (
    SteadyStateEquationDerivatives,
)
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class SteadyStateEquationProtocol(Protocol, Generic[Array]):
    """
    Protocol for steady state equations R(u, p) = 0.

    The residual, its Newton solve and both Jacobians are required. The
    adjoint-weighted second derivatives are optional and come from
    ``derivatives()``.
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

    def nparams(self) -> int:
        """
        Return the number of variables in the system.

        Returns
        -------
        int
            Number of variables.
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

    def derivatives(self) -> SteadyStateEquationDerivatives[Array]:
        """Return the optional derivatives this state equation provides."""
        ...

    def __call__(self, state: Array, param: Array) -> Array:
        """
        Compute the residual value for the given state.

        Parameters
        ----------
        state : Array
            The state.

        Returns
        -------
        Array
            Residual value.
        """
        ...

    def solve(self, init_state: Array, param: Array) -> Array:
        """
        Solve the residual equation for the given initial state and parameters.

        Parameters
        ----------
        init_state : Array
            Initial state.
        param : Array
            Parameters.

        Returns
        -------
        Array
            Solution to the residual equation.
        """
        ...

    def param_jacobian(self, state: Array, param: Array) -> Array:
        """
        Compute the Jacobian of the residual with respect to the parameters.

        Parameters
        ----------
        state : Array
            State of the system.
        param : Array
            Parameters of the system.

        Returns
        -------
        Array
            Jacobian matrix with respect to the parameters.
        """
        ...

    def state_jacobian(self, state: Array, param: Array) -> Array:
        """
        Compute the Jacobian of the residual with respect to the state.

        Parameters
        ----------
        state : Array
            State of the system.
        param : Array
            Parameters of the system.

        Returns
        -------
        Array
            Jacobian matrix with respect to the state.
        """
        ...
