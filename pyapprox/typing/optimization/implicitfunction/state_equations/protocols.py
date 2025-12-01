from typing import Protocol, Generic, runtime_checkable

from pyapprox.typing.util.backend import Array, Backend


@runtime_checkable
class ParameterizedStateEquationProtocolWithJacobian(Protocol, Generic[Array]):
    """
    Protocol for parameterized state equations with adjoint capabilities.
    """

    def nvars(self) -> int:
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

    def set_parameters(self, param: Array) -> None:
        """
        Set the parameters for the residual equation.

        Parameters
        ----------
        param : Array
            Parameters to set.
        """
        ...

    def get_parameters(self) -> Array:
        """
        Get the current parameters of the residual equation.

        Returns
        -------
        Array
            Current parameters.
        """
        ...

    def value(self, init_state: Array) -> Array:
        """
        Compute the residual value for the given initial state.

        Parameters
        ----------
        init_state : Array
            Initial state.

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

    def use_auto_differentiation(self) -> bool:
        """
        Return whether automatic differentiation is enabled.

        Returns
        -------
        bool
            True if automatic differentiation is enabled, False otherwise.
        """
        ...


@runtime_checkable
class ParameterizedStateEquationWithJacobianAndHVPProtocol(
    Protocol, Generic[Array]
):
    """
    Protocol for state equations with Jacobian and Hessian-vector
    product capabilities.
    """

    def nvars(self) -> int:
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

    def bkd(self) -> Backend[Array]:
        """
        Return the backend used for computations.

        Returns
        -------
        Backend
            Backend used for computations.
        """
        ...

    def value(self, state: Array, param: Array) -> Array:
        """
        Compute the residual value for the given state and parameters.

        Parameters
        ----------
        state : Array
            State of the system.
        param : Array
            Parameters of the system.

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

    def state_state_hvp(
        self, state: Array, param: Array, adj_state: Array, wvec: Array
    ) -> Array:
        """
        Compute the Hessian-vector product with respect to the state.

        Parameters
        ----------
        state : Array
            State of the system.
        param : Array
            Parameters of the system.
        adj_state : Array
            Adjoint state.
        wvec : Array
            Vector for Hessian computation.

        Returns
        -------
        Array
            Hessian-vector product result.
        """
        ...

    def param_param_hvp(
        self, state: Array, param: Array, adj_state: Array, vvec: Array
    ) -> Array:
        """
        Compute the Hessian-vector product with respect to the parameters.

        Parameters
        ----------
        state : Array
            State of the system.
        param : Array
            Parameters of the system.
        adj_state : Array
            Adjoint state.
        vvec : Array
            Vector for Hessian computation.

        Returns
        -------
        Array
            Hessian-vector product result.
        """
        ...

    def state_param_hvp(
        self, state: Array, param: Array, adj_state: Array, vvec: Array
    ) -> Array:
        """
        Compute the Hessian-vector product with respect to state and parameters.

        Parameters
        ----------
        state : Array
            State of the system.
        param : Array
            Parameters of the system.
        adj_state : Array
            Adjoint state.
        vvec : Array
            Vector for Hessian computation.

        Returns
        -------
        Array
            Hessian-vector product result.
        """
        ...

    def param_state_hvp(
        self, state: Array, param: Array, adj_state: Array, wvec: Array
    ) -> Array:
        """
        Compute the Hessian-vector product with respect to parameters and state.

        Parameters
        ----------
        state : Array
            State of the system.
        param : Array
            Parameters of the system.
        adj_state : Array
            Adjoint state.
        wvec : Array
            Vector for Hessian computation.

        Returns
        -------
        Array
            Hessian-vector product result.
        """
        ...
