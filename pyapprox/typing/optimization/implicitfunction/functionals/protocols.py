from typing import Protocol, Generic, runtime_checkable
from pyapprox.typing.util.backends.protocols import Array, Backend


@runtime_checkable
class ParameterizedFunctionalWithJacobianProtocol(Protocol, Generic[Array]):
    """
    Protocol for adjoint-enabled functionals that compute quantities of
    interest (QoI).
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

    def state_jacobian(self, state: Array, param: Array) -> Array:
        """
        Compute the Jacobian of the QoI with respect to the state.

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
        Compute the Jacobian of the QoI with respect to the parameters.

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


@runtime_checkable
class ParameterizedFunctionalWithJacobianAndHVPProtocol(
    Protocol, Generic[Array]
):
    """
    Protocol for functionals with Jacobian and Hessian-vector product
    capabilities.
    """

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

    def bkd(self) -> Backend[Array]:
        """
        Return the backend used for computations.

        Returns
        -------
        Backend
            Backend used for computations.
        """
        ...

    def __call__(self, state: Array, param: Array) -> Array:
        """
        Compute the value of the functional.

        Parameters
        ----------
        state : Array
            State of the system.
        param : Array
            Parameters of the system.

        Returns
        -------
        Array
            Value of the functional.
        """
        ...

    def state_jacobian(self, state: Array, param: Array) -> Array:
        """
        Compute the Jacobian of the functional with respect to the state.

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
        Compute the Jacobian of the functional with respect to the parameters.

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
        self, state: Array, param: Array, wvec: Array
    ) -> Array:
        """
        Compute the Hessian-vector product with respect to the state.

        Parameters
        ----------
        state : Array
            State of the system.
        param : Array
            Parameters of the system.
        wvec : Array
            Vector for Hessian computation.

        Returns
        -------
        Array
            Hessian-vector product result.
        """
        ...

    def param_param_hvp(
        self, state: Array, param: Array, vvec: Array
    ) -> Array:
        """
        Compute the Hessian-vector product with respect to the parameters.

        Parameters
        ----------
        state : Array
            State of the system.
        param : Array
            Parameters of the system.
        vvec : Array
            Vector for Hessian computation.

        Returns
        -------
        Array
            Hessian-vector product result.
        """
        ...

    def state_param_hvp(
        self, state: Array, param: Array, vvec: Array
    ) -> Array:
        """
        Compute the Hessian-vector product with respect to state and parameters.

        Parameters
        ----------
        state : Array
            State of the system.
        param : Array
            Parameters of the system.
        vvec : Array
            Vector for Hessian computation.

        Returns
        -------
        Array
            Hessian-vector product result.
        """
        ...

    def param_state_hvp(
        self, state: Array, param: Array, wvec: Array
    ) -> Array:
        """
        Compute the Hessian-vector product with respect to parameters and state.

        Parameters
        ----------
        state : Array
            State of the system.
        param : Array
            Parameters of the system.
        wvec : Array
            Vector for Hessian computation.

        Returns
        -------
        Array
            Hessian-vector product result.
        """
        ...
