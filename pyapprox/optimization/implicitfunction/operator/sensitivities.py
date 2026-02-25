from typing import Generic

from pyapprox.optimization.implicitfunction.functionals.protocols import (
    ParameterizedFunctionalWithJacobianProtocol,
)
from pyapprox.optimization.implicitfunction.operator.storage import (
    AdjointOperatorStorage,
)
from pyapprox.optimization.implicitfunction.state_equations.protocols import (
    ParameterizedStateEquationWithJacobianProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.validation import validate_backends


class VectorAdjointOperatorWithJacobian(Generic[Array]):
    """
    Vector adjoint operator for solving adjoint equations and computing
    sensitivities.

    This class encapsulates adjoint operator functionality for vector
    functionals with Jacobian computations.
    """

    def __init__(
        self,
        state_eq: ParameterizedStateEquationWithJacobianProtocol[Array],
        functional: ParameterizedFunctionalWithJacobianProtocol[Array],
    ):
        """
        Initialize the VectorAdjointOperatorWithJacobian object.

        Parameters
        ----------
        state_eq : ParameterizedStateEquationWithJacobianProtocol
            State equation object implementing the parameterized state
            equation protocol.
        functional : ParameterizedFunctionalWithJacobianProtocol
            Functional object implementing the parameterized functional
            protocol.

        Raises
        ------
        TypeError
            If the state equation or functional are not valid instances of
            their respective protocols.
        """
        self.validate_state_eq(state_eq)
        self.validate_functional(functional)
        validate_backends([functional.bkd(), state_eq.bkd()])
        self._bkd = state_eq.bkd()
        self._state_eq = state_eq
        self._functional = functional
        self._storage = AdjointOperatorStorage(
            self._state_eq.nstates(), self._state_eq.nparams(), self.bkd()
        )

    def validate_state_eq(
        self,
        state_eq: ParameterizedStateEquationWithJacobianProtocol[Array],
    ) -> None:
        """
        Validate the state equation.

        Parameters
        ----------
        state_eq : ParameterizedStateEquationWithJacobianProtocol
            State equation object.

        Raises
        ------
        TypeError
            If the state equation is not a valid instance of
            ParameterizedStateEquationWithJacobianProtocol.
        """
        if not isinstance(state_eq, ParameterizedStateEquationWithJacobianProtocol):
            raise TypeError(
                "state_eq must be an instance of "
                "ParameterizedStateEquationWithJacobianProtocol."
            )

    def validate_functional(
        self, functional: ParameterizedFunctionalWithJacobianProtocol[Array]
    ) -> None:
        """
        Validate the functional.

        Parameters
        ----------
        functional : ParameterizedFunctionalWithJacobianProtocol
            Functional object.

        Raises
        ------
        TypeError
            If the functional is not a valid instance of
            ParameterizedFunctionalWithJacobianProtocol.
        """
        if not isinstance(functional, ParameterizedFunctionalWithJacobianProtocol):
            raise TypeError(
                "functional must be an instance of "
                "ParameterizedFunctionalWithJacobianProtocol."
            )

    def bkd(self) -> Backend[Array]:
        """
        Return the backend used for computations.

        Returns
        -------
        Backend
            Backend used for computations.
        """
        return self._bkd

    def storage(self) -> AdjointOperatorStorage:
        """
        Return the adjoint operator storage.

        Returns
        -------
        AdjointOperatorStorage
            Storage for adjoint operator data.
        """
        return self._storage

    def nparams(self) -> int:
        """
        Return the number of variables in the system.

        Returns
        -------
        int
            Number of variables.
        """
        return self._state_eq.nparams()

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest (QoI).

        Returns
        -------
        int
            Number of quantities of interest.
        """
        return self._functional.nqoi()

    def __call__(self, init_fwd_state: Array, param: Array) -> Array:
        """
        Compute the value of the functional.

        Parameters
        ----------
        init_fwd_state : Array
            Initial forward state.
        param : Array
            Parameters.

        Returns
        -------
        Array
            Value of the functional.
        """
        fwd_state = self._get_forward_state(init_fwd_state, param)
        return self._functional(fwd_state, param)

    def _get_forward_state(self, init_fwd_state: Array, param: Array) -> Array:
        """
        Get the forward state, computing it if necessary.

        Parameters
        ----------
        init_fwd_state : Array
            Initial forward state.
        param : Array
            Parameters.

        Returns
        -------
        Array
            Forward state.
        """
        if (
            not self._storage.has_parameter(param)
            or not self._storage.has_forward_state()
        ):
            self._storage._clear()
            fwd_state = self._state_eq.solve(init_fwd_state, param)
            self._storage.set_forward_state(fwd_state)
        return self._storage.get_forward_state()

    def _sensitivities(self, fwd_state: Array, param: Array) -> Array:
        """
        Compute the Jacobian of the state with respect to the parameters.

        Parameters
        ----------
        fwd_state : Array
            Forward state.
        param : Array
            Parameters.

        Returns
        -------
        Array
            Sensitivities (Jacobian of the state with respect to the
            parameters).
        """
        drdy = self._state_eq.state_jacobian(fwd_state, param)
        drdp = self._state_eq.param_jacobian(fwd_state, param)
        sens = self._bkd.solve(drdy, -drdp)
        return sens

    def jacobian(self, init_fwd_state: Array, param: Array) -> Array:
        """
        Compute the Jacobian of the functional with respect to the parameters.

        Parameters
        ----------
        init_fwd_state : Array
            Initial forward state.
        param : Array
            Parameters.

        Returns
        -------
        Array
            Jacobian of the functional with respect to the parameters.
        """
        fwd_state = self._get_forward_state(init_fwd_state, param)
        sens = self._sensitivities(fwd_state, param)
        dqdy = self._functional.state_jacobian(fwd_state, param)
        dqdp = self._functional.param_jacobian(fwd_state, param)
        return dqdy @ sens + dqdp

    def state_equation(
        self,
    ) -> ParameterizedStateEquationWithJacobianProtocol[Array]:
        """
        Return the state equation object.

        Returns
        -------
        ParameterizedStateEquationWithJacobianProtocol
            State equation object.
        """
        return self._state_eq

    def functional(self) -> ParameterizedFunctionalWithJacobianProtocol[Array]:
        """
        Return the functional object.

        Returns
        -------
        ParameterizedFunctionalWithJacobianProtocol
            Functional object.
        """
        return self._functional

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the object for debugging.
        """
        return (
            f"{self.__class__.__name__}("
            f"nstates={self.nstates()}, "
            f"nparams={self.nparams()}, "
            f"bkd={type(self._bkd).__name__})"
        )
