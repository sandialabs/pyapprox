from typing import Generic

from pyapprox.optimization.implicitfunction.functionals.derivatives import (
    SteadyFunctionalJacobianFn,
)
from pyapprox.optimization.implicitfunction.functionals.protocols import (
    SteadyFunctionalProtocol,
)
from pyapprox.optimization.implicitfunction.operator.derivatives import (
    ImplicitOperatorDerivatives,
)
from pyapprox.optimization.implicitfunction.operator.storage import (
    AdjointOperatorStorage,
)
from pyapprox.optimization.implicitfunction.state_equations.protocols import (
    SteadyStateEquationProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.validation import validate_backends
from pyapprox.util.linalg.sparse_dispatch import solve_maybe_sparse


class ForwardSensitivityOperator(Generic[Array]):
    """
    Operator for a vector-valued functional of the solution of a steady
    state equation, differentiated by forward sensitivities.

    Solves ``(dR/du) S = -dR/dp`` for the state sensitivities ``S`` (one
    linear solve per parameter) and returns ``dQ/du S + dQ/dp``, so it
    suits functionals with many QoI. It provides no Hessian-vector
    product.
    """

    def __init__(
        self,
        state_eq: SteadyStateEquationProtocol[Array],
        functional: SteadyFunctionalProtocol[Array],
    ):
        """
        Initialize the ForwardSensitivityOperator object.

        Parameters
        ----------
        state_eq : SteadyStateEquationProtocol
            State equation object implementing the parameterized state
            equation protocol.
        functional : SteadyFunctionalProtocol
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
        derivs = functional.derivatives()
        if derivs.state_jacobian is None or derivs.param_jacobian is None:
            raise TypeError(
                f"{type(functional).__name__} provides no Jacobians "
                "(derivatives().state_jacobian is None); the sensitivities "
                "need them"
            )
        self._q_state_jacobian: SteadyFunctionalJacobianFn[Array] = (
            derivs.state_jacobian
        )
        self._q_param_jacobian: SteadyFunctionalJacobianFn[Array] = (
            derivs.param_jacobian
        )
        self._storage = AdjointOperatorStorage(
            self._state_eq.nstates(), self._state_eq.nparams(), self.bkd()
        )

    def validate_state_eq(
        self,
        state_eq: SteadyStateEquationProtocol[Array],
    ) -> None:
        """
        Validate the state equation.

        Parameters
        ----------
        state_eq : SteadyStateEquationProtocol
            State equation object.

        Raises
        ------
        TypeError
            If the state equation is not a valid instance of
            SteadyStateEquationProtocol.
        """
        if not isinstance(state_eq, SteadyStateEquationProtocol):
            raise TypeError(
                "state_eq must be an instance of "
                "SteadyStateEquationProtocol."
            )

    def validate_functional(
        self, functional: SteadyFunctionalProtocol[Array]
    ) -> None:
        """
        Validate the functional.

        Parameters
        ----------
        functional : SteadyFunctionalProtocol
            Functional object.

        Raises
        ------
        TypeError
            If the functional is not a valid instance of
            SteadyFunctionalProtocol.
        """
        if not isinstance(functional, SteadyFunctionalProtocol):
            raise TypeError(
                "functional must be an instance of "
                "SteadyFunctionalProtocol."
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

    def derivatives(self) -> ImplicitOperatorDerivatives[Array]:
        """Return the optional derivatives this operator provides (none)."""
        return ImplicitOperatorDerivatives.none()

    def storage(self) -> AdjointOperatorStorage[Array]:
        """
        Return the adjoint operator storage.

        Returns
        -------
        AdjointOperatorStorage
            Storage for adjoint operator data.
        """
        return self._storage

    def nstates(self) -> int:
        """
        Return the number of states in the system.

        Returns
        -------
        int
            Number of states.
        """
        return self._state_eq.nstates()

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
        # Sparse-aware: galerkin state equations return sparse
        # Jacobians (matrix right-hand side supported).
        sens = solve_maybe_sparse(self._bkd, drdy, -drdp)
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
        dqdy = self._q_state_jacobian(fwd_state, param)
        dqdp = self._q_param_jacobian(fwd_state, param)
        return dqdy @ sens + dqdp

    def state_equation(
        self,
    ) -> SteadyStateEquationProtocol[Array]:
        """
        Return the state equation object.

        Returns
        -------
        SteadyStateEquationProtocol
            State equation object.
        """
        return self._state_eq

    def functional(self) -> SteadyFunctionalProtocol[Array]:
        """
        Return the functional object.

        Returns
        -------
        SteadyFunctionalProtocol
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
