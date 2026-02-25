from typing import Generic

from pyapprox.optimization.implicitfunction.functionals.protocols import (
    ParameterizedFunctionalWithJacobianAndHVPProtocol,
)
from pyapprox.optimization.implicitfunction.operator.operator_with_jacobian import (
    AdjointOperatorWithJacobian,
)
from pyapprox.optimization.implicitfunction.operator.storage import (
    AdjointOperatorStorage,
)
from pyapprox.optimization.implicitfunction.state_equations.protocols import (
    ParameterizedStateEquationWithJacobianAndHVPProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.validation import validate_backends


class AdjointOperatorWithJacobianAndHVP(Generic[Array]):
    """
    Scalar adjoint operator with Hessian computations.

    This class encapsulates adjoint operator functionality for scalar functionals
    (nqoi = 1) with Hessian computations. It uses a ScalarAdjointOperator instance
    for basic adjoint functionality and extends it with Hessian-related computations.
    """

    def __init__(
        self,
        state_eq: ParameterizedStateEquationWithJacobianAndHVPProtocol[Array],
        functional: ParameterizedFunctionalWithJacobianAndHVPProtocol[Array],
    ):
        """
        Initialize the ScalarAdjointOperatorWithHessian object.

        Parameters
        ----------
        state_eq : AdjointParameterizedStateEquationProtocol
            Residual equation object implementing the adjoint residual equation
            protocol.
        functional : ParameterizedFunctionalWithJacobianAndHVPProtocol
            Functional object implementing the adjoint functional protocol.

        Raises
        ------
        TypeError
            If the residual equation or functional are not valid instances of their
            respective protocols.
        """
        self._validate_state_eq(state_eq)
        self._validate_functional(functional)
        validate_backends([functional.bkd(), state_eq.bkd()])
        self._jacobian_operator = AdjointOperatorWithJacobian(state_eq, functional)
        self._bkd = self._jacobian_operator.bkd()
        self._state_eq = state_eq
        self._functional = functional

    def _validate_state_eq(
        self,
        state_eq: ParameterizedStateEquationWithJacobianAndHVPProtocol[Array],
    ) -> None:
        """
        Validate the state equation.

        Parameters
        ----------
        state_eq : ParameterizedStateEquationWithJacobianAndHVPProtocol
            State equation object.

        Raises
        ------
        TypeError
            If the state equation is not a valid instance of "
        "ParameterizedStateEquationWithJacobianAndHVPProtocol.
        """
        if not isinstance(
            state_eq, ParameterizedStateEquationWithJacobianAndHVPProtocol
        ):
            raise TypeError(
                "state_eq must be an instance of "
                "ParameterizedStateEquationWithJacobianAndHVPProtocol."
            )

    def _validate_functional(
        self,
        functional: ParameterizedFunctionalWithJacobianAndHVPProtocol[Array],
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
            If the functional is not a valid instance of "
            "ParameterizedFunctionalWithJacobianAndHVPProtocol or nqoi != 1.
        """
        if not isinstance(
            functional, ParameterizedFunctionalWithJacobianAndHVPProtocol
        ):
            raise TypeError(
                "functional must be an instance of "
                "ParameterizedFunctionalWithJacobianAndHVPProtocol."
            )
        if functional.nqoi() != 1:
            raise ValueError("functional must have nqoi == 1.")

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
        return self._jacobian_operator.storage()

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
        return self._jacobian_operator.nparams()

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
        return self._jacobian_operator(init_fwd_state, param)

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
        return self._jacobian_operator.jacobian(init_fwd_state, param)

    def _forward_hessian_solve(
        self,
        fwd_state: Array,
        param: Array,
        drdy: Array,
        drdp: Array,
        vvec: Array,
    ) -> Array:
        """
        Solve the forward Hessian equation.

        Parameters
        ----------
        fwd_state : Array
            Forward state.
        param : Array
            Parameters.
        drdy : Array
            State Jacobian of the residual equation.
        drdp : Array
            Parameter Jacobian of the residual equation.
        vvec : Array
            Vector for Hessian computation.

        Returns
        -------
        Array
            Solution to the forward Hessian equation.
        """
        return self._bkd.solve(drdy, drdp @ vvec)

    def _lagrangian_state_state_hvp(
        self, fwd_state: Array, param: Array, adj_state: Array, wvec: Array
    ) -> Array:
        """
        Compute the state-state Hessian-vector product of the Lagrangian.

        Parameters
        ----------
        fwd_state : Array
            Forward state.
        param : Array
            Parameters.
        adj_state : Array
            Adjoint state.
        wvec : Array
            Vector for Hessian computation.

        Returns
        -------
        Array
            State-state Hessian-vector product.
        """
        return self._functional.state_state_hvp(
            fwd_state, param, wvec
        ) + self._state_eq.state_state_hvp(fwd_state, param, adj_state, wvec)

    def _lagrangian_state_param_hvp(
        self, fwd_state: Array, param: Array, adj_state: Array, vvec: Array
    ) -> Array:
        """
        Compute the state-parameter Hessian-vector product of the Lagrangian.

        Parameters
        ----------
        fwd_state : Array
            Forward state.
        param : Array
            Parameters.
        adj_state : Array
            Adjoint state.
        vvec : Array
            Vector for Hessian computation.

        Returns
        -------
        Array
            State-parameter Hessian-vector product.
        """
        return self._functional.state_param_hvp(
            fwd_state, param, vvec
        ) + self._state_eq.state_param_hvp(fwd_state, param, adj_state, vvec)

    def _lagrangian_param_state_hvp(
        self, fwd_state: Array, param: Array, adj_state: Array, wvec: Array
    ) -> Array:
        """
        Compute the parameter-state Hessian-vector product of the Lagrangian.

        Parameters
        ----------
        fwd_state : Array
            Forward state.
        param : Array
            Parameters.
        adj_state : Array
            Adjoint state.
        wvec : Array
            Vector for Hessian computation.

        Returns
        -------
        Array
            Parameter-state Hessian-vector product.
        """
        qps_hvp = self._functional.param_state_hvp(fwd_state, param, wvec)
        if qps_hvp.ndim != 2 or qps_hvp.shape[1] != 1:
            raise RuntimeError("qps_hvp must be a 2D array with shape[1] == 1")
        rps_hvp = self._state_eq.param_state_hvp(fwd_state, param, adj_state, wvec)
        if rps_hvp.ndim != 2 or rps_hvp.shape[1] != 1:
            raise RuntimeError("rps_hvp must be a 2D array with shape[1] == 1")
        return qps_hvp + rps_hvp

    def _lagrangian_param_param_hvp(
        self, fwd_state: Array, param: Array, adj_state: Array, vvec: Array
    ) -> Array:
        """
        Compute the parameter-parameter Hessian-vector product of the Lagrangian.

        Parameters
        ----------
        fwd_state : Array
            Forward state.
        param : Array
            Parameters.
        adj_state : Array
            Adjoint state.
        vvec : Array
            Vector for Hessian computation.

        Returns
        -------
        Array
            Parameter-parameter Hessian-vector product.
        """
        qpp_hvp = self._functional.param_param_hvp(fwd_state, param, vvec)
        if qpp_hvp.ndim != 2 or qpp_hvp.shape[1] != 1:
            raise RuntimeError("qpp_hvp must be a 2D array with shape[1] == 1")
        rpp_hvp = self._state_eq.param_param_hvp(fwd_state, param, adj_state, vvec)
        if rpp_hvp.ndim != 2 or rpp_hvp.shape[1] != 1:
            raise RuntimeError(
                "rpp_hvp returned by {0} must be a 2D array with shape[1] == 1".format(
                    self._state_eq
                )
            )
        return qpp_hvp + rpp_hvp

    def _adjoint_hessian_solve(
        self,
        fwd_state: Array,
        param: Array,
        adj_state: Array,
        drdy: Array,
        wvec: Array,
        vvec: Array,
    ) -> Array:
        """
        Solve the adjoint Hessian equation.

        Parameters
        ----------
        fwd_state : Array
            Forward state.
        param : Array
            Parameters.
        adj_state : Array
            Adjoint state.
        drdy : Array
            State Jacobian of the residual equation.
        wvec : Array
            Vector for forward Hessian computation.
        vvec : Array
            Vector for adjoint Hessian computation.

        Returns
        -------
        Array
            Solution to the adjoint Hessian equation.
        """
        return self._bkd.solve(
            drdy.T,
            self._lagrangian_state_state_hvp(fwd_state, param, adj_state, wvec)
            - self._lagrangian_state_param_hvp(fwd_state, param, adj_state, vvec),
        )

    def _get_adjoint_state(self, init_fwd_state: Array, param: Array) -> Array:
        if (
            not self.storage().has_parameter(param)
            or not self.storage().has_adjoint_state()
        ):
            fwd_state = self._jacobian_operator._get_forward_state(
                init_fwd_state, param
            )
            adj_state = self._jacobian_operator.solve_adjoint_equation(fwd_state, param)
            self.storage().set_adjoint_state(adj_state)
        return self.storage().get_adjoint_state()

    def hvp(self, init_fwd_state: Array, param: Array, vvec: Array) -> Array:
        """
        Apply the Hessian to a vector.

        Parameters
        ----------
        init_fwd_state : Array
            Initial forward state.
        param : Array
            Parameters.
        vvec : Array
            Vector for Hessian computation.

        Returns
        -------
        Array
            Result of applying the Hessian to the vector.
        """

        if vvec.shape != (self.nparams(), 1):
            raise ValueError(
                f"vvec has shape {vvec.shape} but must be {(self.nparams(), 1)}"
            )

        # Load or compute forward state
        fwd_state = self._jacobian_operator._get_forward_state(init_fwd_state, param)

        # Load or compute adjoint state
        adj_state = self._get_adjoint_state(init_fwd_state, param)

        # Load drdy (state Jacobian), guaranteed to exist after adjoint solve
        drdy = self.storage().get_state_eq_state_jacobian()

        # Load or compute drdp (parameter Jacobian)
        drdp = self._jacobian_operator._get_state_eq_param_jacobian(fwd_state, param)

        # Compute forward Hessian state
        wvec = self._forward_hessian_solve(fwd_state, param, drdy, drdp, vvec)

        # Compute adjoint Hessian state
        svec = self._adjoint_hessian_solve(
            fwd_state, param, adj_state, drdy, wvec, vvec
        )
        lps_hvp = self._lagrangian_param_state_hvp(fwd_state, param, adj_state, wvec)
        lpp_hvp = self._lagrangian_param_param_hvp(fwd_state, param, adj_state, vvec)
        hvp = drdp.T @ svec - lps_hvp + lpp_hvp
        return hvp

    def state_equation(
        self,
    ) -> ParameterizedStateEquationWithJacobianAndHVPProtocol[Array]:
        """
        Return the state equation object.

        Returns
        -------
        ParameterizedStateEquationWithJacobianAndHVPProtocol
            State equation object.
        """
        return self._state_eq

    def functional(
        self,
    ) -> ParameterizedFunctionalWithJacobianAndHVPProtocol[Array]:
        """
        Return the functional object.

        Returns
        -------
        ParameterizedFunctionalWithJacobianAndHVPProtocol
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
