from typing import Generic
from pyapprox.typing.util.backend import Array, Backend
from pyapprox.typing.optimization.implicitfunction.state_equations.protocols import (
    AdjointParameterizedStateEquationProtocol,
)
from pyapprox.typing.optimization.implicitfunction.functionals.protocols import (
    AdjointFunctionalProtocol,
)
from pyapprox.typing.optimization.implicitfunction.operator.storage import (
    AdjointOperatorStorage,
)
from pyapprox.typing.optimization.implicitfunction.operator.jacobian import (
    AdjointOperatorWithJacobian,
)


class ScalarAdjointOperatorWithHessian(Generic[Array]):
    """
    Scalar adjoint operator with Hessian computations.

    This class encapsulates adjoint operator functionality for scalar functionals
    (nqoi = 1) with Hessian computations. It uses a ScalarAdjointOperator instance
    for basic adjoint functionality and extends it with Hessian-related computations.
    """

    def __init__(
        self,
        residual_eq: AdjointParameterizedStateEquationProtocol[Array],
        functional: AdjointFunctionalProtocol[Array],
    ):
        """
        Initialize the ScalarAdjointOperatorWithHessian object.

        Parameters
        ----------
        residual_eq : AdjointParameterizedStateEquationProtocol
            Residual equation object implementing the adjoint residual equation protocol.
        functional : AdjointFunctionalProtocol
            Functional object implementing the adjoint functional protocol.

        Raises
        ------
        TypeError
            If the residual equation or functional are not valid instances of their respective protocols.
        """
        self._jacobian_operator = AdjointOperatorWithJacobian(
            residual_eq, functional
        )
        self._bkd = self._jacobian_operator.bkd()
        self._residual_eq = residual_eq
        self._functional = functional
        self._adjoint_data = self._jacobian_operator.adjoint_data()

    def bkd(self) -> Backend[Array]:
        """
        Return the backend used for computations.

        Returns
        -------
        Backend
            Backend used for computations.
        """
        return self._bkd

    def adjoint_data(self) -> AdjointOperatorStorage:
        """
        Return the adjoint operator storage.

        Returns
        -------
        AdjointOperatorStorage
            Storage for adjoint operator data.
        """
        return self._adjoint_data

    def nvars(self) -> int:
        """
        Return the number of variables in the system.

        Returns
        -------
        int
            Number of variables.
        """
        return self._jacobian_operator.nvars()

    def value(self, init_fwd_state: Array, param: Array) -> Array:
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
        return self._jacobian_operator.value(init_fwd_state, param)

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
        ) + self._residual_eq.state_state_hvp(
            fwd_state, param, adj_state, wvec
        )

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
        ) + self._residual_eq.state_param_hvp(
            fwd_state, param, adj_state, vvec
        )

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
        if qps_hvp.ndim != 1:
            raise RuntimeError("qps_hvp must be a 1D array")
        rps_hvp = self._residual_eq.param_state_hvp(
            fwd_state, param, adj_state, wvec
        )
        if rps_hvp.ndim != 1:
            raise RuntimeError("rps_hvp must be a 1D array")
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
        if qpp_hvp.ndim != 1:
            raise RuntimeError("qpp_hvp must be a 1D array")
        rpp_hvp = self._residual_eq.param_param_hvp(
            fwd_state, param, adj_state, vvec
        )
        if rpp_hvp.ndim != 1:
            raise RuntimeError(
                "rpp_hvp returned by {0} must be a 1D array".format(
                    self._residual_eq
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
            - self._lagrangian_state_param_hvp(
                fwd_state, param, adj_state, vvec
            ),
        )

    def apply_hessian(
        self, init_fwd_state: Array, param: Array, vvec: Array
    ) -> Array:
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
        self._residual_eq._check_state_param_shapes(init_fwd_state, param)
        if vvec.shape != (self.nvars(),):
            raise ValueError(
                f"vvec has shape {vvec.shape} but must be {(self.nvars(),)}"
            )

        # Load or compute forward state
        fwd_state = self._jacobian_operator._get_forward_state(
            init_fwd_state, param
        )

        # Load or compute adjoint state
        adj_state = self._jacobian_operator._get_adjoint_state(
            init_fwd_state, param
        )

        # Load drdy (state Jacobian), guaranteed to exist after adjoint solve
        drdy = self._adjoint_data.get_residual_eq_state_jacobian()

        # Load or compute drdp (parameter Jacobian)
        drdp = self._jacobian_operator._get_state_eq_param_jacobian(
            fwd_state, param
        )

        # Compute forward Hessian state
        wvec = self._forward_hessian_solve(fwd_state, param, drdy, drdp, vvec)

        # Compute adjoint Hessian state
        svec = self._adjoint_hessian_solve(
            fwd_state, param, adj_state, drdy, wvec, vvec
        )
        lps_hvp = self._lagrangian_param_state_hvp(
            fwd_state, param, adj_state, wvec
        )
        lpp_hvp = self._lagrangian_param_param_hvp(
            fwd_state, param, adj_state, vvec
        )
        hvp = drdp.T @ svec - lps_hvp + lpp_hvp
        return hvp
