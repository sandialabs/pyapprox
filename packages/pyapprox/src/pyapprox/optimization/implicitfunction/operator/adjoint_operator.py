from typing import Generic, Optional

from pyapprox.optimization.implicitfunction.functionals.derivatives import (
    SteadyFunctionalHVPFn,
    SteadyFunctionalJacobianFn,
)
from pyapprox.optimization.implicitfunction.functionals.protocols import (
    ParameterizedFunctionalWithJacobianProtocol,
)
from pyapprox.optimization.implicitfunction.operator.derivatives import (
    ImplicitOperatorDerivatives,
)
from pyapprox.optimization.implicitfunction.operator.storage import (
    AdjointOperatorStorage,
)
from pyapprox.optimization.implicitfunction.state_equations.derivatives import (
    StateEquationHVPFn,
)
from pyapprox.optimization.implicitfunction.state_equations.protocols import (
    ParameterizedStateEquationWithJacobianProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.validation import validate_backends
from pyapprox.util.linalg.sparse_dispatch import solve_maybe_sparse


class _LagrangianHVPs(Generic[Array]):
    """The four second derivatives of the Lagrangian
    ``L = Q(u, p) + lambda^T R(u, p)``, each the functional's term plus
    the state equation's adjoint-weighted term."""

    def __init__(
        self,
        q_state_state: SteadyFunctionalHVPFn[Array],
        q_state_param: SteadyFunctionalHVPFn[Array],
        q_param_state: SteadyFunctionalHVPFn[Array],
        q_param_param: SteadyFunctionalHVPFn[Array],
        r_state_state: StateEquationHVPFn[Array],
        r_state_param: StateEquationHVPFn[Array],
        r_param_state: StateEquationHVPFn[Array],
        r_param_param: StateEquationHVPFn[Array],
    ) -> None:
        self._q_state_state = q_state_state
        self._q_state_param = q_state_param
        self._q_param_state = q_param_state
        self._q_param_param = q_param_param
        self._r_state_state = r_state_state
        self._r_state_param = r_state_param
        self._r_param_state = r_param_state
        self._r_param_param = r_param_param

    def state_state(
        self, fwd_state: Array, param: Array, adj_state: Array, wvec: Array
    ) -> Array:
        """``(d^2 L/du^2) w``. Shape: ``(nstates, 1)``."""
        return self._q_state_state(
            fwd_state, param, wvec
        ) + self._r_state_state(fwd_state, param, adj_state, wvec)

    def state_param(
        self, fwd_state: Array, param: Array, adj_state: Array, vvec: Array
    ) -> Array:
        """``(d^2 L/du dp) v``. Shape: ``(nstates, 1)``."""
        return self._q_state_param(
            fwd_state, param, vvec
        ) + self._r_state_param(fwd_state, param, adj_state, vvec)

    def param_state(
        self, fwd_state: Array, param: Array, adj_state: Array, wvec: Array
    ) -> Array:
        """``(d^2 L/dp du) w``. Shape: ``(nparams, 1)``."""
        qps_hvp = self._q_param_state(fwd_state, param, wvec)
        if qps_hvp.ndim != 2 or qps_hvp.shape[1] != 1:
            raise RuntimeError("qps_hvp must be a 2D array with shape[1] == 1")
        rps_hvp = self._r_param_state(fwd_state, param, adj_state, wvec)
        if rps_hvp.ndim != 2 or rps_hvp.shape[1] != 1:
            raise RuntimeError("rps_hvp must be a 2D array with shape[1] == 1")
        return qps_hvp + rps_hvp

    def param_param(
        self, fwd_state: Array, param: Array, adj_state: Array, vvec: Array
    ) -> Array:
        """``(d^2 L/dp^2) v``. Shape: ``(nparams, 1)``."""
        qpp_hvp = self._q_param_param(fwd_state, param, vvec)
        if qpp_hvp.ndim != 2 or qpp_hvp.shape[1] != 1:
            raise RuntimeError("qpp_hvp must be a 2D array with shape[1] == 1")
        rpp_hvp = self._r_param_param(fwd_state, param, adj_state, vvec)
        if rpp_hvp.ndim != 2 or rpp_hvp.shape[1] != 1:
            raise RuntimeError("rpp_hvp must be a 2D array with shape[1] == 1")
        return qpp_hvp + rpp_hvp


class AdjointOperator(Generic[Array]):
    """
    Adjoint operator for a scalar functional of the solution of a steady
    state equation.

    Computes the value of ``Q(u(p), p)``, where ``R(u(p), p) = 0``, and its
    parameter gradient by the adjoint method. When the state equation and
    the functional both provide their second derivatives, the
    ``derivatives()`` bundle also carries the second-order adjoint
    Hessian-vector product; otherwise its ``hvp`` is ``None``.

    Parameters
    ----------
    state_eq : ParameterizedStateEquationWithJacobianProtocol
        State equation ``R(u, p) = 0``.
    functional : ParameterizedFunctionalWithJacobianProtocol
        Scalar (``nqoi == 1``) functional providing its Jacobians.

    Raises
    ------
    TypeError
        If either argument does not satisfy its protocol, or the
        functional provides no Jacobians.
    ValueError
        If the functional is not scalar.
    """

    def __init__(
        self,
        state_eq: ParameterizedStateEquationWithJacobianProtocol[Array],
        functional: ParameterizedFunctionalWithJacobianProtocol[Array],
    ):
        self._validate_state_eq(state_eq)
        self._validate_functional(functional)
        q_derivs = functional.derivatives()
        if q_derivs.state_jacobian is None or q_derivs.param_jacobian is None:
            raise TypeError(
                f"{type(functional).__name__} provides no Jacobians "
                "(derivatives().state_jacobian is None); the adjoint "
                "gradient needs them"
            )
        self._q_state_jacobian: SteadyFunctionalJacobianFn[Array] = (
            q_derivs.state_jacobian
        )
        self._q_param_jacobian: SteadyFunctionalJacobianFn[Array] = (
            q_derivs.param_jacobian
        )
        validate_backends([functional.bkd(), state_eq.bkd()])
        self._bkd = state_eq.bkd()
        self._state_eq = state_eq
        self._functional = functional
        self._storage = AdjointOperatorStorage(
            self._state_eq.nstates(), self._state_eq.nparams(), self.bkd()
        )
        # The second-order adjoint contracts the second derivatives of
        # both the functional and the state equation, so it exists only
        # when both bundles carry them.
        r_derivs = state_eq.derivatives()
        self._lagrangian: Optional[_LagrangianHVPs[Array]] = None
        self._derivs: ImplicitOperatorDerivatives[Array] = (
            ImplicitOperatorDerivatives.none()
        )
        if (
            q_derivs.state_state_hvp is not None
            and q_derivs.state_param_hvp is not None
            and q_derivs.param_state_hvp is not None
            and q_derivs.param_param_hvp is not None
            and r_derivs.state_state_hvp is not None
            and r_derivs.state_param_hvp is not None
            and r_derivs.param_state_hvp is not None
            and r_derivs.param_param_hvp is not None
        ):
            self._lagrangian = _LagrangianHVPs(
                q_derivs.state_state_hvp,
                q_derivs.state_param_hvp,
                q_derivs.param_state_hvp,
                q_derivs.param_param_hvp,
                r_derivs.state_state_hvp,
                r_derivs.state_param_hvp,
                r_derivs.param_state_hvp,
                r_derivs.param_param_hvp,
            )
            self._derivs = ImplicitOperatorDerivatives.second_order(self._hvp)

    def _validate_state_eq(
        self, state_eq: ParameterizedStateEquationWithJacobianProtocol[Array]
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

    def _validate_functional(
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
        ValueError
            If nqoi != 1.
        """
        if not isinstance(functional, ParameterizedFunctionalWithJacobianProtocol):
            raise TypeError(
                "functional must be an instance of "
                "ParameterizedFunctionalWithJacobianProtocol."
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

    def derivatives(self) -> ImplicitOperatorDerivatives[Array]:
        """Return the optional derivatives this operator provides."""
        return self._derivs

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

    def solve_adjoint_equation(self, fwd_state: Array, param: Array) -> Array:
        """
        Solve the adjoint equation.

        Parameters
        ----------
        fwd_state : Array
            Forward state.
        param : Array
            Parameters.

        Returns
        -------
        Array
            Adjoint state.
        """
        drdy = self._state_eq.state_jacobian(fwd_state, param)
        dqdy = self._q_state_jacobian(fwd_state, param)
        # Sparse-aware: galerkin state equations return sparse Jacobians.
        adj_state = solve_maybe_sparse(
            self._bkd, drdy.T, self._bkd.flatten(-dqdy.T)
        )[:, None]
        self._storage.set_adjoint_state(adj_state)
        self._storage.set_state_eq_state_jacobian(drdy)
        self._storage.set_qoi_state_jacobian(dqdy)
        return adj_state

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

    def _get_adjoint_state(self, init_fwd_state: Array, param: Array) -> Array:
        """Get the adjoint state, solving for it if necessary."""
        if (
            not self._storage.has_parameter(param)
            or not self._storage.has_adjoint_state()
        ):
            fwd_state = self._get_forward_state(init_fwd_state, param)
            self.solve_adjoint_equation(fwd_state, param)
        return self._storage.get_adjoint_state()

    def _get_state_eq_param_jacobian(self, fwd_state: Array, param: Array) -> Array:
        """
        Get the parameter Jacobian of the state equation, computing it if necessary.

        Parameters
        ----------
        fwd_state : Array
            Forward state.
        param : Array
            Parameters.

        Returns
        -------
        Array
            Parameter Jacobian of the state equation.
        """
        if not self._storage.has_state_eq_param_jacobian():
            drdp = self._state_eq.param_jacobian(fwd_state, param)
            self._storage.set_state_eq_param_jacobian(drdp)
        return self._storage.get_state_eq_param_jacobian()

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
        adj_state = self.solve_adjoint_equation(fwd_state, param)
        drdp = self._get_state_eq_param_jacobian(fwd_state, param)
        jacobian = (
            self._q_param_jacobian(fwd_state, param) + adj_state.T @ drdp
        )
        return jacobian

    def _hvp(self, init_fwd_state: Array, param: Array, vvec: Array) -> Array:
        """
        Apply the Hessian of the functional with respect to the parameters
        to a vector, by the second-order adjoint method.

        Parameters
        ----------
        init_fwd_state : Array
            Initial forward state.
        param : Array
            Parameters.
        vvec : Array
            Direction. Shape: ``(nparams, 1)``.

        Returns
        -------
        Array
            Hessian-vector product. Shape: ``(nparams, 1)``.
        """
        lagrangian = self._lagrangian
        if lagrangian is None:
            raise RuntimeError(
                "hvp is unavailable; check derivatives() before calling"
            )
        if vvec.shape != (self.nparams(), 1):
            raise ValueError(
                f"vvec has shape {vvec.shape} but must be {(self.nparams(), 1)}"
            )
        fwd_state = self._get_forward_state(init_fwd_state, param)
        adj_state = self._get_adjoint_state(init_fwd_state, param)
        # The state Jacobian is stored by the adjoint solve.
        drdy = self._storage.get_state_eq_state_jacobian()
        drdp = self._get_state_eq_param_jacobian(fwd_state, param)
        # Forward (tangent) state: dR/du w = dR/dp v.
        # Sparse-aware: galerkin state equations return sparse Jacobians.
        wvec = solve_maybe_sparse(
            self._bkd, drdy, self._bkd.flatten(drdp @ vvec)
        )[:, None]
        # Second adjoint state: (dR/du)^T s = L_uu w - L_up v.
        rhs = lagrangian.state_state(
            fwd_state, param, adj_state, wvec
        ) - lagrangian.state_param(fwd_state, param, adj_state, vvec)
        svec = solve_maybe_sparse(
            self._bkd, drdy.T, self._bkd.flatten(rhs)
        )[:, None]
        lps_hvp = lagrangian.param_state(fwd_state, param, adj_state, wvec)
        lpp_hvp = lagrangian.param_param(fwd_state, param, adj_state, vvec)
        return drdp.T @ svec - lps_hvp + lpp_hvp

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
