"""
Derivative checker for time-dependent adjoint operators.

Use this instead of plain DerivativeChecker when validating derivatives in
time-dependent adjoint/HVP pipelines. A DerivativeChecker on the final
functional output would only verify the composed map from parameters to QoI.
This checker validates each layer independently:

1. ODE residual: jacobian, param_jacobian, and all four HVP blocks
2. Time stepping residual: param_jacobian and HVP methods at each time step,
   verifying that the time stepper correctly wraps the ODE residual
3. Full HVP accumulation over time: the assembled Hessian-vector product
   that the adjoint operator produces by marching backward

Checks are performed at multiple time steps to catch time-dependent bugs
(e.g., wrong deltat scaling, stale cached state).

Usage::

    checker = TimeAdjointDerivativeChecker(adjoint_operator)
    checker.check_all_derivatives(init_state, param, tol=1e-6)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, List, Optional, cast

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
    FunctionWithJVPFromCallable,
)
from pyapprox.ode.protocols.ode_residual import (
    ODEResidualWithHVPProtocol,
)
from pyapprox.ode.protocols.time_stepping import (
    HVPEnabledTimeSteppingResidualProtocol,
)
from pyapprox.util.backends.protocols import Array

if TYPE_CHECKING:
    from pyapprox.ode.operator.time_adjoint_hvp import (
        TimeAdjointOperatorWithHVP,
    )


class TimeAdjointDerivativeChecker(Generic[Array]):
    """
    Derivative checker for time-dependent adjoint operators.

    This class checks derivatives at three levels:
    1. ODE Residual level - tests the underlying ODE functions
    2. Time Residual level - tests the time stepping residual functions
    3. Accumulation level - tests the full HVP accumulation

    All checks are performed at multiple time steps to verify time dependence.
    This checker is reusable for any native residual and time stepper.
    """

    def __init__(self, adjoint_operator: TimeAdjointOperatorWithHVP[Array]) -> None:
        """
        Initialize the derivative checker.

        Parameters
        ----------
        adjoint_operator : TimeAdjointOperatorWithHVP
            The time adjoint operator to check.

        Raises
        ------
        TypeError
            If the integrator's time residual does not support HVP methods.
        """
        self._operator = adjoint_operator
        self._integrator = adjoint_operator._integrator
        self._functional = adjoint_operator._functional
        hvp_tr = self._integrator.hvp_time_residual()
        if hvp_tr is None:
            raise TypeError(
                "TimeAdjointDerivativeChecker requires an HVP-capable stepper. "
                f"Got {type(self._integrator.time_residual()).__name__}."
            )
        self._time_residual: HVPEnabledTimeSteppingResidualProtocol[Array] = hvp_tr
        self._ode_residual: ODEResidualWithHVPProtocol[Array] = cast(
            ODEResidualWithHVPProtocol[Array],
            hvp_tr.native_residual,
        )
        self._bkd = self._ode_residual.bkd()

    def _get_fd_eps(self, fd_eps: Optional[Array] = None) -> Array:
        """Get default finite difference step sizes if not provided."""
        if fd_eps is None:
            return self._bkd.flip(self._bkd.logspace(-12, -1, 12))
        return fd_eps

    def _to_2d(self, arr: Array) -> Array:
        """Convert 1D array to 2D column vector for DerivativeChecker."""
        if arr.ndim == 1:
            return arr[:, None]
        return arr

    def _from_2d(self, arr: Array) -> Array:
        """Convert 2D column vector to 1D array."""
        if arr.ndim == 2 and arr.shape[1] == 1:
            return self._bkd.flatten(arr)
        return arr

    def _check_and_return(
        self,
        checker: DerivativeChecker[Array],
        sample: Array,
        fd_eps: Optional[Array],
        verbosity: int,
    ) -> Array:
        """Run check_derivatives and return the error array."""
        results: List[Array] = checker.check_derivatives(
            sample,
            fd_eps=self._get_fd_eps(fd_eps),
            direction=None,
            relative=True,
            verbosity=verbosity,
        )
        return results[0]

    def _assert_derivatives_close(
        self, errors: Array, tol: float, label: str
    ) -> None:
        """Assert finite-difference error ratio is below tolerance.

        Parameters
        ----------
        errors : Array
            Error array from check_derivatives (one entry per FD step size).
        tol : float
            Maximum acceptable min/max error ratio (< 1e-6 typical).
        label : str
            Description for the assertion message.
        """
        ratio = self._bkd.to_float(
            self._bkd.min(errors) / self._bkd.max(errors)
        )
        assert ratio <= tol, (
            f"{label}: error ratio {ratio:.2e} exceeds tolerance {tol:.2e}"
        )

    # =========================================================================
    # ODE Residual Checks
    # =========================================================================

    def check_ode_jacobian(
        self,
        state: Array,
        time: float = 0.0,
        fd_eps: Optional[Array] = None,
        verbosity: int = 0,
    ) -> Array:
        """
        Check ODE state Jacobian: df/dy.

        Parameters
        ----------
        state : Array
            State at which to check. Shape: (nstates,) or (nstates, 1)
        time : float
            Time at which to evaluate.
        fd_eps : Optional[Array]
            Finite difference step sizes.
        verbosity : int
            Verbosity level.

        Returns
        -------
        Array
            Finite difference errors.
        """
        if verbosity > 0:
            print(f"ODE Residual Jacobian check at t={time}")

        self._ode_residual.set_time(time)
        state_1d = self._from_2d(state)
        nstates = state_1d.shape[0]

        def fun(y_2d: Array) -> Array:
            y_1d = self._from_2d(y_2d)
            result = self._ode_residual(y_1d)
            return self._to_2d(result)

        def jac(y_2d: Array) -> Array:
            y_1d = self._from_2d(y_2d)
            return self._ode_residual.jacobian(y_1d)

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=nstates,
            nvars=nstates,
            fun=fun,
            jacobian=jac,
            bkd=self._bkd,
        )
        return self._check_and_return(
            DerivativeChecker(wrapper), self._to_2d(state_1d), fd_eps, verbosity
        )

    def check_ode_param_jacobian(
        self,
        state: Array,
        param: Array,
        time: float = 0.0,
        fd_eps: Optional[Array] = None,
        verbosity: int = 0,
    ) -> Array:
        """
        Check ODE parameter Jacobian: df/dp.

        Parameters
        ----------
        state : Array
            State at which to check. Shape: (nstates,) or (nstates, 1)
        param : Array
            Parameters. Shape: (nparams,) or (nparams, 1)
        time : float
            Time at which to evaluate.
        fd_eps : Optional[Array]
            Finite difference step sizes.
        verbosity : int
            Verbosity level.

        Returns
        -------
        Array
            Finite difference errors.
        """
        if verbosity > 0:
            print(f"ODE Param Jacobian check at t={time}")

        state_1d = self._from_2d(state)
        param_2d = self._to_2d(param)
        nstates = state_1d.shape[0]
        nparams = param_2d.shape[0]

        def fun(p_2d: Array) -> Array:
            self._ode_residual.set_param(p_2d)
            self._ode_residual.set_time(time)
            return self._to_2d(self._ode_residual(state_1d))

        def jac(p_2d: Array) -> Array:
            self._ode_residual.set_param(p_2d)
            self._ode_residual.set_time(time)
            return self._ode_residual.param_jacobian(state_1d)

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=nstates,
            nvars=nparams,
            fun=fun,
            jacobian=jac,
            bkd=self._bkd,
        )
        return self._check_and_return(
            DerivativeChecker(wrapper), param_2d, fd_eps, verbosity
        )

    def check_ode_state_state_hvp(
        self,
        state: Array,
        adj_state: Array,
        time: float = 0.0,
        fd_eps: Optional[Array] = None,
        verbosity: int = 0,
    ) -> Array:
        """
        Check ODE state-state HVP: adj^T @ (d2f/dy2) @ w.

        This is the directional derivative of (df/dy)^T @ adj w.r.t. state.
        """
        if verbosity > 0:
            print(f"ODE state_state_hvp check at t={time}")

        self._ode_residual.set_time(time)
        state_1d = self._from_2d(state)
        adj_1d = self._from_2d(adj_state)
        nstates = state_1d.shape[0]

        def fun(y_2d: Array) -> Array:
            y_1d = self._from_2d(y_2d)
            result = self._ode_residual.jacobian(y_1d).T @ adj_1d
            return self._to_2d(result)

        def jvp(y_2d: Array, w_2d: Array) -> Array:
            y_1d = self._from_2d(y_2d)
            w_1d = self._from_2d(w_2d)
            result = self._ode_residual.state_state_hvp(y_1d, adj_1d, w_1d)
            return self._to_2d(result)

        wrapper = FunctionWithJVPFromCallable(
            nqoi=nstates,
            nvars=nstates,
            fun=fun,
            jvp=jvp,
            bkd=self._bkd,
        )
        return self._check_and_return(
            DerivativeChecker(wrapper), self._to_2d(state_1d), fd_eps, verbosity
        )

    def check_ode_state_param_hvp(
        self,
        state: Array,
        param: Array,
        adj_state: Array,
        time: float = 0.0,
        fd_eps: Optional[Array] = None,
        verbosity: int = 0,
    ) -> Array:
        """
        Check ODE state-param HVP: adj^T @ (d2f/dy dp) @ v.

        This is the directional derivative of (df/dy)^T @ adj w.r.t. param.
        """
        if verbosity > 0:
            print(f"ODE state_param_hvp check at t={time}")

        state_1d = self._from_2d(state)
        param_2d = self._to_2d(param)
        adj_1d = self._from_2d(adj_state)
        nstates = state_1d.shape[0]
        nparams = param_2d.shape[0]

        def fun(p_2d: Array) -> Array:
            self._ode_residual.set_param(p_2d)
            self._ode_residual.set_time(time)
            result = self._ode_residual.jacobian(state_1d).T @ adj_1d
            return self._to_2d(result)

        def jvp(p_2d: Array, v_2d: Array) -> Array:
            self._ode_residual.set_param(p_2d)
            self._ode_residual.set_time(time)
            result = self._ode_residual.state_param_hvp(state_1d, adj_1d, v_2d)
            return self._to_2d(result)

        wrapper = FunctionWithJVPFromCallable(
            nqoi=nstates,
            nvars=nparams,
            fun=fun,
            jvp=jvp,
            bkd=self._bkd,
        )
        return self._check_and_return(
            DerivativeChecker(wrapper), param_2d, fd_eps, verbosity
        )

    def check_ode_param_state_hvp(
        self,
        state: Array,
        param: Array,
        adj_state: Array,
        time: float = 0.0,
        fd_eps: Optional[Array] = None,
        verbosity: int = 0,
    ) -> Array:
        """
        Check ODE param-state HVP: adj^T @ (d2f/dp dy) @ w.

        This is the directional derivative of (df/dp)^T @ adj w.r.t. state.
        """
        if verbosity > 0:
            print(f"ODE param_state_hvp check at t={time}")

        state_1d = self._from_2d(state)
        param_2d = self._to_2d(param)
        adj_1d = self._from_2d(adj_state)
        nstates = state_1d.shape[0]
        nparams = param_2d.shape[0]

        self._ode_residual.set_param(param_2d)

        def fun(y_2d: Array) -> Array:
            y_1d = self._from_2d(y_2d)
            self._ode_residual.set_time(time)
            result = self._ode_residual.param_jacobian(y_1d).T @ adj_1d
            return self._to_2d(result)

        def jvp(y_2d: Array, w_2d: Array) -> Array:
            y_1d = self._from_2d(y_2d)
            w_1d = self._from_2d(w_2d)
            self._ode_residual.set_time(time)
            result = self._ode_residual.param_state_hvp(y_1d, adj_1d, w_1d)
            return self._to_2d(result)

        wrapper = FunctionWithJVPFromCallable(
            nqoi=nparams,
            nvars=nstates,
            fun=fun,
            jvp=jvp,
            bkd=self._bkd,
        )
        return self._check_and_return(
            DerivativeChecker(wrapper), self._to_2d(state_1d), fd_eps, verbosity
        )

    def check_ode_param_param_hvp(
        self,
        state: Array,
        param: Array,
        adj_state: Array,
        time: float = 0.0,
        fd_eps: Optional[Array] = None,
        verbosity: int = 0,
    ) -> Array:
        """
        Check ODE param-param HVP: adj^T @ (d2f/dp2) @ v.

        This is the directional derivative of (df/dp)^T @ adj w.r.t. param.
        """
        if verbosity > 0:
            print(f"ODE param_param_hvp check at t={time}")

        state_1d = self._from_2d(state)
        param_2d = self._to_2d(param)
        adj_1d = self._from_2d(adj_state)
        nparams = param_2d.shape[0]

        def fun(p_2d: Array) -> Array:
            self._ode_residual.set_param(p_2d)
            self._ode_residual.set_time(time)
            result = self._ode_residual.param_jacobian(state_1d).T @ adj_1d
            return self._to_2d(result)

        def jvp(p_2d: Array, v_2d: Array) -> Array:
            self._ode_residual.set_param(p_2d)
            self._ode_residual.set_time(time)
            result = self._ode_residual.param_param_hvp(state_1d, adj_1d, v_2d)
            return self._to_2d(result)

        wrapper = FunctionWithJVPFromCallable(
            nqoi=nparams,
            nvars=nparams,
            fun=fun,
            jvp=jvp,
            bkd=self._bkd,
        )
        return self._check_and_return(
            DerivativeChecker(wrapper), param_2d, fd_eps, verbosity
        )

    # =========================================================================
    # Time Residual Checks
    # =========================================================================

    def check_time_residual_param_jacobian(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        param: Array,
        time: float,
        deltat: float,
        fd_eps: Optional[Array] = None,
        verbosity: int = 0,
    ) -> Array:
        """
        Check time residual parameter Jacobian: dR/dp.

        Parameters
        ----------
        fsol_nm1 : Array
            Forward solution at previous time step. Shape: (nstates,)
        fsol_n : Array
            Forward solution at current time step. Shape: (nstates,)
        param : Array
            Parameters. Shape: (nparams,) or (nparams, 1)
        time : float
            Time at start of step (t_{n-1}).
        deltat : float
            Time step size.
        fd_eps : Optional[Array]
            Finite difference step sizes.
        verbosity : int
            Verbosity level.

        Returns
        -------
        Array
            Finite difference errors.
        """
        if verbosity > 0:
            print(f"Time Residual param_jacobian check at t={time}")

        fsol_nm1_1d = self._from_2d(fsol_nm1)
        fsol_n_1d = self._from_2d(fsol_n)
        param_2d = self._to_2d(param)
        nstates = fsol_n_1d.shape[0]
        nparams = param_2d.shape[0]

        def fun(p_2d: Array) -> Array:
            self._ode_residual.set_param(p_2d)
            self._time_residual.set_time(time, deltat, fsol_nm1_1d)
            return self._to_2d(self._time_residual(fsol_n_1d))

        def jac(p_2d: Array) -> Array:
            self._ode_residual.set_param(p_2d)
            self._time_residual.set_time(time, deltat, fsol_nm1_1d)
            return self._time_residual.param_jacobian(fsol_nm1_1d, fsol_n_1d)

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=nstates,
            nvars=nparams,
            fun=fun,
            jacobian=jac,
            bkd=self._bkd,
        )
        return self._check_and_return(
            DerivativeChecker(wrapper), param_2d, fd_eps, verbosity
        )

    def check_time_residual_state_state_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        param: Array,
        adj_state: Array,
        time: float,
        deltat: float,
        fd_eps: Optional[Array] = None,
        verbosity: int = 0,
    ) -> Array:
        """
        Check time residual state-state HVP: adj^T @ (d2R/dy_{n-1}^2) @ w.

        This checks the derivative of (dR/dy_{n-1})^T @ adj w.r.t. y_{n-1}.
        """
        if verbosity > 0:
            print(f"Time Residual state_state_hvp check at t={time}")

        fsol_nm1_1d = self._from_2d(fsol_nm1)
        fsol_n_1d = self._from_2d(fsol_n)
        param_2d = self._to_2d(param)
        adj_1d = self._from_2d(adj_state)
        nstates = fsol_n_1d.shape[0]

        self._ode_residual.set_param(param_2d)

        def fun(y_nm1_2d: Array) -> Array:
            y_nm1_1d = self._from_2d(y_nm1_2d)
            self._time_residual.set_time(time, deltat, y_nm1_1d)
            result = (
                self._time_residual.adjoint_off_diag_jacobian(y_nm1_1d, deltat).T
                @ adj_1d
            )
            return self._to_2d(result)

        def jvp(y_nm1_2d: Array, w_2d: Array) -> Array:
            y_nm1_1d = self._from_2d(y_nm1_2d)
            w_1d = self._from_2d(w_2d)
            self._time_residual.set_time(time, deltat, y_nm1_1d)
            result = self._time_residual.state_state_hvp(
                y_nm1_1d, fsol_n_1d, adj_1d, w_1d
            )
            return self._to_2d(result)

        wrapper = FunctionWithJVPFromCallable(
            nqoi=nstates,
            nvars=nstates,
            fun=fun,
            jvp=jvp,
            bkd=self._bkd,
        )
        return self._check_and_return(
            DerivativeChecker(wrapper), self._to_2d(fsol_nm1_1d), fd_eps, verbosity
        )

    def check_time_residual_param_param_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        param: Array,
        adj_state: Array,
        time: float,
        deltat: float,
        fd_eps: Optional[Array] = None,
        verbosity: int = 0,
    ) -> Array:
        """
        Check time residual param-param HVP: adj^T @ (d2R/dp^2) @ v.

        This checks the derivative of (dR/dp)^T @ adj w.r.t. p.
        """
        if verbosity > 0:
            print(f"Time Residual param_param_hvp check at t={time}")

        fsol_nm1_1d = self._from_2d(fsol_nm1)
        fsol_n_1d = self._from_2d(fsol_n)
        param_2d = self._to_2d(param)
        adj_1d = self._from_2d(adj_state)
        nparams = param_2d.shape[0]

        def fun(p_2d: Array) -> Array:
            self._ode_residual.set_param(p_2d)
            self._time_residual.set_time(time, deltat, fsol_nm1_1d)
            jac = self._time_residual.param_jacobian(fsol_nm1_1d, fsol_n_1d)
            result = jac.T @ adj_1d
            return self._to_2d(result)

        def jvp(p_2d: Array, v_2d: Array) -> Array:
            self._ode_residual.set_param(p_2d)
            self._time_residual.set_time(time, deltat, fsol_nm1_1d)
            result = self._time_residual.param_param_hvp(
                fsol_nm1_1d, fsol_n_1d, adj_1d, v_2d
            )
            return self._to_2d(result)

        wrapper = FunctionWithJVPFromCallable(
            nqoi=nparams,
            nvars=nparams,
            fun=fun,
            jvp=jvp,
            bkd=self._bkd,
        )
        return self._check_and_return(
            DerivativeChecker(wrapper), param_2d, fd_eps, verbosity
        )

    def check_time_residual_state_param_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        param: Array,
        adj_state: Array,
        time: float,
        deltat: float,
        fd_eps: Optional[Array] = None,
        verbosity: int = 0,
    ) -> Array:
        """
        Check time residual state-param HVP: adj^T @ (d2R/dy_{n-1} dp) @ v.

        This checks the derivative of (dR/dy_{n-1})^T @ adj w.r.t. p.
        """
        if verbosity > 0:
            print(f"Time Residual state_param_hvp check at t={time}")

        fsol_nm1_1d = self._from_2d(fsol_nm1)
        fsol_n_1d = self._from_2d(fsol_n)
        param_2d = self._to_2d(param)
        adj_1d = self._from_2d(adj_state)
        nstates = fsol_n_1d.shape[0]
        nparams = param_2d.shape[0]

        def fun(p_2d: Array) -> Array:
            self._ode_residual.set_param(p_2d)
            self._time_residual.set_time(time, deltat, fsol_nm1_1d)
            result = (
                self._time_residual.adjoint_off_diag_jacobian(fsol_nm1_1d, deltat).T
                @ adj_1d
            )
            return self._to_2d(result)

        def jvp(p_2d: Array, v_2d: Array) -> Array:
            self._ode_residual.set_param(p_2d)
            self._time_residual.set_time(time, deltat, fsol_nm1_1d)
            result = self._time_residual.state_param_hvp(
                fsol_nm1_1d, fsol_n_1d, adj_1d, v_2d
            )
            return self._to_2d(result)

        wrapper = FunctionWithJVPFromCallable(
            nqoi=nstates,
            nvars=nparams,
            fun=fun,
            jvp=jvp,
            bkd=self._bkd,
        )
        return self._check_and_return(
            DerivativeChecker(wrapper), param_2d, fd_eps, verbosity
        )

    def check_time_residual_param_state_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        param: Array,
        adj_state: Array,
        time: float,
        deltat: float,
        fd_eps: Optional[Array] = None,
        verbosity: int = 0,
    ) -> Array:
        """
        Check time residual param-state HVP: adj^T @ (d2R/dp dy_{n-1}) @ w.

        This checks the derivative of (dR/dp)^T @ adj w.r.t. y_{n-1}.
        """
        if verbosity > 0:
            print(f"Time Residual param_state_hvp check at t={time}")

        fsol_nm1_1d = self._from_2d(fsol_nm1)
        fsol_n_1d = self._from_2d(fsol_n)
        param_2d = self._to_2d(param)
        adj_1d = self._from_2d(adj_state)
        nstates = fsol_n_1d.shape[0]
        nparams = param_2d.shape[0]

        self._ode_residual.set_param(param_2d)

        def fun(y_nm1_2d: Array) -> Array:
            y_nm1_1d = self._from_2d(y_nm1_2d)
            self._time_residual.set_time(time, deltat, y_nm1_1d)
            jac = self._time_residual.param_jacobian(y_nm1_1d, fsol_n_1d)
            result = jac.T @ adj_1d
            return self._to_2d(result)

        def jvp(y_nm1_2d: Array, w_2d: Array) -> Array:
            y_nm1_1d = self._from_2d(y_nm1_2d)
            w_1d = self._from_2d(w_2d)
            self._time_residual.set_time(time, deltat, y_nm1_1d)
            result = self._time_residual.param_state_hvp(
                y_nm1_1d, fsol_n_1d, adj_1d, w_1d
            )
            return self._to_2d(result)

        wrapper = FunctionWithJVPFromCallable(
            nqoi=nparams,
            nvars=nstates,
            fun=fun,
            jvp=jvp,
            bkd=self._bkd,
        )
        return self._check_and_return(
            DerivativeChecker(wrapper), self._to_2d(fsol_nm1_1d), fd_eps, verbosity
        )

    # =========================================================================
    # Full Operator Checks
    # =========================================================================

    def check_jacobian(
        self,
        init_state: Array,
        param: Array,
        fd_eps: Optional[Array] = None,
        verbosity: int = 0,
    ) -> Array:
        """
        Check the full Jacobian dQ/dp via adjoint method.

        Parameters
        ----------
        init_state : Array
            Initial state. Shape: (nstates,) or (nstates, 1)
        param : Array
            Parameters. Shape: (nparams,) or (nparams, 1)
        fd_eps : Optional[Array]
            Finite difference step sizes.
        verbosity : int
            Verbosity level.

        Returns
        -------
        Array
            Finite difference errors.
        """
        if verbosity > 0:
            print("Full Jacobian check")

        init_state_1d = self._from_2d(init_state)
        param_2d = self._to_2d(param)
        nparams = param_2d.shape[0]

        def fun(p_2d: Array) -> Array:
            self._ode_residual.set_param(p_2d)
            self._operator.storage()._clear()
            fwd_sols, times = self._operator._get_forward_trajectory(
                init_state_1d, p_2d
            )
            return self._functional(fwd_sols, p_2d)

        def jac(p_2d: Array) -> Array:
            self._ode_residual.set_param(p_2d)
            self._operator.storage()._clear()
            return self._operator.jacobian(init_state_1d, p_2d)

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=1,
            nvars=nparams,
            fun=fun,
            jacobian=jac,
            bkd=self._bkd,
        )
        return self._check_and_return(
            DerivativeChecker(wrapper), param_2d, fd_eps, verbosity
        )

    def check_hvp(
        self,
        init_state: Array,
        param: Array,
        fd_eps: Optional[Array] = None,
        verbosity: int = 0,
    ) -> Array:
        """
        Check the full HVP via second-order adjoints.

        Parameters
        ----------
        init_state : Array
            Initial state. Shape: (nstates,) or (nstates, 1)
        param : Array
            Parameters. Shape: (nparams,) or (nparams, 1)
        fd_eps : Optional[Array]
            Finite difference step sizes.
        verbosity : int
            Verbosity level.

        Returns
        -------
        Array
            Finite difference errors.
        """
        if verbosity > 0:
            print("Full HVP check")

        init_state_1d = self._from_2d(init_state)
        param_2d = self._to_2d(param)
        nparams = param_2d.shape[0]

        def fun(p_2d: Array) -> Array:
            self._ode_residual.set_param(p_2d)
            self._operator.storage()._clear()
            return self._operator.jacobian(init_state_1d, p_2d).T

        def jvp(p_2d: Array, v_2d: Array) -> Array:
            self._ode_residual.set_param(p_2d)
            self._operator.storage()._clear()
            # Must recompute jacobian to populate storage before HVP
            _ = self._operator.jacobian(init_state_1d, p_2d)
            return self._operator.hvp(init_state_1d, p_2d, v_2d)

        wrapper = FunctionWithJVPFromCallable(
            nqoi=nparams,
            nvars=nparams,
            fun=fun,
            jvp=jvp,
            bkd=self._bkd,
        )
        return self._check_and_return(
            DerivativeChecker(wrapper), param_2d, fd_eps, verbosity
        )

    # =========================================================================
    # Convenience methods
    # =========================================================================

    def check_all_ode_derivatives(
        self,
        state: Array,
        param: Array,
        adj_state: Optional[Array] = None,
        time: float = 0.0,
        fd_eps: Optional[Array] = None,
        verbosity: int = 0,
    ) -> List[Array]:
        """Check all ODE-level derivatives."""
        if adj_state is None:
            adj_state = self._bkd.ones(self._from_2d(state).shape)

        return [
            self.check_ode_jacobian(state, time, fd_eps, verbosity),
            self.check_ode_param_jacobian(state, param, time, fd_eps, verbosity),
            self.check_ode_state_state_hvp(
                state, adj_state, time, fd_eps, verbosity
            ),
            self.check_ode_state_param_hvp(
                state, param, adj_state, time, fd_eps, verbosity
            ),
            self.check_ode_param_state_hvp(
                state, param, adj_state, time, fd_eps, verbosity
            ),
            self.check_ode_param_param_hvp(
                state, param, adj_state, time, fd_eps, verbosity
            ),
        ]

    def check_all_time_residual_derivatives(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        param: Array,
        adj_state: Optional[Array] = None,
        time: float = 0.0,
        deltat: float = 0.01,
        fd_eps: Optional[Array] = None,
        verbosity: int = 0,
    ) -> List[Array]:
        """Check all time-residual-level derivatives."""
        if adj_state is None:
            adj_state = self._bkd.ones(self._from_2d(fsol_n).shape)

        return [
            self.check_time_residual_param_jacobian(
                fsol_nm1, fsol_n, param, time, deltat, fd_eps, verbosity
            ),
            self.check_time_residual_state_state_hvp(
                fsol_nm1, fsol_n, param, adj_state, time, deltat, fd_eps, verbosity
            ),
            self.check_time_residual_param_param_hvp(
                fsol_nm1, fsol_n, param, adj_state, time, deltat, fd_eps, verbosity
            ),
            self.check_time_residual_state_param_hvp(
                fsol_nm1, fsol_n, param, adj_state, time, deltat, fd_eps, verbosity
            ),
            self.check_time_residual_param_state_hvp(
                fsol_nm1, fsol_n, param, adj_state, time, deltat, fd_eps, verbosity
            ),
        ]
