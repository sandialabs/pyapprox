"""
Derivative checker for time-dependent adjoint operators.

This module provides comprehensive derivative checking for:
1. ODE residual functions (jacobian, param_jacobian, HVP methods)
2. Time stepping residual functions (param_jacobian, HVP methods at each step)
3. Full HVP accumulation over time

The checks are performed at multiple time steps to ensure residuals correctly
depend on time.

Usage:
    # For any native ODE residual and time stepper
    checker = TimeAdjointDerivativeChecker(adjoint_operator)
    checker.check_all_derivatives(init_state, param, tol=1e-6)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, List, Optional

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
    FunctionWithJVPFromCallable,
)
from pyapprox.util.backends.protocols import Array

if TYPE_CHECKING:
    from pyapprox.pde.time.operator.time_adjoint_hvp import (
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
        """
        self._operator = adjoint_operator
        self._integrator = adjoint_operator._integrator
        self._functional = adjoint_operator._functional
        self._time_residual = self._integrator._newton_solver._residual
        self._ode_residual = self._time_residual._residual
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
        checker = DerivativeChecker(wrapper)
        return checker.check_derivatives(
            self._to_2d(state_1d),
            fd_eps=self._get_fd_eps(fd_eps),
            direction=None,
            relative=True,
            verbosity=verbosity,
        )[0]

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
        checker = DerivativeChecker(wrapper)
        return checker.check_derivatives(
            param_2d,
            fd_eps=self._get_fd_eps(fd_eps),
            direction=None,
            relative=True,
            verbosity=verbosity,
        )[0]

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
        checker = DerivativeChecker(wrapper)
        return checker.check_derivatives(
            self._to_2d(state_1d),
            fd_eps=self._get_fd_eps(fd_eps),
            direction=None,
            relative=True,
            verbosity=verbosity,
        )[0]

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
        checker = DerivativeChecker(wrapper)
        return checker.check_derivatives(
            param_2d,
            fd_eps=self._get_fd_eps(fd_eps),
            direction=None,
            relative=True,
            verbosity=verbosity,
        )[0]

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
        checker = DerivativeChecker(wrapper)
        return checker.check_derivatives(
            self._to_2d(state_1d),
            fd_eps=self._get_fd_eps(fd_eps),
            direction=None,
            relative=True,
            verbosity=verbosity,
        )[0]

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
        checker = DerivativeChecker(wrapper)
        return checker.check_derivatives(
            param_2d,
            fd_eps=self._get_fd_eps(fd_eps),
            direction=None,
            relative=True,
            verbosity=verbosity,
        )[0]

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
            return self._time_residual._param_jacobian(fsol_nm1_1d, fsol_n_1d)

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=nstates,
            nvars=nparams,
            fun=fun,
            jacobian=jac,
            bkd=self._bkd,
        )
        checker = DerivativeChecker(wrapper)
        return checker.check_derivatives(
            param_2d,
            fd_eps=self._get_fd_eps(fd_eps),
            direction=None,
            relative=True,
            verbosity=verbosity,
        )[0]

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
            result = self._time_residual._state_state_hvp(
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
        checker = DerivativeChecker(wrapper)
        return checker.check_derivatives(
            self._to_2d(fsol_nm1_1d),
            fd_eps=self._get_fd_eps(fd_eps),
            direction=None,
            relative=True,
            verbosity=verbosity,
        )[0]

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
            jac = self._time_residual._param_jacobian(fsol_nm1_1d, fsol_n_1d)
            result = jac.T @ adj_1d
            return self._to_2d(result)

        def jvp(p_2d: Array, v_2d: Array) -> Array:
            self._ode_residual.set_param(p_2d)
            self._time_residual.set_time(time, deltat, fsol_nm1_1d)
            result = self._time_residual._param_param_hvp(
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
        checker = DerivativeChecker(wrapper)
        return checker.check_derivatives(
            param_2d,
            fd_eps=self._get_fd_eps(fd_eps),
            direction=None,
            relative=True,
            verbosity=verbosity,
        )[0]

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
            result = self._time_residual._state_param_hvp(
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
        checker = DerivativeChecker(wrapper)
        return checker.check_derivatives(
            param_2d,
            fd_eps=self._get_fd_eps(fd_eps),
            direction=None,
            relative=True,
            verbosity=verbosity,
        )[0]

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
            jac = self._time_residual._param_jacobian(y_nm1_1d, fsol_n_1d)
            result = jac.T @ adj_1d
            return self._to_2d(result)

        def jvp(y_nm1_2d: Array, w_2d: Array) -> Array:
            y_nm1_1d = self._from_2d(y_nm1_2d)
            w_1d = self._from_2d(w_2d)
            self._time_residual.set_time(time, deltat, y_nm1_1d)
            result = self._time_residual._param_state_hvp(
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
        checker = DerivativeChecker(wrapper)
        return checker.check_derivatives(
            self._to_2d(fsol_nm1_1d),
            fd_eps=self._get_fd_eps(fd_eps),
            direction=None,
            relative=True,
            verbosity=verbosity,
        )[0]

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
        checker = DerivativeChecker(wrapper)
        return checker.check_derivatives(
            param_2d,
            fd_eps=self._get_fd_eps(fd_eps),
            direction=None,
            relative=True,
            verbosity=verbosity,
        )[0]

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
        checker = DerivativeChecker(wrapper)
        return checker.check_derivatives(
            param_2d,
            fd_eps=self._get_fd_eps(fd_eps),
            direction=None,
            relative=True,
            verbosity=verbosity,
        )[0]

    # =========================================================================
    # Comprehensive Check
    # =========================================================================

    def check_all_derivatives(
        self,
        init_state: Array,
        param: Array,
        tol: float = 1e-6,
        times: Optional[List[float]] = None,
        fd_eps: Optional[Array] = None,
        verbosity: int = 0,
    ) -> None:
        """
        Check all derivatives at multiple time points.

        Parameters
        ----------
        init_state : Array
            Initial state. Shape: (nstates,) or (nstates, 1)
        param : Array
            Parameters. Shape: (nparams,) or (nparams, 1)
        tol : float
            Tolerance for error ratio (min/max errors).
        times : Optional[List[float]]
            Times at which to check ODE residual derivatives.
            Default: [0.0, dt, 2*dt] where dt is the time step.
        fd_eps : Optional[Array]
            Finite difference step sizes.
        verbosity : int
            Verbosity level.

        Raises
        ------
        AssertionError
            If any derivative check fails.
        """
        init_state_1d = self._from_2d(init_state)
        param_2d = self._to_2d(param)

        self._ode_residual.set_param(param_2d)
        deltat = self._integrator._deltat

        # Default times to check
        if times is None:
            times = [0.0, deltat, 2 * deltat]

        # Get forward trajectory for test states
        self._operator.storage()._clear()
        fwd_sols, time_pts = self._operator._get_forward_trajectory(
            init_state_1d, param_2d
        )
        adj_sols = self._operator._get_adjoint_trajectory(init_state_1d, param_2d)

        nstates = init_state_1d.shape[0]
        adj_state = self._bkd.asarray([1.0] + [0.0] * (nstates - 1))

        # 1. Check ODE residual derivatives at multiple times
        if verbosity > 0:
            print("\n" + "=" * 60)
            print("CHECKING ODE RESIDUAL DERIVATIVES")
            print("=" * 60)

        for t in times:
            errors = self.check_ode_jacobian(init_state_1d, t, fd_eps, verbosity)
            self._assert_derivatives_close(errors, tol, f"ODE jacobian at t={t}")

            errors = self.check_ode_param_jacobian(
                init_state_1d, param_2d, t, fd_eps, verbosity
            )
            self._assert_derivatives_close(errors, tol, f"ODE param_jacobian at t={t}")

            errors = self.check_ode_state_state_hvp(
                init_state_1d, adj_state, t, fd_eps, verbosity
            )
            self._assert_derivatives_close(errors, tol, f"ODE state_state_hvp at t={t}")

            errors = self.check_ode_state_param_hvp(
                init_state_1d, param_2d, adj_state, t, fd_eps, verbosity
            )
            self._assert_derivatives_close(errors, tol, f"ODE state_param_hvp at t={t}")

            errors = self.check_ode_param_state_hvp(
                init_state_1d, param_2d, adj_state, t, fd_eps, verbosity
            )
            self._assert_derivatives_close(errors, tol, f"ODE param_state_hvp at t={t}")

            errors = self.check_ode_param_param_hvp(
                init_state_1d, param_2d, adj_state, t, fd_eps, verbosity
            )
            self._assert_derivatives_close(errors, tol, f"ODE param_param_hvp at t={t}")

        # 2. Check time residual derivatives at each step
        if verbosity > 0:
            print("\n" + "=" * 60)
            print("CHECKING TIME RESIDUAL DERIVATIVES")
            print("=" * 60)

        ntimes = time_pts.shape[0]
        for nn in range(1, min(ntimes, 3)):  # Check first few steps
            t = float(time_pts[nn - 1])
            fsol_nm1 = fwd_sols[:, nn - 1]
            fsol_n = fwd_sols[:, nn]
            adj_n = adj_sols[:, nn]

            errors = self.check_time_residual_param_jacobian(
                fsol_nm1, fsol_n, param_2d, t, deltat, fd_eps, verbosity
            )
            self._assert_derivatives_close(
                errors, tol, f"Time param_jacobian step {nn}"
            )

            errors = self.check_time_residual_param_param_hvp(
                fsol_nm1, fsol_n, param_2d, adj_n, t, deltat, fd_eps, verbosity
            )
            self._assert_derivatives_close(
                errors, tol, f"Time param_param_hvp step {nn}"
            )

            errors = self.check_time_residual_state_param_hvp(
                fsol_nm1, fsol_n, param_2d, adj_n, t, deltat, fd_eps, verbosity
            )
            self._assert_derivatives_close(
                errors, tol, f"Time state_param_hvp step {nn}"
            )

            errors = self.check_time_residual_param_state_hvp(
                fsol_nm1, fsol_n, param_2d, adj_n, t, deltat, fd_eps, verbosity
            )
            self._assert_derivatives_close(
                errors, tol, f"Time param_state_hvp step {nn}"
            )

        # 3. Check full operator derivatives
        if verbosity > 0:
            print("\n" + "=" * 60)
            print("CHECKING FULL OPERATOR DERIVATIVES")
            print("=" * 60)

        errors = self.check_jacobian(init_state_1d, param_2d, fd_eps, verbosity)
        self._assert_derivatives_close(errors, tol, "Full Jacobian")

        errors = self.check_hvp(init_state_1d, param_2d, fd_eps, verbosity)
        self._assert_derivatives_close(errors, tol, "Full HVP")

        if verbosity > 0:
            print("\n" + "=" * 60)
            print("ALL DERIVATIVE CHECKS PASSED")
            print("=" * 60)

    def _assert_derivatives_close(
        self, errors: Array, tol: float, name: str = ""
    ) -> None:
        """
        Assert that finite difference errors show proper convergence.

        The error ratio (min/max) should be small, indicating the analytical
        derivative matches finite differences at the optimal step size.
        """
        min_err = self._bkd.to_float(self._bkd.min(errors))
        max_err = self._bkd.to_float(self._bkd.max(errors))

        if min_err == max_err:
            # All errors are the same - either all zero or no convergence
            if min_err > tol:
                raise AssertionError(
                    f"{name}: Constant error {min_err:.2e} > tol {tol:.2e}"
                )
        else:
            ratio = min_err / max_err
            if ratio > tol:
                raise AssertionError(
                    f"{name}: Error ratio {ratio:.2e} > tol {tol:.2e} "
                    f"(min={min_err:.2e}, max={max_err:.2e})"
                )
