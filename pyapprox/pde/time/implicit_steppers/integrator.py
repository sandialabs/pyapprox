"""
Time integrator with adjoint support for gradient computation.

Provides forward solve, adjoint solve, and gradient computation via
the discrete adjoint method.
"""

from typing import Generic, Optional, Tuple

from pyapprox.pde.sparse_utils import solve_maybe_sparse
from pyapprox.pde.time.functionals.protocols import (
    TransientFunctionalWithJacobianProtocol,
)
from pyapprox.pde.time.protocols.time_stepping import (
    AdjointEnabledTimeSteppingResidualProtocol,
    HVPEnabledTimeSteppingResidualProtocol,
)
from pyapprox.pde.time.protocols.type_guards import is_hvp_enabled
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.rootfinding.newton import NewtonSolver


class TimeIntegrator(Generic[Array]):
    """
    Time integrator with adjoint support for gradient computation.

    This class integrates time-dependent problems forward in time using a
    Newton solver, and computes gradients via the discrete adjoint method.

    Parameters
    ----------
    init_time : float
        Initial time.
    final_time : float
        Final time.
    deltat : float
        Time step size.
    newton_solver : NewtonSolver
        Newton solver for solving the residual equations. The solver's
        residual must implement AdjointEnabledTimeSteppingResidualProtocol.
    verbosity : int, optional
        Verbosity level. Defaults to 0.
    """

    def __init__(
        self,
        init_time: float,
        final_time: float,
        deltat: float,
        newton_solver: NewtonSolver[Array],
        verbosity: int = 0,
    ):
        time_residual = newton_solver.residual()
        if not isinstance(
            time_residual, AdjointEnabledTimeSteppingResidualProtocol
        ):
            raise TypeError(
                f"Newton solver's residual must satisfy "
                f"AdjointEnabledTimeSteppingResidualProtocol, "
                f"got {type(time_residual).__name__}"
            )
        self._time_residual: AdjointEnabledTimeSteppingResidualProtocol[Array] = (
            time_residual
        )
        self._bkd = self._time_residual.bkd()
        self._init_time = init_time
        self._final_time = final_time
        self._deltat = deltat
        self._verbosity = verbosity
        self._newton_solver = newton_solver
        self._functional: Optional[
            TransientFunctionalWithJacobianProtocol[Array]
        ] = None

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def set_functional(
        self, functional: TransientFunctionalWithJacobianProtocol[Array]
    ) -> None:
        """
        Set the functional for gradient computation.

        Parameters
        ----------
        functional : TransientFunctionalWithJacobianProtocol
            Functional defining the quantity of interest.
        """
        self._functional = functional

    def step(self, state: Array, deltat: float) -> Array:
        """
        Perform a single forward time step.

        Parameters
        ----------
        state : Array
            Current state. Shape: (nstates,)
        deltat : float
            Time step size.

        Returns
        -------
        Array
            State at the next time step. Shape: (nstates,)
        """
        self._time_residual.set_time(self._time, deltat, state)
        state = self._newton_solver.solve(state)
        self._time += deltat
        if self._verbosity >= 1:
            print("Time:", self._time)
        return state

    def solve(self, init_state: Array) -> Tuple[Array, Array]:
        """
        Solve the time-dependent problem forward in time.

        Parameters
        ----------
        init_state : Array
            Initial state. Shape: (nstates,)

        Returns
        -------
        Tuple[Array, Array]
            states : Array
                Solution trajectory. Shape: (nstates, ntimes)
            times : Array
                Time points. Shape: (ntimes,)
        """
        states_list: list[Array] = []
        times_list: list[float] = []
        self._time = self._init_time
        times_list.append(self._time)
        state = init_state
        states_list.append(init_state)
        while self._time < self._final_time - 1e-12:
            deltat = min(self._deltat, self._final_time - self._time)
            state = self.step(state, deltat)
            states_list.append(state)
            times_list.append(self._time)
        states = self._bkd.stack(states_list, axis=1)
        return states, self._bkd.asarray(times_list)

    def time_residual(self) -> AdjointEnabledTimeSteppingResidualProtocol[Array]:
        """Return the time stepping residual."""
        return self._time_residual

    def hvp_time_residual(
        self,
    ) -> Optional[HVPEnabledTimeSteppingResidualProtocol[Array]]:
        """Return the time residual narrowed to HVP capability, or None."""
        if is_hvp_enabled(self._time_residual):
            return self._time_residual
        return None

    # =========================================================================
    # Adjoint Methods
    # =========================================================================

    def adjoint_step(
        self,
        fsol_n: Array,
        asol_np1: Array,
        dqdu_n: Array,
        deltat_n: float,
        deltat_np1: float,
        time_n: float,
    ) -> Array:
        """
        Perform a single backward adjoint step.

        Solves: (dR/dy_n)^T · λ_n = -(dR/dy_{n+1})^T · λ_{n+1} - dQ/dy_n

        Parameters
        ----------
        fsol_n : Array
            Forward solution at time step n. Shape: (nstates,)
        asol_np1 : Array
            Adjoint solution at time step n+1. Shape: (nstates,)
        dqdu_n : Array
            Gradient dQ/dy at time step n. Shape: (nstates,)
        deltat_n : float
            Time step size from n-1 to n.
        deltat_np1 : float
            Time step size from n to n+1.
        time_n : float
            Time at step n.

        Returns
        -------
        Array
            Adjoint solution at time step n. Shape: (nstates,)
        """
        self._time = time_n
        self._time_residual.set_time(time_n, deltat_n, fsol_n)

        # Adjoint Jacobian functions already apply transpose
        drduT_diag = self._time_residual.adjoint_diag_jacobian(fsol_n)
        drduT_offdiag = self._time_residual.adjoint_off_diag_jacobian(
            fsol_n, deltat_np1
        )

        rhs = -drduT_offdiag @ asol_np1 - dqdu_n
        return solve_maybe_sparse(self._bkd, drduT_diag, rhs)

    def solve_adjoint(self, fwd_sols: Array, times: Array, param: Array) -> Array:
        """
        Solve the adjoint equations backward in time.

        Parameters
        ----------
        fwd_sols : Array
            Forward solution trajectory. Shape: (nstates, ntimes)
        times : Array
            Time points. Shape: (ntimes,)
        param : Array
            Parameters. Shape: (nparams, 1)

        Returns
        -------
        Array
            Adjoint solution trajectory. Shape: (nstates, ntimes)
        """
        if not self._bkd.allclose(
            times[-1], self._bkd.asarray(self._final_time), atol=1e-12
        ):
            raise ValueError("times array is inconsistent with final_time")

        if self._functional is None:
            raise RuntimeError("Must call set_functional() first")

        # Compute dQ/dy at all time steps
        dqdu = self._functional.state_jacobian(fwd_sols, param)

        adj_sols = self._bkd.zeros(fwd_sols.shape)

        # Initial condition for adjoint at final time
        # deltat_n = t_N - t_{N-1}
        self._time = float(times[-2])
        deltat_n = float(times[-1] - times[-2])
        self._time_residual.set_time(self._time, deltat_n, fwd_sols[:, -2])

        dqdu_final = self._time_residual.zero_adjoint_rhs(dqdu[:, -1])

        adj_sols = self._bkd.copy(adj_sols)
        adj_sols[:, -1] = self._time_residual.adjoint_initial_condition(
            fwd_sols[:, -1], dqdu_final
        )

        # Backward sweep from N-2 to 1
        for nn in range(fwd_sols.shape[1] - 2, 0, -1):
            deltat_np1 = deltat_n
            deltat_n = float(times[nn] - times[nn - 1])

            dqdu_n = self._time_residual.zero_adjoint_rhs(dqdu[:, nn])

            adj_sols[:, nn] = self.adjoint_step(
                fwd_sols[:, nn],
                adj_sols[:, nn + 1],
                dqdu_n,
                deltat_n,
                deltat_np1,
                float(times[nn]),
            )

        # Final adjoint step at t=0
        deltat_np1 = deltat_n
        deltat_n = float(times[1] - times[0])
        self._time = float(times[0])
        self._time_residual.set_time(self._time, deltat_n, fwd_sols[:, 0])

        dqdu_0 = self._time_residual.zero_adjoint_rhs(dqdu[:, 0])

        adj_sols[:, 0] = self._time_residual.adjoint_final_solution(
            fwd_sols[:, 0],
            adj_sols[:, 1],
            dqdu_0,
            deltat_np1,
        )

        return adj_sols

    def gradient(self, fwd_sols: Array, times: Array, param: Array) -> Array:
        """
        Compute the gradient dQ/dp via the adjoint method.

        Parameters
        ----------
        fwd_sols : Array
            Forward solution trajectory. Shape: (nstates, ntimes)
        times : Array
            Time points. Shape: (ntimes,)
        param : Array
            Parameters. Shape: (nparams, 1)

        Returns
        -------
        Array
            Gradient dQ/dp. Shape: (1, nparams)
        """
        adj_sols = self.solve_adjoint(fwd_sols, times, param)
        return self.gradient_from_adjoint_sols(adj_sols, fwd_sols, times, param)

    def gradient_from_adjoint_sols(
        self,
        adj_sols: Array,
        fwd_sols: Array,
        times: Array,
        param: Array,
    ) -> Array:
        """
        Compute gradient from pre-computed adjoint solutions.

        dQ/dp = dQ/dp|_direct + Σ_n λ_n^T · dR_n/dp

        Parameters
        ----------
        adj_sols : Array
            Adjoint solution trajectory. Shape: (nstates, ntimes)
        fwd_sols : Array
            Forward solution trajectory. Shape: (nstates, ntimes)
        times : Array
            Time points. Shape: (ntimes,)
        param : Array
            Parameters. Shape: (nparams, 1)

        Returns
        -------
        Array
            Gradient dQ/dp. Shape: (1, nparams)
        """
        if self._functional is None:
            raise RuntimeError("Must call set_functional() first")
        # Direct parameter dependence of functional
        dqdp = self._functional.param_jacobian(fwd_sols, param)
        grad = self._bkd.copy(dqdp)

        # Initial condition contribution (if initial condition depends on params)
        self._time_residual.set_time(
            float(times[0]),
            float(times[1] - times[0]),
            fwd_sols[:, 0],
        )

        drdp_init = self._time_residual.initial_param_jacobian()
        # Prepend zeros for functional-only parameters
        n_unique = self._functional.nunique_params()
        if n_unique > 0:
            zeros = self._bkd.zeros((drdp_init.shape[0], n_unique))
            drdp_init = self._bkd.hstack((zeros, drdp_init))
        grad += adj_sols[:, 0:1].T @ drdp_init

        # Accumulate contributions from each time step
        for ii in range(len(times) - 1):
            self._time_residual.set_time(
                float(times[ii]),
                float(times[ii + 1] - times[ii]),
                fwd_sols[:, ii],
            )

            drdp = self._time_residual.param_jacobian(
                fwd_sols[:, ii], fwd_sols[:, ii + 1]
            )
            # Prepend zeros for functional-only parameters
            n_unique = self._functional.nunique_params()
            if n_unique > 0:
                zeros = self._bkd.zeros((drdp.shape[0], n_unique))
                drdp = self._bkd.hstack((zeros, drdp))
            grad += adj_sols[:, ii + 1 : ii + 2].T @ drdp

        return grad

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"init_time={self._init_time}, "
            f"final_time={self._final_time}, "
            f"deltat={self._deltat})"
        )


# Alias for backward compatibility
ImplicitTimeIntegrator = TimeIntegrator
