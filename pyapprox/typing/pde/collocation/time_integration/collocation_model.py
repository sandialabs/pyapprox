"""Collocation model for time-dependent PDE problems.

Provides a high-level interface for solving time-dependent PDEs using
spectral collocation with various time integration methods.
"""

from dataclasses import dataclass
from typing import Generic, Optional, Callable, Tuple, Literal

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.pde.collocation.protocols import PhysicsProtocol
from pyapprox.typing.pde.collocation.time_integration.physics_adapter import (
    PhysicsToODEResidualAdapter,
)


@dataclass
class TimeIntegrationConfig:
    """Configuration for time integration.

    Parameters
    ----------
    method : str
        Time integration method. One of:
        - "forward_euler": Explicit first-order
        - "backward_euler": Implicit first-order (A-stable)
        - "crank_nicolson": Implicit second-order
        - "heun": Explicit second-order (RK2)
    init_time : float
        Initial time. Default: 0.0
    final_time : float
        Final time.
    deltat : float
        Time step size.
    newton_tol : float
        Newton solver tolerance for implicit methods. Default: 1e-10
    newton_maxiter : int
        Newton solver maximum iterations. Default: 20
    verbosity : int
        Verbosity level. Default: 0
    """

    method: Literal[
        "forward_euler", "backward_euler", "crank_nicolson", "heun"
    ] = "backward_euler"
    init_time: float = 0.0
    final_time: float = 1.0
    deltat: float = 0.01
    newton_tol: float = 1e-10
    newton_maxiter: int = 20
    verbosity: int = 0


class CollocationModel(Generic[Array]):
    """High-level model for spectral collocation PDE problems.

    Provides a unified interface for solving steady and time-dependent
    PDE problems using spectral collocation methods.

    The model wraps physics, boundary conditions, and time integration
    into a single coherent interface.

    Parameters
    ----------
    physics : PhysicsProtocol
        Physics object defining the PDE.
    bkd : Backend
        Computational backend.

    Examples
    --------
    >>> # Create mesh and basis
    >>> mesh = TransformedMesh1D(20, bkd)
    >>> basis = ChebyshevBasis1D(mesh, bkd)
    >>> physics = AdvectionDiffusionReaction(basis, bkd, diffusion=0.1)
    >>> physics.set_boundary_conditions([bc_left, bc_right])
    >>>
    >>> # Create model
    >>> model = CollocationModel(physics, bkd)
    >>>
    >>> # Solve steady problem
    >>> u_steady = model.solve_steady(initial_guess, tol=1e-10)
    >>>
    >>> # Solve transient problem
    >>> config = TimeIntegrationConfig(
    ...     method="backward_euler", final_time=1.0, deltat=0.01
    ... )
    >>> u_all, times = model.solve_transient(u0, config)
    """

    def __init__(
        self,
        physics: PhysicsProtocol[Array],
        bkd: Backend[Array],
    ):
        self._physics = physics
        self._bkd = bkd
        self._adapter = PhysicsToODEResidualAdapter(physics, bkd)

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def physics(self) -> PhysicsProtocol[Array]:
        """Return the physics object."""
        return self._physics

    def adapter(self) -> PhysicsToODEResidualAdapter[Array]:
        """Return the ODE residual adapter."""
        return self._adapter

    def nstates(self) -> int:
        """Return number of states."""
        return self._physics.nstates()

    def _apply_boundary_conditions(
        self, residual: Array, jacobian: Array, state: Array, time: float
    ) -> Tuple[Array, Array]:
        """Apply boundary conditions to residual and Jacobian.

        This method applies boundary conditions directly to the Newton residual
        for transient problems, enforcing u[boundary] = g(t) at boundaries.

        Parameters
        ----------
        residual : Array
            Newton residual. Shape: (nstates,)
        jacobian : Array
            Physics Jacobian (not the Newton Jacobian). Shape: (nstates, nstates)
        state : Array
            Current state. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Tuple[Array, Array]
            Modified residual and Jacobian with boundary conditions applied.
        """
        if self._physics.boundary_conditions():
            residual, jacobian = self._physics.apply_boundary_conditions(
                residual, jacobian, state, time
            )
        return residual, jacobian

    def solve_steady(
        self,
        initial_guess: Array,
        tol: float = 1e-10,
        maxiter: int = 50,
        verbosity: int = 0,
    ) -> Array:
        """Solve the steady-state problem.

        Finds u such that residual(u, t=0) = 0 with boundary conditions.

        Uses Newton's method with damping for robustness.

        Parameters
        ----------
        initial_guess : Array
            Initial guess for solution. Shape: (nstates,)
        tol : float
            Convergence tolerance on residual norm.
        maxiter : int
            Maximum Newton iterations.
        verbosity : int
            Verbosity level.

        Returns
        -------
        Array
            Steady-state solution. Shape: (nstates,)

        Raises
        ------
        RuntimeError
            If Newton iteration fails to converge.
        """
        bkd = self._bkd

        state = bkd.copy(initial_guess)
        self._adapter.set_time(0.0)

        for iteration in range(maxiter):
            # Compute residual and Jacobian
            residual = self._adapter(state)
            jacobian = self._adapter.jacobian(state)

            # Apply boundary conditions
            residual, jacobian = self._apply_boundary_conditions(
                residual, jacobian, state, 0.0
            )

            # Check convergence
            res_norm = float(bkd.norm(residual))
            if verbosity >= 1:
                print(f"Newton iter {iteration}: ||res|| = {res_norm:.3e}")

            if res_norm < tol:
                if verbosity >= 1:
                    print(f"Converged in {iteration} iterations")
                return state

            # Solve for update
            delta = bkd.solve(jacobian, -residual)

            # Line search with backtracking
            alpha = 1.0
            for _ in range(10):
                state_new = state + alpha * delta
                res_new = self._adapter(state_new)
                if float(bkd.norm(res_new)) < res_norm:
                    break
                alpha *= 0.5
            else:
                if verbosity >= 1:
                    print(f"Warning: Line search failed at iteration {iteration}")

            state = state_new

        raise RuntimeError(
            f"Newton iteration failed to converge after {maxiter} iterations. "
            f"Final residual norm: {res_norm:.3e}"
        )

    def solve_transient(
        self,
        initial_condition: Array,
        config: TimeIntegrationConfig,
    ) -> Tuple[Array, Array]:
        """Solve the time-dependent problem.

        Integrates du/dt = f(u, t) from init_time to final_time.

        Parameters
        ----------
        initial_condition : Array
            Initial state u(t=0). Shape: (nstates,)
        config : TimeIntegrationConfig
            Time integration configuration.

        Returns
        -------
        Tuple[Array, Array]
            solutions : Array
                Solution trajectory. Shape: (nstates, ntimes)
            times : Array
                Time points. Shape: (ntimes,)
        """
        bkd = self._bkd

        # Build time grid
        times = [config.init_time]
        t = config.init_time
        while t < config.final_time - 1e-12:
            dt = min(config.deltat, config.final_time - t)
            t += dt
            times.append(t)
        times = bkd.asarray(times)
        ntimes = len(times)

        # Allocate solution storage
        solutions = bkd.zeros((self.nstates(), ntimes))
        solutions = bkd.copy(solutions)
        solutions[:, 0] = initial_condition

        # Time stepping
        state = bkd.copy(initial_condition)

        for ii in range(ntimes - 1):
            t_n = float(times[ii])
            dt = float(times[ii + 1] - times[ii])

            state = self._time_step(state, t_n, dt, config)
            solutions[:, ii + 1] = state

            if config.verbosity >= 1:
                print(f"Time {float(times[ii + 1]):.4f}")

        return solutions, times

    def _time_step(
        self,
        state: Array,
        time: float,
        deltat: float,
        config: TimeIntegrationConfig,
    ) -> Array:
        """Perform a single time step.

        Parameters
        ----------
        state : Array
            Current state. Shape: (nstates,)
        time : float
            Current time.
        deltat : float
            Time step size.
        config : TimeIntegrationConfig
            Time integration configuration.

        Returns
        -------
        Array
            State at next time. Shape: (nstates,)
        """
        method = config.method

        if method == "forward_euler":
            return self._forward_euler_step(state, time, deltat)
        elif method == "heun":
            return self._heun_step(state, time, deltat)
        elif method == "backward_euler":
            return self._backward_euler_step(state, time, deltat, config)
        elif method == "crank_nicolson":
            return self._crank_nicolson_step(state, time, deltat, config)
        else:
            raise ValueError(f"Unknown time integration method: {method}")

    def _forward_euler_step(
        self, state: Array, time: float, deltat: float
    ) -> Array:
        """Forward Euler: y_{n+1} = y_n + dt * f(y_n, t_n)."""
        self._adapter.set_time(time)
        f_n = self._adapter(state)
        return state + deltat * f_n

    def _heun_step(self, state: Array, time: float, deltat: float) -> Array:
        """Heun's method (RK2): predictor-corrector."""
        bkd = self._bkd

        # Predictor (Forward Euler)
        self._adapter.set_time(time)
        k1 = self._adapter(state)
        y_pred = state + deltat * k1

        # Corrector
        self._adapter.set_time(time + deltat)
        k2 = self._adapter(y_pred)

        return state + 0.5 * deltat * (k1 + k2)

    def _backward_euler_step(
        self,
        state: Array,
        time: float,
        deltat: float,
        config: TimeIntegrationConfig,
    ) -> Array:
        """Backward Euler: y_{n+1} = y_n + dt * f(y_{n+1}, t_{n+1}).

        Requires Newton iteration.

        For transient Dirichlet BCs, boundary conditions are applied to the
        Newton residual (not the physics residual), enforcing y[boundary] = g(t)
        directly rather than integrating du/dt at boundaries.
        """
        bkd = self._bkd
        t_np1 = time + deltat
        self._adapter.set_time(t_np1)

        # Newton iteration
        y = bkd.copy(state)  # Initial guess

        for _ in range(config.newton_maxiter):
            # Residual: R = y - y_n - dt * f(y)
            # Note: adapter returns physics residual WITHOUT BC modifications
            f_y = self._adapter(y)
            residual = y - state - deltat * f_y

            # Apply boundary conditions to Newton residual
            # This enforces y[boundary] = g(t_{n+1}) at boundaries
            jac_f = self._adapter.jacobian(y)
            residual, jac_f = self._apply_boundary_conditions(
                residual, jac_f, y, t_np1
            )

            # Check convergence
            if float(bkd.norm(residual)) < config.newton_tol:
                return y

            # Jacobian: dR/dy = I - dt * df/dy
            jac_R = bkd.eye(self.nstates()) - deltat * jac_f

            # Newton update
            delta = bkd.solve(jac_R, -residual)
            y = y + delta

        return y

    def _crank_nicolson_step(
        self,
        state: Array,
        time: float,
        deltat: float,
        config: TimeIntegrationConfig,
    ) -> Array:
        """Crank-Nicolson: y_{n+1} = y_n + dt/2 * (f(y_n) + f(y_{n+1})).

        Requires Newton iteration.

        For transient Dirichlet BCs, boundary conditions are applied to the
        Newton residual (not the physics residual), enforcing y[boundary] = g(t)
        directly rather than integrating du/dt at boundaries.
        """
        bkd = self._bkd
        t_n = time
        t_np1 = time + deltat

        # f(y_n, t_n)
        self._adapter.set_time(t_n)
        f_n = self._adapter(state)

        # Newton iteration
        self._adapter.set_time(t_np1)
        y = bkd.copy(state)  # Initial guess

        for _ in range(config.newton_maxiter):
            # Residual: R = y - y_n - dt/2 * (f_n + f(y))
            # Note: adapter returns physics residual WITHOUT BC modifications
            f_y = self._adapter(y)
            residual = y - state - 0.5 * deltat * (f_n + f_y)

            # Apply boundary conditions to Newton residual
            # This enforces y[boundary] = g(t_{n+1}) at boundaries
            jac_f = self._adapter.jacobian(y)
            residual, jac_f = self._apply_boundary_conditions(
                residual, jac_f, y, t_np1
            )

            # Check convergence
            if float(bkd.norm(residual)) < config.newton_tol:
                return y

            # Jacobian: dR/dy = I - dt/2 * df/dy
            jac_R = bkd.eye(self.nstates()) - 0.5 * deltat * jac_f

            # Newton update
            delta = bkd.solve(jac_R, -residual)
            y = y + delta

        return y

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"physics={self._physics.__class__.__name__}, "
            f"nstates={self.nstates()})"
        )
