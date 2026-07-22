"""Galerkin model for time-dependent PDE problems.

Provides a high-level interface for solving time-dependent PDEs using
Galerkin finite element methods with various time integration methods.

Analogous to CollocationModel but for weak-form (Galerkin) physics.
"""

from typing import Generic, Optional, Tuple

import numpy as np

from pyapprox.ode.config import TimeIntegrationConfig
from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
from pyapprox.ode.step_context import StepContext
from pyapprox.ode.stepper_table import (
    EXPLICIT_METHOD_NAMES,
    create_stepper,
)
from pyapprox.pde.galerkin.protocols.physics import (
    GalerkinPhysicsProtocol,
)
from pyapprox.pde.galerkin.solvers.steady_state import SteadyStateSolver
from pyapprox.pde.galerkin.time_integration.bc_time_residual_adapter import (
    create_galerkin_bc_enforcing_residual,
)
from pyapprox.pde.galerkin.time_integration.explicit_adapter import (
    GalerkinExplicitODEAdapter,
)
from pyapprox.pde.galerkin.time_integration.physics_adapter import (
    GalerkinPhysicsToODEResidualAdapter,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.rootfinding.newton import NewtonSolver


class GalerkinModel(Generic[Array]):
    """High-level model for Galerkin FEM PDE problems.

    Provides a unified interface for solving steady and time-dependent
    PDE problems using Galerkin finite element methods.

    Reuses existing time stepping residuals from pde.time and the
    GalerkinPhysicsToODEResidualAdapter for mass matrix handling.

    Parameters
    ----------
    physics : GalerkinPhysicsProtocol
        Physics object defining the PDE in weak form.
    bkd : Backend
        Computational backend.

    Examples
    --------
    >>> model = GalerkinModel(physics, bkd)
    >>> config = TimeIntegrationConfig(
    ...     method="backward_euler", init_time=0.0, final_time=1.0,
    ...     deltat=0.01, newton_tol=1e-10, newton_maxiter=20,
    ...     lumped_mass=False, verbosity=0,
    ... )
    >>> solutions, times = model.solve_transient(u0, config)
    """

    def __init__(
        self,
        physics: GalerkinPhysicsProtocol[Array],
        bkd: Backend[Array],
    ):
        self._physics = physics
        self._bkd = bkd
        self._adapter = GalerkinPhysicsToODEResidualAdapter(physics)
        self._last_integrator: Optional[TimeIntegrator[Array]] = None

    def last_integrator(self) -> TimeIntegrator[Array]:
        """Return the TimeIntegrator from the most recent implicit solve."""
        if self._last_integrator is None:
            raise RuntimeError(
                "no transient solve has been run yet; call "
                "solve_transient with an implicit method first"
            )
        return self._last_integrator

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def physics(self) -> GalerkinPhysicsProtocol[Array]:
        """Return the physics object."""
        return self._physics

    def nstates(self) -> int:
        """Return number of states."""
        return self._physics.nstates()

    def solve_steady(
        self,
        initial_guess: Array,
        tol: float = 1e-10,
        maxiter: int = 50,
        time: float = 0.0,
    ) -> Array:
        """Solve the steady-state problem.

        Finds u such that residual(u, t) = 0 with boundary conditions.

        Parameters
        ----------
        initial_guess : Array
            Initial guess for solution. Shape: (nstates,)
        tol : float
            Convergence tolerance on residual norm.
        maxiter : int
            Maximum Newton iterations.
        time : float
            Time to evaluate at. Default: 0.0.

        Returns
        -------
        Array
            Steady-state solution. Shape: (nstates,)

        Raises
        ------
        RuntimeError
            If Newton iteration fails to converge.
        """
        solver = SteadyStateSolver(self._physics, tol=tol, max_iter=maxiter)
        result = solver.solve(initial_guess, time=time)
        if not result.converged:
            raise RuntimeError(f"Newton iteration failed to converge: {result.message}")
        return result.solution

    def solve_transient(
        self,
        initial_condition: Array,
        config: TimeIntegrationConfig[Array],
    ) -> Tuple[Array, Array]:
        """Solve the time-dependent problem.

        Integrates M * du/dt = F(u, t) from init_time to final_time.

        Implicit methods run the TimeIntegrator pipeline: raw ODE
        adapter -> stepper -> BC-enforcing residual wrapper (constraint
        rows applied via the physics' DirichletConstraintSet) -> Newton.
        Explicit methods use GalerkinExplicitODEAdapter, which provides
        BC-clean f(y,t) = M_bc^{-1} * spatial_residual with Dirichlet
        values injected after each step.

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
        method = config.method
        if isinstance(method, str) and method in EXPLICIT_METHOD_NAMES:
            return self._solve_transient_explicit(initial_condition, config)

        # Implicit pipeline: adapter -> stepper -> BC residual -> Newton
        # -> integrator (custom StepperFactory handles share this path;
        # unknown string names error inside create_stepper).
        stepper = create_stepper(method, self._adapter)
        bc_residual = create_galerkin_bc_enforcing_residual(
            stepper, self._physics, self._bkd
        )
        newton = NewtonSolver(bc_residual)
        newton.set_options(
            maxiters=config.newton_maxiter,
            atol=config.newton_tol,
            rtol=0.0,
            verbosity=max(0, config.verbosity - 1),
        )
        integrator = TimeIntegrator(
            config.init_time,
            config.final_time,
            config.deltat,
            newton,
            verbosity=config.verbosity,
        )
        init_state = self._physics.constraint_set().inject(
            initial_condition, config.init_time
        )
        solutions, times = integrator.solve(init_state)
        self._last_integrator = integrator
        return solutions, times

    def _solve_transient_explicit(
        self,
        initial_condition: Array,
        config: TimeIntegrationConfig[Array],
    ) -> Tuple[Array, Array]:
        """Explicit stepping with post-step Dirichlet injection.

        Legacy path retained until explicit methods route through the
        BC-enforcing wrapper with a consistent mass solve.
        """
        bkd = self._bkd

        explicit_adapter = GalerkinExplicitODEAdapter(
            self._physics, lumped_mass=config.lumped_mass
        )
        stepper = create_stepper(config.method, explicit_adapter)

        # Build time grid
        times_list = [config.init_time]
        t = config.init_time
        while t < config.final_time - 1e-12:
            dt = min(config.deltat, config.final_time - t)
            t += dt
            times_list.append(t)
        times = bkd.asarray(times_list)
        ntimes = len(times_list)

        # Allocate solution storage
        solutions = bkd.zeros((self.nstates(), ntimes))
        solutions = bkd.copy(solutions)
        solutions[:, 0] = initial_condition

        state = bkd.copy(initial_condition)

        for ii in range(ntimes - 1):
            t_n = float(times[ii])
            dt = float(times[ii + 1] - times[ii])

            ctx = StepContext(t_prev=t_n, deltat=dt, y_prev=state)
            stepper.bind(ctx)
            state = state - stepper(state)
            # Inject Dirichlet values at t_{n+1}
            t_np1 = t_n + dt
            d_dofs, d_vals = self._physics.dirichlet_dof_info(t_np1)
            d_dofs_np = bkd.to_numpy(d_dofs).astype(np.intp)
            if len(d_dofs_np) > 0:
                state_np = bkd.to_numpy(state).copy()
                d_vals_np = bkd.to_numpy(d_vals)
                state_np[d_dofs_np] = d_vals_np
                state = bkd.asarray(state_np.astype(np.float64))

            solutions[:, ii + 1] = state

            if config.verbosity >= 1:
                print(f"Time {float(times[ii + 1]):.4f}")

        return solutions, times

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"physics={self._physics.__class__.__name__}, "
            f"nstates={self.nstates()})"
        )
