"""Galerkin model for time-dependent PDE problems.

Provides a high-level interface for solving time-dependent PDEs using
Galerkin finite element methods with various time integration methods.

Analogous to CollocationModel but for weak-form (Galerkin) physics.
"""

from typing import Generic, Tuple

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.pde.galerkin.protocols.physics import (
    GalerkinPhysicsProtocol,
)
from pyapprox.typing.pde.galerkin.time_integration.physics_adapter import (
    GalerkinPhysicsODEAdapter,
)
from pyapprox.typing.pde.galerkin.time_integration.explicit_adapter import (
    GalerkinExplicitODEAdapter,
)
from pyapprox.typing.pde.galerkin.solvers.steady_state import SteadyStateSolver
from pyapprox.typing.pde.time.config import TimeIntegrationConfig
from pyapprox.typing.pde.time.explicit_steppers import (
    ForwardEulerResidual,
    HeunResidual,
)
from pyapprox.typing.pde.time.implicit_steppers import (
    BackwardEulerResidual,
    CrankNicolsonResidual,
)
from pyapprox.typing.optimization.rootfinding.newton import NewtonSolver


_STEPPER_MAP = {
    "forward_euler": ForwardEulerResidual,
    "backward_euler": BackwardEulerResidual,
    "crank_nicolson": CrankNicolsonResidual,
    "heun": HeunResidual,
}


class GalerkinModel(Generic[Array]):
    """High-level model for Galerkin FEM PDE problems.

    Provides a unified interface for solving steady and time-dependent
    PDE problems using Galerkin finite element methods.

    Reuses existing time stepping residuals from pde.time and the
    GalerkinPhysicsODEAdapter for mass matrix handling.

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
    ...     method="backward_euler", final_time=1.0, deltat=0.01
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
        self._adapter = GalerkinPhysicsODEAdapter(physics)

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
        solver = SteadyStateSolver(
            self._physics, tol=tol, max_iter=maxiter
        )
        result = solver.solve(initial_guess, time=time)
        if not result.converged:
            raise RuntimeError(
                f"Newton iteration failed to converge: {result.message}"
            )
        return result.solution

    def solve_transient(
        self,
        initial_condition: Array,
        config: TimeIntegrationConfig,
    ) -> Tuple[Array, Array]:
        """Solve the time-dependent problem.

        Integrates M * du/dt = F(u, t) from init_time to final_time.

        For explicit methods, uses GalerkinExplicitODEAdapter which provides
        BC-clean f(y,t) = M_bc^{-1} * spatial_residual. Dirichlet values are
        injected after each step. For implicit methods, uses the standard
        GalerkinPhysicsODEAdapter with Newton solver.

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

        stepper, is_explicit = self._create_stepper(config)

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

        # Setup Newton solver for implicit methods
        if not is_explicit:
            newton = NewtonSolver(stepper)
            newton.set_options(
                maxiters=config.newton_maxiter,
                atol=config.newton_tol,
                rtol=0.0,
            )

        for ii in range(ntimes - 1):
            t_n = float(times[ii])
            dt = float(times[ii + 1] - times[ii])

            stepper.set_time(t_n, dt, state)

            if is_explicit:
                state = state - stepper(state)
                # Inject Dirichlet values at t_{n+1}
                t_np1 = t_n + dt
                d_dofs, d_vals = self._physics.dirichlet_dof_info(t_np1)
                d_dofs_np = bkd.to_numpy(d_dofs).astype(np.intp)
                if len(d_dofs_np) > 0:
                    state_np = bkd.to_numpy(state).copy()
                    d_vals_np = bkd.to_numpy(d_vals)
                    state_np[d_dofs_np] = d_vals_np
                    state = bkd.asarray(
                        state_np.astype(np.float64)
                    )
            else:
                state = newton.solve(state)

            solutions[:, ii + 1] = state

            if config.verbosity >= 1:
                print(f"Time {float(times[ii + 1]):.4f}")

        return solutions, times

    def _create_stepper(self, config: TimeIntegrationConfig):
        """Create a time stepping residual for the given method.

        For explicit methods, uses GalerkinExplicitODEAdapter (BC-clean).
        For implicit methods, uses GalerkinPhysicsODEAdapter (standard).

        Parameters
        ----------
        config : TimeIntegrationConfig
            Time integration configuration.

        Returns
        -------
        Tuple[TimeSteppingResidualBase, bool]
            stepper : TimeSteppingResidualBase
                The time stepping residual.
            is_explicit : bool
                Whether the method is explicit.
        """
        method = config.method
        if method not in _STEPPER_MAP:
            raise ValueError(
                f"Unknown time integration method: {method}. "
                f"Supported: {list(_STEPPER_MAP.keys())}"
            )

        stepper_cls = _STEPPER_MAP[method]
        is_explicit = method in ("forward_euler", "heun")

        if is_explicit:
            lumped = getattr(config, "lumped_mass", False)
            adapter = GalerkinExplicitODEAdapter(
                self._physics, lumped_mass=lumped
            )
        else:
            adapter = self._adapter

        return stepper_cls(adapter), is_explicit

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"physics={self._physics.__class__.__name__}, "
            f"nstates={self.nstates()})"
        )
