"""Collocation model for time-dependent PDE problems.

Provides a high-level interface for solving time-dependent PDEs using
spectral collocation with various time integration methods.
"""

from typing import Generic, Optional, Callable, Tuple

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.pde.collocation.protocols import PhysicsProtocol
from pyapprox.pde.collocation.time_integration.physics_adapter import (
    PhysicsToODEResidualAdapter,
)
from pyapprox.pde.parameterizations.protocol import (
    ParameterizationProtocol,
)
from pyapprox.pde.collocation.time_integration.bc_time_residual_adapter import (
    BCEnforcingTimeResidual,
)
from pyapprox.pde.time.config import TimeIntegrationConfig
from pyapprox.pde.time.implicit_steppers.backward_euler import (
    BackwardEulerResidual,
)
from pyapprox.pde.time.implicit_steppers.crank_nicolson import (
    CrankNicolsonResidual,
)
from pyapprox.pde.time.explicit_steppers.forward_euler import (
    ForwardEulerResidual,
)
from pyapprox.pde.time.explicit_steppers.heun import HeunResidual
from pyapprox.pde.time.implicit_steppers.integrator import (
    TimeIntegrator,
)
from pyapprox.optimization.rootfinding.newton import NewtonSolver


_STEPPER_REGISTRY = {
    "backward_euler": BackwardEulerResidual,
    "crank_nicolson": CrankNicolsonResidual,
    "forward_euler": ForwardEulerResidual,
    "heun": HeunResidual,
}


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
        parameterization: Optional[ParameterizationProtocol[Array]] = None,
    ):
        if parameterization is not None and not isinstance(
            parameterization, ParameterizationProtocol
        ):
            raise TypeError(
                f"parameterization must satisfy ParameterizationProtocol, "
                f"got {type(parameterization).__name__}"
            )
        self._physics = physics
        self._bkd = bkd
        self._adapter = PhysicsToODEResidualAdapter(
            physics, bkd, parameterization=parameterization
        )
        self._mass_matrix = physics.mass_matrix()
        self._last_integrator = None

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

    def last_integrator(self) -> TimeIntegrator[Array]:
        """Return the integrator from the most recent solve_transient call.

        Returns
        -------
        TimeIntegrator
            The time integrator (with adjoint support).

        Raises
        ------
        RuntimeError
            If solve_transient has not been called.
        """
        if self._last_integrator is None:
            raise RuntimeError(
                "solve_transient() must be called before last_integrator()"
            )
        return self._last_integrator

    def _apply_bc_to_residual(
        self, residual: Array, state: Array, time: float
    ) -> Array:
        """Apply boundary conditions to residual only.

        Used by the line search where the Jacobian is not needed.
        """
        if self._physics.boundary_conditions():
            for bc in self._physics.boundary_conditions():
                residual = bc.apply_to_residual(residual, state, time)
        return residual

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
                res_new = self._apply_bc_to_residual(
                    res_new, state_new, 0.0
                )
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

        Integrates du/dt = f(u, t) from init_time to final_time using the
        TimeIntegrator framework with BC enforcement.

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

        Raises
        ------
        ValueError
            If config.method is not a recognized time integration method.
        """
        if config.method not in _STEPPER_REGISTRY:
            raise ValueError(
                f"Unknown time integration method: '{config.method}'. "
                f"Available: {list(_STEPPER_REGISTRY.keys())}"
            )

        # Build pipeline: adapter → stepper → BC residual → Newton → integrator
        stepper_cls = _STEPPER_REGISTRY[config.method]
        stepper = stepper_cls(self._adapter)
        bc_residual = BCEnforcingTimeResidual(
            stepper, self._physics, self._bkd
        )
        newton = NewtonSolver(bc_residual)
        newton.set_options(
            maxiters=config.newton_maxiter,
            atol=config.newton_tol,
            rtol=1e-8,
            verbosity=max(0, config.verbosity - 1),
        )
        integrator = TimeIntegrator(
            config.init_time,
            config.final_time,
            config.deltat,
            newton,
            verbosity=config.verbosity,
        )

        solutions, times = integrator.solve(initial_condition)
        self._last_integrator = integrator
        return solutions, times

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"physics={self._physics.__class__.__name__}, "
            f"nstates={self.nstates()})"
        )
