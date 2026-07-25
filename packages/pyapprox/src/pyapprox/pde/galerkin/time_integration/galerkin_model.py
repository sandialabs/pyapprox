"""Galerkin model for time-dependent PDE problems.

Provides a high-level interface for solving time-dependent PDEs using
Galerkin finite element methods with various time integration methods.

Analogous to CollocationModel but for weak-form (Galerkin) physics.
"""

from typing import Generic, Optional, Tuple

from pyapprox.ode.config import TimeIntegrationConfig
from pyapprox.ode.functionals.protocols import (
    TransientFunctionalWithJacobianAndHVPProtocol,
    TransientFunctionalWithJacobianProtocol,
)
from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
from pyapprox.ode.operator.time_adjoint_hvp import (
    TimeAdjointOperatorWithHVP,
)
from pyapprox.ode.stepper_table import create_stepper
from pyapprox.pde.galerkin.protocols.physics import (
    GalerkinPhysicsProtocol,
)
from pyapprox.pde.galerkin.solvers.steady_state import SteadyStateSolver
from pyapprox.pde.galerkin.time_integration.bc_time_residual_adapter import (
    create_galerkin_bc_enforcing_residual,
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
    adapter : GalerkinPhysicsToODEResidualAdapter, optional
        ODE-residual adapter to drive the time integration. Omitted for
        pure forward solves (the model builds the base adapter); the
        models layer injects a parameterized tier here so
        ``gradient``/``hvp_operator`` gain the dR/dp surface.

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
        adapter: Optional[GalerkinPhysicsToODEResidualAdapter[Array]] = None,
    ):
        if adapter is not None:
            if not isinstance(adapter, GalerkinPhysicsToODEResidualAdapter):
                raise TypeError(
                    "adapter must be a GalerkinPhysicsToODEResidualAdapter, "
                    f"got {type(adapter).__name__}"
                )
            if adapter.physics() is not physics:
                raise ValueError(
                    "adapter wraps a different physics instance than the "
                    "one passed to GalerkinModel"
                )
        self._physics = physics
        self._bkd = bkd
        self._adapter_injected = adapter is not None
        if adapter is None:
            adapter = GalerkinPhysicsToODEResidualAdapter(physics)
        self._adapter = adapter
        self._last_integrator: Optional[TimeIntegrator[Array]] = None

    def adapter(self) -> GalerkinPhysicsToODEResidualAdapter[Array]:
        """Return the ODE residual adapter."""
        return self._adapter

    def last_integrator(self) -> TimeIntegrator[Array]:
        """Return the TimeIntegrator from the most recent transient solve."""
        if self._last_integrator is None:
            raise RuntimeError(
                "no transient solve has been run yet; call "
                "solve_transient first"
            )
        return self._last_integrator

    def set_functional(
        self, functional: TransientFunctionalWithJacobianProtocol[Array]
    ) -> None:
        """Set the QoI functional on the most recent solve's integrator.

        Forwarded to :meth:`TimeIntegrator.set_functional`; required
        before :meth:`gradient`.
        """
        self.last_integrator().set_functional(functional)

    def gradient(
        self, fwd_sols: Array, times: Array, param: Array
    ) -> Array:
        """Compute dQ/dp by the adjoint method on the last solve.

        Delegates to :meth:`TimeIntegrator.gradient`; the integrator
        raises an actionable TypeError when the stepper was created
        without adjoint support.

        Parameters
        ----------
        fwd_sols : Array
            Forward trajectory from solve_transient.
            Shape: (nstates, ntimes)
        times : Array
            Time points. Shape: (ntimes,)
        param : Array
            Parameters. Shape: (nparams, 1)

        Returns
        -------
        Array
            Gradient dQ/dp. Shape: (1, nparams)
        """
        return self.last_integrator().gradient(fwd_sols, times, param)

    def hvp_operator(
        self,
        functional: TransientFunctionalWithJacobianAndHVPProtocol[Array],
    ) -> TimeAdjointOperatorWithHVP[Array]:
        """Build the second-order adjoint operator on the last solve.

        The returned operator exposes ``jacobian`` and ``hvp``; the
        underlying wrapper raises an actionable TypeError when the
        stepper was created without HVP support.
        """
        return TimeAdjointOperatorWithHVP(
            self.last_integrator(), functional
        )

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

        All methods run one TimeIntegrator pipeline: raw ODE adapter ->
        stepper -> BC-enforcing residual wrapper (constraint rows
        applied via the physics' DirichletConstraintSet) -> Newton.
        Explicit steppers are one-step solvable (the constraint rows
        are linear), so Newton reduces to a single linear solve with a
        cached factorization of the constant BC-modified mass;
        ``config.lumped_mass`` swaps in the row-sum lumped mass.

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
        # One pipeline for all methods: adapter -> stepper -> BC
        # residual -> Newton -> integrator (custom StepperFactory
        # handles share this path; unknown string names error inside
        # create_stepper).
        if config.lumped_mass:
            if self._adapter_injected:
                raise ValueError(
                    "config.lumped_mass=True would discard the injected "
                    "adapter; construct the injected adapter with "
                    "lumped_mass=True instead"
                )
            adapter = GalerkinPhysicsToODEResidualAdapter(
                self._physics, lumped_mass=True
            )
        else:
            adapter = self._adapter
        stepper = create_stepper(config.method, adapter)
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

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"physics={self._physics.__class__.__name__}, "
            f"nstates={self.nstates()})"
        )
