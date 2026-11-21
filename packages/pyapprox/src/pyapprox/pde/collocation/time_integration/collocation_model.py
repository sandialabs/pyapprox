"""Collocation model for time-dependent PDE problems.

Provides a high-level interface for solving time-dependent PDEs using
spectral collocation with various time integration methods.

Transient solves route every step through the BC-enforcing
time-residual wrapper (``create_bc_enforcing_residual``), which also
owns the adjoint/sensitivity/HVP row-and-column corrections derived
from ``bc_dof_classification``; the physics holds the boundary
conditions. Steady problems are solved with ``collocation_steady_residual``
and ``SteadySystem`` (``pyapprox.pde.models.collocation``).
"""

from typing import Generic, Optional, Tuple

from pyapprox.ode.config import TimeIntegrationConfig
from pyapprox.ode.implicit_steppers.integrator import (
    TimeIntegrator,
)
from pyapprox.ode.protocols.time_stepping import (
    SensitivityStepperProtocol,
)
from pyapprox.ode.stepper_table import create_stepper
from pyapprox.pde.collocation.protocols import PhysicsProtocol
from pyapprox.pde.collocation.time_integration.bc_time_residual_adapter import (
    create_bc_enforcing_residual,
)
from pyapprox.pde.collocation.time_integration.physics_adapter import (
    CollocationPhysicsToODEResidualAdapter,
)
from pyapprox.pde.ownership import owns
from pyapprox.pde.time_order import refuse_second_order_in_time
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.rootfinding.newton import NewtonSolver


class CollocationModel(Generic[Array]):
    """High-level model for spectral collocation PDE problems.

    Solves time-dependent PDE problems with spectral collocation.

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
    >>> # Solve transient problem
    >>> config = TimeIntegrationConfig(
    ...     method="backward_euler", init_time=0.0, final_time=1.0,
    ...     deltat=0.01, newton_tol=1e-10, newton_maxiter=20, verbosity=0,
    ... )
    >>> u_all, times = model.solve_transient(u0, config)
    """

    def __init__(
        self,
        physics: PhysicsProtocol[Array],
        bkd: Backend[Array],
        adapter: Optional[CollocationPhysicsToODEResidualAdapter[Array]] = None,
    ):
        if adapter is not None:
            if not isinstance(adapter, CollocationPhysicsToODEResidualAdapter):
                raise TypeError(
                    f"adapter must be a "
                    f"CollocationPhysicsToODEResidualAdapter, "
                    f"got {type(adapter).__name__}"
                )
            if not owns(physics, adapter.physics()):
                raise ValueError(
                    "adapter wraps a physics that is not part of the one "
                    "passed to CollocationModel"
                )
        self._physics = physics
        self._bkd = bkd
        if adapter is None:
            adapter = CollocationPhysicsToODEResidualAdapter(physics, bkd)
        self._adapter = adapter
        self._mass_matrix = physics.mass_matrix()
        self._last_integrator: Optional[TimeIntegrator[Array]] = None

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def physics(self) -> PhysicsProtocol[Array]:
        """Return the physics object."""
        return self._physics

    def adapter(self) -> CollocationPhysicsToODEResidualAdapter[Array]:
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

    def solve_transient(
        self,
        initial_condition: Array,
        config: TimeIntegrationConfig[Array],
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
            If config.method is not a recognized time integration
            method.
        TypeError
            If the physics is second order in time: integrating it as
            ``du/dt = f(u, t)`` is a relaxation, not dynamics (see
            ``pyapprox.pde.time_order``).
        """
        refuse_second_order_in_time(self._physics)
        # Build pipeline: adapter → stepper → BC residual → Newton → integrator
        stepper = create_stepper(config.method, self._adapter)
        if not isinstance(stepper, SensitivityStepperProtocol):
            raise TypeError(
                "the constructed stepper must satisfy at least "
                "SensitivityStepperProtocol for BC enforcement, got "
                f"{type(stepper).__name__}"
            )
        bc_residual = create_bc_enforcing_residual(stepper, self._physics, self._bkd)
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
