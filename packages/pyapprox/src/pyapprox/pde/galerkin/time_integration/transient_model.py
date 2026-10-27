"""Transient solves of a Galerkin system through its time-integration adapter.

:class:`GalerkinTransientModel` is built from the adapter alone: the
adapter carries the system, the stack, the assemblers and the solver,
so nothing is passed twice. Steady solves are a different object,
:class:`~pyapprox.pde.galerkin.time_integration.galerkin_model.GalerkinSteadyModel`.
"""

from typing import Generic, Optional, Tuple

from pyapprox.ode.config import TimeIntegrationConfig
from pyapprox.ode.functionals.protocols import (
    TransientFunctionalProtocol,
)
from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
from pyapprox.ode.operator.time_adjoint_hvp import (
    TimeAdjointOperatorWithHVP,
)
from pyapprox.ode.stepper_table import create_stepper
from pyapprox.pde.galerkin.protocols.system import (
    GalerkinTransientSystemProtocol,
)
from pyapprox.pde.galerkin.time_integration.bc_time_residual_adapter import (
    create_galerkin_bc_enforcing_residual,
)
from pyapprox.pde.galerkin.time_integration.physics_adapter import (
    GalerkinODEResidualProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.rootfinding.newton import NewtonSolver


class GalerkinTransientModel(Generic[Array]):
    """Transient solves, gradients and HVPs of a Galerkin system.

    Parameters
    ----------
    adapter : GalerkinODEResidualProtocol
        The system's time-integration adapter, e.g.
        ``skfem_ode_residual(system)``; the models layer passes a
        parameterized one so ``gradient``/``hvp_operator`` gain the
        dR/dp surface.

    Raises
    ------
    TypeError
        If ``adapter`` does not satisfy its protocol.

    Examples
    --------
    >>> model = GalerkinTransientModel(skfem_ode_residual(system))
    >>> config = TimeIntegrationConfig(
    ...     method="backward_euler", init_time=0.0, final_time=1.0,
    ...     deltat=0.01, newton_tol=1e-10, newton_maxiter=20,
    ...     lumped_mass=False, verbosity=0,
    ... )
    >>> solutions, times = model.solve_transient(u0, config)
    """

    def __init__(self, adapter: GalerkinODEResidualProtocol[Array]) -> None:
        if not isinstance(adapter, GalerkinODEResidualProtocol):
            raise TypeError(
                "adapter must satisfy GalerkinODEResidualProtocol, "
                f"got {type(adapter).__name__}"
            )
        self._adapter = adapter
        self._last_integrator: Optional[TimeIntegrator[Array]] = None

    def adapter(self) -> GalerkinODEResidualProtocol[Array]:
        """Return the time-integration adapter."""
        return self._adapter

    def system(self) -> GalerkinTransientSystemProtocol[Array]:
        """Return the adapted system."""
        return self._adapter.system()

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._adapter.bkd()

    def nstates(self) -> int:
        """Return number of states."""
        return self.system().nstates()

    def last_integrator(self) -> TimeIntegrator[Array]:
        """Return the TimeIntegrator from the most recent transient solve."""
        if self._last_integrator is None:
            raise RuntimeError(
                "no transient solve has been run yet; call solve_transient first"
            )
        return self._last_integrator

    def set_functional(self, functional: TransientFunctionalProtocol[Array]) -> None:
        """Set the QoI functional on the most recent solve's integrator.

        Forwarded to :meth:`TimeIntegrator.set_functional`; required
        before :meth:`gradient`.
        """
        self.last_integrator().set_functional(functional)

    def gradient(self, fwd_sols: Array, times: Array, param: Array) -> Array:
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
        functional: TransientFunctionalProtocol[Array],
    ) -> TimeAdjointOperatorWithHVP[Array]:
        """Build the second-order adjoint operator on the last solve.

        The returned operator exposes ``jacobian`` and ``hvp``. It raises
        an actionable TypeError when the functional's ``derivatives()``
        has no HVPs or the stepper was created without HVP support.
        """
        return TimeAdjointOperatorWithHVP(self.last_integrator(), functional)

    def solve_transient(
        self,
        initial_condition: Array,
        config: TimeIntegrationConfig[Array],
    ) -> Tuple[Array, Array]:
        """Solve the time-dependent problem.

        Integrates M * du/dt = F(u, t) from init_time to final_time.

        All methods run one TimeIntegrator pipeline: raw ODE adapter ->
        stepper -> BC-enforcing residual wrapper (constraint rows
        applied via the system's constraint set) -> Newton.
        Explicit steppers are one-step solvable (the constraint rows
        are linear), so Newton reduces to a single linear solve with a
        cached factorization of the constant BC-modified mass. The
        adapter's mass decides lumping; ``config.lumped_mass`` must agree
        with it.

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
            If ``config.lumped_mass`` disagrees with the adapter's mass.
        """
        # One pipeline for all methods: adapter -> stepper -> BC
        # residual -> Newton -> integrator (custom StepperFactory
        # handles share this path; unknown string names error inside
        # create_stepper).
        adapter = self._adapter
        if config.lumped_mass != adapter.lumped_mass():
            raise ValueError(
                f"config.lumped_mass={config.lumped_mass} disagrees with the "
                f"adapter's mass (lumped: {adapter.lumped_mass()}); lumping is "
                "chosen when the adapter is built"
            )
        stepper = create_stepper(config.method, adapter)
        bc_residual = create_galerkin_bc_enforcing_residual(
            stepper, adapter.system().constraint_set(), self.bkd()
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
        init_state = (
            adapter.system()
            .constraint_set()
            .inject(initial_condition, config.init_time)
        )
        solutions, times = integrator.solve(init_state)
        self._last_integrator = integrator
        return solutions, times

    def __repr__(self) -> str:
        return f"{type(self).__name__}(adapter={self._adapter!r})"
