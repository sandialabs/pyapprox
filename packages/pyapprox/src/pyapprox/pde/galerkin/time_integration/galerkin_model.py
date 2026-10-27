"""Galerkin model for steady PDE problems.

Provides a high-level interface for steady solves of a composed Galerkin
system. Transient solves are
:class:`~pyapprox.pde.galerkin.time_integration.transient_model.GalerkinTransientModel`.
"""

from typing import Generic

from pyapprox.pde.galerkin.protocols.system import (
    GalerkinSteadySystemProtocol,
)
from pyapprox.pde.galerkin.solvers.steady_state import SteadyStateSolver
from pyapprox.util.backends.protocols import Array, Backend


class GalerkinSteadyModel(Generic[Array]):
    """Steady solves of a Galerkin system: ``F(u) = 0`` with its essential
    constraints.

    Transient solves are a different object with different inputs:
    :class:`~pyapprox.pde.galerkin.time_integration.transient_model.GalerkinTransientModel`,
    built from a time-integration adapter.

    Parameters
    ----------
    system : GalerkinSteadySystemProtocol
        The composed system with its essential constraints, e.g. from
        ``compose_galerkin_system``. Its backend is the model's.

    Examples
    --------
    >>> model = GalerkinSteadyModel(compose_galerkin_system(physics, bcs))
    >>> solution = model.solve_steady(initial_guess)
    """

    def __init__(self, system: GalerkinSteadySystemProtocol[Array]) -> None:
        if not isinstance(system, GalerkinSteadySystemProtocol):
            raise TypeError(
                "system must satisfy GalerkinSteadySystemProtocol, got "
                f"{type(system).__name__}"
            )
        self._system = system

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._system.bkd()

    def system(self) -> GalerkinSteadySystemProtocol[Array]:
        """Return the composed system this model solves."""
        return self._system

    def nstates(self) -> int:
        """Return number of states."""
        return self._system.nstates()

    def solve_steady(
        self,
        initial_guess: Array,
        tol: float = 1e-10,
        maxiter: int = 50,
        rtol: float = 0.0,
    ) -> Array:
        """Solve the steady-state problem F(u) = 0 with constraints.

        The data must be declared time-invariant. For the steady state of
        time-dependent data frozen at some time, solve
        ``system.steady_snapshot(time)`` with ``SteadyStateSolver``
        directly.

        Parameters
        ----------
        initial_guess : Array
            Initial guess for solution. Shape: (nstates,)
        tol : float
            Absolute tolerance on the residual norm.
        maxiter : int
            Maximum Newton iterations.
        rtol : float
            Tolerance relative to the initial residual norm; Newton stops
            when ``||F|| <= tol + rtol * ||F(u_0)||``. An operator declared
            linear is solved in one step without either test.

        Returns
        -------
        Array
            Steady-state solution. Shape: (nstates,)

        Raises
        ------
        ValueError
            If the data are declared time-dependent.
        RuntimeError
            If Newton iteration fails to converge.
        """
        solver = SteadyStateSolver(
            self._system.steady(), tol=tol, max_iter=maxiter, rtol=rtol
        )
        result = solver.solve(initial_guess)
        if not result.converged:
            raise RuntimeError(f"Newton iteration failed to converge: {result.message}")
        return result.solution

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"system={self._system!r}, "
            f"nstates={self.nstates()})"
        )
