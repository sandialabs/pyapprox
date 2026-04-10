"""Steady-state solver for Galerkin physics.

Solves the steady-state equation F(u) = 0 where F is the Galerkin residual
(spatial discretization without time derivative).

For linear problems: K*u = b (solved directly)
For nonlinear problems: Newton iteration with line search
"""

from dataclasses import dataclass
from typing import Generic

import numpy as np

from pyapprox.pde.galerkin.protocols.physics import GalerkinPhysicsProtocol
from pyapprox.util.backends.protocols import Array
from pyapprox.util.linalg.sparse_dispatch import solve_maybe_sparse


@dataclass
class SolverResult(Generic[Array]):
    """Result from steady-state solver."""

    solution: Array
    """The computed solution."""

    converged: bool
    """Whether the solver converged."""

    iterations: int
    """Number of iterations taken."""

    residual_norm: float
    """Final residual norm."""

    message: str
    """Convergence message."""


class SteadyStateSolver(Generic[Array]):
    """Steady-state solver for Galerkin physics.

    Solves F(u) = 0 where F = residual from Galerkin physics.
    For linear problems (constant Jacobian), this reduces to solving K*u = b.
    For nonlinear problems, Newton iteration with optional line search.

    Parameters
    ----------
    physics : GalerkinPhysicsProtocol
        The Galerkin physics to solve.
    tol : float, optional
        Convergence tolerance on residual norm. Default: 1e-10.
    max_iter : int, optional
        Maximum Newton iterations. Default: 50.
    line_search : bool, optional
        Whether to use line search. Default: True.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.pde.galerkin import (
    ...     StructuredMesh1D, LagrangeBasis, LinearAdvectionDiffusionReaction
    ... )
    >>> from pyapprox.pde.galerkin.solvers import SteadyStateSolver
    >>> bkd = NumpyBkd()
    >>> mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
    >>> basis = LagrangeBasis(mesh, degree=1)
    >>> physics = LinearAdvectionDiffusionReaction(
    ...     basis=basis, diffusivity=0.01, bkd=bkd,
    ...     forcing=lambda x: np.ones(x.shape[1])
    ... )
    >>> solver = SteadyStateSolver(physics)
    >>> u_guess = bkd.zeros(physics.nstates())
    >>> result = solver.solve(u_guess)
    """

    def __init__(
        self,
        physics: GalerkinPhysicsProtocol[Array],
        tol: float = 1e-10,
        max_iter: int = 50,
        line_search: bool = True,
    ):
        self._physics = physics
        self._bkd = physics.bkd()
        self._tol = tol
        self._max_iter = max_iter
        self._line_search = line_search

    def solve(
        self,
        initial_guess: Array,
        time: float = 0.0,
    ) -> SolverResult[Array]:
        """Solve for steady state.

        Parameters
        ----------
        initial_guess : Array
            Initial guess for the solution. Shape: (nstates,)
        time : float, optional
            Time at which to evaluate (for time-dependent coefficients).

        Returns
        -------
        SolverResult
            Solution result with convergence information.
        """
        u = self._bkd.copy(initial_guess)

        for iteration in range(self._max_iter):
            # Compute residual and check convergence
            residual = self._physics.residual(u, time)
            residual_np = self._bkd.to_numpy(residual)
            residual_norm = float(np.linalg.norm(residual_np))

            if residual_norm < self._tol:
                return SolverResult(
                    solution=u,
                    converged=True,
                    iterations=iteration,
                    residual_norm=residual_norm,
                    message=f"Converged in {iteration} iterations",
                )

            # Compute Jacobian and Newton direction
            jacobian = self._physics.jacobian(u, time)
            delta_u = solve_maybe_sparse(self._bkd, jacobian, -residual)

            # Line search for robustness
            if self._line_search:
                alpha = self._line_search_backtrack(u, delta_u, residual_norm, time)
            else:
                alpha = 1.0

            # Update solution
            u = u + alpha * delta_u

        # Did not converge
        residual = self._physics.residual(u, time)
        residual_np = self._bkd.to_numpy(residual)
        final_residual_norm = float(np.linalg.norm(residual_np))

        return SolverResult(
            solution=u,
            converged=False,
            iterations=self._max_iter,
            residual_norm=final_residual_norm,
            message=f"Did not converge in {self._max_iter} iterations",
        )

    def _line_search_backtrack(
        self,
        u: Array,
        delta_u: Array,
        residual_norm: float,
        time: float,
        alpha_init: float = 1.0,
        rho: float = 0.5,
        c: float = 1e-4,
        max_backtracks: int = 10,
    ) -> float:
        """Backtracking line search.

        Finds alpha such that ||F(u + alpha*du)|| < ||F(u)||

        Parameters
        ----------
        u : Array
            Current solution.
        delta_u : Array
            Newton direction.
        residual_norm : float
            Current residual norm.
        time : float
            Current time.
        alpha_init : float
            Initial step size.
        rho : float
            Backtracking reduction factor.
        c : float
            Sufficient decrease parameter.
        max_backtracks : int
            Maximum number of backtracking steps.

        Returns
        -------
        float
            Accepted step size.
        """
        alpha = alpha_init

        for _ in range(max_backtracks):
            u_trial = u + alpha * delta_u
            residual_trial = self._physics.residual(u_trial, time)
            residual_trial_np = self._bkd.to_numpy(residual_trial)
            new_norm = float(np.linalg.norm(residual_trial_np))

            # Armijo condition (sufficient decrease)
            if new_norm < (1 - c * alpha) * residual_norm:
                return alpha

            alpha *= rho

        # Return smallest tried alpha
        return alpha

    def solve_linear(
        self,
        time: float = 0.0,
    ) -> SolverResult[Array]:
        """Solve a linear steady-state problem directly.

        For linear problems where F(u) = b - K*u, solve K*u = b.
        This is more efficient than Newton iteration for linear problems.

        Parameters
        ----------
        time : float, optional
            Time at which to evaluate (for time-dependent coefficients).

        Returns
        -------
        SolverResult
            Solution result.
        """
        # For linear problems: F(u) = b - K*u
        # At u=0: F(0) = b
        # Jacobian J = -K
        # So K = -J and b = F(0)

        nstates = self._physics.nstates()
        # Use double precision for consistency with skfem assembly
        u_zero = self._bkd.asarray(np.zeros(nstates, dtype=np.float64))

        b = self._physics.residual(u_zero, time)
        jacobian = self._physics.jacobian(u_zero, time)
        K = -jacobian  # K = -dF/du

        # Solve K*u = b
        u = solve_maybe_sparse(self._bkd, K, b)

        # Verify
        residual = self._physics.residual(u, time)
        residual_np = self._bkd.to_numpy(residual)
        residual_norm = float(np.linalg.norm(residual_np))

        converged = residual_norm < self._tol

        return SolverResult(
            solution=u,
            converged=converged,
            iterations=1,
            residual_norm=residual_norm,
            message="Linear solve"
            if converged
            else "Linear solve (residual above tolerance)",
        )

    def __repr__(self) -> str:
        return (
            f"SteadyStateSolver(\n"
            f"  physics={self._physics!r},\n"
            f"  tol={self._tol},\n"
            f"  max_iter={self._max_iter},\n"
            f"  line_search={self._line_search},\n"
            f")"
        )
