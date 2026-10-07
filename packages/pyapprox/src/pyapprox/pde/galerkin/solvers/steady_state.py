"""Steady-state solver.

Solves F(u) = 0 for a steady operator: the constrained residual with no
time in it (``SteadyOperatorProtocol``; build one with
``compose_galerkin_system(physics, bcs).steady()`` or ``SteadyView``).

The iteration is the shared ``NewtonSolver``: one exact step for an
operator declared linear, Newton with an optional line search otherwise.
"""

from dataclasses import replace
from typing import Generic, Optional

from pyapprox.pde.steady_view import SteadyOperatorProtocol
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.sparse_dispatch import solve_maybe_sparse
from pyapprox.util.rootfinding.newton import NewtonResult, NewtonSolver


class _SteadyNewtonResidual(Generic[Array]):
    """A steady operator as a ``NewtonSolver`` residual.

    The Newton step solves ``J delta = F`` with the constrained Jacobian;
    the solve dispatches on sparse or dense storage.
    """

    def __init__(self, operator: SteadyOperatorProtocol[Array]) -> None:
        self._operator = operator

    def bkd(self) -> Backend[Array]:
        return self._operator.bkd()

    def __call__(self, iterate: Array, /) -> Array:
        return self._operator.steady_residual(iterate)

    def linsolve(self, state: Array, prev_residual: Array) -> Array:
        return solve_maybe_sparse(
            self._operator.bkd(),
            self._operator.steady_jacobian(state),
            prev_residual,
        )

    def is_one_step_solvable(self) -> bool:
        return self._operator.is_linear()


class SteadyStateSolver(Generic[Array]):
    """Steady-state solver for a time-free steady operator.

    Solves F(u) = 0 where F is the constrained steady residual, with the
    shared ``NewtonSolver``. Newton stops when
    ``||F(u_k)|| <= tol + rtol * ||F(u_0)||``; an operator declared
    linear (``operator.is_linear()``) is solved by one exact step with no
    tolerance test, since the residual left is round-off, which grows
    with the conditioning.

    Parameters
    ----------
    operator : SteadyOperatorProtocol
        The steady operator to solve
        (``compose_galerkin_system(physics, bcs).steady()``).
    tol : float, optional
        Absolute tolerance on the residual norm. Default: 1e-10.
    max_iter : int, optional
        Maximum Newton iterations. Default: 50.
    line_search : bool, optional
        Whether to use a backtracking line search. Default: True.
    rtol : float, optional
        Tolerance relative to the initial residual norm. Default: 0.0.

    Examples
    --------
    >>> import numpy as np
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.pde.constitutive.coefficient_functions import (
    ...     TimeIndependent,
    ... )
    >>> from pyapprox.pde.galerkin import (
    ...     StructuredMesh1D, LagrangeBasis, LinearAdvectionDiffusionReaction,
    ...     compose_galerkin_system,
    ... )
    >>> from pyapprox.pde.galerkin.boundary import DirichletBC
    >>> from pyapprox.pde.galerkin.solvers import SteadyStateSolver
    >>> bkd = NumpyBkd()
    >>> mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
    >>> basis = LagrangeBasis(mesh, degree=1)
    >>> physics = LinearAdvectionDiffusionReaction(
    ...     basis=basis, diffusivity=0.01, bkd=bkd,
    ...     forcing=TimeIndependent(lambda x: np.ones(x.shape[1])),
    ... )
    >>> bcs = [
    ...     DirichletBC(basis, "left", 0.0, bkd),
    ...     DirichletBC(basis, "right", 0.0, bkd),
    ... ]
    >>> solver = SteadyStateSolver(compose_galerkin_system(physics, bcs).steady())
    >>> u_guess = bkd.zeros(physics.nstates())
    >>> result = solver.solve(u_guess)
    """

    def __init__(
        self,
        operator: SteadyOperatorProtocol[Array],
        tol: float = 1e-10,
        max_iter: int = 50,
        line_search: bool = True,
        rtol: float = 0.0,
    ):
        if not isinstance(operator, SteadyOperatorProtocol):
            raise TypeError(
                "operator must satisfy SteadyOperatorProtocol (build one "
                "with compose_galerkin_system(physics, bcs).steady()), got "
                f"{type(operator).__name__}"
            )
        self._operator = operator
        self._bkd = operator.bkd()
        self._tol = tol
        self._rtol = rtol
        self._max_iter = max_iter
        self._line_search = line_search
        self._newton = NewtonSolver(_SteadyNewtonResidual(operator))
        self._newton.set_options(
            maxiters=max_iter,
            atol=tol,
            rtol=rtol,
            line_search=line_search,
            linesearch_maxiters=10,
        )

    def solve(
        self, initial_guess: Optional[Array] = None
    ) -> NewtonResult[Array]:
        """Solve for steady state.

        Parameters
        ----------
        initial_guess : Array, optional
            Initial guess for the solution. Shape: (nstates,). Default:
            zeros (exact after one step for an operator declared linear).

        Returns
        -------
        NewtonResult
            The solution with convergence information; the residual norm
            is always evaluated, including after the one-step path.
        """
        guess = (
            self._bkd.zeros((self._operator.nstates(),))
            if initial_guess is None
            else self._bkd.copy(initial_guess)
        )
        result = self._newton.solve_with_info(guess)
        if result.residual_norm is None:
            result = replace(
                result, residual_norm=self._residual_norm(result.solution)
            )
        return result

    def _residual_norm(self, state: Array) -> Array:
        return self._bkd.norm(self._operator.steady_residual(state))

    def __repr__(self) -> str:
        return (
            f"SteadyStateSolver(\n"
            f"  operator={self._operator!r},\n"
            f"  tol={self._tol},\n"
            f"  rtol={self._rtol},\n"
            f"  max_iter={self._max_iter},\n"
            f"  line_search={self._line_search},\n"
            f")"
        )
