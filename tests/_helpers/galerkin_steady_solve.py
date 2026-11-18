"""Solve a parameter-free Galerkin system for its steady state, on the
steady system path.

For tests that need only the solution of a composed system:
:func:`steady_solve` solves the steady residual with
``solve_galerkin_steady`` and ``Newton`` (sparse LU each iteration), and
returns a
:class:`SteadySolveResult`: the solution, whether it converged, and the
residual norm there. Its defaults are an absolute tolerance of 1e-10, no
relative one, 50 iterations and a line search of at most 10 halvings per
step.

Requires skfem; import only after skipping when it is absent.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.galerkin.time_integration.skfem_discretization import (
    skfem_discretization,
)
from pyapprox.pde.models.galerkin.steady_residual import solve_galerkin_steady
from pyapprox.util.linalg.scipy_stack import ScipySparseLU
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.adjoint_checks import NumpyArray


@dataclass(frozen=True)
class SteadySolveResult:
    """The result of :func:`steady_solve`.

    Attributes
    ----------
    solution : NumpyArray
        The last iterate, the solution when ``converged``. Shape:
        ``(nstates,)``.
    converged : bool
        Whether the residual norm reached the tolerance.
    residual_norm : float
        The steady residual's 2-norm at ``solution``.
    """

    solution: NumpyArray
    converged: bool
    residual_norm: float


def steady_solve(
    system: GalerkinSystem[NumpyArray],
    tol: float = 1e-10,
    max_iter: int = 50,
    line_search: bool = True,
    rtol: float = 0.0,
    initial_guess: Optional[NumpyArray] = None,
    time: Optional[float] = None,
) -> SteadySolveResult:
    """Solve ``system``'s steady residual at ``time`` by Newton.

    Parameters
    ----------
    system : GalerkinSystem
        A composed system without parameters.
    tol, rtol : float
        Converged when the residual norm is at most
        ``tol + rtol * ||R(u_0)||``.
    max_iter : int
        Newton steps allowed.
    line_search : bool
        Backtrack each step (at most 10 halvings) until the residual norm
        decreases sufficiently.
    initial_guess : NumpyArray, optional
        Shape ``(nstates,)``; zeros when omitted.
    time : float, optional
        The time the data is frozen at; omitted, the data must be
        declared time-invariant.
    """
    guess = (
        None
        if initial_guess is None
        else np.array(initial_guess, dtype=float).reshape(-1, 1)
    )
    newton: Newton[NumpyArray, NumpyArray] = Newton(
        atol=tol,
        rtol=rtol,
        maxiters=max_iter,
        line_search=line_search,
        linesearch_maxiters=10,
    )
    residual, point = solve_galerkin_steady(
        skfem_discretization(system),
        RecomputeAlways(ScipySparseLU()),
        newton,
        time=time,
        initial_guess=guess,
    )
    state = point.state()
    return SteadySolveResult(
        solution=np.array(state[:, 0]),
        converged=point.converged(),
        residual_norm=float(np.linalg.norm(residual(state))),
    )
