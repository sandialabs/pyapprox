"""DtN solver for domain decomposition.

High-level Newton solver for finding interface values that satisfy
flux conservation across all interfaces.
"""

from typing import Generic, Dict, Tuple, Optional, NamedTuple

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.pde.decomposition.solver.dtn_residual import DtNResidual
from pyapprox.pde.decomposition.solver.dtn_jacobian import (
    DtNJacobian,
    create_jacobian,
)


class DtNSolverResult(NamedTuple):
    """Result from DtN solver."""

    interface_dofs: Array
    """Converged interface DOF values. Shape: (total_dofs,)"""

    subdomain_solutions: Dict[int, Array]
    """Subdomain solutions keyed by subdomain ID."""

    converged: bool
    """Whether the solver converged."""

    iterations: int
    """Number of Newton iterations."""

    residual_norm: float
    """Final residual norm."""

    residual_history: list
    """Residual norm at each iteration."""


class DtNSolver(Generic[Array]):
    """Newton solver for DtN domain decomposition.

    Solves for interface DOFs λ such that flux is conserved:
        R(λ) = sum_interfaces (flux_left + flux_right) = 0

    Uses Newton iteration:
        λ^{n+1} = λ^n - J^{-1} R(λ^n)

    where J is the Jacobian of the residual.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    residual : DtNResidual
        Residual computation object.
    jacobian_method : str, optional
        Method for Jacobian computation. Default: "finite_difference"
    max_iters : int, optional
        Maximum Newton iterations. Default: 20
    tol : float, optional
        Convergence tolerance for residual norm. Default: 1e-10
    fd_epsilon : float, optional
        Finite difference epsilon for Jacobian. Default: 1e-7
    verbose : bool, optional
        Print convergence information. Default: False
    """

    def __init__(
        self,
        bkd: Backend[Array],
        residual: DtNResidual[Array],
        jacobian_method: str = "finite_difference",
        max_iters: int = 20,
        tol: float = 1e-10,
        fd_epsilon: float = 1e-7,
        verbose: bool = False,
    ):
        self._bkd = bkd
        self._residual = residual
        self._jacobian = create_jacobian(bkd, residual, jacobian_method, fd_epsilon)
        self._max_iters = max_iters
        self._tol = tol
        self._verbose = verbose

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def solve(
        self, initial_guess: Optional[Array] = None
    ) -> DtNSolverResult:
        """Solve for interface DOFs using Newton iteration.

        Parameters
        ----------
        initial_guess : Array, optional
            Initial guess for interface DOFs. Shape: (total_dofs,)
            If None, uses zeros.

        Returns
        -------
        DtNSolverResult
            Solution results including interface DOFs and subdomain solutions.
        """
        bkd = self._bkd
        n = self._residual.total_dofs()

        # Initialize
        if initial_guess is None:
            lambda_n = bkd.zeros((n,))
        else:
            lambda_n = bkd.copy(initial_guess)

        residual_history = []
        converged = False

        for iteration in range(self._max_iters):
            # Compute residual
            r = self._residual(lambda_n)
            res_norm = float(bkd.norm(r))
            residual_history.append(res_norm)

            if self._verbose:
                print(f"Iteration {iteration}: residual norm = {res_norm:.2e}")

            # Check convergence
            if res_norm < self._tol:
                converged = True
                break

            # Compute Jacobian
            J = self._jacobian(lambda_n)

            # Newton step: solve J * delta = -r
            delta = bkd.solve(J, -r)

            # Update
            lambda_n = lambda_n + delta

        # Final residual
        final_res_norm = float(bkd.norm(self._residual(lambda_n)))

        # Get subdomain solutions
        subdomain_solutions = {}
        for sid, solver in self._residual._subdomain_solvers.items():
            subdomain_solutions[sid] = solver.solution()

        return DtNSolverResult(
            interface_dofs=lambda_n,
            subdomain_solutions=subdomain_solutions,
            converged=converged,
            iterations=iteration + 1,
            residual_norm=final_res_norm,
            residual_history=residual_history,
        )


def create_dtn_solver(
    bkd: Backend[Array],
    residual: DtNResidual[Array],
    **kwargs,
) -> DtNSolver[Array]:
    """Create a DtN solver.

    Convenience function for creating a DtN solver with common options.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    residual : DtNResidual
        Residual computation object.
    **kwargs
        Additional arguments passed to DtNSolver.

    Returns
    -------
    DtNSolver
        Configured solver.
    """
    return DtNSolver(bkd, residual, **kwargs)
