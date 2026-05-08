"""Preconditioned Conjugate Gradient solver.

Implements PCG algorithm for symmetric positive definite systems with
optional preconditioning.
"""

from typing import Callable, Generic, Optional, Tuple

from pyapprox.optimization.linear_solvers.protocols import (
    PreconditionerProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class PreconditionedConjugateGradient(Generic[Array]):
    """Preconditioned Conjugate Gradient iterative solver.

    Solves Ax = b for symmetric positive definite A using the PCG algorithm
    with optional preconditioning.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    preconditioner : Optional[PreconditionerProtocol]
        Preconditioner that approximates A^{-1}. If None, uses identity
        (equivalent to standard CG).
    """

    def __init__(
        self,
        bkd: Backend[Array],
        preconditioner: Optional[PreconditionerProtocol[Array]] = None,
    ):
        self._bkd = bkd
        self._preconditioner = preconditioner

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def preconditioner(self) -> Optional[PreconditionerProtocol[Array]]:
        """Return the preconditioner (or None)."""
        return self._preconditioner

    def _apply_preconditioner(self, r: Array) -> Array:
        """Apply preconditioner to residual vector."""
        if self._preconditioner is None:
            return self._bkd.copy(r)
        return self._preconditioner.apply(r)

    def solve(
        self,
        A: Array,
        b: Array,
        x0: Optional[Array] = None,
        tol: float = 1e-6,
        maxiter: int = 100,
    ) -> Tuple[Array, int, bool]:
        """Solve linear system Ax = b using Preconditioned Conjugate Gradient.

        Parameters
        ----------
        A : Array
            System matrix. Must be symmetric positive definite.
            Shape: (n, n)
        b : Array
            Right-hand side. Shape: (n,)
        x0 : Optional[Array]
            Initial guess. Shape: (n,). Defaults to zeros.
        tol : float
            Convergence tolerance for relative residual norm.
        maxiter : int
            Maximum number of iterations.

        Returns
        -------
        x : Array
            Solution. Shape: (n,)
        niter : int
            Number of iterations performed.
        converged : bool
            True if converged within tolerance.
        """
        bkd = self._bkd
        n = b.shape[0]

        # Initial guess
        if x0 is None:
            x = bkd.zeros((n,))
        else:
            x = bkd.copy(x0)

        # Initial residual r = b - A @ x
        r = b - A @ x

        # Check for zero RHS
        b_norm = bkd.norm(b)
        if b_norm < 1e-15:
            return x, 0, True

        # Apply preconditioner: z = M^{-1} @ r
        z = self._apply_preconditioner(r)

        # Initial search direction
        p = bkd.copy(z)

        # Initial dot product r^T @ z
        rz_old = bkd.dot(r, z)

        tol_sq = (tol * b_norm) ** 2

        for k in range(maxiter):
            # Matrix-vector product
            Ap = A @ p
            pAp = bkd.dot(p, Ap)

            # Avoid division by zero
            if abs(float(pAp)) < 1e-30:
                return x, k + 1, False

            # Step length
            alpha = rz_old / pAp

            # Update solution and residual
            x = x + alpha * p
            r = r - alpha * Ap

            # Check convergence
            r_norm_sq = bkd.dot(r, r)
            if float(r_norm_sq) < float(tol_sq):
                return x, k + 1, True

            # Apply preconditioner to new residual
            z = self._apply_preconditioner(r)

            # New dot product
            rz_new = bkd.dot(r, z)

            # Update search direction
            beta = rz_new / rz_old
            p = z + beta * p

            rz_old = rz_new

        return x, maxiter, False

    def solve_matvec(
        self,
        matvec: Callable[[Array], Array],
        b: Array,
        x0: Optional[Array] = None,
        tol: float = 1e-6,
        maxiter: int = 100,
    ) -> Tuple[Array, int, bool]:
        """Solve linear system using matrix-vector product (matrix-free).

        Parameters
        ----------
        matvec : Callable[[Array], Array]
            Function that computes A @ x for given x.
        b : Array
            Right-hand side. Shape: (n,)
        x0 : Optional[Array]
            Initial guess. Shape: (n,).
        tol : float
            Convergence tolerance.
        maxiter : int
            Maximum iterations.

        Returns
        -------
        x : Array
            Solution. Shape: (n,)
        niter : int
            Number of iterations.
        converged : bool
            True if converged.
        """
        bkd = self._bkd
        n = b.shape[0]

        if x0 is None:
            x = bkd.zeros((n,))
        else:
            x = bkd.copy(x0)

        r = b - matvec(x)

        b_norm = bkd.norm(b)
        if b_norm < 1e-15:
            return x, 0, True

        z = self._apply_preconditioner(r)
        p = bkd.copy(z)
        rz_old = bkd.dot(r, z)

        tol_sq = (tol * b_norm) ** 2

        for k in range(maxiter):
            Ap = matvec(p)
            pAp = bkd.dot(p, Ap)

            if abs(float(pAp)) < 1e-30:
                return x, k + 1, False

            alpha = rz_old / pAp
            x = x + alpha * p
            r = r - alpha * Ap

            r_norm_sq = bkd.dot(r, r)
            if float(r_norm_sq) < float(tol_sq):
                return x, k + 1, True

            z = self._apply_preconditioner(r)
            rz_new = bkd.dot(r, z)
            beta = rz_new / rz_old
            p = z + beta * p
            rz_old = rz_new

        return x, maxiter, False


def pcg_solve(
    A: Array,
    b: Array,
    bkd: Backend[Array],
    preconditioner: Optional[PreconditionerProtocol[Array]] = None,
    x0: Optional[Array] = None,
    tol: float = 1e-6,
    maxiter: int = 100,
) -> Tuple[Array, int, bool]:
    """Solve Ax = b using Preconditioned Conjugate Gradient (functional interface).

    Parameters
    ----------
    A : Array
        System matrix (SPD). Shape: (n, n)
    b : Array
        Right-hand side. Shape: (n,)
    bkd : Backend
        Computational backend.
    preconditioner : Optional[PreconditionerProtocol]
        Preconditioner to use. If None, equivalent to standard CG.
    x0 : Optional[Array]
        Initial guess.
    tol : float
        Convergence tolerance.
    maxiter : int
        Maximum iterations.

    Returns
    -------
    x : Array
        Solution.
    niter : int
        Number of iterations.
    converged : bool
        True if converged.
    """
    solver = PreconditionedConjugateGradient(bkd, preconditioner)
    return solver.solve(A, b, x0, tol, maxiter)
