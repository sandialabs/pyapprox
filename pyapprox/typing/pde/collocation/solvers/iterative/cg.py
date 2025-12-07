"""Conjugate Gradient solver for spectral collocation methods.

Implements the standard CG algorithm for symmetric positive definite systems.
"""

from typing import Generic, Optional, Tuple, Callable

from pyapprox.typing.util.backends.protocols import Array, Backend


class ConjugateGradient(Generic[Array]):
    """Conjugate Gradient iterative solver.

    Solves Ax = b for symmetric positive definite A using the CG algorithm.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]):
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def solve(
        self,
        A: Array,
        b: Array,
        x0: Optional[Array] = None,
        tol: float = 1e-6,
        maxiter: int = 100,
    ) -> Tuple[Array, int, bool]:
        """Solve linear system Ax = b using Conjugate Gradient.

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
        p = bkd.copy(r)
        rs_old = bkd.dot(r, r)

        # Tolerance based on initial residual
        b_norm = bkd.norm(b)
        if b_norm < 1e-15:
            # b is zero, solution is zero
            return x, 0, True

        tol_sq = (tol * b_norm) ** 2

        for k in range(maxiter):
            Ap = A @ p
            pAp = bkd.dot(p, Ap)

            # Avoid division by zero
            if abs(float(pAp)) < 1e-30:
                return x, k + 1, False

            alpha = rs_old / pAp
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = bkd.dot(r, r)

            # Check convergence
            if float(rs_new) < float(tol_sq):
                return x, k + 1, True

            beta = rs_new / rs_old
            p = r + beta * p
            rs_old = rs_new

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
        p = bkd.copy(r)
        rs_old = bkd.dot(r, r)

        b_norm = bkd.norm(b)
        if b_norm < 1e-15:
            return x, 0, True

        tol_sq = (tol * b_norm) ** 2

        for k in range(maxiter):
            Ap = matvec(p)
            pAp = bkd.dot(p, Ap)

            if abs(float(pAp)) < 1e-30:
                return x, k + 1, False

            alpha = rs_old / pAp
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = bkd.dot(r, r)

            if float(rs_new) < float(tol_sq):
                return x, k + 1, True

            beta = rs_new / rs_old
            p = r + beta * p
            rs_old = rs_new

        return x, maxiter, False


def cg_solve(
    A: Array,
    b: Array,
    bkd: Backend[Array],
    x0: Optional[Array] = None,
    tol: float = 1e-6,
    maxiter: int = 100,
) -> Tuple[Array, int, bool]:
    """Solve Ax = b using Conjugate Gradient (functional interface).

    Parameters
    ----------
    A : Array
        System matrix (SPD). Shape: (n, n)
    b : Array
        Right-hand side. Shape: (n,)
    bkd : Backend
        Computational backend.
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
    solver = ConjugateGradient(bkd)
    return solver.solve(A, b, x0, tol, maxiter)
