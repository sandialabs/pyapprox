"""Solver protocols for spectral collocation methods.

Defines interfaces for linear solvers (direct and iterative) and
preconditioners.
"""

from typing import Protocol, Generic, runtime_checkable, Tuple, Optional, Callable

from pyapprox.typing.util.backends.protocols import Array, Backend


@runtime_checkable
class LinearSolverProtocol(Protocol, Generic[Array]):
    """Protocol for linear system solvers.

    Solves Ax = b using any method (direct or iterative).
    """

    def solve(self, A: Array, b: Array) -> Array:
        """Solve linear system Ax = b.

        Parameters
        ----------
        A : Array
            System matrix. Shape: (n, n)
        b : Array
            Right-hand side. Shape: (n,) or (n, m) for multiple RHS

        Returns
        -------
        Array
            Solution x with same shape as b.
        """
        ...


@runtime_checkable
class IterativeSolverProtocol(Protocol, Generic[Array]):
    """Protocol for iterative linear solvers.

    Provides convergence information in addition to solution.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def solve(
        self,
        A: Array,
        b: Array,
        x0: Optional[Array] = None,
        tol: float = 1e-6,
        maxiter: int = 100,
    ) -> Tuple[Array, int, bool]:
        """Solve linear system Ax = b iteratively.

        Parameters
        ----------
        A : Array
            System matrix. Shape: (n, n)
        b : Array
            Right-hand side. Shape: (n,)
        x0 : Optional[Array]
            Initial guess. Shape: (n,). Defaults to zeros.
        tol : float
            Convergence tolerance for residual norm.
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
        ...


@runtime_checkable
class MatrixFreeSolverProtocol(Protocol, Generic[Array]):
    """Protocol for matrix-free iterative solvers.

    Uses a callable matvec instead of explicit matrix.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def solve(
        self,
        matvec: Callable[[Array], Array],
        b: Array,
        x0: Optional[Array] = None,
        tol: float = 1e-6,
        maxiter: int = 100,
    ) -> Tuple[Array, int, bool]:
        """Solve linear system using matrix-vector product.

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
        ...


@runtime_checkable
class PreconditionerProtocol(Protocol, Generic[Array]):
    """Protocol for preconditioners.

    A preconditioner approximates the inverse of a matrix to
    accelerate iterative solvers.
    """

    def apply(self, r: Array) -> Array:
        """Apply preconditioner to vector.

        Computes M^{-1} @ r where M approximates A.

        Parameters
        ----------
        r : Array
            Input vector. Shape: (n,)

        Returns
        -------
        Array
            Preconditioned vector. Shape: (n,)
        """
        ...


@runtime_checkable
class PreconditionerWithSetupProtocol(Protocol, Generic[Array]):
    """Protocol for preconditioners requiring setup.

    Some preconditioners (like ILU) need to analyze the matrix
    before they can be applied.
    """

    def setup(self, A: Array) -> None:
        """Setup preconditioner for given matrix.

        Parameters
        ----------
        A : Array
            System matrix to precondition. Shape: (n, n)
        """
        ...

    def apply(self, r: Array) -> Array:
        """Apply preconditioner to vector.

        Parameters
        ----------
        r : Array
            Input vector. Shape: (n,)

        Returns
        -------
        Array
            Preconditioned vector. Shape: (n,)
        """
        ...
