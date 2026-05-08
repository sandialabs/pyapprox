"""Direct linear solver.

Wraps backend solve function for direct solution of linear systems Ax = b.
"""

from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend

# TODO: should we implement a direct solver for sparse matrices.
# Does np and torch both support sparse solve?

class DirectSolver(Generic[Array]):
    """Direct linear solver using backend's solve function.

    Solves Ax = b using direct factorization (e.g., LU decomposition).

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
        return self._bkd.solve(A, b)


def direct_solve(A: Array, b: Array, bkd: Backend[Array]) -> Array:
    """Solve linear system Ax = b directly (functional interface).

    Parameters
    ----------
    A : Array
        System matrix. Shape: (n, n)
    b : Array
        Right-hand side. Shape: (n,) or (n, m)
    bkd : Backend
        Computational backend.

    Returns
    -------
    Array
        Solution x.
    """
    return bkd.solve(A, b)
