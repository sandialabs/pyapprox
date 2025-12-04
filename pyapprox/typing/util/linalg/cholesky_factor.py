from typing import Any, Generic, Tuple

from pyapprox.typing.util.backends.protocols import Array, Backend


class CholeskyFactor(Generic[Array]):
    """
    A class to encapsulate operations related to a Cholesky factor.

    Parameters
    ----------
    L : Array
        The Cholesky factor of a matrix.
    bkd : Backend, optional
        Backend for numerical computations
    """

    def __init__(self, L: Array, bkd: Backend) -> None:
        self._L = L
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def shape(self) -> Tuple:
        return self._L.shape

    def factor(self) -> Array:
        """
        Return the stored Cholesky factor.

        Returns
        -------
        L : Array
            The Cholesky factor of the matrix.
        """
        return self._L

    def log_determinant(self) -> Any:
        """
        Compute the log determinant of the matrix from its Cholesky factor.

        Returns
        -------
        log_det : Any
            The log determinant of the matrix.
        """
        return 2.0 * self._bkd.sum(self._bkd.log(self._bkd.diag(self._L)))

    def matrix_inverse(self) -> Array:
        """
        Compute the inverse inv(A) of the matrix A=L@L.T using its Cholesky factor.

        Returns
        -------
        A_inv : Array
            The inverse of the matrix.
        """
        rhs = self._bkd.eye(self._L.shape[0])
        A_inv = self._bkd.cholesky_solve(self._L, rhs, lower=True)
        return A_inv

    def factor_inverse(self) -> Array:
        """
        Compute the inverse of the Cholesky factor.

        Returns
        -------
        L_inv : Array
            The inverse of the Cholesky factor.
        """
        rhs = self._bkd.eye(self._L.shape[0])
        L_inv = self._bkd.solve_triangular(self._L, rhs, lower=True)
        return L_inv

    def solve(self, rhs: Array) -> Array:
        """
        Solve the linear system LL'x = b using forward and backward substitution.

        Parameters
        ----------
        rhs : Array
            The right-hand side of the linear system.

        Returns
        -------
        x : Array
            The solution to the linear system.
        """
        # Use forward substitution to solve Ly = b
        y = self._bkd.solve_triangular(self._L, rhs, lower=True)
        # Use backward substitution to solve L'x = y
        x = self._bkd.solve_triangular(self._L.T, y, lower=False)
        return x

    def __repr__(self) -> str:
        """
        Return a string representation of the CholeskyFactor class.

        Returns
        -------
        repr : str
            String representation of the class.
        """
        return "{0}(N={1}, backend={2})".format(
            self.__class__.__name__,
            self._L.shape,
            self.bkd().__class__.__name__,
        )
