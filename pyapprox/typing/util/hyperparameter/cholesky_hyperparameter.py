from typing import Union, Tuple
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.hyperparameter.hyperparameter import HyperParameter


class CholeskyHyperParameter(HyperParameter[Array]):
    """
    A hyperparameter that stores the nonzero elements of the lower triangular
    part of a Cholesky factor internally but allows users to interact with the
    full Cholesky factor.

    Parameters
    ----------
    name : str
        Name of the hyperparameter.
    nrows : int
        Number of rows (and columns) in the Cholesky factor.
    user_values : Array
        Full Cholesky factor provided by the user.
    user_bounds : Array
        Flattened bounds corresponding to the nonzero elements of the lower triangular part.
    bkd : Backend
        Backend for numerical computations.
    fixed : bool, optional
        Whether the hyperparameter is fixed (default is False).
    """

    def __init__(
        self,
        name: str,
        nrows: int,
        user_values: Array,
        user_bounds: Array,
        bkd: Backend[Array],
        fixed: bool = False,
    ):
        self._nrows = nrows

        # Create mask for lower triangular part
        self._mask = bkd.tril(bkd.ones((self._nrows, self._nrows), dtype=bool))

        # Extract nonzero elements from the user-provided Cholesky factor
        flattened_values = user_values[self._mask]

        # Ensure user_bounds is a flattened array
        if user_bounds.shape[0] != bkd.nonzero(self._mask)[0].shape[0]:
            raise ValueError(
                f"user_bounds shape {user_bounds.shape} inconsistent with "
                "the number of nonzero elements "
                f"{bkd.nonzero(self._mask)[0].shape[0]}"
            )

        # Initialize the base HyperParameter class with flattened values
        super().__init__(
            name=name,
            nparams=bkd.nonzero(self._mask)[0].shape[0],
            values=flattened_values,
            bounds=user_bounds,
            bkd=bkd,
            fixed=fixed,
        )

    def nrows(self) -> int:
        """
        Return the number of rows (and columns) in the Cholesky factor.

        Returns
        -------
        nrows : int
            Number of rows (and columns) in the Cholesky factor.
        """
        return self._nrows

    def factor(self) -> Array:
        """
        Get the full Cholesky factor.

        Returns
        -------
        chol : Array
            Full Cholesky factor reconstructed from the stored nonzero
           elements.
        """
        chol = self._bkd.zeros((self._nrows, self._nrows))
        chol[self._mask] = self.get_values()
        return chol
