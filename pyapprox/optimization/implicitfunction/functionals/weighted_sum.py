from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend


class WeightedSumFunctional(Generic[Array]):
    """
    Weighted sum functional with Jacobian and Hessian capabilities.
    """

    def __init__(self, weights: Array, nparams: int, backend: Backend[Array]):
        """
        Initialize the WeightedSumFunctional object.

        Parameters
        ----------
        weights : Array
            Weights for the weighted sum.
        nparams : int
            Number of uncertain variables.
        backend : Backend
            Backend used for computations.

        Raises
        ------
        ValueError
            If weights is not a 1D array.
        """
        if weights.ndim != 2:
            raise ValueError("weights must be a 2D array")
        self._nstates = weights.shape[0]
        self._nparams = nparams
        self._weights = weights
        self._bkd = backend

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nqoi(self) -> int:
        return 1

    def nstates(self) -> int:
        return self._nstates

    def nparams(self) -> int:
        return self._nparams

    def nunique_params(self) -> int:
        return 0

    def __call__(self, state: Array, param: Array) -> Array:
        return self._bkd.sum(state * self._weights, keepdims=True)

    def param_jacobian(self, state: Array, param: Array) -> Array:
        return self._bkd.zeros((1, self.nparams()))

    def state_jacobian(self, state: Array, param: Array) -> Array:
        return self._weights.T

    def param_param_hvp(
        self, state: Array, param: Array, vvec: Array
    ) -> Array:
        return self._bkd.zeros((self.nparams(), 1))

    def state_state_hvp(
        self, state: Array, param: Array, vvec: Array
    ) -> Array:
        return self._bkd.zeros((self.nstates(), 1))

    def param_state_hvp(
        self, state: Array, param: Array, vvec: Array
    ) -> Array:
        return self._bkd.zeros((self.nparams(), 1))

    def state_param_hvp(
        self, state: Array, param: Array, vvec: Array
    ) -> Array:
        return self._bkd.zeros((self.nstates(), 1))

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the object for debugging.
        """
        return (
            f"{self.__class__.__name__}("
            f"nstates={self.nstates()}, "
            f"nparams={self.nparams()}, "
            f"bkd={type(self._bkd).__name__})"
        )
