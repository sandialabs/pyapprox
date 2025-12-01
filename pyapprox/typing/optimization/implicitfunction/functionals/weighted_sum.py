from typing import Generic

from pyapprox.typing.util.backend import Array, Backend


class WeightedSumFunctional(Generic[Array]):
    """
    Weighted sum functional with Jacobian and Hessian capabilities.
    """

    def __init__(self, weights: Array, nvars: int, backend: Backend[Array]):
        """
        Initialize the WeightedSumFunctional object.

        Parameters
        ----------
        weights : Array
            Weights for the weighted sum.
        nvars : int
            Number of uncertain variables.
        backend : Backend
            Backend used for computations.

        Raises
        ------
        ValueError
            If weights is not a 1D array.
        """
        if weights.ndim != 1:
            raise ValueError("weights must be a 1D array")
        self._nstates = weights.shape[0]
        self._nvars = nvars
        self._weights = weights
        self._bkd = backend

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nstates(self) -> int:
        return self._nstates

    def nvars(self) -> int:
        return self._nvars

    def nunique_vars(self) -> int:
        return 0

    def value(self, state: Array, param: Array) -> float:
        return self._bkd.sum(state * self._weights)

    def param_jacobian(self, state: Array, param: Array) -> Array:
        return self._bkd.zeros((1, self.nvars()))

    def state_jacobian(self, state: Array, param: Array) -> Array:
        return self._weights[None, :]

    def param_param_hvp(
        self, state: Array, param: Array, vvec: Array
    ) -> Array:
        return self._bkd.zeros((self.nvars(),))

    def state_state_hvp(
        self, state: Array, param: Array, vvec: Array
    ) -> Array:
        return self._bkd.zeros((self.nstates(),))

    def param_state_hvp(
        self, state: Array, param: Array, vvec: Array
    ) -> Array:
        return self._bkd.zeros((self.nvars(),))

    def state_param_hvp(
        self, state: Array, param: Array, vvec: Array
    ) -> Array:
        return self._bkd.zeros((self.nstates(),))
