from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend


class SubsetOfStatesAdjointFunctional(Generic[Array]):
    """
    Subset vector adjoint functional with Jacobian capabilities.

    This class implements a functional that extracts a subset of state
    variables.
    """

    def __init__(
        self,
        nstates: int,
        nparams: int,
        subset: Array,
        backend: Backend[Array],
    ):
        """
        Initialize the SubsetVectorAdjointFunctional object.

        Parameters
        ----------
        nstates : int
            Number of state variables.
        nparams : int
            Number of uncertain variables.
        subset : Array
            Indices of the subset of state variables to extract.
        backend : Backend
            Backend used for computations.

        Raises
        ------
        ValueError
            If the subset indices exceed the number of state variables.
        """
        self._nstates = nstates
        self._nparams = nparams
        if subset.shape[0] > self.nstates():
            raise ValueError(
                "subset index must be smaller than self.nstates()"
            )
        self._subset = subset
        self._bkd = backend

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nstates(self) -> int:
        return self._nstates

    def nparams(self) -> int:
        return self._nparams

    def nunique_params(self) -> int:
        return 0

    def __call__(self, state: Array, param: Array) -> Array:
        return state[self._subset]

    def nqoi(self) -> int:
        return self._subset.shape[0]

    def param_jacobian(self, state: Array, param: Array) -> Array:
        return self._bkd.zeros((self.nqoi(), self.nparams()))

    def state_jacobian(self, state: Array, param: Array) -> Array:
        jac = self._bkd.zeros((self.nqoi(), self.nstates()))
        row_idx = self._bkd.arange(self.nqoi())
        jac[row_idx, self._subset] = 1.0
        return jac
