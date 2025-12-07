"""
Endpoint functional for transient problems.

Computes Q = y_k(T) where y_k is a specific state variable at final time T.
"""

from typing import Generic

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.validation import validate_backend


class EndpointFunctional(Generic[Array]):
    """
    Endpoint functional Q = y_k(T).

    Evaluates a single state variable at the final time.
    Useful for optimal control problems targeting a specific final state.

    Parameters
    ----------
    state_idx : int
        Index of the state variable to evaluate at final time.
    nstates : int
        Total number of state variables.
    nparams : int
        Total number of parameters.
    bkd : Backend
        Backend for array operations.
    """

    def __init__(
        self,
        state_idx: int,
        nstates: int,
        nparams: int,
        bkd: Backend[Array],
    ):
        validate_backend(bkd)
        if state_idx < 0 or state_idx >= nstates:
            raise ValueError(
                f"state_idx {state_idx} out of range [0, {nstates})"
            )
        self._state_idx = state_idx
        self._nstates = nstates
        self._nparams = nparams
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def nqoi(self) -> int:
        """Return the number of QoI outputs."""
        return 1

    def nstates(self) -> int:
        """Return the number of state variables."""
        return self._nstates

    def nparams(self) -> int:
        """Return the total number of parameters."""
        return self._nparams

    def nunique_params(self) -> int:
        """Return the number of parameters unique to the functional."""
        return 0

    def __call__(self, sol: Array, param: Array) -> Array:
        """
        Evaluate the functional.

        Parameters
        ----------
        sol : Array
            Solution trajectory. Shape: (nstates, ntimes)
        param : Array
            Parameters. Shape: (nparams, 1)

        Returns
        -------
        Array
            Q = y_k(T). Shape: (1, 1)
        """
        return self._bkd.atleast_2d(sol[self._state_idx, -1])

    def state_jacobian(self, sol: Array, param: Array) -> Array:
        """
        Compute dQ/dy.

        Only non-zero at the selected state index at final time.

        Parameters
        ----------
        sol : Array
            Solution trajectory. Shape: (nstates, ntimes)
        param : Array
            Parameters. Shape: (nparams, 1)

        Returns
        -------
        Array
            State Jacobian. Shape: (nstates, ntimes)
        """
        dqdu = self._bkd.zeros(sol.shape)
        dqdu = self._bkd.copy(dqdu)
        dqdu[self._state_idx, -1] = 1.0
        return dqdu

    def param_jacobian(self, sol: Array, param: Array) -> Array:
        """
        Compute dQ/dp.

        Zero since Q does not depend on parameters directly.

        Parameters
        ----------
        sol : Array
            Solution trajectory. Shape: (nstates, ntimes)
        param : Array
            Parameters. Shape: (nparams, 1)

        Returns
        -------
        Array
            Parameter Jacobian. Shape: (1, nparams)
        """
        return self._bkd.zeros((1, self._nparams))

    # =========================================================================
    # HVP Methods
    # =========================================================================

    def state_state_hvp(
        self, sol: Array, param: Array, time_idx: int, wvec: Array
    ) -> Array:
        """
        Compute (d^2Q/dy^2)·w.

        Zero for linear functional.
        """
        return self._bkd.zeros((self._nstates, 1))

    def state_param_hvp(
        self, sol: Array, param: Array, time_idx: int, vvec: Array
    ) -> Array:
        """
        Compute (d^2Q/dy dp)·v.

        Zero since Q does not depend on parameters.
        """
        return self._bkd.zeros((self._nstates, 1))

    def param_state_hvp(
        self, sol: Array, param: Array, time_idx: int, wvec: Array
    ) -> Array:
        """
        Compute (d^2Q/dp dy)·w.

        Zero since Q does not depend on parameters.
        """
        return self._bkd.zeros((self._nparams, 1))

    def param_param_hvp(
        self, sol: Array, param: Array, vvec: Array
    ) -> Array:
        """
        Compute (d^2Q/dp^2)·v.

        Zero since Q does not depend on parameters.
        """
        return self._bkd.zeros((self._nparams, 1))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"state_idx={self._state_idx}, "
            f"nstates={self._nstates}, "
            f"nparams={self._nparams})"
        )
