"""
All-states endpoint functional for transient problems.

Computes Q = y(T), returning all state variables at the final time.
nqoi = nstates.
"""

from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.validation import validate_backend


class AllStatesEndpointFunctional(Generic[Array]):
    """
    All-states endpoint functional Q = y(T).

    Returns the full state vector at the final time. Useful as the default
    functional for TransientForwardModel, analogous to the identity
    functional used for SteadyForwardModel.

    Parameters
    ----------
    nstates : int
        Total number of state variables.
    nparams : int
        Total number of parameters.
    bkd : Backend
        Backend for array operations.
    """

    def __init__(
        self,
        nstates: int,
        nparams: int,
        bkd: Backend[Array],
    ):
        validate_backend(bkd)
        self._nstates = nstates
        self._nparams = nparams
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def nqoi(self) -> int:
        """Return the number of QoI outputs."""
        return self._nstates

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
            Q = y(T). Shape: (nstates, 1)
        """
        return sol[:, -1:]

    def state_jacobian(self, sol: Array, param: Array) -> Array:
        """
        Compute dQ/dy.

        Non-zero only at the final time (identity block).

        Note: This returns shape (nstates, ntimes) following the transient
        functional protocol. For vector QoI, the full Jacobian would be
        (nqoi, nstates, ntimes), but this shape is only used for scalar
        QoI adjoint accumulation. For vector QoI, use forward sensitivities.

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
        dqdu[:, -1] = 1.0
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
            Parameter Jacobian. Shape: (nqoi, nparams)
        """
        return self._bkd.zeros((self._nstates, self._nparams))

    # =========================================================================
    # HVP Methods (all zero — linear functional)
    # =========================================================================

    def state_state_hvp(
        self, sol: Array, param: Array, time_idx: int, wvec: Array
    ) -> Array:
        """Compute (d^2Q/dy^2)·w. Zero for linear functional."""
        return self._bkd.zeros((self._nstates, 1))

    def state_param_hvp(
        self, sol: Array, param: Array, time_idx: int, vvec: Array
    ) -> Array:
        """Compute (d^2Q/dy dp)·v. Zero for linear functional."""
        return self._bkd.zeros((self._nstates, 1))

    def param_state_hvp(
        self, sol: Array, param: Array, time_idx: int, wvec: Array
    ) -> Array:
        """Compute (d^2Q/dp dy)·w. Zero for linear functional."""
        return self._bkd.zeros((self._nparams, 1))

    def param_param_hvp(
        self, sol: Array, param: Array, vvec: Array
    ) -> Array:
        """Compute (d^2Q/dp^2)·v. Zero for linear functional."""
        return self._bkd.zeros((self._nparams, 1))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"nstates={self._nstates}, "
            f"nparams={self._nparams})"
        )
