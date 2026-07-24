"""
Weighted endpoint functional for transient problems.

Computes :math:`Q = c^T y(T)`, a weighted sum of all state variables at
final time T.
"""

from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.validation import validate_backend


class WeightedEndpointFunctional(Generic[Array]):
    """
    Weighted endpoint functional :math:`Q = c^T y(T)`.

    Evaluates a weighted sum of the state variables at the final time.
    Generalizes ``EndpointFunctional`` (a one-hot weight vector) to
    arbitrary weights, e.g. a mass-weighted subdomain average
    ``c = M @ indicator``.

    Parameters
    ----------
    weights : Array
        Weight vector c. Shape: ``(nstates, 1)``.
    nparams : int
        Total number of parameters.
    bkd : Backend
        Backend for array operations.
    """

    def __init__(
        self,
        weights: Array,
        nparams: int,
        bkd: Backend[Array],
    ):
        validate_backend(bkd)
        if weights.ndim != 2 or weights.shape[1] != 1:
            raise ValueError(
                f"weights must have shape (nstates, 1), got {weights.shape}"
            )
        self._weights = weights
        self._nstates = weights.shape[0]
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

    def weights(self) -> Array:
        """Return the weight vector. Shape: ``(nstates, 1)``."""
        return self._weights

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
            :math:`Q = c^T y(T)`. Shape: (1, 1)
        """
        if sol.shape[0] != self._nstates:
            raise ValueError(
                f"sol has {sol.shape[0]} states but weights have "
                f"{self._nstates}"
            )
        return self._bkd.reshape(
            self._bkd.sum(self._weights[:, 0] * sol[:, -1]), (1, 1)
        )

    def state_jacobian(self, sol: Array, param: Array) -> Array:
        """
        Compute dQ/dy.

        Only non-zero at the final time, where it equals the weights.

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
        dqdu[:, -1] = self._weights[:, 0]
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

    def param_param_hvp(self, sol: Array, param: Array, vvec: Array) -> Array:
        """
        Compute (d^2Q/dp^2)·v.

        Zero since Q does not depend on parameters.
        """
        return self._bkd.zeros((self._nparams, 1))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"nstates={self._nstates}, "
            f"nparams={self._nparams})"
        )
