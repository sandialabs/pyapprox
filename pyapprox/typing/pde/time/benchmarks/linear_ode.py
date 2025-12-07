"""
Linear ODE benchmark for testing adjoint gradient computation.

Implements the linear ODE:
    dy/dt = A·y + B·p

where A is a stability matrix and B maps parameters to forcing.
The analytical solution is available for verification.
"""

from typing import Generic

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.validation import validate_backend


class LinearODEResidual(Generic[Array]):
    """
    Linear ODE residual: f(y, t; p) = A·y + B·p.

    This is a simple benchmark problem for testing time integrators
    and adjoint gradient computation.

    Parameters
    ----------
    Amat : Array
        Stability matrix. Shape: (nstates, nstates)
    Bmat : Array
        Parameter-to-forcing matrix. Shape: (nstates, nparams)
    bkd : Backend
        Backend for array operations.
    """

    def __init__(
        self,
        Amat: Array,
        Bmat: Array,
        bkd: Backend[Array],
    ):
        validate_backend(bkd)
        self._Amat = Amat
        self._Bmat = Bmat
        self._bkd = bkd
        self._time = 0.0
        self._param = None
        self._nstates = Amat.shape[0]
        self._nparams = Bmat.shape[1]

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def nparams(self) -> int:
        """Return the number of parameters."""
        return self._nparams

    def set_time(self, time: float) -> None:
        """Set the current time."""
        self._time = time

    def set_param(self, param: Array) -> None:
        """
        Set the parameters.

        Parameters
        ----------
        param : Array
            Parameters. Shape: (nparams, 1) or (nparams,)
        """
        if param.ndim == 1:
            param = param[:, None]
        self._param = param

    def __call__(self, state: Array) -> Array:
        """
        Evaluate the ODE right-hand side.

        f(y) = A·y + B·p

        Parameters
        ----------
        state : Array
            State. Shape: (nstates,)

        Returns
        -------
        Array
            f(y). Shape: (nstates,)
        """
        forcing = (self._Bmat @ self._param).flatten()
        return self._Amat @ state + forcing

    def jacobian(self, state: Array) -> Array:
        """
        Compute df/dy = A.

        Parameters
        ----------
        state : Array
            State. Shape: (nstates,)

        Returns
        -------
        Array
            Jacobian matrix. Shape: (nstates, nstates)
        """
        return self._Amat

    def mass_matrix(self, nstates: int) -> Array:
        """
        Return the identity mass matrix.

        Parameters
        ----------
        nstates : int
            Number of states.

        Returns
        -------
        Array
            Mass matrix. Shape: (nstates, nstates)
        """
        return self._bkd.eye(nstates)

    def param_jacobian(self, state: Array) -> Array:
        """
        Compute df/dp = B.

        Parameters
        ----------
        state : Array
            State. Shape: (nstates,)

        Returns
        -------
        Array
            Parameter Jacobian. Shape: (nstates, nparams)
        """
        return self._Bmat

    def initial_param_jacobian(self) -> Array:
        """
        Compute derivative of initial condition with respect to parameters.

        For this benchmark, initial condition is fixed, so this is zero.

        Returns
        -------
        Array
            Shape: (nstates, nparams)
        """
        return self._bkd.zeros((self._nstates, self._nparams))

    # =========================================================================
    # HVP Methods for second-order adjoints
    # =========================================================================

    def state_state_hvp(
        self, state: Array, adj_state: Array, wvec: Array
    ) -> Array:
        """
        Compute (d^2f/dy^2)·w contracted with adjoint.

        For linear ODE, this is zero (no second derivatives).
        """
        return self._bkd.zeros(state.shape)

    def state_param_hvp(
        self, state: Array, adj_state: Array, vvec: Array
    ) -> Array:
        """
        Compute (d^2f/dy dp)·v contracted with adjoint.

        For linear ODE, this is zero.
        """
        return self._bkd.zeros(state.shape)

    def param_state_hvp(
        self, state: Array, adj_state: Array, wvec: Array
    ) -> Array:
        """
        Compute (d^2f/dp dy)·w contracted with adjoint.

        For linear ODE, this is zero.
        """
        return self._bkd.zeros((self._nparams,))

    def param_param_hvp(
        self, state: Array, adj_state: Array, vvec: Array
    ) -> Array:
        """
        Compute (d^2f/dp^2)·v contracted with adjoint.

        For linear ODE, this is zero.
        """
        return self._bkd.zeros((self._nparams,))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"nstates={self._nstates}, "
            f"nparams={self._nparams})"
        )
