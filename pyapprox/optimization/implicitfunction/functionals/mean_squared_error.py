from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.validation import validate_backend


class MSEFunctional(Generic[Array]):
    """
    Mean Squared Error (MSE) functional with Jacobian and Hessian capabilities.
    """

    def __init__(self, nstates: int, nparams: int, bkd: Backend[Array]):
        """
        Initialize the MSEFunctional object.

        Parameters
        ----------
        nstates : int
            Number of state variables.
        nparams : int
            Number of uncertain variables.
        bkd : Backend
            Backend used for computations.
        """
        validate_backend(bkd)
        self._nstates = nstates
        self._nparams = nparams
        self._bkd = bkd

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest (QoI).

        Returns
        -------
        int
            Number of quantities of interest.
        """
        return 1

    def bkd(self) -> Backend[Array]:
        """
        Return the backend used for computations.

        Returns
        -------
        Backend
            Backend used for computations.
        """
        return self._bkd

    def nstates(self) -> int:
        """
        Return the number of state variables.

        Returns
        -------
        int
            Number of state variables.
        """
        return self._nstates

    def nparams(self) -> int:
        """
        Return the number of uncertain variables.

        Returns
        -------
        int
            Number of uncertain variables.
        """
        return self._nparams

    def nunique_params(self) -> int:
        """
        Return the number of unique parameters in the functional.

        Returns
        -------
        int
            Number of unique variables.
        """
        return 0

    def set_observations(self, obs: Array) -> None:
        """
        Set the observations for the functional.

        Parameters
        ----------
        obs : Array
            Observations array.

        Raises
        ------
        ValueError
            If the shape of `obs` is inconsistent with the number of states.
        """
        if obs.shape != (self.nstates(), 1):
            raise ValueError(
                f"obs has shape {obs.shape} but must have shape {(self.nstates(), 1)}"
            )
        self._obs = obs

    def __call__(self, state: Array, param: Array) -> Array:
        """
        Compute the value of the functional.

        Parameters
        ----------
        state : Array
            State of the system.
        param : Array
            Parameters of the system.

        Returns
        -------
        Array
            Value of the functional.
        """
        return self._bkd.sum((self._obs - state) ** 2, keepdims=True) / 2.0

    def param_jacobian(self, state: Array, param: Array) -> Array:
        """
        Compute the Jacobian of the functional with respect to the parameters.

        Returns
        -------
        Array
            Jacobian matrix with respect to the parameters.
        """
        return self._bkd.zeros((1, self.nparams()))

    def state_jacobian(self, state: Array, param: Array) -> Array:
        """
        Compute the Jacobian of the functional with respect to the state.

        Returns
        -------
        Array
            Jacobian matrix with respect to the state.
        """
        return (state - self._obs).T

    def param_param_hvp(self, state: Array, param: Array, vvec: Array) -> Array:
        """
        Compute the Hessian-vector product with respect to the parameters.

        Returns
        -------
        Array
            Hessian-vector product result.
        """
        return self._bkd.zeros((self.nparams(), 1))

    def state_state_hvp(self, state: Array, param: Array, vvec: Array) -> Array:
        """
        Compute the Hessian-vector product with respect to the state.

        Returns
        -------
        Array
            Hessian-vector product result.
        """
        return vvec

    def param_state_hvp(self, state: Array, param: Array, vvec: Array) -> Array:
        """
        Compute the Hessian-vector product with respect to parameters and state.

        Returns
        -------
        Array
            Hessian-vector product result.
        """
        return self._bkd.zeros((self.nparams(), 1))

    def state_param_hvp(self, state: Array, param: Array, vvec: Array) -> Array:
        """
        Compute the Hessian-vector product with respect to state and parameters.

        Returns
        -------
        Array
            Hessian-vector product result.
        """
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
