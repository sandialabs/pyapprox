from typing import Generic

from pyapprox.typing.util.backend import Array, Backend


class MSEFunctional(Generic[Array]):
    """
    Mean Squared Error (MSE) functional with Jacobian and Hessian capabilities.
    """

    def __init__(self, nstates: int, nvars: int, backend: Backend[Array]):
        """
        Initialize the MSEFunctional object.

        Parameters
        ----------
        nstates : int
            Number of state variables.
        nvars : int
            Number of uncertain variables.
        backend : Backend
            Backend used for computations.
        """
        self._nstates = nstates
        self._nvars = nvars
        self._bkd = backend

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

    def nvars(self) -> int:
        """
        Return the number of uncertain variables.

        Returns
        -------
        int
            Number of uncertain variables.
        """
        return self._nvars

    def nunique_vars(self) -> int:
        """
        Return the number of unique variables.

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
        if obs.shape != (self.nstates(),):
            raise ValueError(
                f"obs has shape {obs.shape} but must have "
                f"shape {(self.nstates(),)}"
            )
        self._obs = obs

    def value(self, state: Array, param: Array) -> Array:
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
        return self._bkd.sum((self._obs - state) ** 2) / 2.0

    def param_jacobian(self, state: Array, param: Array) -> Array:
        """
        Compute the Jacobian of the functional with respect to the parameters.

        Returns
        -------
        Array
            Jacobian matrix with respect to the parameters.
        """
        return self._bkd.zeros((1, self.nvars()))

    def state_jacobian(self, state: Array, param: Array) -> Array:
        """
        Compute the Jacobian of the functional with respect to the state.

        Returns
        -------
        Array
            Jacobian matrix with respect to the state.
        """
        return (state - self._obs)[None, :]

    def param_param_hvp(
        self, state: Array, param: Array, vvec: Array
    ) -> Array:
        """
        Compute the Hessian-vector product with respect to the parameters.

        Returns
        -------
        Array
            Hessian-vector product result.
        """
        return self._bkd.zeros((self.nvars(),))

    def state_state_hvp(
        self, state: Array, param: Array, vvec: Array
    ) -> Array:
        """
        Compute the Hessian-vector product with respect to the state.

        Returns
        -------
        Array
            Hessian-vector product result.
        """
        return vvec

    def param_state_hvp(
        self, state: Array, param: Array, vvec: Array
    ) -> Array:
        """
        Compute the Hessian-vector product with respect to parameters and state.

        Returns
        -------
        Array
            Hessian-vector product result.
        """
        return self._bkd.zeros((self.nvars(),))

    def state_param_hvp(
        self, state: Array, param: Array, vvec: Array
    ) -> Array:
        """
        Compute the Hessian-vector product with respect to state and parameters.

        Returns
        -------
        Array
            Hessian-vector product result.
        """
        return self._bkd.zeros((self.nstates(),))
