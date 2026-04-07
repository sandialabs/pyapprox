from typing import Generic

from pyapprox.interface.functions.protocols.validation import (
    validate_sample,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.validation import validate_backend


class LinearStateEquation(Generic[Array]):
    """
    Linear state equation with Jacobian and Hessian capabilities.

    This class implements a linear residual equation of the form:
        r(state, param) = state - Amat @ param - bvec

    Parameters
    ----------
    Amat : Array
        Matrix defining the linear system.
    bvec : Array
        Vector defining the linear system.
    bkd : Backend
        Backend used for numerical computations.
    """

    def __init__(self, Amat: Array, bvec: Array, bkd: Backend[Array]) -> None:
        """
        Initialize the linear state equation.

        Parameters
        ----------
        Amat : Array
            Matrix defining the linear system.
        bvec : Array
            Vector defining the linear system.
        bkd : Backend
            Backend used for numerical computations.

        Raises
        ------
        ValueError
            If `bvec` is not a 1D array or if the dimensions of `Amat` and `bvec` are
            inconsistent.
        """
        validate_backend(bkd)
        self._bkd = bkd
        if bvec.ndim != 2 or bvec.shape[1] != 1:
            raise ValueError(
                f"bvec must be a 2D array wioth 1 column but has shape {bvec.shape}"
            )
        if Amat.shape[0] != bvec.shape[0]:
            raise ValueError(
                "Amat and bvec must have the same number of rows"
                f"but had shapes {Amat.shape} and {bvec.shape}"
            )
        self._Amat = Amat
        self._bvec = bvec

    def bkd(self) -> Backend[Array]:
        """
        Return the backend used for computations.

        Returns
        -------
        Backend
            Backend used for computations.
        """
        return self._bkd

    def nparams(self) -> int:
        """
        Return the number of uncertain variables.

        Returns
        -------
        int
            Number of uncertain variables.
        """
        return self._Amat.shape[1]

    def nstates(self) -> int:
        """
        Return the number of state variables.

        Returns
        -------
        int
            Number of state variables.
        """
        return self._Amat.shape[0]

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest (QoI).

        Returns
        -------
        int
            Number of state variables.
        """
        return self.nstates()

    def __call__(self, state: Array, param: Array) -> Array:
        """
        Compute the residual value.

        Parameters
        ----------
        state : Array
            State of the system.
        param : Array
            Parameters of the system.

        Returns
        -------
        Array
            Residual value.
        """
        validate_sample(self.nstates(), state)
        validate_sample(self.nparams(), param)
        return state - self._Amat @ param - self._bvec

    def solve(self, init_state: Array, param: Array) -> Array:
        """
        Solve the residual equation for the given initial state and parameters.

        Parameters
        ----------
        init_state : Array
            Initial state (ignored for this linear problem).
        param : Array
            Parameters of the system.

        Returns
        -------
        Array
            Solution to the residual equation.
        """
        validate_sample(self.nstates(), init_state)
        validate_sample(self.nparams(), param)
        return self._Amat @ param + self._bvec

    def param_jacobian(self, state: Array, param: Array) -> Array:
        """
        Compute the Jacobian of the residual with respect to the parameters.

        Parameters
        ----------
        state : Array
            State of the system.
        param : Array
            Parameters of the system.

        Returns
        -------
        Array
            Jacobian matrix with respect to the parameters.
        """
        validate_sample(self.nstates(), state)
        validate_sample(self.nparams(), param)
        return -self._Amat

    def state_jacobian(self, state: Array, param: Array) -> Array:
        """
        Compute the Jacobian of the residual with respect to the state.

        Parameters
        ----------
        state : Array
            State of the system.
        param : Array
            Parameters of the system.

        Returns
        -------
        Array
            Jacobian matrix with respect to the state.
        """
        validate_sample(self.nstates(), state)
        validate_sample(self.nparams(), param)
        return self._bkd.eye(self.nstates())

    def param_param_hvp(
        self, state: Array, param: Array, adj_state: Array, vvec: Array
    ) -> Array:
        """
        Compute the Hessian-vector product with respect to the parameters.

        Parameters
        ----------
        state : Array
            State of the system.
        param : Array
            Parameters of the system.
        adj_state : Array
            Adjoint state.
        vvec : Array
            Vector for Hessian computation.

        Returns
        -------
        Array
            Hessian-vector product result.
        """
        validate_sample(self.nstates(), state)
        validate_sample(self.nparams(), param)
        return self._bkd.zeros((self.nparams(), 1))

    def state_state_hvp(
        self, state: Array, param: Array, adj_state: Array, wvec: Array
    ) -> Array:
        """
        Compute the Hessian-vector product with respect to the state.

        Parameters
        ----------
        state : Array
            State of the system.
        param : Array
            Parameters of the system.
        adj_state : Array
            Adjoint state.
        wvec : Array
            Vector for Hessian computation.

        Returns
        -------
        Array
            Hessian-vector product result.
        """
        validate_sample(self.nstates(), state)
        validate_sample(self.nparams(), param)
        return self._bkd.zeros((self.nstates(), 1))

    def param_state_hvp(
        self, state: Array, param: Array, adj_state: Array, wvec: Array
    ) -> Array:
        """
        Compute the Hessian-vector product with respect to parameters and state.

        Parameters
        ----------
        state : Array
            State of the system.
        param : Array
            Parameters of the system.
        adj_state : Array
            Adjoint state.
        wvec : Array
            Vector for Hessian computation.

        Returns
        -------
        Array
            Hessian-vector product result.
        """
        validate_sample(self.nstates(), state)
        validate_sample(self.nparams(), param)
        return self._bkd.zeros((self.nparams(), 1))

    def state_param_hvp(
        self, state: Array, param: Array, adj_state: Array, vvec: Array
    ) -> Array:
        """
        Compute the Hessian-vector product with respect to state and parameters.

        Parameters
        ----------
        state : Array
            State of the system.
        param : Array
            Parameters of the system.
        adj_state : Array
            Adjoint state.
        vvec : Array
            Vector for Hessian computation.

        Returns
        -------
        Array
            Hessian-vector product result.
        """
        validate_sample(self.nstates(), state)
        validate_sample(self.nparams(), param)
        return self._bkd.zeros((self.nstates(), 1))
