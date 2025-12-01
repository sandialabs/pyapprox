from pyapprox.typing.util.backend import Array, Backend
from pyapprox.typing.optimization.implicitfunction.state_equations.protocols import (
    ParameterizedStateEquationWithJacobianAndHVPProtocol,
)


class LinearStateEquation(
    ParameterizedStateEquationWithJacobianAndHVPProtocol[Array]
):
    """
    Linear state equation with Jacobian and Hessian capabilities.

    This class implements a linear state equation of the form:
        R(state, param) = state - Amat @ param
    """

    def __init__(self, Amat: Array, bvec: Array, backend: Backend[Array]):
        """
        Initialize the LinearStateEquation object.

        Parameters
        ----------
        Amat : Array
            Matrix defining the linear state equation.
        bvec : Array
            Vector defining the linear state equation.
        backend : Backend
            Backend used for computations.

        Raises
        ------
        ValueError
            If the dimensions of Amat and bvec are inconsistent.
        """
        self._bkd = backend
        if bvec.ndim != 1:
            raise ValueError(
                f"bvec must be a 1D array but has shape {bvec.shape}"
            )
        if Amat.shape[0] != bvec.shape[0]:
            raise ValueError(
                "Amat and bvec must have the same number of rows "
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

    def nstates(self) -> int:
        """
        Return the number of states in the system.

        Returns
        -------
        int
            Number of states.
        """
        return self._Amat.shape[0]

    def nvars(self) -> int:
        """
        Return the number of variables in the system.

        Returns
        -------
        int
            Number of variables.
        """
        return self._Amat.shape[1]

    def value(self, state: Array, param: Array) -> Array:
        """
        Compute the residual value for the given state and parameters.

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
        return state - self._Amat @ param

    def solve(self, init_state: Array, param: Array) -> Array:
        """
        Solve the state equation for the given initial state and parameters.

        Parameters
        ----------
        init_state : Array
            Initial state (ignored for this linear problem).
        param : Array
            Parameters of the system.

        Returns
        -------
        Array
            Solution to the state equation.
        """
        return self._Amat @ param

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
        return self._bkd.eye(self.nstates())

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
        return -self._Amat

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
        return self._bkd.zeros((self.nstates(),))

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
        return self._bkd.zeros((self.nvars(),))

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
        return self._bkd.zeros((self.nstates(),))

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
        return self._bkd.zeros((self.nvars(),))
