from pyapprox.typing.util.backend import Array, Backend
from pyapprox.typing.optimization.implicitfunction.state_equations.protocols import (
    ParameterizedStateEquationWithJacobianAndHVPProtocol,
)


class NonLinearCoupledStateEquations(
    ParameterizedStateEquationWithJacobianAndHVPProtocol[Array]
):
    r"""
    Nonlinear coupled equations residual with Jacobian and Hessian capabilities.

    This class implements the residual equations for a nonlinear coupled system
    with adjustable powers for the parameters. The powers of the parameters are
    set automatically during initialization.

    The system is governed by the following equations:

    .. math::
        f_1(x_1, x_2) = a^p \cdot x_1^2 + x_2^2 - 1 \\
        f_2(x_1, x_2) = x_1^2 - b^q \cdot x_2^2 - 1

    where:

    - :math:`x_1, x_2`: State variables.
    - :math:`a, b`: Parameters.
    - :math:`p, q`: Powers of the parameters.

    Parameters
    ----------
    backend : Backend
        Backend for numerical computations.
    """

    def __init__(self, backend: Backend[Array]) -> None:
        """
        Initialize the nonlinear coupled equations residual.

        Parameters
        ----------
        backend : Backend
            Backend for numerical computations.

        Raises
        ------
        RuntimeError
            If the powers of the parameters are less than 1.
        """
        self._bkd = backend
        self._set_param_powers()
        if self._apow < 1 or self._bpow < 1:
            raise RuntimeError("apow and bpow must be >= 1")

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
        Return the number of state variables in the residual equation.

        Returns
        -------
        nstates : int
            The number of state variables, which is 2 in this case.
        """
        return 2

    def nvars(self) -> int:
        """
        Return the number of uncertain variables in the model.

        Returns
        -------
        nvars : int
            Number of uncertain variables in the model. For this model, it is
            always 2.
        """
        return 2

    def _set_param_powers(self) -> None:
        """
        Set the powers of the parameters.

        This method sets the powers of the parameters :math:`a` and :math:`b`
        to predefined values.
        """
        self._apow = 1
        self._bpow = 1

    def value(self, state: Array, param: Array) -> Array:
        r"""
        Compute the residuals for the nonlinear coupled system.

        Parameters
        ----------
        state : Array
            Array of shape (nvars,) containing the state variables.
        param : Array
            Array containing the model parameters.

        Returns
        -------
        residuals : Array
            Array of shape (nvars,) containing the residuals.

        Notes
        -----
        The residual equations are defined as:

        .. math::
            f_1(x_1, x_2) = a^p \cdot x_1^2 + x_2^2 - 1 \\
            f_2(x_1, x_2) = x_1^2 - b^q \cdot x_2^2 - 1
        """
        a, b = param
        return self._bkd.stack(
            [
                a**self._apow * state[0] ** 2 + state[1] ** 2 - 1,
                state[0] ** 2 - b**self._bpow * state[1] ** 2 - 1,
            ],
            axis=0,
        )

    def solve(self, init_state: Array, param: Array) -> Array:
        """
        Solve the residual equation for the given initial state and parameters.

        Parameters
        ----------
        init_state : Array
            Initial state (ignored for this nonlinear problem).
        param : Array
            Parameters of the system.

        Returns
        -------
        Array
            Solution to the residual equation.
        """
        # For this nonlinear problem, solving requires external methods.
        # Here, we simply return the initial state as a placeholder.
        return init_state

    def state_jacobian(self, state: Array, param: Array) -> Array:
        r"""
        Compute the Jacobian of the residuals with respect to the states.

        Parameters
        ----------
        state : Array
            Array of shape (nvars,) containing the state variables.
        param : Array
            Array containing the model parameters.

        Returns
        -------
        jacobian : Array
            Array of shape (nvars, nvars) containing the Jacobian matrix.

        Notes
        -----
        The Jacobian matrix is defined as:

        .. math::
            J = \begin{bmatrix}
                2 a^p x_1 & 2 x_2 \\
                2 x_1 & -2 b^q x_2
            \end{bmatrix}
        """
        a, b = param
        return self._bkd.stack(
            [
                self._bkd.hstack([2 * a**self._apow * state[0], 2 * state[1]]),
                self._bkd.hstack(
                    [2 * state[0], -2 * b**self._bpow * state[1]]
                ),
            ],
            axis=0,
        )

    def param_jacobian(self, state: Array, param: Array) -> Array:
        r"""
        Compute the Jacobian of the residuals with respect to the parameters.

        Parameters
        ----------
        state : Array
            Array of shape (nvars,) containing the state variables.
        param : Array
            Array containing the model parameters.

        Returns
        -------
        param_jacobian : Array
            Array of shape (nvars, nparams) containing the parameter Jacobian
            matrix.

        Notes
        -----
        The parameter Jacobian matrix is defined as:

        .. math::
            J_p = \begin{bmatrix}
                p a^{p-1} x_1^2 & 0 \\
                0 & -q b^{q-1} x_2^2
            \end{bmatrix}
        """
        zero = self._bkd.zeros((1,))
        a, b = param
        return self._bkd.stack(
            [
                self._bkd.hstack(
                    [
                        self._apow * a ** (self._apow - 1) * state[0] ** 2,
                        zero,
                    ]
                ),
                self._bkd.hstack(
                    [
                        zero,
                        -self._bpow * b ** (self._bpow - 1) * state[1] ** 2,
                    ]
                ),
            ],
            axis=0,
        )

    def state_state_hvp(
        self, state: Array, param: Array, adj_state: Array, wvec: Array
    ) -> Array:
        """
        Compute the Hessian-vector product with respect to the state.

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

        Returns
        -------
        Array
            Hessian-vector product result.
        """
        return self._bkd.zeros((self.nvars(),))

    def __repr__(self) -> str:
        """
        Return a string representation of the class.

        Returns
        -------
        repr : str
            String representation of the class, including the parameter values.
        """
        if not hasattr(self, "_a"):
            return "{0}()".format(self.__class__.__name__)
        return "{0}(a={1}, b={2})".format(
            self.__class__.__name__, self._a, self._b
        )
