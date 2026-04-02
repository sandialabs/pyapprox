from typing import Generic

from pyapprox.interface.functions.protocols.validation import (
    validate_sample,
)
from pyapprox.optimization.implicitfunction.state_equations.wrappers import (
    ParameterizedStateEquationAsNewtonEquation,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.validation import validate_backend
from pyapprox.util.rootfinding.newton import (
    NewtonSolver,
    NewtonSolverOptions,
)


class NonLinearCoupledStateEquations(Generic[Array]):
    r"""
    Nonlinear coupled equations residual with Jacobian and Hessian
    capabilities.

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

    def __init__(
        self,
        bkd: Backend[Array],
        newton_options: NewtonSolverOptions = NewtonSolverOptions(),
    ) -> None:
        """
        Initialize the nonlinear coupled equations residual.

        Parameters
        ----------
        bkd : Backend
            Backend for numerical computations.

        Raises
        ------
        RuntimeError
            If the powers of the parameters are less than 1.
        """
        validate_backend(bkd)
        self._bkd = bkd
        self._set_param_powers()
        if self._apow < 1 or self._bpow < 1:
            raise RuntimeError("apow and bpow must be >= 1")
        self._newton_equation = ParameterizedStateEquationAsNewtonEquation(
            self, self.bkd().zeros((self.nparams(), 1))
        )
        self._newton_solver = NewtonSolver(self._newton_equation)
        self._newton_solver.set_options(**vars(newton_options))

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
        Return the number of uncertain variables in the model.

        Returns
        -------
        nparams : int
            Number of uncertain variables in the model. For this model, it is
            always 2.
        """
        return 2

    def nstates(self) -> int:
        """
        Return the number of state variables in the residual equation.

        Returns
        -------
        nstates : int
            The number of state variables, which is 2 in this case.
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

    def __call__(self, state: Array, param: Array) -> Array:
        r"""
        Compute the residuals for the nonlinear coupled system.

        Parameters
        ----------
        state : Array
            Array of shape (nparams,) containing the state variables.
        param : Array
            Array containing the model parameters.

        Returns
        -------
        residuals : Array
            Array of shape (nparams,) containing the residuals.

        Notes
        -----
        The residual equations are defined as:

        .. math::
            f_1(x_1, x_2) = a^p \cdot x_1^2 + x_2^2 - 1 \\
            f_2(x_1, x_2) = x_1^2 - b^q \cdot x_2^2 - 1
        """
        validate_sample(self.nstates(), state)
        validate_sample(self.nparams(), param)
        a, b = param[:, 0]
        value = self._bkd.stack(
            [
                a**self._apow * state[0] ** 2 + state[1] ** 2 - 1,
                state[0] ** 2 - b**self._bpow * state[1] ** 2 - 1,
            ],
            axis=0,
        )
        return value

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
        validate_sample(self.nstates(), init_state)
        validate_sample(self.nparams(), param)
        self._newton_equation.set_parameters(param)
        return self._newton_solver.solve(init_state[:, 0])[:, None]

    def state_jacobian(self, state: Array, param: Array) -> Array:
        r"""
        Compute the Jacobian of the residuals with respect to the states.

        Parameters
        ----------
        state : Array
            Array of shape (nparams,) containing the state variables.
        param : Array
            Array containing the model parameters.

        Returns
        -------
        jacobian : Array
            Array of shape (nparams, nparams) containing the Jacobian matrix.

        Notes
        -----
        The Jacobian matrix is defined as:

        .. math::
            J = \begin{bmatrix}
                2 a^p x_1 & 2 x_2 \\
                2 x_1 & -2 b^q x_2
            \end{bmatrix}
        """
        validate_sample(self.nstates(), state)
        validate_sample(self.nparams(), param)
        a, b = param[:, 0]
        return self._bkd.stack(
            [
                self._bkd.hstack([2 * a**self._apow * state[0], 2 * state[1]]),
                self._bkd.hstack([2 * state[0], -2 * b**self._bpow * state[1]]),
            ],
            axis=0,
        )

    def param_jacobian(self, state: Array, param: Array) -> Array:
        r"""
        Compute the Jacobian of the residuals with respect to the parameters.

        Parameters
        ----------
        state : Array
            Array of shape (nparams,) containing the state variables.
        param : Array
            Array containing the model parameters.

        Returns
        -------
        param_jacobian : Array
            Array of shape (nparams, nparams) containing the parameter Jacobian
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
        validate_sample(self.nstates(), state)
        validate_sample(self.nparams(), param)
        zero = self._bkd.zeros((1,))
        a, b = param[:, 0]
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
        validate_sample(self.nstates(), state)
        validate_sample(self.nparams(), param)
        a, b = param[:, 0]
        w1, w2 = wvec[:, 0]  # Extract components of the input vector

        # Compute Nabla_x (J_x @ w)
        grad_Jx_w = self._bkd.stack(
            [
                self._bkd.hstack([2 * a**self._apow * w1, 2 * w2]),
                self._bkd.hstack([2 * w1, -2 * b**self._bpow * w2]),
            ],
            axis=0,
        )  # Shape (nstates, nstates)

        # Compute delta @ grad_Jx_w
        hvp = adj_state.T @ grad_Jx_w  # Shape (1, nstates)

        # Return the hvp
        return hvp.T

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
        validate_sample(self.nstates(), state)
        validate_sample(self.nparams(), param)
        a, b = param[:, 0]
        h11 = self._apow * (self._apow - 1) * a ** (self._apow - 2) * state[0] ** 2
        h22 = -self._bpow * (self._bpow - 1) * b ** (self._bpow - 2) * state[1] ** 2

        # Compute the Hessian-vector product
        return self._bkd.stack(
            [h11 * adj_state[0] * vvec[0], h22 * adj_state[1] * vvec[1]],
            axis=0,
        )

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
        validate_sample(self.nstates(), state)
        validate_sample(self.nparams(), param)
        a, b = param[:, 0]
        va, vb = vvec  # Extract components of the input vector

        # Compute Nabla_x (J_p @ v)
        grad_Jp_v = self._bkd.stack(
            [
                self._bkd.hstack(
                    [
                        2 * self._apow * a ** (self._apow - 1) * state[0, 0] * va,
                        self._bkd.zeros((1,)),
                    ]
                ),
                self._bkd.hstack(
                    [
                        self._bkd.zeros((1,)),
                        -2 * self._bpow * b ** (self._bpow - 1) * state[1, 0] * vb,
                    ]
                ),
            ],
            axis=0,
        )  # Shape (nstates, nstates)

        # Compute delta @ grad_Jp_v
        hvp = adj_state.T @ grad_Jp_v
        return hvp.T

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
        validate_sample(self.nstates(), state)
        validate_sample(self.nparams(), param)
        a, b = param[:, 0]
        w1, w2 = wvec  # Extract components of the input vector

        # Compute Nabla_p (J_x @ w)
        grad_Jx_w = self._bkd.stack(
            [
                self._bkd.hstack(
                    [
                        2 * self._apow * a ** (self._apow - 1) * state[0, 0] * w1,
                        self._bkd.asarray(0.0),
                    ]
                ),
                self._bkd.hstack(
                    [
                        self._bkd.asarray(0.0),
                        -2 * self._bpow * b ** (self._bpow - 1) * state[1, 0] * w2,
                    ]
                ),
            ],
            axis=0,
        )

        # Compute delta @ grad_Jx_w
        hvp = adj_state.T @ grad_Jx_w

        return hvp.T

    def __repr__(self) -> str:
        """
        Return a string representation of the class.

        Returns
        -------
        repr : str
            String representation of the class, including the parameter values.
        """
        return (
            f"{self.__class__.__name__}("
            f"nstates={self.nstates()}, "
            f"nparams={self.nparams()}, "
            f"bkd={type(self._bkd).__name__})"
        )
