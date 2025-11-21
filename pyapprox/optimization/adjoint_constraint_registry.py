from pyapprox.util.backends.template import Array, BackendMixin
from pyapprox.optimization.adjoint import (
    AdjointResidualEquationWithHessian,
    NewtonResidualWithGradient,
)


class LinearResidualEquation(AdjointResidualEquationWithHessian):
    def __init__(self, Amat: Array, bvec: Array, backend: BackendMixin):
        super().__init__(backend)
        if bvec.ndim != 1:
            raise ValueError(
                f"bvec must be a 1D array but has shape {bvec.shape}"
            )
        if Amat.shape[0] != bvec.shape[0]:
            raise ValueError(
                "Amat and bvec must have the same number of rows"
                f"but had shapes {Amat.shape} and {bvec.shape}"
            )
        self._Amat = Amat
        self._bvec = bvec

    def nstates(self) -> int:
        return self._Amat.shape[0]

    def nvars(self) -> int:
        return self._Amat.shape[1]

    def _value(self, state: Array, param: Array) -> Array:
        return state - self._Amat @ param

    def _solve(self, init_state: Array, param: Array):
        # init_state is ignored for this linear problem
        return self._Amat @ param

    def _param_jacobian(self, state: Array, param: Array):
        return -self._Amat

    def _state_jacobian(self, state: Array, param: Array):
        return self._bkd.eye(self.nstates())

    def param_param_hvp(
        self, fwd_state: Array, param: Array, adj_state: Array, vvec: Array
    ) -> Array:
        return self._bkd.zeros((self.nvars(),))

    def _state_state_hvp(
        self, fwd_state: Array, param: Array, adj_state: Array, wvec: Array
    ) -> Array:
        return self._bkd.zeros((self.nstates(),))

    def _param_state_hvp(
        self, fwd_state: Array, param: Array, adj_state: Array, wvec: Array
    ) -> Array:
        return self._bkd.zeros((self.nvars(),))

    def _state_param_hvp(
        self, fwd_state: Array, param: Array, adj_state: Array, vvec: Array
    ) -> Array:
        return self._bkd.zeros((self.nstates(),))
        pass


class NonLinearCoupledEquationsResidual(NewtonResidualWithGradient):
    r"""
    Nonlinear coupled equations residual with automatic differentiation.

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
    backend : BackendMixin
        Backend for numerical computations.
    """

    def __init__(self, backend: BackendMixin):
        """
        Initialize the nonlinear coupled equations residual with automatic
        differentiation.

        Parameters
        ----------
        backend : BackendMixin
            Backend for numerical computations.

        Raises
        ------
        RuntimeError
            If the powers of the parameters are less than 1.
        """
        super().__init__(backend)
        self._set_param_powers()
        if self._apow < 1 or self._bpow < 1:
            raise RuntimeError("apow and bpow must be >= 1")

    def _set_param_powers(self):
        """
        Set the powers of the parameters.

        This method sets the powers of the parameters :math:`a` and :math:`b`
        to predefined values.
        """
        self._apow = 2
        self._bpow = 3

    def __call__(self, iterate: Array) -> Array:
        r"""
        Compute the residuals for the nonlinear coupled system.

        Parameters
        ----------
        iterate : Array
            Array of shape (nvars,) containing the state variables.

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
        return self._bkd.stack(
            [
                self._a**self._apow * iterate[0] ** 2 + iterate[1] ** 2 - 1,
                iterate[0] ** 2 - self._b**self._bpow * iterate[1] ** 2 - 1,
            ],
            axis=0,
        )

    def set_param(self, param: Array):
        """
        Set the model parameters.

        Parameters
        ----------
        param : Array
            Array containing the model parameters.
        """
        self._param = param
        self._a, self._b = self._param

    def nvars(self) -> int:
        """
        Return the number of uncertain variables in the model.

        Returns
        -------
        nvars : int
            Number of uncertain variables in the model. For this model, it is always 2.
        """
        return 2

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

    def _jacobian(self, iterate: Array) -> Array:
        r"""
        Compute the Jacobian of the residuals with respect to the states.

        Parameters
        ----------
        iterate : Array
            Array of shape (nvars,) containing the state variables.

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
        return self._bkd.stack(
            [
                self._bkd.hstack(
                    [2 * self._a**self._apow * iterate[0], 2 * iterate[1]]
                ),
                self._bkd.hstack(
                    [2 * iterate[0], -2 * self._b**self._bpow * iterate[1]]
                ),
            ],
            axis=0,
        )

    def _param_jacobian(self, iterate: Array) -> Array:
        r"""
        Compute the Jacobian of the residuals with respect to the parameters.

        Parameters
        ----------
        iterate : Array
            Array of shape (nvars,) containing the state variables.

        Returns
        -------
        param_jacobian : Array
            Array of shape (nvars, nparams) containing the parameter Jacobian matrix.

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
        return self._bkd.stack(
            [
                self._bkd.hstack(
                    [
                        self._apow
                        * self._a ** (self._apow - 1)
                        * iterate[0] ** 2,
                        zero,
                    ]
                ),
                self._bkd.hstack(
                    [
                        zero,
                        -self._bpow
                        * self._b ** (self._bpow - 1)
                        * iterate[1] ** 2,
                    ]
                ),
            ],
            axis=0,
        )
