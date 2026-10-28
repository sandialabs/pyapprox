from typing import Generic

from pyapprox.ode.mass_matrix import IdentityMassMatrix
from pyapprox.ode.mixins.default_newton_jacobian import (
    DefaultNewtonJacobianMixin,
)
from pyapprox.ode.residual_derivatives import (
    STATED_ZERO,
    ODEResidualDerivatives,
    backend_spaces,
    dense_jacobian,
    dense_operator,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import DenseFormat


class NonLinearDecoupledODE(DefaultNewtonJacobianMixin[Array], Generic[Array]):
    r"""
    Nonlinear decoupled ODE system.

    The system is defined as:
        f(x, t) = -b * x**2,
    where b is a coefficient that depends on the state index and time:
    :math:`b_i = c^2 i \tau(t)` with :math:`\tau = 2 + t` when
    ``transient_coef`` and 1 otherwise. The parameters are
    :math:`p = (c, y_0)`: the coefficient and the value of every entry of
    the initial state, so :math:`\partial y_0/\partial p = [0, 1]`. The
    derivatives bundle holds all first and second derivatives; the
    initial state is linear in :math:`p`, so its curvature is a stated
    zero.

    Parameters
    ----------
    nstates : int
        Number of states in the system.
    transient_coef : bool
        Whether the coefficient b depends on time.
    backend : Backend[Array]
        Backend used for computations (e.g., NumPy, PyTorch).

    Attributes
    ----------
    _nstates : int
        Number of states in the system.
    _transient_coef : bool
        Whether the coefficient b depends on time.
    _bkd : Backend[Array]
        Backend used for computations.
    _coef : Array
        Coefficient for the ODE system.
    _init_cond : Array
        Initial condition for the ODE system.
    """

    def __init__(self, nstates: int, transient_coef: bool, bkd: Backend[Array]):
        self._parameters_version = 0
        self._nstates = nstates
        self._transient_coef = transient_coef
        self._bkd = bkd
        self.set_param(self._bkd.array([0.0, 0.0]))
        self._mass = IdentityMassMatrix(nstates, bkd)
        states, params = backend_spaces(bkd, nstates, 2)
        self._derivatives = ODEResidualDerivatives.second_order(
            states,
            params,
            jacobian=dense_jacobian(self.jacobian, states, bkd),
            param=dense_operator(self.param_jacobian, params, states, bkd),
            initial_state=dense_operator(
                self._initial_state_matrix, params, states, bkd
            ),
            initial_state_curvature=STATED_ZERO,
            state_state=dense_operator(
                self._state_state_matrix, states, states, bkd, symmetric=True
            ),
            state_param=dense_operator(
                self._state_param_matrix, params, states, bkd
            ),
            param_param=dense_operator(
                self._param_param_matrix, params, params, bkd, symmetric=True
            ),
        )

    def bkd(self) -> Backend[Array]:
        """
        Get the backend used for computations.

        Returns
        -------
        Backend[Array]
            Backend used for computations.
        """
        return self._bkd

    def nvars(self) -> int:
        """
        Get the number of variables in the system.

        Returns
        -------
        int
            Number of variables.
        """
        return self._nstates

    def nparams(self) -> int:
        """The number of parameters, :math:`(c, y_0)`."""
        return 2

    def set_param(self, param: Array) -> None:
        """
        Set the parameters for the ODE system.

        Parameters
        ----------
        param : Array
            Parameter vector [coef, init_cond]. Shape: (2,)
        """
        self._coef = param[0]
        self._init_cond = param[1]
        self._parameters_version += 1

    def parameters_version(self) -> int:
        """The number of ``set_param`` calls so far."""
        return self._parameters_version

    def derivatives(self) -> ODEResidualDerivatives[Array, DenseFormat[Array]]:
        """Every first and second derivative as a dense operator; the
        initial-state curvature is a stated zero."""
        return self._derivatives

    def _rate(self, nstates: int, time: float) -> Array:
        r""":math:`b / c^2 = i \tau(t)` for each state index."""
        rate = self._bkd.arange(1, nstates + 1, dtype=self._bkd.double_dtype())
        if self._transient_coef:
            rate = rate * (2 + time)
        return rate

    def param_jacobian(self, state: Array, time: float) -> Array:
        r""":math:`\partial f/\partial p`, of shape (nstates, 2)."""
        rate = self._rate(state.shape[0], time)
        dcoef = -2 * self._coef * rate * state**2
        return self._bkd.stack([dcoef, self._bkd.zeros(state.shape)], axis=1)

    def _initial_state_matrix(self) -> Array:
        r""":math:`\partial y_0/\partial p = [0, 1]`."""
        bkd = self._bkd
        return bkd.stack(
            [bkd.zeros((self._nstates,)), bkd.ones((self._nstates,))], axis=1
        )

    def _state_state_matrix(self, state: Array, time: float, adj: Array) -> Array:
        r""":math:`\lambda^\top \partial^2 f/\partial y^2 = \mathrm{diag}(-2 b
        \lambda)`."""
        rate = self._rate(state.shape[0], time)
        return self._bkd.diag(-2 * self._coef**2 * rate * adj)

    def _state_param_matrix(self, state: Array, time: float, adj: Array) -> Array:
        r""":math:`\lambda^\top \partial^2 f/\partial y\,\partial p`, of shape
        (nstates, 2)."""
        rate = self._rate(state.shape[0], time)
        dcoef = -4 * self._coef * rate * state * adj
        return self._bkd.stack([dcoef, self._bkd.zeros(state.shape)], axis=1)

    def _param_param_matrix(self, state: Array, time: float, adj: Array) -> Array:
        r""":math:`\lambda^\top \partial^2 f/\partial p^2`; only the
        coefficient's entry is nonzero."""
        bkd = self._bkd
        rate = self._rate(state.shape[0], time)
        dcoef2 = bkd.sum(-2 * rate * state**2 * adj)
        zero = bkd.zeros(())
        return bkd.stack([bkd.stack([dcoef2, zero]), bkd.stack([zero, zero])])

    def __call__(self, state: Array, time: float) -> Array:
        """
        Evaluate the ODE residual.

        Parameters
        ----------
        state : Array
            Current solution.
        time : float
            Current time.

        Returns
        -------
        Array
            Residual evaluated at the current solution.
        """
        b = self._coef**2 * self._rate(state.shape[0], time)
        return -b * state**2

    def jacobian(self, state: Array, time: float) -> Array:
        """
        Compute the Jacobian of the ODE residual with respect to the solution.

        Parameters
        ----------
        state : Array
            Current solution.
        time : float
            Current time.

        Returns
        -------
        Array
            Jacobian matrix.
        """
        b = self._coef**2 * self._rate(state.shape[0], time)
        return self._bkd.diag(-2 * b * state)

    def mass_matrix(self) -> IdentityMassMatrix[Array]:
        return self._mass
