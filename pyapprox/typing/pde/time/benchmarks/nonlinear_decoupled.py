from typing import Generic
from pyapprox.typing.util.backend import Array, Backend
from pyapprox.typing.pde.time.implicit_steppers.protocols import (
    ImplicitODEResidualProtocol,
)


class NonLinearDecoupledODE(
    Generic[Array], ImplicitODEResidualProtocol[Array]
):
    """
    Nonlinear decoupled ODE system.

    The system is defined as:
        f(x, t) = -b * x**2,
    where b is a coefficient that depends on the state index and time.

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
    _time : float
        Current time.
    _coef : float
        Coefficient for the ODE system.
    _init_cond : float
        Initial condition for the ODE system.
    """

    def __init__(
        self, nstates: int, transient_coef: bool, bkd: Backend[Array]
    ):
        self._nstates = nstates
        self._transient_coef = transient_coef
        self._bkd = bkd
        self._time = 0.0
        self.set_parameters(self._bkd.array([0.0, 0.0]))
        self._init_cond = None

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

    def set_time(self, time: float) -> None:
        """
        Set the current time.

        Parameters
        ----------
        time : float
            Current time.
        """
        self._time = time

    def set_parameters(self, param: Array) -> None:
        """
        Set the parameters for the ODE system.

        Parameters
        ----------
        param : Array
            Parameter vector [coef, init_cond].
        """
        self._coef = param[0]
        self._init_cond = param[1]

    def __call__(self, sol: Array) -> Array:
        """
        Evaluate the ODE residual.

        Parameters
        ----------
        sol : Array
            Current solution.

        Returns
        -------
        Array
            Residual evaluated at the current solution.
        """
        b = self._coef**2 * self._bkd.arange(
            1, sol.shape[0] + 1, dtype=self._bkd.double_dtype()
        )
        if self._transient_coef:
            b *= 2 + self._time
        return -b * sol**2

    def jacobian(self, sol: Array) -> Array:
        """
        Compute the Jacobian of the ODE residual with respect to the solution.

        Parameters
        ----------
        sol : Array
            Current solution.

        Returns
        -------
        Array
            Jacobian matrix.
        """
        b = self._coef**2 * self._bkd.arange(
            1, sol.shape[0] + 1, dtype=self._bkd.double_dtype()
        )
        if self._transient_coef:
            b *= 2 + self._time
        return self._bkd.diag(-2 * b * sol)

    def mass_matrix(self, nvars: int) -> Array:
        """
        Get the mass matrix for the ODE system.

        Parameters
        ----------
        nvars : int
            Number of variables.

        Returns
        -------
        Array
            Mass matrix.
        """
        return self._bkd.eye(nvars, dtype=self._bkd.double_dtype())
