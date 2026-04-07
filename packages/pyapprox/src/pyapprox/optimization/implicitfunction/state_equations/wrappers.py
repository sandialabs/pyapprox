from typing import Generic

from pyapprox.interface.functions.protocols.validation import (
    validate_sample,
)
from pyapprox.optimization.implicitfunction.state_equations.protocols import (
    ParameterizedStateEquationWithJacobianProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class ParameterizedStateEquationAsNewtonEquation(Generic[Array]):
    def __init__(
        self,
        state_equations: ParameterizedStateEquationWithJacobianProtocol[Array],
        param: Array,
    ) -> None:
        """
        Initialize the parameterized state equation as a Newton equation.

        Parameters
        ----------
        state_equations : ParameterizedStateEquationWithJacobianProtocol
            State equations with Jacobian.
        param : Array
            Parameters for the state equations.
        """
        self._validate_state_equations(state_equations)
        self._state_eq = state_equations
        self._bkd = self._state_eq.bkd()
        self.set_parameters(param)

    def _validate_state_equations(
        self,
        state_equations: ParameterizedStateEquationWithJacobianProtocol[Array],
    ) -> None:
        """
        Validate that the provided state equations conform to the required
        protocol.

        Parameters
        ----------
        state_equations : ParameterizedStateEquationWithJacobianProtocol
            State equations to validate.

        Raises
        ------
        TypeError
            If the state equations do not conform to the required protocol.
        """
        if not isinstance(
            state_equations, ParameterizedStateEquationWithJacobianProtocol
        ):
            raise TypeError(
                "The provided state equations do not conform to the "
                "ParameterizedStateEquationWithJacobianProtocol."
            )

    def bkd(self) -> Backend[Array]:
        """
        Get the backend associated with the state equations.

        Returns
        -------
        Backend[Array]
            Backend for array operations.
        """
        return self._bkd

    def set_parameters(self, param: Array) -> None:
        """
        Set the parameters for the state equations.

        Parameters
        ----------
        param : Array
            Parameters for the state equations.
        """
        validate_sample(self._state_eq.nparams(), param)
        self._param = param

    def __call__(self, iterate1d: Array) -> Array:
        """
        Evaluate the state equations at the given iterate.

        Parameters
        ----------
        iterate : Array
            Current iterate.

        Returns
        -------
        Array
            Residual of the state equations.
        """
        if iterate1d.shape != (self._state_eq.nstates(),):
            raise ValueError(f"newton method uses 1D arrays but {iterate1d.shape=}")
        val = self._state_eq(iterate1d[:, None], self._param)
        if val.shape != (self._state_eq.nstates(), 1):
            raise ValueError(f"state equations mustretur 2D arrays but {val.shape=}")
        return val

    def jacobian(self, iterate1d: Array) -> Array:
        """
        Compute the Jacobian of the state equations at the given iterate.

        Parameters
        ----------
        iterate1d : Array
            Current iterate.

        Returns
        -------
        Array
            Jacobian matrix.
        """
        return self._state_eq.state_jacobian(iterate1d[:, None], self._param)

    def linsolve(self, iterate1d: Array, prev_residual: Array) -> Array:
        """
        Solve the linear system using the Jacobian and the previous residual.

        Parameters
        ----------
        iterate1d : Array
            Current iterate.
        prev_residual : Array
            Previous residual.

        Returns
        -------
        Array
            Solution to the linear system.
        """
        jacobian = self.jacobian(iterate1d)
        # newton requires 1D array as output of linsolve
        return self._bkd.solve(jacobian, prev_residual)[:, 0]
