from typing import Tuple

from pyapprox.typing.util.backends.protocols import Array
from pyapprox.typing.pde.time.implicit_steppers.protocols import (
    ImplicitTimeSteppingResidualBase,
)


class BackwardEulerResidual(ImplicitTimeSteppingResidualBase[Array]):
    def __call__(self, state: Array) -> Array:
        self._residual.set_time(self._time + self._deltat)
        return state - self._prev_state - self._deltat * self._residual(state)

    def jacobian(self, state: Array) -> Array:
        self._residual.set_time(self._time + self._deltat)
        return self._residual.mass_matrix(
            state.shape[0]
        ) - self._deltat * self._residual.jacobian(state)

    def jacobian_wrt_parameters(self, fsol_nm1: Array, fsol_n: Array) -> Array:
        """
        Compute the Jacobian of the residual with respect to parameters.

        Parameters
        ----------
        fsol_nm1 : Array
            Solution at the previous time step.
        fsol_n : Array
            Solution at the current time step.

        Returns
        -------
        Array
            Parameter Jacobian matrix.
        """
        self._residual.set_time(self._time + self._deltat)
        return -self._deltat * self._residual.param_jacobian(fsol_n)

    def quadrature_samples_weights(self, times: Array) -> Tuple[Array, Array]:
        """
        Compute quadrature samples and weights for the time integration.

        Parameters
        ----------
        times : Array
            Time points.

        Returns
        -------
        Tuple[Array, Array]
            Quadrature samples and weights.
        """
        # Use PiecewiseConstantRight class for quadrature computation
        piecewise_constant_right = PiecewiseConstantRight(times, self._bkd)
        quadrature_points, quadrature_weights = (
            piecewise_constant_right.quadrature_rule()
        )
        return quadrature_points, quadrature_weights

    def adjoint_diagonal_jacobian(self, fsol_n: Array) -> Array:
        """
        Compute the diagonal Jacobian for adjoint computations.

        Parameters
        ----------
        fsol_n : Array
            Solution at the current time step.

        Returns
        -------
        Array
            Diagonal Jacobian matrix for adjoint computations.
        """
        self._residual.set_time(self._time)
        return (
            self._residual.mass_matrix(fsol_n.shape[0])
            - self._deltat * self._residual.jacobian(fsol_n)
        ).T

    def adjoint_off_diagonal_jacobian(
        self, fsol_n: Array, deltat_np1: float
    ) -> Array:
        """
        Compute the off-diagonal Jacobian for adjoint computations.

        Parameters
        ----------
        fsol_n : Array
            Solution at the current time step.
        deltat_np1 : float
            Time step size for the next time step.

        Returns
        -------
        Array
            Off-diagonal Jacobian matrix for adjoint computations.
        """
        return -self._residual.mass_matrix(fsol_n.shape[0]).T
