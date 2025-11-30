from pyapprox.typing.util.backend import Array
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
