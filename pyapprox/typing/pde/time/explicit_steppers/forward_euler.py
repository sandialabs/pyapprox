from typing import Generic

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.pde.time.explicit_steppers.protocols import (
    ExplicitODEResidualProtocol,
)


class ForwardEulerResidual(Generic[Array]):
    def __init__(self, residual: ExplicitODEResidualProtocol[Array]):
        self._residual = residual
        self._bkd = residual.bkd()

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def set_time(self, time: float, deltat: float, prev_state: Array) -> None:
        self._time = time
        self._deltat = deltat
        self._prev_state = prev_state

    def __call__(self, state: Array) -> Array:
        self._residual.set_time(self._time)
        return self._deltat * self._residual(self._prev_state)
