"""ROL optimizer result wrapper satisfying OptimizerResultProtocol."""

from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend


class ROLOptimizerResult(Generic[Array]):
    """Wrapper for ROL optimization results.

    Satisfies OptimizerResultProtocol by providing fun(), success(), and
    optima() methods.
    """

    def __init__(
        self,
        x_array: Array,
        fun_value: float,
        success_flag: bool,
        bkd: Backend[Array],
        msg: str = "",
    ) -> None:
        self._x = bkd.asarray(x_array)[:, None]
        self._fun = fun_value
        self._success = success_flag
        self._bkd = bkd
        self._message = msg

    def fun(self) -> float:
        return self._fun

    def success(self) -> bool:
        return self._success

    def optima(self) -> Array:
        return self._x

    def message(self) -> str:
        return self._message
