"""Generic optimizer result satisfying OptimizerResultProtocol."""

from typing import Generic

from pyapprox.util.backends.protocols import Array


class OptimizerResult(Generic[Array]):
    """Backend-generic optimizer result.

    Satisfies ``OptimizerResultProtocol`` without depending on scipy.

    Parameters
    ----------
    optima : Array
        Optimal parameter vector, shape (nvars, 1).
    fun : float
        Objective value at the optimum.
    success : bool
        Whether the optimizer converged.
    message : str
        Termination message.
    nit : int
        Number of iterations performed.
    """

    def __init__(
        self,
        optima: Array,
        fun: float,
        success: bool,
        message: str = "",
        nit: int = 0,
    ) -> None:
        self._optima = optima
        self._fun = fun
        self._success = success
        self._message = message
        self._nit = nit

    def fun(self) -> float:
        return self._fun

    def success(self) -> bool:
        return self._success

    def optima(self) -> Array:
        return self._optima

    def message(self) -> str:
        return self._message

    def nit(self) -> int:
        return self._nit

    def __repr__(self) -> str:
        return (
            f"OptimizerResult(fun={self._fun}, success={self._success}, "
            f"nit={self._nit})"
        )
