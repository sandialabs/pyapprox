from typing import Protocol, Generic, runtime_checkable, cast

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class BisectionResidualProtocol(Protocol, Generic[Array]):
    def bkd(self) -> Backend[Array]: ...

    def __call__(self, iterate: Array) -> Array: ...


class BisectionSearch(Generic[Array]):
    def __init__(self, residual: BisectionResidualProtocol[Array]) -> None:
        if not isinstance(residual, BisectionResidualProtocol):
            raise ValueError("residual must be an instance of NewtonResidual")
        self._residual = residual
        self._bkd = residual.bkd()

    def _bisection_search(self, lb: Array, ub: Array) -> Array:
        it = 0
        residual_lb = self._residual(lb)
        residual_ub = self._residual(ub)
        if self._bkd.any_bool(
            cast(
                Array,
                self._bkd.sign(residual_lb) == self._bkd.sign(residual_ub),
            )
        ):
            raise RuntimeError
        while it < self._maxiters:
            center = (lb + ub) / 2
            residual_center = self._residual(center)
            residual_norm = self._bkd.norm(residual_center)
            if self._verbosity > 0:
                print("iter {0}: {1}".format(it, residual_norm))
            if residual_norm < self._atol:
                return center
            idx1 = self._bkd.nonzero(
                cast(
                    Array,
                    self._bkd.sign(residual_center)
                    == self._bkd.sign(residual_lb),
                )
            )[0]
            idx2 = self._bkd.nonzero(
                cast(
                    Array,
                    self._bkd.sign(residual_center)
                    != self._bkd.sign(residual_lb),
                )
            )[0]
            lb[idx1] = center[idx1]
            ub[idx2] = center[idx2]
            it += 1
        return center

    def solve(
        self,
        bounds: Array,
        maxiters: int = 30,
        verbosity: int = 0,
        atol: float = 1e-7,
    ) -> Array:
        if bounds.shape[1] != 2:
            raise ValueError("Must provide upper and lower bound")
        self._maxiters = maxiters
        self._verbosity = verbosity
        self._atol = atol
        return self._bisection_search(*bounds.T)
