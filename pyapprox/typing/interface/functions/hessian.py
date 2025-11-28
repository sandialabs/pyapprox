from typing import Callable

from pyapprox.typing.util.backend import Array, Backend
from pyapprox.typing.interface.functions.function import validate_sample
from pyapprox.typing.interface.functions.jacobian import (
    FunctionWithJacobianFromCallable,
    validate_vector_for_apply,
)


def validate_hvp(nvars: int, hvp: Array) -> None:
    if hvp.shape != (nvars, 1):
        raise ValueError(
            f"Hvp shape mismatch: expected " f"({nvars, 1}), got {hvp.shape}"
        )


class FunctionWithJacobianAndHVPFromCallable(
    FunctionWithJacobianFromCallable[Array]
):
    def __init__(
        self,
        nvars: int,
        fun: Callable[[Array], Array],
        jacobian: Callable[[Array], Array],
        hvp: Callable[[Array, Array], Array],
        bkd: Backend[Array],
    ):
        super().__init__(1, nvars, fun, jacobian, bkd)
        if not callable(hvp):
            raise ValueError(
                "The provided 'hvp' object must be callable. "
                "Expected a callable object that takes two inputs of type "
                "'Array' and returns an output of type 'Array'. "
                f"Got an object of type {type(hvp).__name__}. "
                f"Object details: {self}"
            )
        self._hvp: Callable[[Array, Array], Array] = hvp

    def hvp(self, sample: Array, vec: Array) -> Array:
        validate_sample(self.nvars(), sample)
        validate_vector_for_apply(self.nvars(), vec)
        hvp = self._hvp(sample, vec)
        validate_hvp(self.nvars(), hvp)
        return hvp


class FunctionWithJacobianAndWeightedHVPFromCallable(
    FunctionWithJacobianFromCallable[Array]
):
    def __init__(
        self,
        nvars: int,
        fun: Callable[[Array], Array],
        jacobian: Callable[[Array], Array],
        whvp: Callable[[Array, Array, Array], Array],
        bkd: Backend[Array],
    ):
        super().__init__(1, nvars, fun, jacobian, bkd)
        if not callable(whvp):
            raise ValueError(
                "The provided 'whvp' object must be callable. "
                "Expected a callable object that takes three inputs of type "
                "'Array' and returns an output of type 'Array'. "
                f"Got an object of type {type(whvp).__name__}. "
                f"Object details: {self}"
            )
        self._whvp: Callable[[Array, Array, Array], Array] = whvp

    def weighted_hvp(self, sample: Array, vec: Array, weights: Array) -> Array:
        validate_sample(self.nvars(), sample)
        validate_vector_for_apply(self.nvars(), vec)
        validate_sample(self.nqoi(), weights)
        whvp = self._whvp(sample, vec, weights)
        validate_hvp(self.nvars(), whvp)
        return whvp

    def hvp(self, sample: Array, vec: Array) -> Array:
        return self.weighted_hvp(sample, vec, self.bkd().array([[1.0]]))
