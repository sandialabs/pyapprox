from typing import Callable

from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.interface.functions.protocols.validation import (
    validate_hvp,
    validate_sample,
    validate_vector_for_apply,
)
from pyapprox.util.backends.protocols import Array, Backend


class FunctionWithJacobianAndHVPFromCallable(FunctionWithJacobianFromCallable[Array]):
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
