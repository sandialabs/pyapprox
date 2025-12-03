from typing import Callable

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.interface.functions.fromcallable.function import (
    FunctionFromCallable,
)
from pyapprox.typing.interface.functions.protocols.validation import (
    validate_sample,
    validate_samples,
    validate_jacobian,
    validate_jacobians,
)


class FunctionWithJacobianFromCallable(FunctionFromCallable[Array]):
    def __init__(
        self,
        nqoi: int,
        nvars: int,
        fun: Callable[[Array], Array],
        jacobian: Callable[[Array], Array],
        bkd: Backend[Array],
    ):
        super().__init__(nqoi, nvars, fun, bkd)
        if not callable(jacobian):
            raise ValueError(
                "The provided 'jacobian' object must be callable. "
                "Expected a callable object that takes an input of type "
                "'Array' and returns an output of type 'Array'. "
                f"Got an object of type {type(jacobian).__name__}. "
                f"Object details: {self}"
            )
        self._jacobian: Callable[[Array], Array] = jacobian

    def nvars(self) -> int:
        return self._nvars

    def nqoi(self) -> int:
        return self._nqoi

    def jacobian(self, sample: Array) -> Array:
        validate_sample(self.nvars(), sample)
        jac = self._jacobian(sample)
        validate_jacobian(self.nqoi(), self.nvars(), jac)
        return jac

    def jacobians(self, samples: Array) -> Array:
        validate_samples(self.nvars(), samples)
        jacs = self._bkd.stack([self.jacobian(sample) for sample in samples.T])
        validate_jacobians(self.nqoi(), self.nvars(), samples, jacs)
        return jacs


class FunctionWithJVPFromCallable(FunctionFromCallable[Array]):
    def __init__(
        self,
        nqoi: int,
        nvars: int,
        fun: Callable[[Array], Array],
        jvp: Callable[[Array, Array], Array],
        bkd: Backend[Array],
    ):
        super().__init__(nqoi, nvars, fun, bkd)
        if not callable(jvp):
            raise ValueError(
                "The provided 'jvp' object must be callable. "
                "Expected a callable object that takes an input of type "
                "'Array' and returns an output of type 'Array'. "
                f"Got an object of type {type(jvp).__name__}. "
                f"Object details: {self}"
            )
        self._jvp: Callable[[Array, Array], Array] = jvp

    def nvars(self) -> int:
        return self._nvars

    def nqoi(self) -> int:
        return self._nqoi

    def jvp(self, sample: Array, vec: Array) -> Array:
        validate_sample(self.nvars(), sample)
        validate_sample(self.nvars(), vec)
        jvp = self._jvp(sample, vec)
        validate_sample(self.nqoi(), jvp)
        return jvp
