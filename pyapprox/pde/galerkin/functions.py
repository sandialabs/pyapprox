from abc import ABC, abstractmethod
from functools import partial

import numpy as np


class FEMScalarFunction(ABC):

    @abstractmethod
    def _values(self, xx: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, xx: np.ndarray) -> np.ndarray:
        vals = self._values(xx)
        if xx.ndim - 1 != vals.ndim:
            raise ValueError(
                f"vals.ndim is incorrect. Was {vals.ndim} "
                f"should be {xx.ndim-1}"
            )
        return vals

    def __repr__(self) -> str:
        return "{0}".format(self.__class__.__name__)


class FromCallableMixin:
    def __init__(self, fun: callable, name: str = None):
        self._fun = fun
        self._name = name

    def _values(self, xx: np.ndarray) -> np.ndarray:
        return self._fun(xx)

    def __repr__(self):
        return "{0}(name={1})".format(self.__class__.__name__, self._name)


class FEMScalarFunctionFromCallable(FromCallableMixin, FEMScalarFunction):
    pass


class FEMFunctionTransientMixin:
    def set_time(self, time: float):
        self._time = time

    def __repr__(self) -> str:
        return "{0}(name={1}, time={2})".format(
            self.__class__.__name__, self._name, self._time
        )


class FEMTransientScalarFunction(FEMFunctionTransientMixin, FEMScalarFunction):
    pass


class FEMTransientScalarFunctionFromCallable(
    FEMTransientScalarFunction, FromCallableMixin
):
    def _values(self, samples: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_time"):
            raise ValueError(f"{self}: must call set_time before calling eval")
        return self._partial_fun(samples)
        # return self._eval(samples)

    def set_time(self, time: float):
        super().set_time(time)
        self._partial_fun = partial(self._fun, time=time)

    # def _eval(self, samples: np.ndarray) -> np.ndarray:
    #     if self._time is None:
    #         raise ValueError("Must call set_time before calling eval")
    #     return self._partial_fun(samples)


class FEMVectorFunction:
    def __init__(self, swapaxes: bool = True):
        self._swapaxes = swapaxes

    @abstractmethod
    def _values(self, xx: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, xx: np.ndarray) -> np.ndarray:
        vals = self._values(xx)
        # if self._fun is not implemented correctly this will fail
        # E.g. when manufactured solution, diff etc. string does not have x
        # in it. If not dependent on x then must use 1e-16*x
        if xx.ndim != vals.ndim:
            raise ValueError(
                f"vals.ndim is incorrect. Was {vals.ndim} "
                f"should be {xx.ndim}"
            )
        # put vals in format required by FEM
        if self._swapaxes:
            vals = np.swapaxes(vals, 0, 1)
        return vals

    def __repr__(self) -> str:
        return "{0}".format(self.__class__.__name__)


class FEMVectorFunctionFromCallable(FromCallableMixin, FEMVectorFunction):
    def __init__(self, fun: callable, name: str = None, swapaxes: bool = True):
        FEMVectorFunction.__init__(self, swapaxes)
        FromCallableMixin.__init__(self, fun, name)


class FEMTransientVectorFunctionFromCallable(
    FEMFunctionTransientMixin, FEMVectorFunctionFromCallable
):
    def _values(self, samples: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_time"):
            raise ValueError(f"{self}: must call set_time before calling eval")
        return self._partial_fun(samples)
        # return self._eval(samples)

    def set_time(self, time: float):
        super().set_time(time)
        self._partial_fun = partial(self._fun, time=time)


class FEMNonLinearOperator(ABC):
    @abstractmethod
    def __call__(self, xx: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def jacobian(self, xx: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class FEMNonLinearOperatorFromCallable(FEMNonLinearOperator):
    def __init__(self, fun: callable, fun_prime: callable):
        self._fun = fun
        if fun_prime is None:
            fun_prime = self._zero_fun
        self._fun_prime = fun_prime

    def _zero_fun(self, x: np.ndarray, *args) -> np.ndarray:
        return x[0] * 0

    def __call__(self, xx: np.ndarray, sol: np.ndarray) -> np.ndarray:
        return self._fun(xx, sol)

    def jacobian(self, xx: np.ndarray, sol: np.ndarray) -> np.ndarray:
        return self._fun_prime(xx, sol)
