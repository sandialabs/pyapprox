from abc import ABC, abstractmethod
from typing import Callable, Generic

from pyapprox.typing.util.backend import Array, Backend


from typing import Protocol, Generic
from pyapprox.interface.functions.backend import Array, Backend


class FunctionProtocol(Protocol, Generic[Array]):
    """
    A protocol defining the required interface for a Function.
    """

    _bkd: Backend[Array]

    def nvars(self) -> int:
        """
        Return the number of variables in the function.
        """
        ...

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest in the function.
        """
        ...

    def __call__(self, samples: Array) -> Array:
        """
        Evaluate the function with the given samples.
        """
        ...


class Function(ABC, Generic[Array]):
    def __init__(self, bkd: Backend[Array]):
        self._bkd = bkd

    @abstractmethod
    def nvars(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def nqoi(self) -> int:
        raise NotImplementedError

    def validate_samples(self, samples: Array) -> None:
        if samples.shape != (self.nvars(), self.nqoi()):
            raise ValueError

    def validate_values(self, samples: Array, values: Array) -> None:
        if values.shape != (samples.shape[1], self.nqoi()):
            raise ValueError

    @abstractmethod
    def __call__(self, samples: Array) -> Array:
        raise NotImplementedError


class FunctionFromCallable(Function[Array]):
    def __init__(self, nqoi: int, nvars: int, fun: Callable[[Array], Array]):
        self._nqoi = nqoi
        self._nvars = nvars
        self._fun = fun

    def nvars(self) -> int:
        return self._nvars

    @abstractmethod
    def nqoi(self) -> int:
        return self._nqoi

    def __call__(self, samples: Array) -> Array:
        self.validate_samples(samples)
        values = self._fun(samples)
        self.validate_values(samples, values)
        return values


def validate_jacobian(nqoi: int, nvars: int, jac: Array) -> None:
    if jac.shape != (nqoi, nvars):
        raise ValueError(
            f"Jacobian shape mismatch: expected ({nqoi, nvars}), "
            f"got {jac.shape}"
        )


def validate_jacobians(
    nqoi: int, nvars: int, samples: Array, jac: Array
) -> None:
    if jac.shape != (samples.shape[1], nqoi, nvars):
        raise ValueError(
            f"Jacobian shape mismatch: expected "
            f"({samples.shape[1], nqoi, nvars}), got {jac.shape}"
        )


class FunctionWithJacobian(Function[Array]):
    @abstractmethod
    def jacobian(self, sample: Array) -> Array:
        raise NotImplementedError

    def jacobians(self, samples: Array) -> Array:
        return self._bkd.stack(
            [self.jacobian(samples) for sample in samples.T]
        )


class FunctionWithJacobianFromCallable(FunctionFromCallable[Array]):
    def __init__(
        self,
        nqoi: int,
        nvars: int,
        fun: Callable[[Array], Array],
        jacobian: Callable[[Array], Array],
    ):
        self._nqoi = nqoi
        self._nvars = nvars
        self._fun = fun
        self._jacobian = jacobian

    def nvars(self) -> int:
        return self._nvars

    @abstractmethod
    def nqoi(self) -> int:
        return self._nqoi

    def jacobian(self, samples: Array) -> Array:
        self.validate_samples(samples)
        values = self._jacobian(samples)
        validate_jacobian(self.nqoi(), self.nvars(), values)
        return values

    def jacobians(self, samples: Array) -> Array:
        self.validate_samples(samples)
        jacs = self._bkd.stack(
            [self.jacobian(samples) for sample in samples.T]
        )
        validate_jacobians(self.nqoi(), self.nvars(), samples, jacs)
        return jacs


class FunctionWithHVP(FunctionWithJacobian[Array]):
    @abstractmethod
    def apply_hessian(self, samples: Array, vec: Array) -> Array:
        raise NotImplementedError
