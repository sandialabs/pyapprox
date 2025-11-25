from typing import Protocol, Callable, runtime_checkable

from pyapprox.typing.util.backend import Array, Backend
from pyapprox.typing.interface.functions.function import (
    FunctionFromCallable,
    FunctionProtocol,
    validate_sample,
)


class JacobianProtocol(Protocol):
    """
    Protocol for functions with Jacobian functionality.
    """

    def jacobian(self, sample: Array) -> Array:
        pass


class ApplyJacobianProtocol(Protocol):
    """
    Protocol for functions that implement Jacobian-vector product.
    """

    def apply_jacobian(self, sample: Array, vec: Array) -> Array:
        """
        Apply the Jacobian to a vector.

        Parameters
        ----------
        x : Array
            Input array of shape (nvars,).
        vce : Array
            Vector to apply the Jacobian to, of shape (nvars,).

        Returns
        -------
        Array
            Result of Jacobian-vector product of shape (nqoi,).
        """
        ...


@runtime_checkable
class FunctionWithJacobianProtocol(
    FunctionProtocol, JacobianProtocol, Protocol
):
    pass


@runtime_checkable
class FunctionWithApplyJacobianProtocol(
    FunctionProtocol, ApplyJacobianProtocol, Protocol
):
    pass


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


def validate_vector_for_apply(nvars: int, vec: Array) -> None:
    """
    Validate that the vector has the correct shape for apply operations
    (e.g., apply_jacobian).

    Parameters
    ----------
    nvars : int
        The expected number of variables (length of the vector).
    vec : Array
        The input vector to validate.

    Raises
    ------
    ValueError
        If the vector does not have the expected shape.
    """
    if vec.shape != (nvars, 1):
        raise ValueError(
            f"Invalid vector shape for apply operation: expected ({nvars}, 1), "
            f"got {vec.shape}."
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
        self.validate_samples(sample)
        jac = self._jacobian(sample)
        validate_jacobian(self.nqoi(), self.nvars(), jac)
        return jac

    def jacobians(self, samples: Array) -> Array:
        self.validate_samples(samples)
        jacs = self._bkd.stack([self.jacobian(sample) for sample in samples.T])
        validate_jacobians(self.nqoi(), self.nvars(), samples, jacs)
        return jacs
