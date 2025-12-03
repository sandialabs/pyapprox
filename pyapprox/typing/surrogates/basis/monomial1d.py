from typing import Generic

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.interface.functions.protocols.validation import (
    validate_samples,
)


def validate_basis_matrix(samples: Array, nterms: int, basis_matrix):
    if basis_matrix.shape != (samples.shape[1], nterms):
        raise ValueError(
            "Invalid basis matrix shape: expected "
            f"{(samples.shape[1], nterms)}, "
            f"but got {basis_matrix.shape}."
        )


class MonomialBasis1D(Generic[Array]):
    """Univariate monomial basis."""

    def __init__(self, bkd: Backend[Array]):
        self._bkd = bkd
        self.set_nterms(0)

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def set_nterms(self, nterms: int):
        self._nterms = nterms

    def nterms(self) -> int:
        return self._nterms

    def __call__(self, samples: Array) -> Array:
        validate_samples(1, samples)
        basis_matrix = samples.T ** self._bkd.reshape(
            self._bkd.arange(self._nterms), (1, -1)
        )
        validate_basis_matrix(samples, self.nterms(), basis_matrix)
        return basis_matrix

    def _derivatives(self, samples: Array, order: int) -> Array:
        powers = self._bkd.hstack(
            (
                self._bkd.zeros((order,)),
                self._bkd.arange(self._nterms - order),
            )
        )
        # 1 x x^2 x^3  x^4 vals
        # 0 1 2x  3x^2 4x^3 1st derivs
        # 0 0 2   6x   12x^2  2nd derivs
        consts = self._bkd.hstack(
            (
                self._bkd.zeros((order,)),
                order * self._bkd.arange(1, self._nterms - order + 1),
            )
        )
        return (samples.T ** powers[None, :]) * consts

    def jacobians(self, samples: Array) -> Array:
        return self._derivatives(samples, 1)

    def hessians(self, samples: Array) -> Array:
        return self._derivatives(samples, 2)

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the object for debugging.
        """
        return (
            f"{self.__class__.__name__}("
            f"nterms={self.nterms()}, "
            f"bkd={type(self._bkd).__name__})"
        )
