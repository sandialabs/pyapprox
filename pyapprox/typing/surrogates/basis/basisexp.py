from typing import Protocol, Generic, runtime_checkable

from pyapprox.typing.util.backend import Array, Backend
from pyapprox.typing.interface.functions.function import (
    validate_samples,
    validate_sample,
)


@runtime_checkable
class BasisProtocol(Protocol, Generic[Array]):
    """
    Protocol for basis objects used in BasisExpansion.

    Defines the required components for the basis object.
    """

    def bkd(self) -> Backend[Array]: ...

    def nterms(self) -> int: ...

    def nvars(self) -> int: ...

    def __call__(self, samples: Array) -> Array: ...


class BasisExpansion(Generic[Array]):
    def __init__(self, nqoi, basis: BasisProtocol):
        self._nqoi = nqoi
        if not isinstance(basis, BasisProtocol):
            raise ValueError(
                f"The provided basis must satisfy the 'BasisProtocol'. "
                f"Got an object of type {type(basis).__name__}."
            )
        self._bkd = basis.bkd()
        self._basis = basis

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._basis.nvars()

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest (QoI).
        """
        return self._nqoi

    def nterms(self) -> int:
        """
        Return the number of terms in the expansion.
        """
        return self._basis.nterms()

    def set_parameters(self, params: Array) -> None:
        if params.ndim != 2 or params.shape != (
            self._basis.nterms(),
            self.nqoi(),
        ):
            raise ValueError(
                "params shape is {0} but must be {1}".format(
                    params.shape, (self.nterms(), self.nqoi())
                )
            )
        self._params = params

    def get_parameters(self) -> Array:
        return self._params

    def __call__(self, samples: Array) -> Array:
        validate_samples(self.nvars(), samples)
        return self._basis(samples) @ self.get_parameters()

    # def jacobians(self, samples: Array) -> Array:
    #     return self._bkd.einsum(
    #         "ijk, jl->ilk",
    #         self._basis.jacobians(samples),
    #         self.get_coefficients(),
    #     )

    # def jacobian(self, sample: Array) -> Array:
    #     validate_sample(sample)
    #     return self.jacobians(sample)

    # def hessians(self, samples: Array) -> Array:
    #     hess = self._basis.hessians(samples)
    #     # hess shape is (nsamples, nterms, nvars, nvars)
    #     # coef shape is (nterms, nqoi)
    #     return self._bkd.einsum(
    #         "ijkl, jm->imkl", hess, self.get_coefficients()
    #     )
