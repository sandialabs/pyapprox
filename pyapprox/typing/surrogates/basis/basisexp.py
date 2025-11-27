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

    def basis(self) -> BasisProtocol:
        return self._basis

    def __repr__(self):
        return "{0}(basis={1}, nqoi={2})".format(
            self.__class__.__name__, self.basis(), self.nqoi()
        )


@runtime_checkable
class BasisWithJacobianProtocol(Protocol, Generic[Array]):
    """
    Protocol for basis objects used in BasisExpansionWithJacobian.

    Defines the required components for the basis object.
    """

    def bkd(self) -> Backend[Array]: ...

    def nterms(self) -> int: ...

    def nvars(self) -> int: ...

    def __call__(self, samples: Array) -> Array: ...

    def jacobians(self, samples: Array) -> Array: ...


class BasisExpansionWithJacobian(Generic[Array]):
    def __init__(self, nqoi, basis: BasisWithJacobianProtocol):
        self._nqoi = nqoi
        if not isinstance(basis, BasisWithJacobianProtocol):
            raise ValueError(
                f"The provided basis must satisfy the "
                "'BasisWithJacobianProtocol'. "
                f"Got an object of type {type(basis).__name__}."
            )
        self._bexp: BasisExpansion = BasisExpansion(nqoi, basis)
        self._basis = basis

    def bkd(self) -> Backend[Array]:
        return self._bexp.bkd()

    def nvars(self) -> int:
        return self._bexp.nvars()

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest (QoI).
        """
        return self._bexp.nqoi()

    def nterms(self) -> int:
        """
        Return the number of terms in the expansion.
        """
        return self._bexp.nterms()

    def set_parameters(self, params: Array) -> None:
        self._bexp.set_parameters(params)

    def get_parameters(self) -> Array:
        return self._bexp.get_parameters()

    def __call__(self, samples: Array) -> Array:
        return self._bexp(samples)

    def basis(self) -> BasisWithJacobianProtocol:
        return self._basis

    def jacobians(self, samples: Array) -> Array:
        return self.bkd().einsum(
            "ijk, jl->ilk",
            self._basis.jacobians(samples),
            self.get_parameters(),
        )

    def jacobian(self, sample: Array) -> Array:
        validate_sample(self.nvars(), sample)
        return self.jacobians(sample)

    def __repr__(self):
        return "{0}(basis={1}, nqoi={2})".format(
            self.__class__.__name__, self.basis(), self.nqoi()
        )


@runtime_checkable
class BasisWithJacobianAndHVPProtocol(Protocol, Generic[Array]):
    """
    Protocol for basis objects used in BasisExpansionWithJacobianAndHVP.

    Defines the required components for the basis object.
    """

    def bkd(self) -> Backend[Array]: ...

    def nterms(self) -> int: ...

    def nvars(self) -> int: ...

    def __call__(self, samples: Array) -> Array: ...

    def jacobians(self, samples: Array) -> Array: ...

    def hessians(self, samples: Array) -> Array: ...


class BasisExpansionWithJacobianAndHVP(Generic[Array]):
    def __init__(self, nqoi, basis: BasisWithJacobianAndHVPProtocol):
        self._nqoi = nqoi
        if not isinstance(basis, BasisWithJacobianAndHVPProtocol):
            raise ValueError(
                f"The provided basis must satisfy the "
                "'BasisWithJacobianAndHVPProtocol'. "
                f"Got an object of type {type(basis).__name__}."
            )
        self._bexp: BasisExpansionWithJacobian = BasisExpansionWithJacobian(
            nqoi, basis
        )
        self._basis = basis

    def bkd(self) -> Backend[Array]:
        return self._bexp.bkd()

    def nvars(self) -> int:
        return self._bexp.nvars()

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest (QoI).
        """
        return self._bexp.nqoi()

    def nterms(self) -> int:
        """
        Return the number of terms in the expansion.
        """
        return self._bexp.nterms()

    def set_parameters(self, params: Array) -> None:
        self._bexp.set_parameters(params)

    def get_parameters(self) -> Array:
        return self._bexp.get_parameters()

    def __call__(self, samples: Array) -> Array:
        return self._bexp(samples)

    def basis(self) -> BasisWithJacobianAndHVPProtocol:
        return self._basis

    def jacobians(self, samples: Array) -> Array:
        return self._bexp.jacobians(samples)

    def jacobian(self, sample: Array) -> Array:
        return self._bexp.jacobian(sample)

    def hessians(self, samples: Array) -> Array:
        hess = self._basis.hessians(samples)
        # hess shape is (nsamples, nterms, nvars, nvars)
        # coef shape is (nterms, nqoi)
        return self.bkd().einsum("ijkl, jm->imkl", hess, self.get_parameters())

    def weighted_hvp(self, sample: Array, vec: Array, weights: Array) -> Array:
        validate_sample(self.nvars(), sample)
        validate_sample(self.nvars(), vec)
        validate_sample(self.nqoi(), weights)

        hess = self._basis.hessians(sample)
        coef = self.get_parameters()

        # "jkl": Represents the indices of the hess tensor:
        #     j: Basis term index (nterms).
        #     k: First variable index (nvars)
        #     l: Second variable index (nvars).
        # "jm": Represents the indices of the coef tensor:
        #     j: Basis term index (nterms).
        #     m: QoI index (nqoi).
        # "l": Represents the indices of the vector tensor:
        #     l: Second variable index (nvars).
        # "mk": Specifies the output tensor:
        #     m: QoI index (nqoi).
        #     k: First variable index (nvars).

        return self.bkd().einsum(
            "jkl, jm, m, ln -> kn", hess[0], coef, weights[:, 0], vec
        )
        return

    def hvp(self, sample: Array, vec: Array) -> Array:
        if self.nqoi() != 1:
            raise ValueError(
                f"hvp can only be used with nqoi=1 but {self.nqoi()=}"
            )
        return self.weighted_hvp(sample, vec, self.bkd().ones((1, 1)))

    def __repr__(self):
        return "{0}(basis={1}, nqoi={2})".format(
            self.__class__.__name__, self.basis(), self.nqoi()
        )
