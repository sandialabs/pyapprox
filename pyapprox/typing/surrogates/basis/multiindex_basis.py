from typing import (
    Protocol,
    Generic,
    runtime_checkable,
    Sequence,
    List,
    Union,
    overload,
)

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.interface.functions.protocols.validation import (
    validate_samples,
)


@runtime_checkable
class Basis1DProtocol(Protocol, Generic[Array]):
    def bkd(self) -> Backend[Array]: ...

    def set_nterms(self, nterms: int) -> None: ...

    def nterms(self) -> int: ...

    def __call__(self, samples: Array) -> Array: ...


class MultiIndexBasis(Generic[Array]):
    def __init__(
        self,
        univariate_bases: Sequence[Basis1DProtocol[Array]],
        indices: Array,
    ):
        self._validate_univariate_bases(univariate_bases)
        self._bases_1d = univariate_bases
        self._bkd = univariate_bases[0].bkd()

        self.set_indices(indices)

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def _validate_univariate_bases(
        self, univariate_bases: Sequence[Basis1DProtocol[Array]]
    ) -> None:
        """
        Validate the univariate bases.

        Parameters
        ----------
        univariate_bases : Sequence[Basis1DProtocol[Array]]
            The sequence of univariate bases to validate.

        Raises
        ------
        ValueError
            If any of the univariate bases do not satisfy the Basis1DProtocol.
        """
        for basis in univariate_bases:
            if not isinstance(basis, Basis1DProtocol):
                raise ValueError(
                    f"All univariate bases must satisfy Basis1DProtocol. "
                    f"Got an object of type {type(basis).__name__}."
                )

    def _set_nterms(self, nterms: Array) -> None:
        """
        Set the number of terms for each univariate basis.

        Parameters
        ----------
        nterms : Array
            The number of terms for each univariate basis.
        """
        for dd, basis_1d in enumerate(self._bases_1d):
            basis_1d.set_nterms(nterms[dd])

    def _validate_indices(self, indices: Array) -> None:
        if indices.ndim != 2:
            raise ValueError("indices must have two dimensions")
        if indices.shape[0] != len(self._bases_1d):
            raise ValueError(
                "indices.shape[0] {0} doesnt match len(bases_1d) {1}".format(
                    indices.shape[0], len(self._bases_1d)
                )
            )
        if indices.dtype != self.bkd().int64_dtype():
            raise ValueError("indices must be int64")

    def set_indices(self, indices: Array) -> None:
        self._validate_indices(indices)
        self._indices = indices
        self._set_nterms(self._bkd.max(self._indices, axis=1) + 1)

    def get_indices(self) -> Array:
        """Return the indices defining the basis terms."""
        return self._indices

    def nterms(self) -> int:
        if self._indices is None:
            return 0
        return self._indices.shape[1]

    def nvars(self) -> int:
        return len(self._bases_1d)

    def _1d_basis_vals(self, samples: Array) -> List[Array]:
        return [
            basis_1d(samples[dd : dd + 1, :])
            for dd, basis_1d in enumerate(self._bases_1d)
        ]

    def __call__(self, samples: Array) -> Array:
        validate_samples(self.nvars(), samples)
        basis_vals_1d = self._1d_basis_vals(samples)
        basis_matrix = basis_vals_1d[0][:, self._indices[0, :]]
        for dd in range(1, self.nvars()):
            basis_matrix *= basis_vals_1d[dd][:, self._indices[dd, :]]
        return basis_matrix

    def __repr__(self) -> str:
        return "{0}(nvars={1}, nterms={2})".format(
            self.__class__.__name__, self.nvars(), self.nterms()
        )


@runtime_checkable
class Basis1DWithJacobiansProtocol(Protocol, Generic[Array]):
    def bkd(self) -> Backend[Array]: ...

    def set_nterms(self, nterms: int) -> None: ...

    def nterms(self) -> int: ...

    def __call__(self, samples: Array) -> Array: ...

    def jacobians(self, samples: Array) -> Array: ...


class MultiIndexBasisWithJacobian(Generic[Array]):
    def __init__(
        self,
        univariate_bases: Sequence[Basis1DWithJacobiansProtocol[Array]],
        indices: Array,
    ):
        self._basis = MultiIndexBasis(univariate_bases, indices)
        self._univariate_bases = univariate_bases

    def bkd(self) -> Backend[Array]:
        return self._basis.bkd()

    def set_indices(self, indices: Array) -> None:
        self._basis.set_indices(indices)

    def get_indices(self) -> Array:
        return self._basis.get_indices()

    def nterms(self) -> int:
        return self._basis.nterms()

    def nvars(self) -> int:
        return self._basis.nvars()

    def __call__(self, samples: Array) -> Array:
        return self._basis(samples)

    def __repr__(self) -> str:
        return "{0}(nvars={1}, nterms={2})".format(
            self.__class__.__name__, self.nvars(), self.nterms()
        )

    def _1d_basis_jacobians(self, samples: Array) -> List[Array]:
        return [
            basis1d.jacobians(samples[dd : dd + 1, :])
            for dd, basis1d in enumerate(self._univariate_bases)
        ]

    def jacobians(self, samples: Array) -> Array:
        indices = self.get_indices()
        basis_vals_1d = self._basis._1d_basis_vals(samples)
        basis_jacobians_1d = self._1d_basis_jacobians(samples)
        jac = []
        for dd in range(self.nvars()):
            jac_dd = basis_jacobians_1d[dd][:, indices[dd, :]]
            for kk in range(self.nvars()):
                if kk != dd:
                    jac_dd *= basis_vals_1d[kk][:, indices[kk, :]]
            jac.append(jac_dd)
        return self.bkd().moveaxis(self.bkd().stack(jac, axis=0), 0, -1)


@runtime_checkable
class Basis1DWithJacobiansAndHessiansProtocol(Protocol, Generic[Array]):
    def bkd(self) -> Backend[Array]: ...

    def set_nterms(self, nterms: int) -> None: ...

    def nterms(self) -> int: ...

    def __call__(self, samples: Array) -> Array: ...

    def jacobians(self, samples: Array) -> Array: ...

    def hessians(self, samples: Array) -> Array: ...


class MultiIndexBasisWithJacobianAndHVP(Generic[Array]):
    def __init__(
        self,
        univariate_bases: Sequence[
            Basis1DWithJacobiansAndHessiansProtocol[Array]
        ],
        indices: Array,
    ):
        self._basis = MultiIndexBasisWithJacobian(univariate_bases, indices)
        self._univariate_bases = univariate_bases

    def bkd(self) -> Backend[Array]:
        return self._basis.bkd()

    def set_indices(self, indices: Array) -> None:
        self._basis.set_indices(indices)

    def get_indices(self) -> Array:
        return self._basis.get_indices()

    def nterms(self) -> int:
        return self._basis.nterms()

    def nvars(self) -> int:
        return self._basis.nvars()

    def __call__(self, samples: Array) -> Array:
        return self._basis(samples)

    def __repr__(self) -> str:
        return "{0}(nvars={1}, nterms={2})".format(
            self.__class__.__name__, self.nvars(), self.nterms()
        )

    def jacobians(self, samples: Array) -> Array:
        return self._basis.jacobians(samples)

    def _1d_basis_hessians(self, samples: Array) -> List[Array]:
        return [
            basis1d.hessians(samples[dd : dd + 1, :])
            for dd, basis1d in enumerate(self._univariate_bases)
        ]

    def hessian(self, samples: Array) -> Array:
        indices = self.get_indices()
        basis_vals_1d = self._basis._basis._1d_basis_vals(samples)
        fir_derivs_1d = self._basis._1d_basis_jacobians(samples)
        sec_derivs_1d = self._basis._1d_basis_hessians(samples)
        hess_items: List[List[Array]] = [
            [[] for kk in range(self.nvars())] for dd in range(self.nvars())
        ]
        for dd in range(self.nvars()):
            for kk in range(dd, self.nvars()):
                if kk == dd:
                    hess_dk = sec_derivs_1d[kk][:, indices[kk, :]]
                else:
                    hess_dk = fir_derivs_1d[dd][:, indices[dd, :]]
                    hess_dk *= fir_derivs_1d[kk][:, indices[kk, :]]
                for ll in range(self.nvars()):
                    if ll == kk or ll == dd:
                        continue
                    hess_dk *= basis_vals_1d[ll][:, indices[ll, :]]
                hess_items[dd][kk] = hess_dk
                hess_items[kk][dd] = hess_dk
        hess = self.bkd().stack(
            [
                self.bkd().stack(hess_items[dd], axis=-1)
                for dd in range(self.nvars())
            ],
            axis=-1,
        )
        return hess
