from typing import Protocol, Generic, runtime_checkable, Sequence, List

from pyapprox.typing.util.backend import Array, Backend
from pyapprox.typing.interface.functions.function import (
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

    def _basis_vals_1d(self, samples: Array) -> List[Array]:
        return [
            basis_1d(samples[dd : dd + 1, :])
            for dd, basis_1d in enumerate(self._bases_1d)
        ]

    def __call__(self, samples: Array) -> Array:
        validate_samples(self.nvars(), samples)
        basis_vals_1d = self._basis_vals_1d(samples)
        basis_matrix = basis_vals_1d[0][:, self._indices[0, :]]
        for dd in range(1, self.nvars()):
            basis_matrix *= basis_vals_1d[dd][:, self._indices[dd, :]]
        return basis_matrix

    def __repr__(self) -> str:
        return "{0}(nvars={1}, nterms={2})".format(
            self.__class__.__name__, self.nvars(), self.nterms()
        )
