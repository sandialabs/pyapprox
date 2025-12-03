from typing import Union, overload, Sequence

from pyapprox.typing.util.backends.protocols import Array

from pyapprox.typing.surrogates.basis.multiindex_basis import (
    Basis1DProtocol,
    Basis1DWithJacobiansProtocol,
    Basis1DWithJacobiansAndHessiansProtocol,
    MultiIndexBasis,
    MultiIndexBasisWithJacobian,
    MultiIndexBasisWithJacobianAndHVP,
)


@overload
def multiindex_basis_factory(
    univariate_bases: Sequence[Basis1DProtocol[Array]],
    indices: Array,
) -> MultiIndexBasis[Array]: ...


@overload
def multiindex_basis_factory(
    univariate_bases: Sequence[Basis1DWithJacobiansProtocol[Array]],
    indices: Array,
) -> MultiIndexBasisWithJacobian[Array]: ...


@overload
def multiindex_basis_factory(
    univariate_bases: Sequence[Basis1DWithJacobiansAndHessiansProtocol[Array]],
    indices: Array,
) -> MultiIndexBasisWithJacobianAndHVP[Array]: ...


def multiindex_basis_factory(
    univariate_bases: Union[
        Sequence[Basis1DProtocol[Array]],
        Sequence[Basis1DWithJacobiansProtocol[Array]],
        Sequence[Basis1DWithJacobiansAndHessiansProtocol[Array]],
    ],
    indices: Array,
) -> Union[
    MultiIndexBasis[Array],
    MultiIndexBasisWithJacobian[Array],
    MultiIndexBasisWithJacobianAndHVP[Array],
]:
    """
    Factory function to create a MultiIndexBasis object based on
    the type of univariate bases.

    Parameters
    ----------
    univariate_bases : Union[
        Sequence[Basis1DProtocol[Array]],
        Sequence[Basis1DWithJacobiansProtocol[Array]],
        Sequence[Basis1DWithJacobiansAndHessiansProtocol[Array]],
    ]
        The sequence of univariate bases.
    indices : Array
        The indices defining the basis terms.

    Returns
    -------
    Union[
        MultiIndexBasis[Array],
        MultiIndexBasisWithJacobian[Array],
        MultiIndexBasisWithJacobianAndHVP[Array],
    ]
        The appropriate MultiIndexBasis object based on the type of
    univariate bases.

    Raises
    ------
    ValueError
        If the type of univariate bases is not recognized.
    """
    if all(
        isinstance(basis, Basis1DWithJacobiansAndHessiansProtocol)
        for basis in univariate_bases
    ):
        return MultiIndexBasisWithJacobianAndHVP(univariate_bases, indices)  # type: ignore

    if all(
        isinstance(basis, Basis1DWithJacobiansProtocol)
        for basis in univariate_bases
    ):
        return MultiIndexBasisWithJacobian(univariate_bases, indices)  # type: ignore

    if all(isinstance(basis, Basis1DProtocol) for basis in univariate_bases):
        return MultiIndexBasis(univariate_bases, indices)

    raise ValueError(
        "All univariate bases must satisfy one of the following protocols: "
        "'Basis1DProtocol', 'Basis1DWithJacobiansProtocol', or "
        "'Basis1DWithJacobiansAndHessiansProtocol'."
    )
