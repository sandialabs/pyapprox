from typing import Union, overload

from pyapprox.typing.util.backend import Array

from pyapprox.typing.surrogates.basis.basisexp import (
    BasisProtocol,
    BasisWithJacobianProtocol,
    BasisWithJacobianAndHVPProtocol,
    BasisExpansion,
    BasisExpansionWithJacobian,
    BasisExpansionWithJacobianAndHVP,
)


@overload
def basis_expansion_factory(
    nqoi: int,
    basis: BasisProtocol[Array],
) -> BasisExpansion[Array]: ...


@overload
def basis_expansion_factory(
    nqoi: int,
    basis: BasisWithJacobianProtocol[Array],
) -> BasisExpansionWithJacobian[Array]: ...


@overload
def basis_expansion_factory(
    nqoi: int,
    basis: BasisWithJacobianAndHVPProtocol[Array],
) -> BasisExpansionWithJacobianAndHVP[Array]: ...


def basis_expansion_factory(
    nqoi: int,
    basis: Union[
        BasisProtocol[Array],
        BasisWithJacobianProtocol[Array],
        BasisWithJacobianAndHVPProtocol[Array],
    ],
) -> Union[
    BasisExpansion[Array],
    BasisExpansionWithJacobian[Array],
    BasisExpansionWithJacobianAndHVP[Array],
]:
    """
    Factory function to create a BasisExpansion object based on the type of basis.

    Parameters
    ----------
    nqoi : int
        Number of quantities of interest (QoI).
    basis : Union[
        BasisProtocol[Array],
        BasisWithJacobianProtocol[Array],
        BasisWithJacobianAndHVPProtocol[Array],
    ]
        The basis object.

    Returns
    -------
    Union[
        BasisExpansion[Array],
        BasisExpansionWithJacobian[Array],
        BasisExpansionWithJacobianAndHVP[Array],
    ]
        The appropriate BasisExpansion object based on the type of basis.

    Raises
    ------
    ValueError
        If the type of basis is not recognized.
    """
    if isinstance(basis, BasisWithJacobianAndHVPProtocol):
        return BasisExpansionWithJacobianAndHVP(nqoi, basis)

    if isinstance(basis, BasisWithJacobianProtocol):
        return BasisExpansionWithJacobian(nqoi, basis)

    if isinstance(basis, BasisProtocol):
        return BasisExpansion(nqoi, basis)

    raise ValueError(
        "The provided basis must satisfy one of the following protocols: "
        "'BasisProtocol', 'BasisWithJacobianProtocol', or "
        "'BasisWithJacobianAndHVPProtocol'. "
        f"Got an object of type {type(basis).__name__}."
    )
