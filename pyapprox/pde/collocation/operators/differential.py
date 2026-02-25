"""Differential operators for spectral collocation methods.

Provides gradient, divergence, and Laplacian operators that act on
scalar and vector fields.
"""

from typing import Generic, List, Tuple

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.pde.collocation.protocols.basis import (
    TensorProductBasisProtocol,
)
from pyapprox.pde.collocation.operators.field import Field


class Gradient(Generic[Array]):
    """Gradient operator for scalar fields.

    Computes the gradient of a scalar field, returning a list of scalar
    fields representing each component of the gradient vector.

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        The collocation basis providing derivative matrices.
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self,
        basis: TensorProductBasisProtocol[Array],
        bkd: Backend[Array],
    ):
        self._basis = basis
        self._bkd = bkd

    def __call__(self, field: Field[Array]) -> List[Field[Array]]:
        """Compute gradient of scalar field.

        Parameters
        ----------
        field : Field
            Input scalar field.

        Returns
        -------
        List[Field]
            List of scalar fields, one per spatial dimension.
            grad[i] = df/dx_i
        """
        ndim = self._basis.ndim()
        return [field.deriv(dim=i) for i in range(ndim)]


class Divergence(Generic[Array]):
    """Divergence operator for vector fields.

    Computes the divergence of a vector field represented as a list
    of scalar fields.

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        The collocation basis providing derivative matrices.
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self,
        basis: TensorProductBasisProtocol[Array],
        bkd: Backend[Array],
    ):
        self._basis = basis
        self._bkd = bkd

    def __call__(
        self, vector_field: List[Field[Array]]
    ) -> Field[Array]:
        """Compute divergence of vector field.

        Parameters
        ----------
        vector_field : List[Field]
            List of scalar fields representing vector components.
            Must have length equal to ndim.

        Returns
        -------
        Field
            Divergence: div(v) = sum_i dv_i/dx_i
        """
        ndim = self._basis.ndim()
        if len(vector_field) != ndim:
            raise ValueError(
                f"Vector field must have {ndim} components, got {len(vector_field)}"
            )

        # div(v) = sum_i dv_i/dx_i
        result = vector_field[0].deriv(dim=0)
        for i in range(1, ndim):
            result = result + vector_field[i].deriv(dim=i)
        return result


class Laplacian(Generic[Array]):
    """Laplacian operator for scalar fields.

    Computes the Laplacian (sum of second derivatives) of a scalar field.

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        The collocation basis providing derivative matrices.
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self,
        basis: TensorProductBasisProtocol[Array],
        bkd: Backend[Array],
    ):
        self._basis = basis
        self._bkd = bkd

    def __call__(self, field: Field[Array]) -> Field[Array]:
        """Compute Laplacian of scalar field.

        Parameters
        ----------
        field : Field
            Input scalar field.

        Returns
        -------
        Field
            Laplacian: nabla^2 f = sum_i d^2f/dx_i^2
        """
        ndim = self._basis.ndim()
        # nabla^2 f = sum_i d^2f/dx_i^2
        result = field.deriv(dim=0, order=2)
        for i in range(1, ndim):
            result = result + field.deriv(dim=i, order=2)
        return result


def gradient(
    field: Field[Array],
    basis: TensorProductBasisProtocol[Array],
    bkd: Backend[Array],
) -> List[Field[Array]]:
    """Compute gradient of scalar field (functional interface).

    Parameters
    ----------
    field : Field
        Input scalar field.
    basis : TensorProductBasisProtocol
        The collocation basis.
    bkd : Backend
        Computational backend.

    Returns
    -------
    List[Field]
        Gradient components.
    """
    return Gradient(basis, bkd)(field)


def divergence(
    vector_field: List[Field[Array]],
    basis: TensorProductBasisProtocol[Array],
    bkd: Backend[Array],
) -> Field[Array]:
    """Compute divergence of vector field (functional interface).

    Parameters
    ----------
    vector_field : List[Field]
        Vector field components.
    basis : TensorProductBasisProtocol
        The collocation basis.
    bkd : Backend
        Computational backend.

    Returns
    -------
    Field
        Divergence.
    """
    return Divergence(basis, bkd)(vector_field)


def laplacian(
    field: Field[Array],
    basis: TensorProductBasisProtocol[Array],
    bkd: Backend[Array],
) -> Field[Array]:
    """Compute Laplacian of scalar field (functional interface).

    Parameters
    ----------
    field : Field
        Input scalar field.
    basis : TensorProductBasisProtocol
        The collocation basis.
    bkd : Backend
        Computational backend.

    Returns
    -------
    Field
        Laplacian.
    """
    return Laplacian(basis, bkd)(field)
