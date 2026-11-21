"""The dimension-generic Chebyshev basis and per-direction derivative
builders.

A 4D basis differentiates a polynomial exactly, a mesh may take another
builder in one direction, the 1D/2D/3D bases are the generic one, and the
refusals.
"""

from typing import Generic

import pytest

from pyapprox.pde.collocation.basis import (
    ChebyshevBasis,
    ChebyshevBasis1D,
    ChebyshevBasis2D,
    ChebyshevBasis3D,
    ChebyshevDerivativeMatrix1D,
    TensorProductBasis,
)
from pyapprox.pde.collocation.mesh import (
    TransformedMesh,
    TransformedMesh1D,
    TransformedMesh2D,
    TransformedMesh3D,
)
from pyapprox.pde.collocation.mesh.transforms import AffineTransform
from pyapprox.util.backends.protocols import Array, Backend


def test_four_dimensional_derivatives_are_exact(bkd: Backend[Array]) -> None:
    """On a 3 x 4 x 3 x 5 mesh of a 4D box, u = x0^2 x1 + x2 x3^3 has
    degree at most n_d - 1 in each direction, so the first and second
    derivative matrices reproduce its derivatives to rounding."""
    transform = AffineTransform([(0.0, 1.0), (-1.0, 2.0), (0.5, 1.5), (-2.0, 0.0)], bkd)
    mesh = TransformedMesh((3, 4, 3, 5), bkd, transform)
    basis = ChebyshevBasis(mesh, bkd)
    x0, x1, x2, x3 = (mesh.points()[d] for d in range(4))
    u = x0**2 * x1 + x2 * x3**3
    first = [2 * x0 * x1, x0**2, x3**3, 3 * x2 * x3**2]
    second = [2 * x1, 0 * x1, 0 * x1, 6 * x2 * x3]
    assert basis.ndim() == 4 and basis.npts() == 180
    for d in range(4):
        bkd.assert_allclose(basis.derivative_matrix(1, d) @ u, first[d], atol=1e-11)
        bkd.assert_allclose(basis.derivative_matrix(2, d) @ u, second[d], atol=1e-10)


class _Scaled(Generic[Array]):
    """Twice the Chebyshev derivative matrix: a stand-in for another
    family's builder."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        self._chebyshev = ChebyshevDerivativeMatrix1D(bkd)

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def compute(self, nodes: Array) -> Array:
        return 2.0 * self._chebyshev.compute(nodes)


def test_derivative_builder_per_direction(bkd: Backend[Array]) -> None:
    """A builder per direction: the second direction's 1D matrix is the
    scaled one, the first's the Chebyshev one."""
    mesh = TransformedMesh2D(4, 5, bkd)
    chebyshev = ChebyshevBasis2D(mesh, bkd)
    basis = TensorProductBasis(
        mesh, [ChebyshevDerivativeMatrix1D(bkd), _Scaled(bkd)], bkd
    )
    bkd.assert_allclose(
        basis.derivative_matrix_1d(0), chebyshev.derivative_matrix_1d(0)
    )
    bkd.assert_allclose(
        basis.derivative_matrix_1d(1), 2.0 * chebyshev.derivative_matrix_1d(1)
    )
    bkd.assert_allclose(
        basis.derivative_matrix(1, 1), 2.0 * chebyshev.derivative_matrix(1, 1)
    )


def test_per_dimension_bases_are_the_generic_basis(bkd: Backend[Array]) -> None:
    """The 1D/2D/3D bases are ChebyshevBasis with the same matrices."""
    for basis, mesh in [
        (ChebyshevBasis1D(TransformedMesh1D(5, bkd), bkd), TransformedMesh1D(5, bkd)),
        (
            ChebyshevBasis2D(TransformedMesh2D(3, 4, bkd), bkd),
            TransformedMesh2D(3, 4, bkd),
        ),
        (
            ChebyshevBasis3D(TransformedMesh3D(3, 2, 4, bkd), bkd),
            TransformedMesh3D(3, 2, 4, bkd),
        ),
    ]:
        assert isinstance(basis, ChebyshevBasis)
        generic = ChebyshevBasis(mesh, bkd)
        for d in range(mesh.ndim()):
            bkd.assert_allclose(
                basis.derivative_matrix(1, d), generic.derivative_matrix(1, d)
            )


def test_refusals(bkd: Backend[Array]) -> None:
    """A builder list of the wrong length, a non-builder, a mesh of
    another dimension for a per-dimension basis."""
    mesh = TransformedMesh2D(3, 3, bkd)
    with pytest.raises(ValueError, match="1 derivative builders for 2 directions"):
        TensorProductBasis(mesh, [ChebyshevDerivativeMatrix1D(bkd)], bkd)
    with pytest.raises(TypeError, match="DerivativeMatrix1DProtocol"):
        TensorProductBasis(
            mesh,
            [ChebyshevDerivativeMatrix1D(bkd), 3],  # type: ignore[list-item]
            bkd,
        )
    with pytest.raises(ValueError, match="a 3D basis needs a 3D mesh, got 2D"):
        ChebyshevBasis3D(mesh, bkd)
    with pytest.raises(ValueError, match="a 1D basis needs a 1D mesh, got 2D"):
        ChebyshevBasis1D(mesh, bkd)
