"""The dimension-generic Chebyshev basis and per-direction 1D bases.

A 4D basis differentiates a polynomial exactly, each direction is
differentiated by the 1D basis that made its nodes (another basis in one
direction, a Gauss direction beside a Gauss-Lobatto one), the 1D/2D/3D
bases are the generic one, and the refusals.
"""

from typing import Generic

import pytest

from pyapprox.pde.collocation.basis import (
    ChebyshevBasis,
    ChebyshevBasis1D,
    ChebyshevBasis2D,
    ChebyshevBasis3D,
    ChebyshevGauss1D,
    ChebyshevGaussLobatto1D,
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
    """Chebyshev-Gauss-Lobatto nodes with twice the derivative matrix: a
    stand-in for another family's 1D basis."""

    def __init__(self) -> None:
        self._chebyshev = ChebyshevGaussLobatto1D[Array]()

    def nodes(self, npts: int, bkd: Backend[Array]) -> Array:
        return self._chebyshev.nodes(npts, bkd)

    def derivative_matrix(self, npts: int, bkd: Backend[Array]) -> Array:
        return 2.0 * self._chebyshev.derivative_matrix(npts, bkd)

    def interpolation_matrix(
        self, npts: int, points: Array, bkd: Backend[Array]
    ) -> Array:
        return self._chebyshev.interpolation_matrix(npts, points, bkd)


def test_basis_1d_per_direction(bkd: Backend[Array]) -> None:
    """A 1D basis per direction: the second direction's 1D matrix is the
    scaled one, the first's the Chebyshev one."""
    chebyshev = ChebyshevBasis2D(TransformedMesh2D(4, 5, bkd), bkd)
    mesh = TransformedMesh(
        (4, 5), bkd, bases=[ChebyshevGaussLobatto1D[Array](), _Scaled[Array]()]
    )
    basis = TensorProductBasis(mesh, bkd)
    bkd.assert_allclose(
        basis.derivative_matrix_1d(0), chebyshev.derivative_matrix_1d(0)
    )
    bkd.assert_allclose(
        basis.derivative_matrix_1d(1), 2.0 * chebyshev.derivative_matrix_1d(1)
    )
    bkd.assert_allclose(
        basis.derivative_matrix(1, 1), 2.0 * chebyshev.derivative_matrix(1, 1)
    )


def test_gauss_direction_beside_a_lobatto_one(bkd: Backend[Array]) -> None:
    """On [0, 2] x [1, 3], 5 Gauss-Lobatto nodes in x and 4 Gauss nodes
    in y (a staggered pressure grid's direction): u = x^3 y^2 - y^3 has
    degree at most n_d - 1 per direction, so both first derivatives and
    the second y derivative are exact."""
    transform = AffineTransform([(0.0, 2.0), (1.0, 3.0)], bkd)
    mesh = TransformedMesh(
        (5, 4),
        bkd,
        transform,
        bases=[ChebyshevGaussLobatto1D[Array](), ChebyshevGauss1D[Array]()],
    )
    basis = ChebyshevBasis(mesh, bkd)
    x, y = mesh.points()[0], mesh.points()[1]
    u = x**3 * y**2 - y**3
    bkd.assert_allclose(basis.derivative_matrix(1, 0) @ u, 3 * x**2 * y**2, atol=1e-11)
    bkd.assert_allclose(
        basis.derivative_matrix(1, 1) @ u, 2 * x**3 * y - 3 * y**2, atol=1e-11
    )
    bkd.assert_allclose(basis.derivative_matrix(2, 1) @ u, 2 * x**3 - 6 * y, atol=1e-10)


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
    """A mesh of another dimension for a per-dimension basis (the 1D
    bases' own refusals are the mesh's)."""
    mesh = TransformedMesh2D(3, 3, bkd)
    with pytest.raises(ValueError, match="a 3D basis needs a 3D mesh, got 2D"):
        ChebyshevBasis3D(mesh, bkd)
    with pytest.raises(ValueError, match="a 1D basis needs a 1D mesh, got 2D"):
        ChebyshevBasis1D(mesh, bkd)
