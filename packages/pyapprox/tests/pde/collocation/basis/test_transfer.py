"""Transfer of nodal values between a Gauss-Lobatto and a Gauss mesh.

The interpolant on the source mesh is exact on polynomials of degree
below the source's node count per direction, so on an affine map both
transfers of a staggered pair (n Gauss-Lobatto, n - 2 Gauss nodes per
direction) reproduce polynomials of the pressure's degree exactly, in 2D
and 3D and on anisotropic sizes; the matrix has the target's rows and the
source's columns, and its rows sum to one. Refused: meshes with different
numbers of directions or maps, and objects that are not meshes.
"""

from typing import Callable, List, Sequence, Tuple

import pytest

from pyapprox.pde.collocation.basis import ChebyshevGauss1D, ChebyshevGaussLobatto1D
from pyapprox.pde.collocation.basis.transfer import grid_transfer_matrix
from pyapprox.pde.collocation.mesh import TransformedMesh
from pyapprox.pde.collocation.mesh.transforms.affine import AffineTransform
from pyapprox.util.backends.protocols import Array, Backend

_Polynomial = Callable[[Array], Array]


def _pair(
    bkd: Backend[Array],
    bounds: Sequence[Tuple[float, float]],
    velocity: Tuple[int, ...],
    pressure: Tuple[int, ...],
) -> Tuple[TransformedMesh[Array], TransformedMesh[Array]]:
    transform = AffineTransform(list(bounds), bkd)
    lobatto = TransformedMesh(
        velocity, bkd, transform, bases=[ChebyshevGaussLobatto1D[Array]()] * len(bounds)
    )
    gauss = TransformedMesh(
        pressure, bkd, transform, bases=[ChebyshevGauss1D[Array]()] * len(bounds)
    )
    return lobatto, gauss


def _check_both_ways(
    bkd: Backend[Array],
    lobatto: TransformedMesh[Array],
    gauss: TransformedMesh[Array],
    polynomials: List[_Polynomial[Array]],
) -> None:
    to_gauss = grid_transfer_matrix(lobatto, gauss, bkd)
    to_lobatto = grid_transfer_matrix(gauss, lobatto, bkd)
    assert to_gauss.shape == (gauss.npts(), lobatto.npts())
    assert to_lobatto.shape == (lobatto.npts(), gauss.npts())
    for matrix in (to_gauss, to_lobatto):
        bkd.assert_allclose(
            bkd.sum(matrix, axis=1), bkd.ones((matrix.shape[0],)), atol=1e-13
        )
    for u in polynomials:
        bkd.assert_allclose(
            to_gauss @ u(lobatto.points()), u(gauss.points()), atol=1e-12
        )
        bkd.assert_allclose(
            to_lobatto @ u(gauss.points()), u(lobatto.points()), atol=1e-12
        )


def test_two_dimensions(bkd: Backend[Array]) -> None:
    """6 x 6 Gauss-Lobatto and 4 x 4 Gauss nodes on [0, 2] x [1, 3]:
    polynomials of degree at most 3 per direction both ways."""
    lobatto, gauss = _pair(bkd, [(0.0, 2.0), (1.0, 3.0)], (6, 6), (4, 4))
    _check_both_ways(
        bkd,
        lobatto,
        gauss,
        [
            lambda p: p[0] ** 3 * p[1] ** 2 - p[1] ** 3,
            lambda p: (p[0] * p[1]) ** 3 + 1.0,
        ],
    )


def test_anisotropic_sizes(bkd: Backend[Array]) -> None:
    """5 x 7 and 3 x 5 nodes: degree 2 in x, 4 in y both ways."""
    lobatto, gauss = _pair(bkd, [(-1.0, 1.0), (0.0, 0.5)], (5, 7), (3, 5))
    _check_both_ways(bkd, lobatto, gauss, [lambda p: p[0] ** 2 * p[1] ** 4 - p[0]])


def test_three_dimensions(bkd: Backend[Array]) -> None:
    """5 x 5 x 5 and 3 x 3 x 3 nodes on a box: degree 2 per direction."""
    lobatto, gauss = _pair(
        bkd, [(0.0, 1.0), (-1.0, 0.0), (2.0, 3.0)], (5, 5, 5), (3, 3, 3)
    )
    _check_both_ways(
        bkd, lobatto, gauss, [lambda p: p[0] ** 2 * p[1] * p[2] ** 2 + p[1] ** 2]
    )


def test_higher_degree_is_not_reproduced_on_fewer_nodes(bkd: Backend[Array]) -> None:
    """A degree-4 polynomial on 5 Gauss-Lobatto nodes is not seen by the
    3 Gauss nodes: transferring to them and back loses it."""
    lobatto, gauss = _pair(bkd, [(-1.0, 1.0)], (5,), (3,))
    u = lobatto.points()[0] ** 4
    back = grid_transfer_matrix(gauss, lobatto, bkd) @ (
        grid_transfer_matrix(lobatto, gauss, bkd) @ u
    )
    assert bkd.to_float(bkd.max(bkd.abs(back - u))) > 1e-2


def test_refusals(bkd: Backend[Array]) -> None:
    lobatto, gauss = _pair(bkd, [(0.0, 1.0), (0.0, 1.0)], (4, 4), (2, 2))
    line, _ = _pair(bkd, [(0.0, 1.0)], (4,), (2,))
    with pytest.raises(ValueError, match="2 directions, the target 1"):
        grid_transfer_matrix(lobatto, line, bkd)
    _, other = _pair(bkd, [(0.0, 1.0), (0.0, 1.0)], (4, 4), (2, 2))
    with pytest.raises(ValueError, match="share one map"):
        grid_transfer_matrix(lobatto, other, bkd)
    with pytest.raises(TypeError, match="target must satisfy TransferMeshProtocol"):
        grid_transfer_matrix(lobatto, "mesh", bkd)  # type: ignore[arg-type]
