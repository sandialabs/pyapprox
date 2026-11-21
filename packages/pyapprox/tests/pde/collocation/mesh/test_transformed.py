"""Tensor-product meshes mapped by a transform, in any dimension.

Boundary indices against explicit formulas, outward normals against the
analytic ones on a polar ring and a 4D box, a map that reverses
orientation, per-direction node generators, and the refusals.
"""

import math
from typing import Generic, List

import numpy as np
import pytest

from pyapprox.pde.collocation.mesh import (
    ChebyshevGaussLobattoNodes,
    ChebyshevGaussNodes,
    TransformedMesh,
    TransformedMesh1D,
    TransformedMesh2D,
    TransformedMesh3D,
    compute_boundary_indices,
)
from pyapprox.pde.collocation.mesh.transforms import AffineTransform, PolarTransform
from pyapprox.util.backends.protocols import Array, Backend


def _ints(bkd: Backend[Array], values: List[int]) -> Array:
    return bkd.asarray(values, dtype=bkd.int64_dtype())


def test_boundary_indices_3d(bkd: Backend[Array]) -> None:
    """On 3 x 4 x 5 nodes, x fastest: each face lists its nodes in
    increasing index, the formulas the 3D helper used to spell out."""
    nx, ny, nz = 3, 4, 5
    indices = compute_boundary_indices((nx, ny, nz), bkd)
    nxy = nx * ny
    expected = [
        [k * nxy + j * nx for k in range(nz) for j in range(ny)],
        [k * nxy + j * nx + nx - 1 for k in range(nz) for j in range(ny)],
        [k * nxy + i for k in range(nz) for i in range(nx)],
        [k * nxy + (ny - 1) * nx + i for k in range(nz) for i in range(nx)],
        [j * nx + i for j in range(ny) for i in range(nx)],
        [(nz - 1) * nxy + j * nx + i for j in range(ny) for i in range(nx)],
    ]
    assert len(indices) == 6
    for got, want in zip(indices, expected):
        bkd.assert_allclose(got, _ints(bkd, want))


def test_four_dimensional_box(bkd: Backend[Array]) -> None:
    """A 3 x 4 x 2 x 3 mesh of a 4D box: 8 boundaries, boundary 2d + s
    holding the 72 / n_d nodes at the first or last node of direction d,
    with unit normals (2 s - 1) e_d."""
    counts = (3, 4, 2, 3)
    transform = AffineTransform([(0.0, 1.0), (-1.0, 2.0), (0.0, 0.5), (1.0, 4.0)], bkd)
    mesh = TransformedMesh(counts, bkd, transform)
    assert (mesh.ndim(), mesh.npts(), mesh.nboundaries()) == (4, 72, 8)
    assert mesh.points().shape == (4, 72)
    for boundary in range(8):
        direction, side = divmod(boundary, 2)
        nodes = mesh.boundary_indices(boundary)
        assert nodes.shape == (72 // counts[direction],)
        bound = transform.bounds()[direction][side]
        bkd.assert_allclose(
            mesh.points()[direction][nodes], bkd.full(nodes.shape, bound), atol=1e-14
        )
        expected = (
            bkd.ones((nodes.shape[0], 1))
            * bkd.asarray(
                [float(2 * side - 1) if d == direction else 0.0 for d in range(4)]
            )[None, :]
        )
        bkd.assert_allclose(mesh.boundary_normals(boundary), expected, atol=1e-14)


def test_polar_ring_normals(bkd: Backend[Array]) -> None:
    """On the ring 1 <= r <= 2, theta in [0, 3 pi / 2]: the circles'
    normals are -+(cos theta, sin theta), the rays' -+(-sin theta,
    cos theta)."""
    transform = PolarTransform((1.0, 2.0), (0.0, 1.5 * math.pi), bkd)
    mesh = TransformedMesh2D(5, 6, bkd, transform)
    points = mesh.points()
    for boundary, sign, radial in [
        (0, -1, True),
        (1, 1, True),
        (2, -1, False),
        (3, 1, False),
    ]:
        nodes = mesh.boundary_indices(boundary)
        theta = bkd.arctan2(points[1][nodes], points[0][nodes])
        if radial:
            direction = bkd.stack([bkd.cos(theta), bkd.sin(theta)], axis=1)
        else:
            direction = bkd.stack([-bkd.sin(theta), bkd.cos(theta)], axis=1)
        bkd.assert_allclose(
            mesh.boundary_normals(boundary), sign * direction, atol=1e-13
        )


class _Reversed(Generic[Array]):
    """x = -2 xi: a 1D map reversing orientation."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def ndim(self) -> int:
        return 1

    def map_to_physical(self, reference_pts: Array) -> Array:
        return -2.0 * reference_pts

    def map_to_reference(self, physical_pts: Array) -> Array:
        return -0.5 * physical_pts

    def jacobian_matrix(self, reference_pts: Array) -> Array:
        return self._bkd.full((reference_pts.shape[1], 1, 1), -2.0)

    def jacobian_determinant(self, reference_pts: Array) -> Array:
        return self._bkd.full((reference_pts.shape[1],), -2.0)

    def gradient_factors(self, reference_pts: Array) -> Array:
        return self._bkd.full((reference_pts.shape[1], 1, 1), -0.5)


def test_reversed_1d_map_normals_point_outward(bkd: Backend[Array]) -> None:
    """Under x = -2 xi, boundary 0 (xi = -1) sits at x = 2, so its
    outward normal is +1, and boundary 1's is -1."""
    mesh = TransformedMesh1D(5, bkd, _Reversed(bkd))
    bkd.assert_allclose(mesh.points()[0][mesh.boundary_indices(0)], bkd.asarray([2.0]))
    bkd.assert_allclose(mesh.boundary_normals(0), bkd.asarray([[1.0]]))
    bkd.assert_allclose(mesh.boundary_normals(1), bkd.asarray([[-1.0]]))


class _Uniform:
    """Equispaced nodes on [-1, 1]."""

    def nodes(self, npts: int, bkd: Backend[Array]) -> Array:
        return bkd.linspace(-1.0, 1.0, npts)


def test_node_generator_per_direction(bkd: Backend[Array]) -> None:
    """Equispaced nodes in the second direction, Chebyshev in the first:
    each direction uses its own generator, first direction fastest."""
    chebyshev = TransformedMesh2D(4, 5, bkd)
    mesh = TransformedMesh(
        (4, 5), bkd, nodes=[ChebyshevGaussLobattoNodes[Array](), _Uniform()]
    )
    bkd.assert_allclose(mesh.reference_nodes_1d(0), chebyshev.reference_nodes_1d(0))
    bkd.assert_allclose(mesh.reference_nodes_1d(1), bkd.linspace(-1.0, 1.0, 5))
    bkd.assert_allclose(
        mesh.reference_points()[1], bkd.repeat(bkd.linspace(-1.0, 1.0, 5), 4)
    )


@pytest.mark.parametrize("npts", [1, 2, 5, 8])
def test_chebyshev_gauss_nodes(bkd: Backend[Array], npts: int) -> None:
    """Chebyshev-Gauss nodes are the n roots of T_n = cos(n arccos x),
    increasing, symmetric about 0 and strictly inside (-1, 1); one node
    is the midpoint."""
    nodes = ChebyshevGaussNodes[Array]().nodes(npts, bkd)
    assert nodes.shape == (npts,)
    bkd.assert_allclose(
        bkd.cos(npts * bkd.arccos(nodes)), bkd.zeros((npts,)), atol=1e-14
    )
    assert bkd.to_bool(bkd.all(bkd.diff(nodes) > 0))
    bkd.assert_allclose(nodes, -bkd.flip(nodes), atol=1e-15)
    assert bkd.to_bool(bkd.all(bkd.abs(nodes) < 1.0))
    with pytest.raises(ValueError, match="at least 1 node"):
        ChebyshevGaussNodes[Array]().nodes(0, bkd)


def test_mesh_on_gauss_nodes_has_no_boundary(bkd: Backend[Array]) -> None:
    """A 4 x 3 mesh of [0, 2] x [1, 3], Chebyshev-Gauss-Lobatto in x and
    Chebyshev-Gauss in y: the points are the affine images of the
    reference nodes, the x sides exist with their 3 nodes, and the y
    sides, which no node reaches, are refused."""
    gauss = ChebyshevGaussNodes[Array]()
    transform = AffineTransform([(0.0, 2.0), (1.0, 3.0)], bkd)
    mesh = TransformedMesh(
        (4, 3),
        bkd,
        transform,
        nodes=[ChebyshevGaussLobattoNodes[Array](), gauss],
    )
    y = gauss.nodes(3, bkd)
    bkd.assert_allclose(mesh.points()[1], bkd.repeat(y + 2.0, 4), atol=1e-15)
    assert mesh.boundary_indices(0).shape == (3,)
    bkd.assert_allclose(mesh.points()[0][mesh.boundary_indices(1)], bkd.full((3,), 2.0))
    for boundary, end in [(2, "lower"), (3, "upper")]:
        with pytest.raises(ValueError, match=f"do not reach its {end} end"):
            mesh.boundary_indices(boundary)
        with pytest.raises(ValueError, match=f"boundary {boundary} does not exist"):
            mesh.boundary_normals(boundary)


def test_one_gauss_node(bkd: Backend[Array]) -> None:
    """A direction may hold a single Gauss node, the interval's midpoint:
    the coarsest pressure grid under a 3-node velocity direction."""
    gauss = ChebyshevGaussNodes[Array]()
    mesh = TransformedMesh((1, 1), bkd, nodes=[gauss, gauss])
    bkd.assert_allclose(mesh.points(), bkd.zeros((2, 1)), atol=1e-15)


def test_per_dimension_meshes_are_the_generic_mesh(bkd: Backend[Array]) -> None:
    """The 1D/2D/3D classes are TransformedMesh with their counts."""
    for mesh, counts in [
        (TransformedMesh1D(6, bkd), (6,)),
        (TransformedMesh2D(3, 4, bkd), (3, 4)),
        (TransformedMesh3D(3, 2, 4, bkd), (3, 2, 4)),
    ]:
        assert isinstance(mesh, TransformedMesh)
        assert mesh.npts_per_dim() == counts
        assert mesh.npts() == int(np.prod(counts))


def test_refusals(bkd: Backend[Array]) -> None:
    """No directions, a direction with no nodes, one Gauss-Lobatto node
    (refused by its generator), a transform or node list of the wrong
    length, a non-generator, an unknown boundary or direction."""
    with pytest.raises(ValueError, match="at least one direction"):
        TransformedMesh((), bkd)
    with pytest.raises(ValueError, match="at least 1 node"):
        TransformedMesh((3, 0), bkd)
    with pytest.raises(ValueError, match="Lobatto nodes need at least 2 nodes"):
        TransformedMesh((3, 1), bkd)
    with pytest.raises(ValueError, match="the transform has 1 directions, the mesh 2"):
        TransformedMesh((3, 3), bkd, AffineTransform([(0.0, 1.0)], bkd))
    with pytest.raises(ValueError, match="1 node generators for 2 directions"):
        TransformedMesh((3, 3), bkd, nodes=[_Uniform()])
    with pytest.raises(TypeError, match="NodeGeneratorProtocol"):
        TransformedMesh((3,), bkd, nodes=[3])  # type: ignore[list-item]
    mesh = TransformedMesh2D(3, 3, bkd)
    with pytest.raises(ValueError, match=r"boundary_id must be in \[0, 4\)"):
        mesh.boundary_normals(4)
    with pytest.raises(ValueError, match=r"dim must be in \[0, 2\)"):
        mesh.reference_nodes_1d(2)
