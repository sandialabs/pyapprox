r"""Quadrature weights on one side of a collocation mesh; NumPy and Torch.

- On [0, 2] x [0, 3] (6 x 7 nodes), each side's weights integrate
  :math:`t^a`, a < 6 along the side, exactly (1e-12).
- On [0, 1] x [0, 2] x [0, 3] (4 x 5 x 6 nodes), the z = 0 and z = 3
  sides integrate :math:`x^a y^b` (a < 4, b < 5) exactly, and the x = 1
  side has area 6.
- On a quarter annulus (polar, r in [1, 2]), the outer arc has length
  :math:`\pi`, the inner :math:`\pi/2`, a radial side 1, and the outer
  arc integrates :math:`y^2` to :math:`2 r^3 \pi/4 = 2\pi` (1e-10).
- A 1D side is a point of weight 1.
- Refusals: a side the mesh does not have; a mesh outside the protocol;
  a rule outside its protocol; a negative extra point count.

Side quadratures by a rule (:func:`boundary_side_quadrature`), on the box
above:

- The node rule's interpolation is the identity, its points the side's
  nodes and its weights the lumped weights (1e-13), on sides x = 0,
  y = 2, z = 3.
- The Gauss rule integrates the product of two side interpolants of
  degree 2n - 3 per direction exactly (1e-12) on sides x = 1, y = 0,
  z = 0, where the lumped weights miss by more than 1e-4.
- On the quarter annulus with two extra points, the outer arc's length
  and the integral of the interpolated y^2 (1e-10), the points on r = 2.
- A 1D side by any rule is its node, weight 1.
"""

import math

import pytest

from pyapprox.pde.collocation.mesh import (
    TransformedMesh1D,
    TransformedMesh2D,
)
from pyapprox.pde.collocation.mesh.transformed import TransformedMesh3D
from pyapprox.pde.collocation.mesh.transforms import PolarTransform
from pyapprox.pde.collocation.mesh.transforms.affine import (
    AffineTransform1D,
    AffineTransform2D,
    AffineTransform3D,
)
from pyapprox.pde.collocation.quadrature.boundary_quadrature import (
    CollocationNodeRule,
    GaussLegendreSideRule,
    boundary_quadrature_weights,
    boundary_side_quadrature,
)
from pyapprox.util.backends.protocols import Array, Backend


def _close(bkd: Backend[Array], value: Array, expected: float, rtol: float) -> None:
    bkd.assert_allclose(bkd.reshape(value, (1,)), bkd.asarray([expected]), rtol=rtol)


@pytest.mark.parametrize(
    "boundary_id, along, length", [(0, 1, 3.0), (1, 1, 3.0), (2, 0, 2.0), (3, 0, 2.0)]
)
def test_rectangle_sides_integrate_polynomials(
    bkd: Backend[Array], boundary_id: int, along: int, length: float
) -> None:
    mesh = TransformedMesh2D(6, 7, bkd, AffineTransform2D((0.0, 2.0, 0.0, 3.0), bkd))
    weights = boundary_quadrature_weights(mesh, boundary_id)
    t = mesh.points()[along, mesh.boundary_indices(boundary_id)]
    for a in range(6):
        _close(bkd, bkd.sum(weights * t**a), length ** (a + 1) / (a + 1), 1e-12)


def test_box_sides_integrate_polynomials(bkd: Backend[Array]) -> None:
    mesh = TransformedMesh3D(
        4, 5, 6, bkd, AffineTransform3D((0.0, 1.0, 0.0, 2.0, 0.0, 3.0), bkd)
    )
    for boundary_id in (4, 5):
        weights = boundary_quadrature_weights(mesh, boundary_id)
        points = mesh.points()[:, mesh.boundary_indices(boundary_id)]
        for a in range(4):
            for b in range(5):
                integral = bkd.sum(weights * points[0] ** a * points[1] ** b)
                expected = 2.0 ** (b + 1) / ((a + 1) * (b + 1))
                _close(bkd, integral, expected, 1e-12)
    _close(bkd, bkd.sum(boundary_quadrature_weights(mesh, 1)), 6.0, 1e-12)


def test_curved_sides(bkd: Backend[Array]) -> None:
    transform = PolarTransform((1.0, 2.0), (0.0, math.pi / 2.0), bkd)
    mesh = TransformedMesh2D(10, 12, bkd, transform)
    outer = boundary_quadrature_weights(mesh, 1)
    _close(bkd, bkd.sum(outer), math.pi, 1e-10)
    _close(bkd, bkd.sum(boundary_quadrature_weights(mesh, 0)), math.pi / 2.0, 1e-10)
    _close(bkd, bkd.sum(boundary_quadrature_weights(mesh, 2)), 1.0, 1e-10)
    y = mesh.points()[1, mesh.boundary_indices(1)]
    _close(bkd, bkd.sum(outer * y**2), 2.0 * math.pi, 1e-10)


def test_a_1d_side_is_a_unit_point(bkd: Backend[Array]) -> None:
    mesh = TransformedMesh1D(5, bkd, AffineTransform1D((0.0, 3.0), bkd))
    for boundary_id in (0, 1):
        bkd.assert_allclose(
            boundary_quadrature_weights(mesh, boundary_id), bkd.ones((1,)), rtol=1e-14
        )


def test_refusals(bkd: Backend[Array]) -> None:
    mesh = TransformedMesh2D(4, 4, bkd)
    with pytest.raises(ValueError, match=r"\[0, 4\)"):
        boundary_quadrature_weights(mesh, 4)
    with pytest.raises(TypeError, match="MeshWithTransformProtocol"):
        boundary_quadrature_weights(object(), 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"\[0, 4\)"):
        boundary_side_quadrature(mesh, 4, CollocationNodeRule())
    with pytest.raises(TypeError, match="SideRuleProtocol"):
        boundary_side_quadrature(mesh, 0, object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="extra must be >= 0"):
        GaussLegendreSideRule(-1)


def _box(bkd: Backend[Array]) -> TransformedMesh3D[Array]:
    return TransformedMesh3D(
        4, 5, 6, bkd, AffineTransform3D((0.0, 1.0, 0.0, 2.0, 0.0, 3.0), bkd)
    )


@pytest.mark.parametrize("boundary_id", [0, 3, 5])
def test_the_node_rule_is_the_lumped_weights(
    bkd: Backend[Array], boundary_id: int
) -> None:
    mesh = _box(bkd)
    side = boundary_side_quadrature(mesh, boundary_id, CollocationNodeRule())
    nodes = mesh.boundary_indices(boundary_id)
    bkd.assert_allclose(
        side.interpolation, bkd.eye(int(nodes.shape[0])), rtol=0.0, atol=1e-14
    )
    bkd.assert_allclose(side.points, mesh.points()[:, nodes], rtol=1e-14, atol=1e-14)
    bkd.assert_allclose(
        side.weights, boundary_quadrature_weights(mesh, boundary_id), rtol=1e-13
    )


@pytest.mark.parametrize("boundary_id", [1, 2, 4])
def test_the_gauss_rule_integrates_products_of_interpolants(
    bkd: Backend[Array], boundary_id: int
) -> None:
    """f g of degree a + c < 2 n - 1 per direction, f and g sampled at the
    side's nodes and interpolated: exact, though the lumped rule is not."""
    mesh = _box(bkd)
    nodes = mesh.points()[:, mesh.boundary_indices(boundary_id)]
    tangents = [d for d in range(3) if d != boundary_id // 2]
    u, v = tangents
    nu, nv = mesh.npts_per_dim()[u], mesh.npts_per_dim()[v]
    lengths = (1.0, 2.0, 3.0)
    side = boundary_side_quadrature(mesh, boundary_id, GaussLegendreSideRule())
    lumped = boundary_quadrature_weights(mesh, boundary_id)
    f = nodes[u] ** (nu - 1) * nodes[v] ** (nv - 2)
    g = nodes[u] ** (nu - 2) * nodes[v] ** (nv - 1)
    a, b = 2 * nu - 3, 2 * nv - 3
    expected = lengths[u] ** (a + 1) / (a + 1) * lengths[v] ** (b + 1) / (b + 1)
    exact = bkd.sum(side.weights * (side.interpolation @ f) * (side.interpolation @ g))
    _close(bkd, exact, expected, 1e-12)
    error = abs(bkd.to_float(bkd.sum(lumped * f * g)) - expected) / expected
    assert error > 1e-4


def test_the_gauss_rule_on_a_curved_side(bkd: Backend[Array]) -> None:
    """The outer arc of a quarter annulus: its length, and y^2 against it,
    as the lumped weights give them (1e-10); the points lie on the arc."""
    transform = PolarTransform((1.0, 2.0), (0.0, math.pi / 2.0), bkd)
    mesh = TransformedMesh2D(10, 12, bkd, transform)
    side = boundary_side_quadrature(mesh, 1, GaussLegendreSideRule(2))
    _close(bkd, bkd.sum(side.weights), math.pi, 1e-10)
    y = side.interpolation @ mesh.points()[1, mesh.boundary_indices(1)]
    _close(bkd, bkd.sum(side.weights * y**2), 2.0 * math.pi, 1e-10)
    radius = bkd.sqrt(bkd.sum(side.points**2, axis=0))
    bkd.assert_allclose(radius, 2.0 * bkd.ones((14,)), rtol=1e-13)


def test_a_1d_side_by_any_rule_is_its_node(bkd: Backend[Array]) -> None:
    mesh = TransformedMesh1D(5, bkd, AffineTransform1D((0.0, 3.0), bkd))
    side = boundary_side_quadrature(mesh, 1, GaussLegendreSideRule())
    bkd.assert_allclose(side.weights, bkd.ones((1,)), rtol=1e-14)
    bkd.assert_allclose(side.interpolation, bkd.ones((1, 1)), rtol=1e-14)
    bkd.assert_allclose(side.points, bkd.asarray([[3.0]]), rtol=1e-14)
