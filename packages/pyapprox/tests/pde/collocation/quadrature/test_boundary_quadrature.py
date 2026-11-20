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
- Refusals: a side the mesh does not have; a mesh outside the protocol.
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
    boundary_quadrature_weights,
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
