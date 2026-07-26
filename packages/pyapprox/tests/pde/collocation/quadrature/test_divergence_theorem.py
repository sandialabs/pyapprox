"""Divergence-theorem identity on a transformed (polar) domain.

Jointly validates the three transform-sensitive discretization seams
that residual-level transform tests do not cover together: physical
derivative matrices (divergence), Jacobian-weighted domain quadrature,
and curved-boundary normals with the surface measure.

For :math:`F = (x^3, y^3)` on the half annulus
:math:`r \\in (1, 2)`, :math:`\\theta \\in (-\\pi/2, \\pi/2)`:

.. math::

    \\int_\\Omega \\nabla \\cdot F \\, dV
    = \\oint_{\\partial\\Omega} F \\cdot n \\, dS
    = \\frac{45\\pi}{4},

with the ray edges (:math:`x = 0`) contributing exactly zero.
"""

import math

import numpy as np
from pyapprox.pde.collocation.basis import ChebyshevBasis1D, ChebyshevBasis2D
from pyapprox.pde.collocation.mesh import TransformedMesh1D, TransformedMesh2D
from pyapprox.pde.collocation.mesh.transforms import PolarTransform
from pyapprox.pde.collocation.quadrature import (
    CollocationQuadrature1D,
    CollocationQuadrature2D,
)
from pyapprox.util.backends.numpy import NumpyBkd

_NPTS_1D = 24
_R_BOUNDS = (1.0, 2.0)
_THETA_BOUNDS = (-math.pi / 2, math.pi / 2)
_EXACT = 45.0 * math.pi / 4.0


def _build_polar_basis(bkd):
    transform = PolarTransform(
        r_bounds=_R_BOUNDS, theta_bounds=_THETA_BOUNDS, bkd=bkd
    )
    mesh = TransformedMesh2D(_NPTS_1D, _NPTS_1D, bkd, transform)
    return ChebyshevBasis2D(mesh, bkd), mesh


def _arc_integral(bkd, pts, f1, f2, arc_radius, outward_sign):
    """Discrete integral of F.n over the arc r = arc_radius.

    Extracts the arc's tensor-grid points (index bookkeeping in
    numpy), aligns them with a 1D CGL quadrature rule in the reference
    edge coordinate, and applies the surface measure
    :math:`ds = r \\, d\\theta`. Returns a length-1 backend array.
    """
    pts_np = bkd.to_numpy(pts)
    radius = np.sqrt(pts_np[0] ** 2 + pts_np[1] ** 2)
    edge = np.where(np.abs(radius - arc_radius) < 1e-10)[0]
    assert edge.shape[0] == _NPTS_1D
    theta = np.arctan2(pts_np[1][edge], pts_np[0][edge])
    edge_sorted = bkd.array(list(edge[np.argsort(theta)]), dtype=int)

    # F.n with n = outward_sign * (x, y)/r at the arc points.
    x_e = pts[0][edge_sorted]
    y_e = pts[1][edge_sorted]
    f_dot_n = (
        outward_sign
        * (f1[edge_sorted] * x_e + f2[edge_sorted] * y_e)
        / arc_radius
    )

    # 1D CGL weights on the reference edge coordinate, aligned to the
    # theta-sorted points (theta is affine increasing in the reference
    # coordinate).
    basis_1d = ChebyshevBasis1D(TransformedMesh1D(_NPTS_1D, bkd), bkd)
    quad_1d = CollocationQuadrature1D(basis_1d, bkd)
    w_ref = quad_1d.weights(-1.0, 1.0)
    node_order = np.argsort(bkd.to_numpy(basis_1d.nodes()))
    w_sorted = w_ref[bkd.array(list(node_order), dtype=int)]

    dtheta_dxi = (_THETA_BOUNDS[1] - _THETA_BOUNDS[0]) / 2.0
    integral = bkd.sum(w_sorted * f_dot_n) * dtheta_dxi * arc_radius
    return bkd.reshape(integral, (1,))


class TestDivergenceTheoremPolar:
    def test_domain_and_boundary_agree_with_analytic(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        bkd = numpy_bkd
        basis, mesh = _build_polar_basis(bkd)
        pts = mesh.points()
        f1 = pts[0] ** 3
        f2 = pts[1] ** 3
        exact = bkd.asarray([_EXACT])

        # Domain side: physical derivative matrices + |J|-weighted
        # quadrature.
        div_f = (
            basis.derivative_matrix(1, 0) @ f1
            + basis.derivative_matrix(1, 1) @ f2
        )
        weights = CollocationQuadrature2D(basis, bkd).full_domain_weights()
        domain_integral = bkd.reshape(weights @ div_f, (1,))
        bkd.assert_allclose(domain_integral, exact, rtol=1e-9)

        # Boundary side: arcs with curved normals and ds = r dtheta;
        # the ray edges lie on x = 0 where F.n = -x^3 vanishes
        # identically.
        boundary_integral = _arc_integral(
            bkd, pts, f1, f2, _R_BOUNDS[1], 1.0
        ) + _arc_integral(bkd, pts, f1, f2, _R_BOUNDS[0], -1.0)
        bkd.assert_allclose(boundary_integral, exact, rtol=1e-9)

        bkd.assert_allclose(domain_integral, boundary_integral, rtol=1e-9)
