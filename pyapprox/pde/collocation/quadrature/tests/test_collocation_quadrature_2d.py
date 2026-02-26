"""Tests for 2D collocation quadrature weights."""

import math
from typing import Any

import torch
from numpy.typing import NDArray

from pyapprox.pde.collocation.basis import ChebyshevBasis2D
from pyapprox.pde.collocation.mesh import TransformedMesh2D
from pyapprox.pde.collocation.mesh.transforms import PolarTransform
from pyapprox.pde.collocation.mesh.transforms.affine import (
    AffineTransform2D,
)
from pyapprox.pde.collocation.quadrature import (
    CollocationQuadrature2D,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
class TestCollocationQuadrature2D:
    def test_full_domain_weights_shape(self, bkd):
        """Full-domain weights have correct shape and are positive."""
        nx, ny = 8, 6
        mesh = TransformedMesh2D(nx, ny, bkd)
        basis = ChebyshevBasis2D(mesh, bkd)
        quad = CollocationQuadrature2D(basis, bkd)
        w = quad.full_domain_weights()
        assert w.shape == (nx * ny,)
        # On reference [-1,1]^2, all weights should be positive
        assert float(bkd.min(w)) > 0.0

    def test_reference_area(self, bkd):
        """Integral of 1 on reference [-1,1]^2 = 4."""
        nx, ny = 8, 8
        mesh = TransformedMesh2D(nx, ny, bkd)
        basis = ChebyshevBasis2D(mesh, bkd)
        quad = CollocationQuadrature2D(basis, bkd)
        w = quad.full_domain_weights()
        area = bkd.sum(w)
        bkd.assert_allclose(
            bkd.reshape(area, (1,)),
            bkd.asarray([4.0]),
            rtol=1e-12,
        )

    def test_cartesian_area(self, bkd):
        """Integral of 1 on [0,2] x [0,3] = 6."""
        nx, ny = 8, 8
        transform = AffineTransform2D((0.0, 2.0, 0.0, 3.0), bkd)
        mesh = TransformedMesh2D(nx, ny, bkd, transform)
        basis = ChebyshevBasis2D(mesh, bkd)
        quad = CollocationQuadrature2D(basis, bkd)
        w = quad.full_domain_weights()
        area = bkd.sum(w)
        bkd.assert_allclose(
            bkd.reshape(area, (1,)),
            bkd.asarray([6.0]),
            rtol=1e-12,
        )

    def test_polynomial_exactness_cartesian(self, bkd):
        """Integrate x^a * y^b on [0,1]^2 for low-degree polynomials."""
        nx, ny = 10, 10
        transform = AffineTransform2D((0.0, 1.0, 0.0, 1.0), bkd)
        mesh = TransformedMesh2D(nx, ny, bkd, transform)
        basis = ChebyshevBasis2D(mesh, bkd)
        quad = CollocationQuadrature2D(basis, bkd)
        w = quad.full_domain_weights()
        pts = mesh.points()  # (2, npts)
        x = pts[0, :]
        y = pts[1, :]

        for a in range(5):
            for b in range(5):
                f_vals = x**a * y**b
                numerical = bkd.sum(w * f_vals)
                exact = 1.0 / ((a + 1) * (b + 1))
                bkd.assert_allclose(
                    bkd.reshape(numerical, (1,)),
                    bkd.asarray([exact]),
                    rtol=1e-10,
                )

    def test_polar_area(self, bkd):
        """Integral of 1 on quarter-annulus = pi/4 * (Ro^2 - Ri^2)."""
        r_inner, r_outer = 1.0, 2.0
        nx, ny = 10, 10
        transform = PolarTransform(
            (r_inner, r_outer),
            (0.0, math.pi / 2.0),
            bkd,
        )
        mesh = TransformedMesh2D(nx, ny, bkd, transform)
        basis = ChebyshevBasis2D(mesh, bkd)
        quad = CollocationQuadrature2D(basis, bkd)
        w = quad.full_domain_weights()
        area = bkd.sum(w)
        exact_area = math.pi / 4.0 * (r_outer**2 - r_inner**2)
        bkd.assert_allclose(
            bkd.reshape(area, (1,)),
            bkd.asarray([exact_area]),
            rtol=1e-10,
        )

    def test_polar_integral_x2_y2(self, bkd):
        """Integrate x^2 + y^2 = r^2 on quarter-annulus.

        Exact: integral_0^{pi/2} integral_{Ri}^{Ro} r^2 * r dr dtheta
             = pi/2 * (Ro^4 - Ri^4) / 4
        """
        r_inner, r_outer = 1.0, 2.0
        nx, ny = 12, 12
        transform = PolarTransform(
            (r_inner, r_outer),
            (0.0, math.pi / 2.0),
            bkd,
        )
        mesh = TransformedMesh2D(nx, ny, bkd, transform)
        basis = ChebyshevBasis2D(mesh, bkd)
        quad = CollocationQuadrature2D(basis, bkd)
        w = quad.full_domain_weights()
        pts = mesh.points()  # (2, npts)
        f_vals = pts[0, :] ** 2 + pts[1, :] ** 2
        numerical = bkd.sum(w * f_vals)
        exact = (math.pi / 2.0) * (r_outer**4 - r_inner**4) / 4.0
        bkd.assert_allclose(
            bkd.reshape(numerical, (1,)),
            bkd.asarray([exact]),
            rtol=1e-10,
        )

    def test_subdomain_weights_reference(self, bkd):
        """Subdomain weights on a sub-rectangle of reference domain."""
        nx, ny = 10, 10
        mesh = TransformedMesh2D(nx, ny, bkd)
        basis = ChebyshevBasis2D(mesh, bkd)
        quad = CollocationQuadrature2D(basis, bkd)

        # Subdomain [-1, 0] x [-1, 0] => area = 1
        w_sub = quad.weights(x_bounds=(-1.0, 0.0), y_bounds=(-1.0, 0.0))
        area_sub = bkd.sum(w_sub)
        bkd.assert_allclose(
            bkd.reshape(area_sub, (1,)),
            bkd.asarray([1.0]),
            rtol=1e-10,
        )

    def test_subdomain_polar_sub_annulus(self, bkd):
        """Subdomain on inner half of quarter-annulus.

        Inner half: r in [Ri, (Ri+Ro)/2], theta in [0, pi/2]
        In reference: x in [-1, 0], y in [-1, 1] (full theta range)
        Area = pi/4 * ((Ri+Ro)^2/4 - Ri^2)
        """
        r_inner, r_outer = 1.0, 3.0
        r_mid = (r_inner + r_outer) / 2.0
        nx, ny = 12, 12
        transform = PolarTransform(
            (r_inner, r_outer),
            (0.0, math.pi / 2.0),
            bkd,
        )
        mesh = TransformedMesh2D(nx, ny, bkd, transform)
        basis = ChebyshevBasis2D(mesh, bkd)
        quad = CollocationQuadrature2D(basis, bkd)

        # Inner half: x_ref in [-1, 0]
        w_sub = quad.weights(x_bounds=(-1.0, 0.0))
        area_sub = bkd.sum(w_sub)
        exact_area = math.pi / 4.0 * (r_mid**2 - r_inner**2)
        bkd.assert_allclose(
            bkd.reshape(area_sub, (1,)),
            bkd.asarray([exact_area]),
            rtol=1e-10,
        )

    def test_subdomain_sum_equals_full(self, bkd):
        """Left + right subdomain weights sum to full-domain weights."""
        nx, ny = 8, 8
        mesh = TransformedMesh2D(nx, ny, bkd)
        basis = ChebyshevBasis2D(mesh, bkd)
        quad = CollocationQuadrature2D(basis, bkd)

        w_full = quad.full_domain_weights()
        w_left = quad.weights(x_bounds=(-1.0, 0.0))
        w_right = quad.weights(x_bounds=(0.0, 1.0))
        bkd.assert_allclose(w_left + w_right, w_full, rtol=1e-12)
