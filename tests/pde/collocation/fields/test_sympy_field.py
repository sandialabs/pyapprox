"""Tests for SympyField2D and field factory functions."""

import pytest

from pyapprox.pde.collocation.fields.sympy_field import (
    SympyField2D,
    create_beta_surface,
    create_polynomial_surface,
    create_quadratic_bed,
    create_shallow_wave_bed,
)
from pyapprox.pde.collocation.mesh.transformed import TransformedMesh2D
from pyapprox.pde.collocation.mesh.transforms.affine import (
    AffineTransform2D,
)


class TestSympyField2D:
    """Tests for SympyField2D."""

    def test_constant_field(self, bkd):
        """Test constant field evaluation."""
        transform = AffineTransform2D((0.0, 1.0, 0.0, 1.0), bkd)
        mesh = TransformedMesh2D(10, 10, bkd, transform)

        field = SympyField2D("5.0", {}, mesh, bkd)
        vals = field.evaluate()

        bkd.assert_allclose(vals, 5.0 * bkd.ones((mesh.npts(),)), rtol=1e-14)

    def test_linear_field_normalized(self, bkd):
        """Test linear field with normalized coordinates."""
        transform = AffineTransform2D((0.0, 2.0, 0.0, 3.0), bkd)
        mesh = TransformedMesh2D(10, 10, bkd, transform)

        # f = 2*xn + 3*yn
        field = SympyField2D("2*xn + 3*yn", {}, mesh, bkd, coord_type="normalized")
        vals = field.evaluate()

        # Compute expected values
        pts = mesh.points()
        x_phys = pts[0, :]
        y_phys = pts[1, :]
        xn = x_phys / 2.0  # Normalize to [0, 1]
        yn = y_phys / 3.0
        expected = 2.0 * xn + 3.0 * yn

        bkd.assert_allclose(vals, expected, rtol=1e-12)

    def test_linear_field_physical(self, bkd):
        """Test linear field with physical coordinates."""
        transform = AffineTransform2D((0.0, 2.0, 0.0, 3.0), bkd)
        mesh = TransformedMesh2D(10, 10, bkd, transform)

        # f = x + y in physical coords
        field = SympyField2D("x + y", {}, mesh, bkd, coord_type="physical")
        vals = field.evaluate()

        pts = mesh.points()
        expected = pts[0, :] + pts[1, :]

        bkd.assert_allclose(vals, expected, rtol=1e-12)

    def test_quadratic_bed_y_direction(self, bkd):
        """Test quadratic bed matches expected formula."""
        transform = AffineTransform2D((0.0, 1.0, 0.0, 1.0), bkd)
        mesh = TransformedMesh2D(15, 15, bkd, transform)

        bed = create_quadratic_bed(mesh, bkd, mean=10.0, amplitude=1.0, direction="y")
        vals = bed.evaluate()

        # Expected: 10 - 1 * yn * (1 - yn)
        pts = mesh.points()
        yn = pts[1, :]  # On [0,1] domain, yn = y_phys
        expected = 10.0 - 1.0 * yn * (1.0 - yn)

        bkd.assert_allclose(vals, expected, rtol=1e-12)

    def test_quadratic_bed_x_direction(self, bkd):
        """Test quadratic bed in x direction."""
        transform = AffineTransform2D((0.0, 1.0, 0.0, 1.0), bkd)
        mesh = TransformedMesh2D(15, 15, bkd, transform)

        bed = create_quadratic_bed(mesh, bkd, mean=5.0, amplitude=2.0, direction="x")
        vals = bed.evaluate()

        pts = mesh.points()
        xn = pts[0, :]
        expected = 5.0 - 2.0 * xn * (1.0 - xn)

        bkd.assert_allclose(vals, expected, rtol=1e-12)

    def test_quadratic_bed_xy_direction(self, bkd):
        """Test quadratic bed in both directions."""
        transform = AffineTransform2D((0.0, 1.0, 0.0, 1.0), bkd)
        mesh = TransformedMesh2D(15, 15, bkd, transform)

        bed = create_quadratic_bed(mesh, bkd, mean=10.0, amplitude=4.0, direction="xy")
        vals = bed.evaluate()

        pts = mesh.points()
        xn = pts[0, :]
        yn = pts[1, :]
        expected = 10.0 - 4.0 * xn * (1.0 - xn) * yn * (1.0 - yn)

        bkd.assert_allclose(vals, expected, rtol=1e-12)

    def test_polynomial_surface(self, bkd):
        """Test polynomial surface factory."""
        transform = AffineTransform2D((0.0, 1.0, 0.0, 1.0), bkd)
        mesh = TransformedMesh2D(12, 12, bkd, transform)

        # surface = (1 + 2*xn) * (3 + 4*yn)
        surface = create_polynomial_surface(mesh, bkd, [1.0, 2.0], [3.0, 4.0])
        vals = surface.evaluate()

        pts = mesh.points()
        xn = pts[0, :]
        yn = pts[1, :]
        expected = (1.0 + 2.0 * xn) * (3.0 + 4.0 * yn)

        bkd.assert_allclose(vals, expected, rtol=1e-12)

    def test_beta_surface(self, bkd):
        """Test beta function surface factory."""
        transform = AffineTransform2D((0.0, 1.0, 0.0, 1.0), bkd)
        mesh = TransformedMesh2D(15, 15, bkd, transform)

        # Simple beta parameters
        surface = create_beta_surface(
            mesh, bkd, a0=2.0, b0=2.0, a1=2.0, b1=2.0, scale=1.0
        )
        vals = surface.evaluate()

        # Check that values are positive and have maximum in interior
        assert float(bkd.min(vals) >= 0)

        # Find maximum - should be near center for symmetric beta(2,2)
        max_val = float(bkd.max(vals))
        assert max_val > 0

    def test_shallow_wave_bed_legacy(self, bkd):
        """Test legacy shallow wave bed formula."""
        transform = AffineTransform2D((0.0, 10.0, 0.0, 5.0), bkd)
        mesh = TransformedMesh2D(20, 20, bkd, transform)

        bed = create_shallow_wave_bed(mesh, bkd)
        vals = bed.evaluate()

        # Verify formula: -1 + (-0.1 + yn*(yn-1)) * (1 - 0.9*xn)
        pts = mesh.points()
        xn = pts[0, :] / 10.0
        yn = pts[1, :] / 5.0
        expected = -1.0 + (-0.1 + yn * (yn - 1.0)) * (1.0 - 0.9 * xn)

        bkd.assert_allclose(vals, expected, rtol=1e-12)

    def test_gradient_constant(self, bkd):
        """Test gradient of constant field is zero."""
        transform = AffineTransform2D((0.0, 2.0, 0.0, 3.0), bkd)
        mesh = TransformedMesh2D(10, 10, bkd, transform)

        field = SympyField2D("7.5", {}, mesh, bkd)
        grad = field.gradient()

        bkd.assert_allclose(grad[0], bkd.zeros((mesh.npts(),)), atol=1e-14)
        bkd.assert_allclose(grad[1], bkd.zeros((mesh.npts(),)), atol=1e-14)

    def test_gradient_linear_normalized(self, bkd):
        """Test gradient of linear field with normalized coords."""
        Lx, Ly = 2.0, 3.0
        transform = AffineTransform2D((0.0, Lx, 0.0, Ly), bkd)
        mesh = TransformedMesh2D(12, 12, bkd, transform)

        # f = 5*xn + 7*yn
        # df/dx_phys = (df/dxn) * (dxn/dx_phys) = 5 * (1/Lx) = 5/2
        # df/dy_phys = 7 * (1/Ly) = 7/3
        field = SympyField2D("5*xn + 7*yn", {}, mesh, bkd, coord_type="normalized")
        grad = field.gradient()

        expected_dx = 5.0 / Lx
        expected_dy = 7.0 / Ly

        bkd.assert_allclose(grad[0], expected_dx * bkd.ones((mesh.npts(),)), rtol=1e-12)
        bkd.assert_allclose(grad[1], expected_dy * bkd.ones((mesh.npts(),)), rtol=1e-12)

    def test_gradient_quadratic_normalized(self, bkd):
        """Test gradient of quadratic field."""
        Lx, Ly = 2.0, 3.0
        transform = AffineTransform2D((0.0, Lx, 0.0, Ly), bkd)
        mesh = TransformedMesh2D(15, 15, bkd, transform)

        # f = xn^2 + 2*yn^3
        # df/dxn = 2*xn, df/dyn = 6*yn^2
        # df/dx_phys = 2*xn / Lx, df/dy_phys = 6*yn^2 / Ly
        field = SympyField2D("xn**2 + 2*yn**3", {}, mesh, bkd, coord_type="normalized")
        grad = field.gradient()

        pts = mesh.points()
        xn = pts[0, :] / Lx
        yn = pts[1, :] / Ly

        expected_dx = 2.0 * xn / Lx
        expected_dy = 6.0 * yn**2 / Ly

        bkd.assert_allclose(grad[0], expected_dx, rtol=1e-10)
        bkd.assert_allclose(grad[1], expected_dy, rtol=1e-10)

    def test_gradient_physical_coords(self, bkd):
        """Test gradient with physical coordinates."""
        transform = AffineTransform2D((1.0, 3.0, 2.0, 5.0), bkd)
        mesh = TransformedMesh2D(12, 12, bkd, transform)

        # f = x^2 + y in physical coords
        # df/dx = 2*x, df/dy = 1
        field = SympyField2D("x**2 + y", {}, mesh, bkd, coord_type="physical")
        grad = field.gradient()

        pts = mesh.points()
        expected_dx = 2.0 * pts[0, :]
        expected_dy = bkd.ones((mesh.npts(),))

        bkd.assert_allclose(grad[0], expected_dx, rtol=1e-12)
        bkd.assert_allclose(grad[1], expected_dy, rtol=1e-12)

    def test_params_substitution(self, bkd):
        """Test parameter substitution works correctly."""
        transform = AffineTransform2D((0.0, 1.0, 0.0, 1.0), bkd)
        mesh = TransformedMesh2D(10, 10, bkd, transform)

        field = SympyField2D(
            "a * xn + b * yn + c",
            {"a": 2.0, "b": 3.0, "c": 5.0},
            mesh,
            bkd,
        )
        vals = field.evaluate()

        pts = mesh.points()
        xn = pts[0, :]
        yn = pts[1, :]
        expected = 2.0 * xn + 3.0 * yn + 5.0

        bkd.assert_allclose(vals, expected, rtol=1e-12)

    def test_invalid_coord_type_raises(self, bkd):
        """Test invalid coord_type raises ValueError."""
        transform = AffineTransform2D((0.0, 1.0, 0.0, 1.0), bkd)
        mesh = TransformedMesh2D(10, 10, bkd, transform)

        with pytest.raises(ValueError):
            SympyField2D("xn", {}, mesh, bkd, coord_type="invalid")
