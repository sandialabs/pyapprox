"""Tests for user-defined SymPy transforms."""

import math
import unittest
from typing import Any, Generic

import numpy as np
import torch

from numpy.typing import NDArray

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.pde.collocation.mesh.transforms.sympy_transform import (
    SympyTransform2D,
    SympyTransform3D,
)
from pyapprox.typing.pde.collocation.mesh.transforms.polar import PolarTransform
from pyapprox.typing.pde.collocation.mesh.transforms.spherical import (
    SphericalTransform,
)
from pyapprox.typing.pde.collocation.mesh.transforms.elliptical import (
    EllipticalTransform,
)
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401


class TestSympyTransform2D(Generic[Array], unittest.TestCase):
    """Tests for SympyTransform2D."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_polar_via_sympy_matches_native(self):
        """Verify SympyTransform2D matches PolarTransform."""
        bkd = self.bkd()

        # Define polar via sympy with symbolic inverse
        sympy_tf = SympyTransform2D(
            x_expr="r * cos(theta)",
            y_expr="r * sin(theta)",
            params={},
            bkd=bkd,
            coord_names=("r", "theta"),
            bounds=((0.5, 2.0), (-math.pi / 2, math.pi / 2)),
            x_inv_expr="sqrt(x**2 + y**2)",
            y_inv_expr="atan2(y, x)",
        )

        # Native polar transform
        polar_tf = PolarTransform(
            (0.5, 2.0), (-math.pi / 2, math.pi / 2), bkd
        )

        # Test points
        ref_pts = bkd.asarray([[1.0, 1.5, 0.8], [0.0, math.pi / 4, -math.pi / 4]])

        # Compare all methods
        bkd.assert_allclose(
            sympy_tf.map_to_physical(ref_pts),
            polar_tf.map_to_physical(ref_pts),
            rtol=1e-12,
        )
        bkd.assert_allclose(
            sympy_tf.jacobian_matrix(ref_pts),
            polar_tf.jacobian_matrix(ref_pts),
            rtol=1e-12,
        )
        bkd.assert_allclose(
            sympy_tf.jacobian_determinant(ref_pts),
            polar_tf.jacobian_determinant(ref_pts),
            rtol=1e-12,
        )
        bkd.assert_allclose(
            sympy_tf.scale_factors(ref_pts),
            polar_tf.scale_factors(ref_pts),
            rtol=1e-12,
        )
        bkd.assert_allclose(
            sympy_tf.unit_curvilinear_basis(ref_pts),
            polar_tf.unit_curvilinear_basis(ref_pts),
            rtol=1e-12,
        )
        bkd.assert_allclose(
            sympy_tf.gradient_factors(ref_pts),
            polar_tf.gradient_factors(ref_pts),
            rtol=1e-12,
        )

    def test_polar_inverse_via_sympy(self):
        """Test inverse mapping for polar coordinates."""
        bkd = self.bkd()

        sympy_tf = SympyTransform2D(
            x_expr="r * cos(theta)",
            y_expr="r * sin(theta)",
            params={},
            bkd=bkd,
            coord_names=("r", "theta"),
            bounds=((0.5, 2.0), (-math.pi / 2, math.pi / 2)),
            x_inv_expr="sqrt(x**2 + y**2)",
            y_inv_expr="atan2(y, x)",
        )

        ref_pts = bkd.asarray([[1.0, 1.5], [0.3, -0.5]])
        phys_pts = sympy_tf.map_to_physical(ref_pts)
        recovered = sympy_tf.map_to_reference(phys_pts)

        bkd.assert_allclose(recovered, ref_pts, rtol=1e-12)

    def test_elliptical_via_sympy_matches_native(self):
        """Verify SympyTransform2D can reproduce EllipticalTransform."""
        bkd = self.bkd()
        a = 2.0

        # Elliptical via sympy (no inverse provided, relies on validation)
        sympy_tf = SympyTransform2D(
            x_expr="a * cosh(u) * cos(v)",
            y_expr="a * sinh(u) * sin(v)",
            params={"a": a},
            bkd=bkd,
            coord_names=("u", "v"),
            bounds=((0.5, 2.0), (0.1, math.pi - 0.1)),
            validate=True,  # Will validate orthogonality
        )

        # Native elliptical
        ellip_tf = EllipticalTransform(
            (0.5, 2.0), (0.1, math.pi - 0.1), a, bkd
        )

        ref_pts = bkd.asarray([[0.7, 1.0, 1.5], [0.5, 1.0, 2.0]])

        bkd.assert_allclose(
            sympy_tf.map_to_physical(ref_pts),
            ellip_tf.map_to_physical(ref_pts),
            rtol=1e-12,
        )
        bkd.assert_allclose(
            sympy_tf.jacobian_matrix(ref_pts),
            ellip_tf.jacobian_matrix(ref_pts),
            rtol=1e-12,
        )
        bkd.assert_allclose(
            sympy_tf.scale_factors(ref_pts),
            ellip_tf.scale_factors(ref_pts),
            rtol=1e-12,
        )

    def test_orthogonality_validation_passes_for_polar(self):
        """Verify orthogonality validation passes for known orthogonal system."""
        bkd = self.bkd()

        # This should not raise
        tf = SympyTransform2D(
            x_expr="r * cos(theta)",
            y_expr="r * sin(theta)",
            params={},
            bkd=bkd,
            coord_names=("r", "theta"),
            bounds=((0.5, 2.0), (0.0, math.pi)),
            validate=True,
        )
        self.assertEqual(tf.ndim(), 2)

    def test_orthogonality_validation_fails_for_nonorthogonal(self):
        """Verify orthogonality validation fails for non-orthogonal system."""
        bkd = self.bkd()

        # A non-orthogonal coordinate system: x = u + v, y = v
        # Jacobian: [[1, 1], [0, 1]]
        # g = J^T @ J = [[1, 1], [1, 2]] -> off-diagonal != 0
        with self.assertRaises(ValueError) as ctx:
            SympyTransform2D(
                x_expr="u + v",
                y_expr="v",
                params={},
                bkd=bkd,
                coord_names=("u", "v"),
                bounds=((0.0, 1.0), (0.0, 1.0)),
                validate=True,
            )
        self.assertIn("Non-orthogonal", str(ctx.exception))

    def test_numerical_inverse(self):
        """Test numerical inverse when no symbolic inverse provided."""
        bkd = self.bkd()

        # Elliptical without symbolic inverse
        a = 1.5
        tf = SympyTransform2D(
            x_expr="a * cosh(u) * cos(v)",
            y_expr="a * sinh(u) * sin(v)",
            params={"a": a},
            bkd=bkd,
            coord_names=("u", "v"),
            bounds=((0.5, 2.0), (0.3, math.pi - 0.3)),
            validate=False,  # Skip validation to speed up
        )

        # Test a single point (numerical inverse is slow)
        ref_pt = bkd.asarray([[1.0], [0.8]])
        phys_pt = tf.map_to_physical(ref_pt)
        recovered = tf.map_to_reference(phys_pt)

        bkd.assert_allclose(recovered, ref_pt, rtol=1e-8)

    def test_unit_basis_orthonormality(self):
        """Verify unit basis vectors are orthonormal."""
        bkd = self.bkd()

        tf = SympyTransform2D(
            x_expr="r * cos(theta)",
            y_expr="r * sin(theta)",
            params={},
            bkd=bkd,
            coord_names=("r", "theta"),
            bounds=((0.5, 2.0), (0.0, math.pi)),
            validate=False,
        )

        ref_pts = bkd.asarray([[1.0, 1.5], [0.5, 1.5]])
        basis = tf.unit_curvilinear_basis(ref_pts)

        npts = ref_pts.shape[1]

        # Check unit length
        for j in range(2):
            e_j = basis[:, :, j]
            norm_sq = bkd.sum(e_j**2, axis=1)
            bkd.assert_allclose(norm_sq, bkd.ones((npts,)), atol=1e-12)

        # Check orthogonality
        e_0 = basis[:, :, 0]
        e_1 = basis[:, :, 1]
        dot = bkd.sum(e_0 * e_1, axis=1)
        bkd.assert_allclose(dot, bkd.zeros((npts,)), atol=1e-12)


class TestSympyTransform3D(Generic[Array], unittest.TestCase):
    """Tests for SympyTransform3D."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_spherical_via_sympy_map_to_physical(self):
        """Verify SympyTransform3D map_to_physical matches SphericalTransform."""
        bkd = self.bkd()

        # Note: SphericalTransform uses (r, azimuth, elevation) convention
        # We define sympy with standard physics convention (r, theta, phi)
        # where theta is polar angle from z-axis, phi is azimuth
        sympy_tf = SympyTransform3D(
            x_expr="r * sin(theta) * cos(phi)",
            y_expr="r * sin(theta) * sin(phi)",
            z_expr="r * cos(theta)",
            params={},
            bkd=bkd,
            coord_names=("r", "theta", "phi"),
            bounds=((1.0, 2.0), (0.3, math.pi - 0.3), (0.0, math.pi)),
            validate=True,
        )

        ref_pts = bkd.asarray([
            [1.0, 1.5, 2.0],       # r
            [0.5, 1.0, 2.0],       # theta
            [0.3, 1.0, 2.5],       # phi
        ])

        phys_pts = sympy_tf.map_to_physical(ref_pts)

        # Verify against direct computation
        r = bkd.to_numpy(ref_pts[0, :])
        theta = bkd.to_numpy(ref_pts[1, :])
        phi = bkd.to_numpy(ref_pts[2, :])

        x_expected = r * np.sin(theta) * np.cos(phi)
        y_expected = r * np.sin(theta) * np.sin(phi)
        z_expected = r * np.cos(theta)

        bkd.assert_allclose(phys_pts[0, :], bkd.asarray(x_expected), rtol=1e-12)
        bkd.assert_allclose(phys_pts[1, :], bkd.asarray(y_expected), rtol=1e-12)
        bkd.assert_allclose(phys_pts[2, :], bkd.asarray(z_expected), rtol=1e-12)

    def test_spherical_scale_factors(self):
        """Verify spherical coordinate scale factors."""
        bkd = self.bkd()

        tf = SympyTransform3D(
            x_expr="r * sin(theta) * cos(phi)",
            y_expr="r * sin(theta) * sin(phi)",
            z_expr="r * cos(theta)",
            params={},
            bkd=bkd,
            coord_names=("r", "theta", "phi"),
            bounds=((1.0, 2.0), (0.3, math.pi - 0.3), (0.0, 2 * math.pi)),
            validate=False,
        )

        ref_pts = bkd.asarray([[1.5], [1.0], [0.5]])
        scales = tf.scale_factors(ref_pts)

        r = 1.5
        theta = 1.0

        # Spherical scale factors: h_r = 1, h_theta = r, h_phi = r*sin(theta)
        h_r_expected = 1.0
        h_theta_expected = r
        h_phi_expected = r * np.sin(theta)

        bkd.assert_allclose(
            scales[0, 0], bkd.asarray([h_r_expected])[0], rtol=1e-12
        )
        bkd.assert_allclose(
            scales[0, 1], bkd.asarray([h_theta_expected])[0], rtol=1e-12
        )
        bkd.assert_allclose(
            scales[0, 2], bkd.asarray([h_phi_expected])[0], rtol=1e-12
        )

    def test_prolate_spheroidal_scale_factors(self):
        """Test prolate spheroidal coordinates scale factors."""
        bkd = self.bkd()
        a = 2.0

        tf = SympyTransform3D(
            x_expr="a * sinh(xi) * sin(eta) * cos(phi)",
            y_expr="a * sinh(xi) * sin(eta) * sin(phi)",
            z_expr="a * cosh(xi) * cos(eta)",
            params={"a": a},
            bkd=bkd,
            coord_names=("xi", "eta", "phi"),
            bounds=((0.5, 2.0), (0.3, math.pi - 0.3), (0.0, 2 * math.pi)),
            validate=True,
        )

        ref_pts = bkd.asarray([[1.0], [0.8], [1.5]])
        scales = tf.scale_factors(ref_pts)

        xi = 1.0
        eta = 0.8

        # Prolate spheroidal: h_xi = h_eta = a*sqrt(sinh^2(xi) + sin^2(eta))
        #                     h_phi = a * sinh(xi) * sin(eta)
        h_xi_eta = a * np.sqrt(np.sinh(xi)**2 + np.sin(eta)**2)
        h_phi = a * np.sinh(xi) * np.sin(eta)

        bkd.assert_allclose(scales[0, 0], bkd.asarray([h_xi_eta])[0], rtol=1e-10)
        bkd.assert_allclose(scales[0, 1], bkd.asarray([h_xi_eta])[0], rtol=1e-10)
        bkd.assert_allclose(scales[0, 2], bkd.asarray([h_phi])[0], rtol=1e-10)

    def test_unit_basis_orthonormality_3d(self):
        """Verify 3D unit basis vectors are orthonormal."""
        bkd = self.bkd()

        tf = SympyTransform3D(
            x_expr="r * sin(theta) * cos(phi)",
            y_expr="r * sin(theta) * sin(phi)",
            z_expr="r * cos(theta)",
            params={},
            bkd=bkd,
            coord_names=("r", "theta", "phi"),
            bounds=((1.0, 2.0), (0.3, math.pi - 0.3), (0.0, 2 * math.pi)),
            validate=False,
        )

        ref_pts = bkd.asarray([[1.5], [1.0], [0.5]])
        basis = tf.unit_curvilinear_basis(ref_pts)

        # Check orthonormality: e_i . e_j = delta_ij
        for i in range(3):
            for j in range(3):
                e_i = basis[0, :, i]
                e_j = basis[0, :, j]
                dot = float(bkd.sum(e_i * e_j))
                expected = 1.0 if i == j else 0.0
                self.assertAlmostEqual(dot, expected, places=12)

    def test_jacobian_determinant_3d(self):
        """Verify Jacobian determinant for spherical coords."""
        bkd = self.bkd()

        tf = SympyTransform3D(
            x_expr="r * sin(theta) * cos(phi)",
            y_expr="r * sin(theta) * sin(phi)",
            z_expr="r * cos(theta)",
            params={},
            bkd=bkd,
            coord_names=("r", "theta", "phi"),
            bounds=((1.0, 2.0), (0.3, math.pi - 0.3), (0.0, 2 * math.pi)),
            validate=False,
        )

        ref_pts = bkd.asarray([[1.5], [1.0], [0.5]])
        det = tf.jacobian_determinant(ref_pts)

        # For spherical: det = h_r * h_theta * h_phi = 1 * r * r*sin(theta)
        #                    = r^2 * sin(theta)
        r = 1.5
        theta = 1.0
        expected = r**2 * np.sin(theta)

        bkd.assert_allclose(det, bkd.asarray([expected]), rtol=1e-12)


class TestSympyTransform2DNumpy(TestSympyTransform2D[NDArray[Any]]):
    """NumPy backend tests for SympyTransform2D."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestSympyTransform2DTorch(TestSympyTransform2D[torch.Tensor]):
    """PyTorch backend tests for SympyTransform2D."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)


class TestSympyTransform3DNumpy(TestSympyTransform3D[NDArray[Any]]):
    """NumPy backend tests for SympyTransform3D."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestSympyTransform3DTorch(TestSympyTransform3D[torch.Tensor]):
    """PyTorch backend tests for SympyTransform3D."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)


if __name__ == "__main__":
    unittest.main()
