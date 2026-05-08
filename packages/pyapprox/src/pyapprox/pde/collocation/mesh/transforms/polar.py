"""Polar coordinate transform for 2D domains.

Maps from polar coordinates (r, theta) to Cartesian coordinates (x, y):
    x = r * cos(theta)
    y = r * sin(theta)

Scale factors: h_r = 1, h_theta = r
Unit curvilinear basis:
    e_r = (cos(theta), sin(theta))
    e_theta = (-sin(theta), cos(theta))
"""

import math
from typing import Generic, Tuple

from pyapprox.util.backends.protocols import Array, Backend


class PolarTransform(Generic[Array]):
    """Transform from polar (r, theta) to Cartesian (x, y) coordinates.

    When `from_reference=True` (default), the transform maps from the
    reference domain [-1, 1]^2 to Cartesian coordinates via internal
    affine mapping to polar coordinates:
        [-1, 1] -> [r_min, r_max]   (first coordinate)
        [-1, 1] -> [theta_min, theta_max]  (second coordinate)
        Then polar to Cartesian.

    When `from_reference=False`, the transform expects input in polar
    coordinates directly (legacy behavior for testing/debugging).

    Parameters
    ----------
    r_bounds : Tuple[float, float]
        Radial bounds (r_min, r_max). r_min >= 0.
    theta_bounds : Tuple[float, float]
        Angular bounds (theta_min, theta_max) in radians.
        Must satisfy -2*pi <= theta_min < theta_max <= 2*pi.
    bkd : Backend
        Computational backend.
    from_reference : bool, optional
        If True (default), input is in [-1, 1]^2 reference domain.
        If False, input is in (r, theta) polar coordinates directly.

    Notes
    -----
    The transform maps:
        x = r * cos(theta)
        y = r * sin(theta)

    Jacobian matrix J = [[dx/dr, dx/dtheta], [dy/dr, dy/dtheta]]:
        J = [[cos(theta), -r*sin(theta)],
             [sin(theta),  r*cos(theta)]]

    Jacobian determinant: det(J) = r

    Scale factors (for curvilinear gradient):
        h_r = 1, h_theta = r
    """

    def __init__(
        self,
        r_bounds: Tuple[float, float],
        theta_bounds: Tuple[float, float],
        bkd: Backend[Array],
        from_reference: bool = True,
    ):
        if r_bounds[0] < 0:
            raise ValueError("r_min must be >= 0")
        if r_bounds[1] <= r_bounds[0]:
            raise ValueError("r_max must be > r_min")
        if theta_bounds[0] < -2 * math.pi or theta_bounds[1] > 2 * math.pi:
            raise ValueError("theta must be in [-2*pi, 2*pi]")
        if theta_bounds[1] <= theta_bounds[0]:
            raise ValueError("theta_max must be > theta_min")

        self._bkd = bkd
        self._r_min, self._r_max = r_bounds
        self._theta_min, self._theta_max = theta_bounds
        self._from_reference = from_reference

        # Affine mapping coefficients: polar = a * ref + b where ref in [-1, 1]
        # polar = (polar_max - polar_min)/2 * ref + (polar_max + polar_min)/2
        self._r_scale = (self._r_max - self._r_min) / 2.0
        self._r_shift = (self._r_max + self._r_min) / 2.0
        self._theta_scale = (self._theta_max - self._theta_min) / 2.0
        self._theta_shift = (self._theta_max + self._theta_min) / 2.0

    def _ref_to_polar(self, reference_pts: Array) -> Array:
        """Map from [-1, 1]^2 to (r, theta) polar coordinates."""
        xi = reference_pts[0, :]
        eta = reference_pts[1, :]
        r = self._r_scale * xi + self._r_shift
        theta = self._theta_scale * eta + self._theta_shift
        return self._bkd.stack([r, theta], axis=0)

    def _polar_to_ref(self, polar_pts: Array) -> Array:
        """Map from (r, theta) to [-1, 1]^2 reference coordinates."""
        r = polar_pts[0, :]
        theta = polar_pts[1, :]
        xi = (r - self._r_shift) / self._r_scale
        eta = (theta - self._theta_shift) / self._theta_scale
        return self._bkd.stack([xi, eta], axis=0)

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def ndim(self) -> int:
        """Return the number of spatial dimensions."""
        return 2

    def r_bounds(self) -> Tuple[float, float]:
        """Return radial bounds (r_min, r_max)."""
        return (self._r_min, self._r_max)

    def theta_bounds(self) -> Tuple[float, float]:
        """Return angular bounds (theta_min, theta_max)."""
        return (self._theta_min, self._theta_max)

    def map_to_physical(self, reference_pts: Array) -> Array:
        """Map from reference/polar coordinates to Cartesian (x, y).

        Parameters
        ----------
        reference_pts : Array
            If from_reference=True: Reference coordinates in [-1, 1]^2. Shape: (2, npts)
            If from_reference=False: Polar coordinates (r, theta). Shape: (2, npts)

        Returns
        -------
        Array
            Cartesian coordinates. Shape: (2, npts)
        """
        if self._from_reference:
            polar_pts = self._ref_to_polar(reference_pts)
        else:
            polar_pts = reference_pts
        r = polar_pts[0, :]
        theta = polar_pts[1, :]
        x = r * self._bkd.cos(theta)
        y = r * self._bkd.sin(theta)
        return self._bkd.stack([x, y], axis=0)

    def map_to_reference(self, physical_pts: Array) -> Array:
        """Map from Cartesian (x, y) to reference/polar coordinates.

        Parameters
        ----------
        physical_pts : Array
            Cartesian coordinates. Shape: (2, npts)

        Returns
        -------
        Array
            If from_reference=True: Reference coordinates in [-1, 1]^2. Shape: (2, npts)
            If from_reference=False: Polar coordinates (r, theta). Shape: (2, npts)
        """
        x = physical_pts[0, :]
        y = physical_pts[1, :]
        r = self._bkd.sqrt(x**2 + y**2)
        theta = self._bkd.arctan2(y, x)
        polar_pts = self._bkd.stack([r, theta], axis=0)
        if self._from_reference:
            return self._polar_to_ref(polar_pts)
        return polar_pts

    def jacobian_matrix(self, reference_pts: Array) -> Array:
        """Compute Jacobian matrix of the mapping.

        J[i,j] = d(physical_i)/d(reference_j)

        Parameters
        ----------
        reference_pts : Array
            If from_reference=True: Reference coordinates in [-1, 1]^2. Shape: (2, npts)
            If from_reference=False: Polar coordinates. Shape: (2, npts)

        Returns
        -------
        Array
            Jacobian matrices. Shape: (npts, 2, 2)
        """
        npts = reference_pts.shape[1]

        if self._from_reference:
            polar_pts = self._ref_to_polar(reference_pts)
        else:
            polar_pts = reference_pts

        r = polar_pts[0, :]
        theta = polar_pts[1, :]

        cos_theta = self._bkd.cos(theta)
        sin_theta = self._bkd.sin(theta)

        # Jacobian of polar-to-Cartesian: J_polar
        # dx/dr = cos(theta), dx/dtheta = -r*sin(theta)
        # dy/dr = sin(theta), dy/dtheta = r*cos(theta)
        jac_polar = self._bkd.zeros((npts, 2, 2))
        jac_polar[:, 0, 0] = cos_theta  # dx/dr
        jac_polar[:, 0, 1] = -r * sin_theta  # dx/dtheta
        jac_polar[:, 1, 0] = sin_theta  # dy/dr
        jac_polar[:, 1, 1] = r * cos_theta  # dy/dtheta

        if self._from_reference:
            # Chain rule: J_total = J_polar @ J_affine
            # J_affine = diag(r_scale, theta_scale)
            # So J_total[:, :, 0] = J_polar[:, :, 0] * r_scale
            #    J_total[:, :, 1] = J_polar[:, :, 1] * theta_scale
            jac = self._bkd.zeros((npts, 2, 2))
            jac[:, 0, 0] = jac_polar[:, 0, 0] * self._r_scale
            jac[:, 0, 1] = jac_polar[:, 0, 1] * self._theta_scale
            jac[:, 1, 0] = jac_polar[:, 1, 0] * self._r_scale
            jac[:, 1, 1] = jac_polar[:, 1, 1] * self._theta_scale
            return jac
        else:
            return jac_polar

    def jacobian_determinant(self, reference_pts: Array) -> Array:
        """Compute Jacobian determinant.

        For polar transform: det(J_polar) = r
        With affine: det(J_total) = r * r_scale * theta_scale

        Parameters
        ----------
        reference_pts : Array
            If from_reference=True: Reference coordinates in [-1, 1]^2. Shape: (2, npts)
            If from_reference=False: Polar coordinates. Shape: (2, npts)

        Returns
        -------
        Array
            Jacobian determinants. Shape: (npts,)
        """
        if self._from_reference:
            polar_pts = self._ref_to_polar(reference_pts)
            r = polar_pts[0, :]
            return r * self._r_scale * self._theta_scale
        else:
            return reference_pts[0, :]  # r

    def scale_factors(self, reference_pts: Array) -> Array:
        """Compute scale factors for curvilinear coordinates.

        The scale factors h_i relate coordinate differentials to arc lengths:
            ds_i = h_i * dq_i

        For polar: h_r = 1, h_theta = r
        With affine: h_r = r_scale, h_theta = r * theta_scale

        Parameters
        ----------
        reference_pts : Array
            If from_reference=True: Reference coordinates in [-1, 1]^2. Shape: (2, npts)
            If from_reference=False: Polar coordinates. Shape: (2, npts)

        Returns
        -------
        Array
            Scale factors. Shape: (npts, 2)
        """
        npts = reference_pts.shape[1]

        if self._from_reference:
            polar_pts = self._ref_to_polar(reference_pts)
            r = polar_pts[0, :]
            h_r = self._bkd.full((npts,), self._r_scale)
            h_theta = r * self._theta_scale
        else:
            r = reference_pts[0, :]
            h_r = self._bkd.ones((npts,))
            h_theta = r

        return self._bkd.stack([h_r, h_theta], axis=1)

    def unit_curvilinear_basis(self, reference_pts: Array) -> Array:
        """Compute unit vectors in curvilinear coordinates.

        Returns the unit basis vectors expressed in Cartesian coordinates:
            e_r = (cos(theta), sin(theta))
            e_theta = (-sin(theta), cos(theta))

        Parameters
        ----------
        reference_pts : Array
            If from_reference=True: Reference coordinates in [-1, 1]^2. Shape: (2, npts)
            If from_reference=False: Polar coordinates. Shape: (2, npts)

        Returns
        -------
        Array
            Unit basis vectors. Shape: (npts, 2, 2)
            result[:, :, 0] = e_r (radial direction)
            result[:, :, 1] = e_theta (angular direction)
        """
        if self._from_reference:
            polar_pts = self._ref_to_polar(reference_pts)
            theta = polar_pts[1, :]
        else:
            theta = reference_pts[1, :]

        cos_theta = self._bkd.cos(theta)
        sin_theta = self._bkd.sin(theta)

        # e_r = [cos(theta), sin(theta)]
        e_r = self._bkd.stack([cos_theta, sin_theta], axis=1)

        # e_theta = [-sin(theta), cos(theta)]
        e_theta = self._bkd.stack([-sin_theta, cos_theta], axis=1)

        # Shape: (npts, 2, 2) - last axis is basis index
        return self._bkd.stack([e_r, e_theta], axis=2)

    def gradient_factors(self, reference_pts: Array) -> Array:
        """Compute factors for transforming gradients.

        For a scalar field f, the gradient in curvilinear coordinates is:
            grad(f) = sum_i (1/h_i) * (df/dq_i) * e_i

        This returns (1/h_i) * e_i for each coordinate.

        Parameters
        ----------
        reference_pts : Array
            If from_reference=True: Reference coordinates in [-1, 1]^2. Shape: (2, npts)
            If from_reference=False: Polar coordinates. Shape: (2, npts)

        Returns
        -------
        Array
            Gradient factors. Shape: (npts, 2, 2)
            result[:, :, i] = e_i / h_i
        """
        unit_basis = self.unit_curvilinear_basis(reference_pts)
        scale = self.scale_factors(reference_pts)
        # Divide each basis vector by its scale factor
        return unit_basis / scale[:, None, :]

    def from_reference(self) -> bool:
        """Return whether the transform expects reference domain input."""
        return self._from_reference
