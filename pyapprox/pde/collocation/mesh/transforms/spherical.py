"""Spherical coordinate transform for 3D domains.

Maps from spherical coordinates (r, azimuth, elevation) to Cartesian (x, y, z):
    x = r * sin(elevation) * cos(azimuth)
    y = r * sin(elevation) * sin(azimuth)
    z = r * cos(elevation)

Convention:
    r: radial distance from origin, r >= 0
    azimuth: angle in x-y plane from positive x-axis, azimuth in [-pi, pi]
    elevation: angle from positive z-axis (polar angle), elevation in [0, pi]

Scale factors: h_r = 1, h_azimuth = r*sin(elevation), h_elevation = r

Unit curvilinear basis:
    e_r = (sin(el)*cos(az), sin(el)*sin(az), cos(el))
    e_az = (-sin(az), cos(az), 0)
    e_el = (cos(el)*cos(az), cos(el)*sin(az), -sin(el))
"""

import math
from typing import Generic, Tuple

from pyapprox.util.backends.protocols import Array, Backend


class SphericalTransform(Generic[Array]):
    """Transform from spherical (r, azimuth, elevation) to Cartesian (x, y, z).

    Reference domain: spherical coordinates (r, azimuth, elevation)
    Physical domain: (x, y, z) in Cartesian coordinates

    Parameters
    ----------
    r_bounds : Tuple[float, float]
        Radial bounds (r_min, r_max). r_min >= 0.
    azimuth_bounds : Tuple[float, float]
        Azimuthal angle bounds in radians.
        Must satisfy -pi <= azimuth_min < azimuth_max <= pi.
    elevation_bounds : Tuple[float, float]
        Elevation (polar) angle bounds in radians.
        Must satisfy 0 <= elevation_min < elevation_max <= pi.
    bkd : Backend
        Computational backend.

    Notes
    -----
    The transform maps:
        x = r * sin(elevation) * cos(azimuth)
        y = r * sin(elevation) * sin(azimuth)
        z = r * cos(elevation)

    Jacobian determinant: det(J) = r^2 * sin(elevation)

    Scale factors (for curvilinear gradient):
        h_r = 1
        h_azimuth = r * sin(elevation)
        h_elevation = r
    """

    def __init__(
        self,
        r_bounds: Tuple[float, float],
        azimuth_bounds: Tuple[float, float],
        elevation_bounds: Tuple[float, float],
        bkd: Backend[Array],
    ):
        if r_bounds[0] < 0:
            raise ValueError("r_min must be >= 0")
        if r_bounds[1] <= r_bounds[0]:
            raise ValueError("r_max must be > r_min")
        if azimuth_bounds[0] < -math.pi or azimuth_bounds[1] > math.pi:
            raise ValueError("azimuth must be in [-pi, pi]")
        if azimuth_bounds[1] <= azimuth_bounds[0]:
            raise ValueError("azimuth_max must be > azimuth_min")
        if elevation_bounds[0] < 0 or elevation_bounds[1] > math.pi:
            raise ValueError("elevation must be in [0, pi]")
        if elevation_bounds[1] <= elevation_bounds[0]:
            raise ValueError("elevation_max must be > elevation_min")

        self._bkd = bkd
        self._r_min, self._r_max = r_bounds
        self._az_min, self._az_max = azimuth_bounds
        self._el_min, self._el_max = elevation_bounds

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def ndim(self) -> int:
        """Return the number of spatial dimensions."""
        return 3

    def r_bounds(self) -> Tuple[float, float]:
        """Return radial bounds (r_min, r_max)."""
        return (self._r_min, self._r_max)

    def azimuth_bounds(self) -> Tuple[float, float]:
        """Return azimuthal bounds (azimuth_min, azimuth_max)."""
        return (self._az_min, self._az_max)

    def elevation_bounds(self) -> Tuple[float, float]:
        """Return elevation bounds (elevation_min, elevation_max)."""
        return (self._el_min, self._el_max)

    def map_to_physical(self, reference_pts: Array) -> Array:
        """Map from spherical to Cartesian coordinates.

        Parameters
        ----------
        reference_pts : Array
            Spherical coordinates. Shape: (3, npts)
            Row 0: r values
            Row 1: azimuth values
            Row 2: elevation values

        Returns
        -------
        Array
            Cartesian coordinates. Shape: (3, npts)
        """
        r = reference_pts[0, :]
        azimuth = reference_pts[1, :]
        elevation = reference_pts[2, :]

        sin_el = self._bkd.sin(elevation)
        cos_el = self._bkd.cos(elevation)
        sin_az = self._bkd.sin(azimuth)
        cos_az = self._bkd.cos(azimuth)

        x = r * sin_el * cos_az
        y = r * sin_el * sin_az
        z = r * cos_el

        return self._bkd.stack([x, y, z], axis=0)

    def map_to_reference(self, physical_pts: Array) -> Array:
        """Map from Cartesian to spherical coordinates.

        Parameters
        ----------
        physical_pts : Array
            Cartesian coordinates. Shape: (3, npts)

        Returns
        -------
        Array
            Spherical coordinates. Shape: (3, npts)
        """
        x = physical_pts[0, :]
        y = physical_pts[1, :]
        z = physical_pts[2, :]

        r = self._bkd.sqrt(x**2 + y**2 + z**2)
        elevation = self._bkd.arccos(z / (r + 1e-15))  # Avoid division by zero
        azimuth = self._bkd.arctan2(y, x)

        return self._bkd.stack([r, azimuth, elevation], axis=0)

    def jacobian_matrix(self, reference_pts: Array) -> Array:
        """Compute Jacobian matrix of the mapping.

        J[i,j] = d(physical_i)/d(reference_j)

        Parameters
        ----------
        reference_pts : Array
            Spherical coordinates. Shape: (3, npts)

        Returns
        -------
        Array
            Jacobian matrices. Shape: (npts, 3, 3)
        """
        npts = reference_pts.shape[1]
        r = reference_pts[0, :]
        azimuth = reference_pts[1, :]
        elevation = reference_pts[2, :]

        sin_el = self._bkd.sin(elevation)
        cos_el = self._bkd.cos(elevation)
        sin_az = self._bkd.sin(azimuth)
        cos_az = self._bkd.cos(azimuth)

        jac = self._bkd.zeros((npts, 3, 3))

        # dx/dr, dx/daz, dx/del
        jac[:, 0, 0] = sin_el * cos_az  # dx/dr
        jac[:, 0, 1] = -r * sin_el * sin_az  # dx/daz
        jac[:, 0, 2] = r * cos_el * cos_az  # dx/del

        # dy/dr, dy/daz, dy/del
        jac[:, 1, 0] = sin_el * sin_az  # dy/dr
        jac[:, 1, 1] = r * sin_el * cos_az  # dy/daz
        jac[:, 1, 2] = r * cos_el * sin_az  # dy/del

        # dz/dr, dz/daz, dz/del
        jac[:, 2, 0] = cos_el  # dz/dr
        jac[:, 2, 1] = 0.0  # dz/daz
        jac[:, 2, 2] = -r * sin_el  # dz/del

        return jac

    def jacobian_determinant(self, reference_pts: Array) -> Array:
        """Compute Jacobian determinant.

        det(J) = r^2 * sin(elevation)

        Parameters
        ----------
        reference_pts : Array
            Spherical coordinates. Shape: (3, npts)

        Returns
        -------
        Array
            Jacobian determinants. Shape: (npts,)
        """
        r = reference_pts[0, :]
        elevation = reference_pts[2, :]
        return r**2 * self._bkd.sin(elevation)

    def scale_factors(self, reference_pts: Array) -> Array:
        """Compute scale factors for curvilinear coordinates.

        For spherical:
            h_r = 1
            h_azimuth = r * sin(elevation)
            h_elevation = r

        Parameters
        ----------
        reference_pts : Array
            Spherical coordinates. Shape: (3, npts)

        Returns
        -------
        Array
            Scale factors. Shape: (npts, 3)
        """
        npts = reference_pts.shape[1]
        r = reference_pts[0, :]
        elevation = reference_pts[2, :]

        h_r = self._bkd.ones((npts,))
        h_az = r * self._bkd.sin(elevation)
        h_el = r

        return self._bkd.stack([h_r, h_az, h_el], axis=1)

    def unit_curvilinear_basis(self, reference_pts: Array) -> Array:
        """Compute unit vectors in curvilinear coordinates.

        Returns the unit basis vectors expressed in Cartesian coordinates:
            e_r = (sin(el)*cos(az), sin(el)*sin(az), cos(el))
            e_az = (-sin(az), cos(az), 0)
            e_el = (cos(el)*cos(az), cos(el)*sin(az), -sin(el))

        Parameters
        ----------
        reference_pts : Array
            Spherical coordinates. Shape: (3, npts)

        Returns
        -------
        Array
            Unit basis vectors. Shape: (npts, 3, 3)
            result[:, :, 0] = e_r (radial direction)
            result[:, :, 1] = e_azimuth (azimuthal direction)
            result[:, :, 2] = e_elevation (elevation direction)
        """
        npts = reference_pts.shape[1]
        azimuth = reference_pts[1, :]
        elevation = reference_pts[2, :]

        sin_el = self._bkd.sin(elevation)
        cos_el = self._bkd.cos(elevation)
        sin_az = self._bkd.sin(azimuth)
        cos_az = self._bkd.cos(azimuth)

        # e_r = (sin(el)*cos(az), sin(el)*sin(az), cos(el))
        e_r = self._bkd.stack([sin_el * cos_az, sin_el * sin_az, cos_el], axis=1)

        # e_az = (-sin(az), cos(az), 0)
        e_az = self._bkd.stack([-sin_az, cos_az, self._bkd.zeros((npts,))], axis=1)

        # e_el = (cos(el)*cos(az), cos(el)*sin(az), -sin(el))
        e_el = self._bkd.stack([cos_el * cos_az, cos_el * sin_az, -sin_el], axis=1)

        # Shape: (npts, 3, 3) - last axis is basis index
        return self._bkd.stack([e_r, e_az, e_el], axis=2)

    def gradient_factors(self, reference_pts: Array) -> Array:
        """Compute factors for transforming gradients.

        Parameters
        ----------
        reference_pts : Array
            Spherical coordinates. Shape: (3, npts)

        Returns
        -------
        Array
            Gradient factors. Shape: (npts, 3, 3)
            result[:, :, i] = e_i / h_i
        """
        unit_basis = self.unit_curvilinear_basis(reference_pts)
        scale = self.scale_factors(reference_pts)
        # Divide each basis vector by its scale factor
        return unit_basis / scale[:, None, :]
