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

from pyapprox.typing.util.backends.protocols import Array, Backend


class PolarTransform(Generic[Array]):
    """Transform from polar (r, theta) to Cartesian (x, y) coordinates.

    Reference domain: (r, theta) where r in [r_min, r_max], theta in [theta_min, theta_max]
    Physical domain: (x, y) in Cartesian coordinates

    Parameters
    ----------
    r_bounds : Tuple[float, float]
        Radial bounds (r_min, r_max). r_min >= 0.
    theta_bounds : Tuple[float, float]
        Angular bounds (theta_min, theta_max) in radians.
        Must satisfy -pi <= theta_min < theta_max <= pi.
    bkd : Backend
        Computational backend.

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
    ):
        if r_bounds[0] < 0:
            raise ValueError("r_min must be >= 0")
        if r_bounds[1] <= r_bounds[0]:
            raise ValueError("r_max must be > r_min")
        if theta_bounds[0] < -math.pi or theta_bounds[1] > math.pi:
            raise ValueError("theta must be in [-pi, pi]")
        if theta_bounds[1] <= theta_bounds[0]:
            raise ValueError("theta_max must be > theta_min")

        self._bkd = bkd
        self._r_min, self._r_max = r_bounds
        self._theta_min, self._theta_max = theta_bounds

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
        """Map from polar (r, theta) to Cartesian (x, y).

        Parameters
        ----------
        reference_pts : Array
            Polar coordinates. Shape: (2, npts)
            First row: r values
            Second row: theta values

        Returns
        -------
        Array
            Cartesian coordinates. Shape: (2, npts)
        """
        r = reference_pts[0, :]
        theta = reference_pts[1, :]
        x = r * self._bkd.cos(theta)
        y = r * self._bkd.sin(theta)
        return self._bkd.stack([x, y], axis=0)

    def map_to_reference(self, physical_pts: Array) -> Array:
        """Map from Cartesian (x, y) to polar (r, theta).

        Parameters
        ----------
        physical_pts : Array
            Cartesian coordinates. Shape: (2, npts)

        Returns
        -------
        Array
            Polar coordinates. Shape: (2, npts)
        """
        x = physical_pts[0, :]
        y = physical_pts[1, :]
        r = self._bkd.sqrt(x**2 + y**2)
        theta = self._bkd.arctan2(y, x)
        return self._bkd.stack([r, theta], axis=0)

    def jacobian_matrix(self, reference_pts: Array) -> Array:
        """Compute Jacobian matrix of the mapping.

        J[i,j] = d(physical_i)/d(reference_j)

        Parameters
        ----------
        reference_pts : Array
            Polar coordinates. Shape: (2, npts)

        Returns
        -------
        Array
            Jacobian matrices. Shape: (npts, 2, 2)
            J[:, 0, 0] = dx/dr = cos(theta)
            J[:, 0, 1] = dx/dtheta = -r*sin(theta)
            J[:, 1, 0] = dy/dr = sin(theta)
            J[:, 1, 1] = dy/dtheta = r*cos(theta)
        """
        npts = reference_pts.shape[1]
        r = reference_pts[0, :]
        theta = reference_pts[1, :]

        cos_theta = self._bkd.cos(theta)
        sin_theta = self._bkd.sin(theta)

        jac = self._bkd.zeros((npts, 2, 2))
        jac[:, 0, 0] = cos_theta       # dx/dr
        jac[:, 0, 1] = -r * sin_theta  # dx/dtheta
        jac[:, 1, 0] = sin_theta       # dy/dr
        jac[:, 1, 1] = r * cos_theta   # dy/dtheta

        return jac

    def jacobian_determinant(self, reference_pts: Array) -> Array:
        """Compute Jacobian determinant.

        det(J) = r

        Parameters
        ----------
        reference_pts : Array
            Polar coordinates. Shape: (2, npts)

        Returns
        -------
        Array
            Jacobian determinants. Shape: (npts,)
        """
        return reference_pts[0, :]  # r

    def scale_factors(self, reference_pts: Array) -> Array:
        """Compute scale factors for curvilinear coordinates.

        The scale factors h_i relate coordinate differentials to arc lengths:
            ds_i = h_i * dq_i

        For polar: h_r = 1, h_theta = r

        Parameters
        ----------
        reference_pts : Array
            Polar coordinates. Shape: (2, npts)

        Returns
        -------
        Array
            Scale factors. Shape: (npts, 2)
            Column 0: h_r = 1
            Column 1: h_theta = r
        """
        npts = reference_pts.shape[1]
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
            Polar coordinates. Shape: (2, npts)

        Returns
        -------
        Array
            Unit basis vectors. Shape: (npts, 2, 2)
            result[:, :, 0] = e_r (radial direction)
            result[:, :, 1] = e_theta (angular direction)
        """
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
            Polar coordinates. Shape: (2, npts)

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
