"""Elliptical coordinate transform for 2D domains.

Maps from elliptical coordinates (u, v) to Cartesian coordinates (x, y):
    x = a * cosh(u) * cos(v)
    y = a * sinh(u) * sin(v)

where a is the semi-focal distance.

Scale factors: h_u = h_v = a * sqrt(sinh^2(u) + sin^2(v))
Unit curvilinear basis:
    e_u = (sinh(u)*cos(v), cosh(u)*sin(v)) / h_u * a
    e_v = (-cosh(u)*sin(v), sinh(u)*cos(v)) / h_v * a

The coordinate lines are:
- Constant u: ellipses with semi-axes a*cosh(u), a*sinh(u)
- Constant v: hyperbolae
"""

import math
from typing import Generic, Tuple

from pyapprox.typing.util.backends.protocols import Array, Backend


class EllipticalTransform(Generic[Array]):
    """Transform from elliptical (u, v) to Cartesian (x, y) coordinates.

    Reference domain: (u, v) where u in [u_min, u_max], v in [v_min, v_max]
    Physical domain: (x, y) in Cartesian coordinates

    Parameters
    ----------
    u_bounds : Tuple[float, float]
        Radial-like bounds (u_min, u_max). u_min > 0 to avoid focal points.
    v_bounds : Tuple[float, float]
        Angular bounds (v_min, v_max) in radians.
        Must satisfy 0 <= v_min < v_max <= 2*pi.
    a : float
        Semi-focal distance. The foci are at (±a, 0).
    bkd : Backend
        Computational backend.

    Notes
    -----
    The transform maps:
        x = a * cosh(u) * cos(v)
        y = a * sinh(u) * sin(v)

    Jacobian matrix J = [[dx/du, dx/dv], [dy/du, dy/dv]]:
        J = [[a*sinh(u)*cos(v), -a*cosh(u)*sin(v)],
             [a*cosh(u)*sin(v),  a*sinh(u)*cos(v)]]

    Jacobian determinant: det(J) = a^2 * (sinh^2(u) + sin^2(v))

    Scale factors (for curvilinear gradient):
        h_u = h_v = a * sqrt(sinh^2(u) + sin^2(v))

    This is an orthogonal coordinate system.
    """

    def __init__(
        self,
        u_bounds: Tuple[float, float],
        v_bounds: Tuple[float, float],
        a: float,
        bkd: Backend[Array],
    ):
        if u_bounds[0] <= 0:
            raise ValueError("u_min must be > 0 to avoid focal point singularity")
        if u_bounds[1] <= u_bounds[0]:
            raise ValueError("u_max must be > u_min")
        if v_bounds[0] < 0 or v_bounds[1] > 2 * math.pi:
            raise ValueError("v must be in [0, 2*pi]")
        if v_bounds[1] <= v_bounds[0]:
            raise ValueError("v_max must be > v_min")
        if a <= 0:
            raise ValueError("Semi-focal distance a must be > 0")

        self._bkd = bkd
        self._u_min, self._u_max = u_bounds
        self._v_min, self._v_max = v_bounds
        self._a = a

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def ndim(self) -> int:
        """Return the number of spatial dimensions."""
        return 2

    def u_bounds(self) -> Tuple[float, float]:
        """Return radial-like bounds (u_min, u_max)."""
        return (self._u_min, self._u_max)

    def v_bounds(self) -> Tuple[float, float]:
        """Return angular bounds (v_min, v_max)."""
        return (self._v_min, self._v_max)

    def a(self) -> float:
        """Return semi-focal distance."""
        return self._a

    def map_to_physical(self, reference_pts: Array) -> Array:
        """Map from elliptical (u, v) to Cartesian (x, y).

        Parameters
        ----------
        reference_pts : Array
            Elliptical coordinates. Shape: (2, npts)
            First row: u values
            Second row: v values

        Returns
        -------
        Array
            Cartesian coordinates. Shape: (2, npts)
        """
        u = reference_pts[0, :]
        v = reference_pts[1, :]
        a = self._a

        x = a * self._bkd.cosh(u) * self._bkd.cos(v)
        y = a * self._bkd.sinh(u) * self._bkd.sin(v)

        return self._bkd.stack([x, y], axis=0)

    def map_to_reference(self, physical_pts: Array) -> Array:
        """Map from Cartesian (x, y) to elliptical (u, v).

        Parameters
        ----------
        physical_pts : Array
            Cartesian coordinates. Shape: (2, npts)

        Returns
        -------
        Array
            Elliptical coordinates. Shape: (2, npts)

        Notes
        -----
        The inverse mapping uses:
            u = arcsinh(sqrt((r - a^2) / (2*a^2) + sqrt(((r - a^2)/(2*a^2))^2 + y^2/a^2)))
        where r = x^2 + y^2, but a simpler approach uses:
            cosh(u) = (r1 + r2) / (2a)
            sinh(u) = (r1 - r2) / (2a)
        where r1, r2 are distances to foci at (a, 0) and (-a, 0).
        """
        x = physical_pts[0, :]
        y = physical_pts[1, :]
        a = self._a

        # Distances to foci at (a, 0) and (-a, 0)
        r1 = self._bkd.sqrt((x - a) ** 2 + y**2)
        r2 = self._bkd.sqrt((x + a) ** 2 + y**2)

        # cosh(u) = (r1 + r2) / (2a), so u = arccosh((r1 + r2) / (2a))
        cosh_u = (r1 + r2) / (2 * a)
        u = self._bkd.arccosh(cosh_u)

        # For v, use atan2(y / sinh(u), x / cosh(u))
        # But sinh(u) might be small, so use: v = atan2(y, x * tanh(u))
        # Actually, from x = a*cosh(u)*cos(v) and y = a*sinh(u)*sin(v):
        # tan(v) = y * cosh(u) / (x * sinh(u)) = y / (x * tanh(u))
        sinh_u = self._bkd.sinh(u)
        v = self._bkd.arctan2(y * cosh_u, x * sinh_u)

        return self._bkd.stack([u, v], axis=0)

    def jacobian_matrix(self, reference_pts: Array) -> Array:
        """Compute Jacobian matrix of the mapping.

        J[i,j] = d(physical_i)/d(reference_j)

        Parameters
        ----------
        reference_pts : Array
            Elliptical coordinates. Shape: (2, npts)

        Returns
        -------
        Array
            Jacobian matrices. Shape: (npts, 2, 2)
            J[:, 0, 0] = dx/du = a*sinh(u)*cos(v)
            J[:, 0, 1] = dx/dv = -a*cosh(u)*sin(v)
            J[:, 1, 0] = dy/du = a*cosh(u)*sin(v)
            J[:, 1, 1] = dy/dv = a*sinh(u)*cos(v)
        """
        npts = reference_pts.shape[1]
        u = reference_pts[0, :]
        v = reference_pts[1, :]
        a = self._a

        sinh_u = self._bkd.sinh(u)
        cosh_u = self._bkd.cosh(u)
        cos_v = self._bkd.cos(v)
        sin_v = self._bkd.sin(v)

        jac = self._bkd.zeros((npts, 2, 2))
        jac[:, 0, 0] = a * sinh_u * cos_v   # dx/du
        jac[:, 0, 1] = -a * cosh_u * sin_v  # dx/dv
        jac[:, 1, 0] = a * cosh_u * sin_v   # dy/du
        jac[:, 1, 1] = a * sinh_u * cos_v   # dy/dv

        return jac

    def jacobian_determinant(self, reference_pts: Array) -> Array:
        """Compute Jacobian determinant.

        det(J) = a^2 * (sinh^2(u) + sin^2(v))

        Parameters
        ----------
        reference_pts : Array
            Elliptical coordinates. Shape: (2, npts)

        Returns
        -------
        Array
            Jacobian determinants. Shape: (npts,)
        """
        u = reference_pts[0, :]
        v = reference_pts[1, :]
        a = self._a

        sinh_u = self._bkd.sinh(u)
        sin_v = self._bkd.sin(v)

        return a**2 * (sinh_u**2 + sin_v**2)

    def scale_factors(self, reference_pts: Array) -> Array:
        """Compute scale factors for curvilinear coordinates.

        The scale factors h_i relate coordinate differentials to arc lengths:
            ds_i = h_i * dq_i

        For elliptical: h_u = h_v = a * sqrt(sinh^2(u) + sin^2(v))

        Parameters
        ----------
        reference_pts : Array
            Elliptical coordinates. Shape: (2, npts)

        Returns
        -------
        Array
            Scale factors. Shape: (npts, 2)
            Column 0: h_u
            Column 1: h_v
        """
        u = reference_pts[0, :]
        v = reference_pts[1, :]
        a = self._a

        sinh_u = self._bkd.sinh(u)
        sin_v = self._bkd.sin(v)

        h = a * self._bkd.sqrt(sinh_u**2 + sin_v**2)

        return self._bkd.stack([h, h], axis=1)

    def unit_curvilinear_basis(self, reference_pts: Array) -> Array:
        """Compute unit vectors in curvilinear coordinates.

        Returns the unit basis vectors expressed in Cartesian coordinates:
            e_u = (sinh(u)*cos(v), cosh(u)*sin(v)) / sqrt(sinh^2(u) + sin^2(v))
            e_v = (-cosh(u)*sin(v), sinh(u)*cos(v)) / sqrt(sinh^2(u) + sin^2(v))

        Parameters
        ----------
        reference_pts : Array
            Elliptical coordinates. Shape: (2, npts)

        Returns
        -------
        Array
            Unit basis vectors. Shape: (npts, 2, 2)
            result[:, :, 0] = e_u (u direction)
            result[:, :, 1] = e_v (v direction)
        """
        u = reference_pts[0, :]
        v = reference_pts[1, :]

        sinh_u = self._bkd.sinh(u)
        cosh_u = self._bkd.cosh(u)
        cos_v = self._bkd.cos(v)
        sin_v = self._bkd.sin(v)

        # Normalization factor
        norm = self._bkd.sqrt(sinh_u**2 + sin_v**2)

        # e_u = (sinh(u)*cos(v), cosh(u)*sin(v)) / norm
        e_u_x = sinh_u * cos_v / norm
        e_u_y = cosh_u * sin_v / norm
        e_u = self._bkd.stack([e_u_x, e_u_y], axis=1)

        # e_v = (-cosh(u)*sin(v), sinh(u)*cos(v)) / norm
        e_v_x = -cosh_u * sin_v / norm
        e_v_y = sinh_u * cos_v / norm
        e_v = self._bkd.stack([e_v_x, e_v_y], axis=1)

        # Shape: (npts, 2, 2) - last axis is basis index
        return self._bkd.stack([e_u, e_v], axis=2)

    def gradient_factors(self, reference_pts: Array) -> Array:
        """Compute factors for transforming gradients.

        For a scalar field f, the gradient in curvilinear coordinates is:
            grad(f) = sum_i (1/h_i) * (df/dq_i) * e_i

        This returns (1/h_i) * e_i for each coordinate.

        Parameters
        ----------
        reference_pts : Array
            Elliptical coordinates. Shape: (2, npts)

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
