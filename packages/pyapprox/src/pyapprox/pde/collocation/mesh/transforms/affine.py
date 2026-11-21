"""Affine (scale and translate) coordinate transforms.

:class:`AffineTransform` maps the reference cube :math:`[-1, 1]^d` onto an
axis-aligned box in any number of directions. ``AffineTransform1D``,
``AffineTransform2D`` and ``AffineTransform3D`` are the same map built
from flat bound tuples ``(a_x, b_x, a_y, b_y, ...)``.
"""

import math
from typing import Generic, Sequence, Tuple

from pyapprox.util.backends.protocols import Array, Backend


class AffineTransform(Generic[Array]):
    """Affine map from the reference cube :math:`[-1, 1]^d` to a box.

    Direction ``d`` maps :math:`\\xi_d \\mapsto s_d \\xi_d + c_d` with
    half-length :math:`s_d = (b_d - a_d) / 2` and center
    :math:`c_d = (a_d + b_d) / 2`.

    Parameters
    ----------
    bounds : Sequence[Tuple[float, float]]
        ``(a_d, b_d)`` per direction, ``a_d < b_d``.
    bkd : Backend
        Computational backend.

    Raises
    ------
    ValueError
        If ``bounds`` is empty or a lower bound is not below its upper.
    """

    def __init__(
        self, bounds: Sequence[Tuple[float, float]], bkd: Backend[Array]
    ) -> None:
        if len(bounds) == 0:
            raise ValueError("an affine transform needs at least one direction")
        for direction, (lower, upper) in enumerate(bounds):
            if not lower < upper:
                raise ValueError(
                    f"direction {direction}: lower bound {lower} is not below "
                    f"upper bound {upper}"
                )
        self._bkd = bkd
        self._bounds = tuple((float(a), float(b)) for a, b in bounds)
        half_lengths = [(b - a) / 2.0 for a, b in self._bounds]
        centers = [(a + b) / 2.0 for a, b in self._bounds]
        self._determinant = math.prod(half_lengths)
        # Columns, so they broadcast over points (one per column).
        self._scale = bkd.reshape(bkd.asarray(half_lengths), (-1, 1))
        self._shift = bkd.reshape(bkd.asarray(centers), (-1, 1))
        self._jacobian = bkd.diag(bkd.asarray(half_lengths))
        self._inverse_jacobian = bkd.diag(bkd.asarray([1.0 / s for s in half_lengths]))

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def ndim(self) -> int:
        """Return the number of directions."""
        return len(self._bounds)

    def bounds(self) -> Tuple[Tuple[float, float], ...]:
        """``(a_d, b_d)`` per direction."""
        return self._bounds

    def map_to_physical(self, reference_pts: Array) -> Array:
        """Map reference points, shape ``(ndim, npts)``, to the box."""
        return self._scale * reference_pts + self._shift

    def map_to_reference(self, physical_pts: Array) -> Array:
        """Map points of the box, shape ``(ndim, npts)``, to the reference
        cube."""
        return (physical_pts - self._shift) / self._scale

    def _per_point(self, matrix: Array, reference_pts: Array) -> Array:
        """``matrix`` repeated at every point: ``(npts, ndim, ndim)``."""
        npts = reference_pts.shape[1]
        return self._bkd.ones((npts, 1, 1)) * matrix[None, :, :]

    def jacobian_matrix(self, reference_pts: Array) -> Array:
        """The constant diagonal Jacobian at every point:
        ``(npts, ndim, ndim)``."""
        return self._per_point(self._jacobian, reference_pts)

    def jacobian_determinant(self, reference_pts: Array) -> Array:
        """The product of half-lengths at every point: ``(npts,)``."""
        return self._bkd.full((reference_pts.shape[1],), self._determinant)

    def scale_factors(self, reference_pts: Array) -> Array:
        """The half-lengths (the Jacobian's diagonal) at every point:
        ``(npts, ndim)``."""
        npts = reference_pts.shape[1]
        return self._bkd.ones((npts, 1)) * self._scale[:, 0][None, :]

    def unit_curvilinear_basis(self, reference_pts: Array) -> Array:
        """The Cartesian unit vectors at every point: ``(npts, ndim,
        ndim)``."""
        return self._per_point(self._bkd.eye(self.ndim()), reference_pts)

    def gradient_factors(self, reference_pts: Array) -> Array:
        """Factors turning reference derivatives into physical ones, at
        every point: ``(npts, ndim, ndim)``.

        Entry ``[i, d, j]`` multiplies the ``j``-th reference derivative
        in the ``d``-th physical derivative at point ``i``: here
        :math:`1 / s_d` on the diagonal.
        """
        return self._per_point(self._inverse_jacobian, reference_pts)


class AffineTransform1D(AffineTransform[Array]):
    """:class:`AffineTransform` of the interval ``(a, b)``.

    Parameters
    ----------
    physical_bounds : Tuple[float, float]
        ``(a, b)``.
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self, physical_bounds: Tuple[float, float], bkd: Backend[Array]
    ) -> None:
        super().__init__([physical_bounds], bkd)

    def physical_bounds(self) -> Tuple[float, float]:
        """``(a, b)``."""
        return self._bounds[0]


class AffineTransform2D(AffineTransform[Array]):
    """:class:`AffineTransform` of the rectangle ``[ax, bx] x [ay, by]``.

    Parameters
    ----------
    physical_bounds : Tuple[float, float, float, float]
        ``(ax, bx, ay, by)``.
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self,
        physical_bounds: Tuple[float, float, float, float],
        bkd: Backend[Array],
    ) -> None:
        ax, bx, ay, by = physical_bounds
        super().__init__([(ax, bx), (ay, by)], bkd)

    def physical_bounds(self) -> Tuple[float, float, float, float]:
        """``(ax, bx, ay, by)``."""
        (ax, bx), (ay, by) = self._bounds
        return (ax, bx, ay, by)


class AffineTransform3D(AffineTransform[Array]):
    """:class:`AffineTransform` of the box
    ``[ax, bx] x [ay, by] x [az, bz]``.

    Parameters
    ----------
    physical_bounds : Tuple[float, float, float, float, float, float]
        ``(ax, bx, ay, by, az, bz)``.
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self,
        physical_bounds: Tuple[float, float, float, float, float, float],
        bkd: Backend[Array],
    ) -> None:
        ax, bx, ay, by, az, bz = physical_bounds
        super().__init__([(ax, bx), (ay, by), (az, bz)], bkd)

    def physical_bounds(self) -> Tuple[float, float, float, float, float, float]:
        """``(ax, bx, ay, by, az, bz)``."""
        (ax, bx), (ay, by), (az, bz) = self._bounds
        return (ax, bx, ay, by, az, bz)
