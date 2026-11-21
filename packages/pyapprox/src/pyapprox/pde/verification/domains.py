"""Domains of verification problems, with named boundaries.

A domain says where a problem is posed, names its coordinates, and names
the parts of its boundary that conditions attach to. It carries no mesh:
a discretization builds its own meshes of the domain and must name their
boundaries the same way.

Coordinates are single letters, so jet symbols such as ``f_xv`` read
unambiguously. Physical space defaults to ``x, y, z``; a phase space
(kinetic equations) names its velocity coordinates too, e.g. ``x, v`` or
``x, y, z, u, v, w``.
"""

import math
from typing import Optional, Protocol, Sequence, Tuple, runtime_checkable

_STANDARD = ("x", "y", "z")


@runtime_checkable
class DomainProtocol(Protocol):
    """A region with named coordinates and named boundary parts."""

    def ndim(self) -> int:
        """The number of coordinates."""
        ...

    def coordinates(self) -> Tuple[str, ...]:
        """The coordinate names, in order."""
        ...

    def boundaries(self) -> Tuple[str, ...]:
        """The names of the boundary parts, which cover the boundary."""
        ...

    def interval(self, coordinate: str) -> Optional[Tuple[float, float]]:
        """``(lower, upper)`` of ``coordinate`` when the domain is a product
        over it (needed to integrate over that coordinate); ``None``
        otherwise."""
        ...


_SIDES = (("left", "right"), ("bottom", "top"), ("back", "front"))


def _check_coordinates(names: Sequence[str]) -> None:
    for name in names:
        if len(name) != 1 or not name.isalpha() or name in ("t", "n"):
            raise ValueError(
                f"coordinates are single letters other than t and n, got {name!r}"
            )
    if len(set(names)) != len(names):
        raise ValueError(f"coordinates must differ, got {list(names)}")


class Box:
    """An axis-aligned box.

    Boundary parts, lower side first in each direction. With the default
    coordinates: ``left``, ``right`` (:math:`x`), ``bottom``, ``top``
    (:math:`y`), ``back``, ``front`` (:math:`z`). With named coordinates:
    ``<name>min``, ``<name>max`` (``xmin``, ``xmax``, ``vmin``, ...).

    Parameters
    ----------
    bounds : Sequence[Tuple[float, float]]
        ``(lower, upper)`` in each direction.
    coordinates : Sequence[str], optional
        The coordinate names, single letters. Default ``x, y, z`` (up to
        three directions).

    Raises
    ------
    ValueError
        If there are no directions, more than three without names, a
        lower bound is not below its upper bound, or the names are not
        distinct single letters other than ``t`` and ``n``.
    """

    def __init__(
        self,
        bounds: Sequence[Tuple[float, float]],
        coordinates: Optional[Sequence[str]] = None,
    ) -> None:
        if coordinates is None and not 1 <= len(bounds) <= 3:
            raise ValueError(
                f"a box without coordinate names has one to three directions, "
                f"got {len(bounds)}"
            )
        names = _STANDARD[: len(bounds)] if coordinates is None else tuple(coordinates)
        if len(names) != len(bounds) or not names:
            raise ValueError(
                f"one coordinate name per direction: {len(bounds)} directions, "
                f"names {list(names)}"
            )
        _check_coordinates(names)
        for lower, upper in bounds:
            if not lower < upper:
                raise ValueError(f"each lower bound must be below its upper: {bounds}")
        self._bounds = tuple((float(lo), float(up)) for lo, up in bounds)
        self._names, self._named = names, coordinates is not None

    def ndim(self) -> int:
        """The number of coordinates."""
        return len(self._bounds)

    def coordinates(self) -> Tuple[str, ...]:
        """The coordinate names."""
        return self._names

    def bounds(self) -> Tuple[Tuple[float, float], ...]:
        """``(lower, upper)`` in each direction."""
        return self._bounds

    def interval(self, coordinate: str) -> Optional[Tuple[float, float]]:
        """``(lower, upper)`` of ``coordinate``.

        Raises
        ------
        ValueError
            If the box has no such coordinate.
        """
        if coordinate not in self._names:
            raise ValueError(f"no coordinate {coordinate!r}; have {list(self._names)}")
        return self._bounds[self._names.index(coordinate)]

    def boundaries(self) -> Tuple[str, ...]:
        """The sides, lower first in each direction."""
        if self._named:
            ends = ("min", "max")
            return tuple(f"{name}{end}" for name in self._names for end in ends)
        return tuple(name for pair in _SIDES[: self.ndim()] for name in pair)

    def side(self, part: str) -> Tuple[str, float]:
        """The coordinate normal to side ``part``, and the sign of its
        outward normal there (-1 on the lower side, +1 on the upper).

        Raises
        ------
        ValueError
            If the box has no such side.
        """
        sides = self.boundaries()
        if part not in sides:
            raise ValueError(f"no side {part!r}; the box has {list(sides)}")
        index = sides.index(part)
        return self._names[index // 2], -1.0 if index % 2 == 0 else 1.0

    def __repr__(self) -> str:
        return f"{type(self).__name__}({list(self._bounds)}, {list(self._names)})"


class Disk:
    """A disk in two dimensions, with one boundary part, ``circle``.

    Parameters
    ----------
    center : Tuple[float, float]
        The center.
    radius : float
        The radius.

    Raises
    ------
    ValueError
        If the radius is not positive.
    """

    def __init__(self, center: Tuple[float, float] = (0.0, 0.0), radius: float = 1.0):
        if not radius > 0.0:
            raise ValueError(f"radius must be positive, got {radius}")
        self._center = (float(center[0]), float(center[1]))
        self._radius = float(radius)

    def ndim(self) -> int:
        """The spatial dimension, two."""
        return 2

    def coordinates(self) -> Tuple[str, ...]:
        """``x, y``."""
        return _STANDARD[:2]

    def interval(self, coordinate: str) -> Optional[Tuple[float, float]]:
        """``None``: a disk is not a product over either coordinate."""
        return None

    def center(self) -> Tuple[float, float]:
        """The center."""
        return self._center

    def radius(self) -> float:
        """The radius."""
        return self._radius

    def boundaries(self) -> Tuple[str, ...]:
        """``circle``."""
        return ("circle",)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(center={self._center}, radius={self._radius})"


class Annulus:
    """The region between two concentric circles in two dimensions, with
    boundary parts ``inner`` and ``outer``.

    Parameters
    ----------
    center : Tuple[float, float]
        The common center.
    r_inner, r_outer : float
        The radii.

    Raises
    ------
    ValueError
        If ``r_inner`` is not positive or not below ``r_outer``.
    """

    def __init__(
        self,
        center: Tuple[float, float] = (0.0, 0.0),
        r_inner: float = 1.0,
        r_outer: float = 2.0,
    ) -> None:
        if not 0.0 < r_inner < r_outer:
            raise ValueError(
                f"radii must satisfy 0 < r_inner < r_outer, got {r_inner}, {r_outer}"
            )
        self._center = (float(center[0]), float(center[1]))
        self._radii = (float(r_inner), float(r_outer))

    def ndim(self) -> int:
        """The spatial dimension, two."""
        return 2

    def coordinates(self) -> Tuple[str, ...]:
        """``x, y``."""
        return _STANDARD[:2]

    def interval(self, coordinate: str) -> Optional[Tuple[float, float]]:
        """``None``: an annulus is not a product over either coordinate."""
        return None

    def center(self) -> Tuple[float, float]:
        """The common center."""
        return self._center

    def radii(self) -> Tuple[float, float]:
        """``(r_inner, r_outer)``."""
        return self._radii

    def boundaries(self) -> Tuple[str, ...]:
        """``inner``, ``outer``."""
        return ("inner", "outer")

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(center={self._center}, "
            f"r_inner={self._radii[0]}, r_outer={self._radii[1]})"
        )


class AnnularSector:
    r"""The part of an annulus between two angles in two dimensions,
    :math:`r_{inner} \le r \le r_{outer}`, :math:`\theta_{start} \le
    \theta \le \theta_{end}` about ``center``, with boundary parts
    ``inner``, ``outer`` (the arcs) and ``start``, ``end`` (the radial
    sides).

    Parameters
    ----------
    center : Tuple[float, float]
        The common center.
    r_inner, r_outer : float
        The radii.
    theta_start, theta_end : float
        The angles, in radians.

    Raises
    ------
    ValueError
        If ``r_inner`` is not positive or not below ``r_outer``, or the
        angles do not satisfy :math:`0 < \theta_{end} - \theta_{start}
        < 2\pi`.
    """

    def __init__(
        self,
        center: Tuple[float, float] = (0.0, 0.0),
        r_inner: float = 1.0,
        r_outer: float = 2.0,
        theta_start: float = 0.0,
        theta_end: float = 0.5 * math.pi,
    ) -> None:
        if not 0.0 < r_inner < r_outer:
            raise ValueError(
                f"radii must satisfy 0 < r_inner < r_outer, got {r_inner}, {r_outer}"
            )
        if not 0.0 < theta_end - theta_start < 2.0 * math.pi:
            raise ValueError(
                "angles must satisfy 0 < theta_end - theta_start < 2 pi, got "
                f"{theta_start}, {theta_end}"
            )
        self._center = (float(center[0]), float(center[1]))
        self._radii = (float(r_inner), float(r_outer))
        self._angles = (float(theta_start), float(theta_end))

    def ndim(self) -> int:
        """The spatial dimension, two."""
        return 2

    def coordinates(self) -> Tuple[str, ...]:
        """``x, y``."""
        return _STANDARD[:2]

    def interval(self, coordinate: str) -> Optional[Tuple[float, float]]:
        """``None``: a sector is not a product over either coordinate."""
        return None

    def center(self) -> Tuple[float, float]:
        """The common center."""
        return self._center

    def radii(self) -> Tuple[float, float]:
        """``(r_inner, r_outer)``."""
        return self._radii

    def angles(self) -> Tuple[float, float]:
        """``(theta_start, theta_end)``."""
        return self._angles

    def boundaries(self) -> Tuple[str, ...]:
        """``inner``, ``outer``, ``start``, ``end``."""
        return ("inner", "outer", "start", "end")

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(center={self._center}, "
            f"r_inner={self._radii[0]}, r_outer={self._radii[1]}, "
            f"theta_start={self._angles[0]}, theta_end={self._angles[1]})"
        )


class EllipticalAnnulus:
    r"""The region between two confocal ellipses centered at the origin,
    foci :math:`(\pm a, 0)`, with boundary parts ``inner`` and ``outer``.

    In elliptical coordinates :math:`x = a\cosh u \cos v`,
    :math:`y = a\sinh u \sin v`, the ellipse :math:`u = c` has semi-axes
    :math:`a\cosh c` and :math:`a\sinh c`; the region is
    :math:`u_{inner} \le u \le u_{outer}`. A filled ellipse would contain
    the focal segment, where these coordinates are singular.

    Parameters
    ----------
    focal : float
        :math:`a`, the distance from the center to each focus.
    u_inner, u_outer : float
        The elliptical radii of the two ellipses.

    Raises
    ------
    ValueError
        If ``focal`` is not positive, or ``u_inner`` is not positive or
        not below ``u_outer``.
    """

    def __init__(
        self, focal: float = 1.0, u_inner: float = 0.5, u_outer: float = 1.0
    ) -> None:
        if not focal > 0.0:
            raise ValueError(f"focal must be positive, got {focal}")
        if not 0.0 < u_inner < u_outer:
            raise ValueError(
                f"elliptical radii must satisfy 0 < u_inner < u_outer, got "
                f"{u_inner}, {u_outer}"
            )
        self._focal = float(focal)
        self._radii = (float(u_inner), float(u_outer))

    def ndim(self) -> int:
        """The spatial dimension, two."""
        return 2

    def coordinates(self) -> Tuple[str, ...]:
        """``x, y``."""
        return _STANDARD[:2]

    def interval(self, coordinate: str) -> Optional[Tuple[float, float]]:
        """``None``: the region is not a product over either coordinate."""
        return None

    def focal(self) -> float:
        """:math:`a`, the distance from the center to each focus."""
        return self._focal

    def radii(self) -> Tuple[float, float]:
        """``(u_inner, u_outer)``."""
        return self._radii

    def boundaries(self) -> Tuple[str, ...]:
        """``inner``, ``outer``."""
        return ("inner", "outer")

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(focal={self._focal}, "
            f"u_inner={self._radii[0]}, u_outer={self._radii[1]})"
        )
