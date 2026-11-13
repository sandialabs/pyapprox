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
