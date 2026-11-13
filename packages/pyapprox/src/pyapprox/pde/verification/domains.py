"""Domains of verification problems, with named boundaries.

A domain says where a problem is posed and names the parts of its boundary
that conditions attach to. It carries no mesh: a discretization builds its
own meshes of the domain and must name their boundaries the same way.
"""

from typing import Protocol, Sequence, Tuple, runtime_checkable


@runtime_checkable
class DomainProtocol(Protocol):
    """A region of space with named boundary parts."""

    def ndim(self) -> int:
        """The spatial dimension."""
        ...

    def boundaries(self) -> Tuple[str, ...]:
        """The names of the boundary parts, which cover the boundary."""
        ...


_SIDES = (("left", "right"), ("bottom", "top"), ("back", "front"))


class Box:
    """An axis-aligned box in one, two or three dimensions.

    Boundary parts, lower side first in each direction: ``left``, ``right``
    (:math:`x`), ``bottom``, ``top`` (:math:`y`), ``back``, ``front``
    (:math:`z`).

    Parameters
    ----------
    bounds : Sequence[Tuple[float, float]]
        ``(lower, upper)`` in each direction.

    Raises
    ------
    ValueError
        If there are not one to three directions, or a lower bound is not
        below its upper bound.
    """

    def __init__(self, bounds: Sequence[Tuple[float, float]]) -> None:
        if not 1 <= len(bounds) <= 3:
            raise ValueError(f"a box has one to three directions, got {len(bounds)}")
        for lower, upper in bounds:
            if not lower < upper:
                raise ValueError(f"each lower bound must be below its upper: {bounds}")
        self._bounds = tuple((float(lo), float(up)) for lo, up in bounds)

    def ndim(self) -> int:
        """The spatial dimension."""
        return len(self._bounds)

    def bounds(self) -> Tuple[Tuple[float, float], ...]:
        """``(lower, upper)`` in each direction."""
        return self._bounds

    def boundaries(self) -> Tuple[str, ...]:
        """``left, right[, bottom, top[, back, front]]``."""
        return tuple(name for pair in _SIDES[: self.ndim()] for name in pair)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({list(self._bounds)})"


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
