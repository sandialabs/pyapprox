r"""Reference shapes of skeleton faces, described by their sub-entities.

A reference face lists its sub-entities (vertices, edges, ..., the face
itself), each with its dimension and its vertices, and gives the
weights of a point of a sub-entity on that sub-entity's vertices. A
partition maps each face's sub-entities to skeleton entities; a nodal
basis says which sub-entity each node lies on. Weights on vertices,
attached to the skeleton's vertex entities, identify a node on a shared
entity the same way from every face, whatever each face's orientation.

Boxes (:class:`ReferenceBox`) and simplices (:class:`ReferenceSimplex`)
are the shapes here; any other is a new class with the same methods.
"""

import itertools
from typing import Sequence


class ReferenceBox:
    r"""The reference box :math:`[-1, 1]^m`.

    A sub-entity fixes each direction at -1 or +1 or spans it (0); the
    sub-entities are numbered by dimension, vertices first, then
    lexicographically in their sign patterns. Weights are multilinear.

    Parameters
    ----------
    ndim
        The dimension :math:`m`, at least 0.
    """

    def __init__(self, ndim: int) -> None:
        if ndim < 0:
            raise ValueError(f"ndim must be >= 0, got {ndim}")
        self._ndim = ndim
        patterns = itertools.product((-1, 0, 1), repeat=ndim)
        self._signs = tuple(sorted(patterns, key=lambda s: (s.count(0), s)))
        self._ids = {s: k for k, s in enumerate(self._signs)}
        self._vertices = tuple(
            tuple(
                self._ids[tuple(v if v != 0 else c for v, c in zip(s, corner))]
                for corner in itertools.product((-1, 1), repeat=ndim)
                if all(v in (0, c) for v, c in zip(s, corner))
            )
            for s in self._signs
        )

    def ndim(self) -> int:
        return self._ndim

    def nsubentities(self) -> int:
        return len(self._signs)

    def subentity_dim(self, subentity: int) -> int:
        return self.signs(subentity).count(0)

    def subentity_vertices(self, subentity: int) -> tuple[int, ...]:
        self.signs(subentity)
        return self._vertices[subentity]

    def vertex_weights(
        self, subentity: int, point: Sequence[float]
    ) -> tuple[float, ...]:
        r"""The multilinear weights of ``point``, which lies on
        ``subentity``, on its vertices: the product over the spanned
        directions of :math:`(1 + c_j x_j)/2`, :math:`c` the vertex."""
        signs = self.signs(subentity)
        if len(point) != self._ndim:
            raise ValueError(
                f"point must have {self._ndim} coordinates, got {len(point)}"
            )
        weights = []
        for vertex in self._vertices[subentity]:
            weight = 1.0
            for s, c, x in zip(signs, self._signs[vertex], point):
                if s == 0:
                    weight *= (1.0 + c * x) / 2.0
            weights.append(weight)
        return tuple(weights)

    def signs(self, subentity: int) -> tuple[int, ...]:
        """The sign pattern of ``subentity``: per direction, -1 or +1 if
        it is fixed at that end, 0 if it spans it."""
        if not 0 <= subentity < len(self._signs):
            raise IndexError(
                f"subentity {subentity} is out of range [0, {len(self._signs)})"
            )
        return self._signs[subentity]

    def subentity(self, signs: Sequence[int]) -> int:
        """The sub-entity with sign pattern ``signs``."""
        key = tuple(signs)
        if key not in self._ids:
            raise ValueError(
                f"signs must give -1, 0 or 1 for each of {self._ndim} "
                f"directions, got {key}"
            )
        return self._ids[key]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ReferenceBox) and other._ndim == self._ndim

    def __hash__(self) -> int:
        return hash((ReferenceBox, self._ndim))

    def __repr__(self) -> str:
        return f"ReferenceBox({self._ndim})"


class ReferenceSimplex:
    r"""The reference simplex with vertices :math:`0, e_1, \ldots, e_m`.

    A sub-entity is a nonempty set of its vertices; the sub-entities are
    numbered by size, vertices first (vertex :math:`k` is sub-entity
    :math:`k`), then lexicographically. Weights are barycentric.

    Parameters
    ----------
    ndim
        The dimension :math:`m`, at least 0.
    """

    def __init__(self, ndim: int) -> None:
        if ndim < 0:
            raise ValueError(f"ndim must be >= 0, got {ndim}")
        self._ndim = ndim
        self._sets = tuple(
            subset
            for size in range(1, ndim + 2)
            for subset in itertools.combinations(range(ndim + 1), size)
        )
        self._ids = {s: k for k, s in enumerate(self._sets)}

    def ndim(self) -> int:
        return self._ndim

    def nsubentities(self) -> int:
        return len(self._sets)

    def subentity_dim(self, subentity: int) -> int:
        return len(self.subentity_vertices(subentity)) - 1

    def subentity_vertices(self, subentity: int) -> tuple[int, ...]:
        if not 0 <= subentity < len(self._sets):
            raise IndexError(
                f"subentity {subentity} is out of range [0, {len(self._sets)})"
            )
        return self._sets[subentity]

    def vertex_weights(
        self, subentity: int, point: Sequence[float]
    ) -> tuple[float, ...]:
        r"""The barycentric coordinates of ``point``, which lies on
        ``subentity``, on its vertices: :math:`1 - \sum_j x_j` on vertex
        0 and :math:`x_j` on vertex :math:`j`."""
        vertices = self.subentity_vertices(subentity)
        if len(point) != self._ndim:
            raise ValueError(
                f"point must have {self._ndim} coordinates, got {len(point)}"
            )
        barycentric = [1.0 - sum(point)] + [float(x) for x in point]
        return tuple(barycentric[v] for v in vertices)

    def subentity(self, vertices: Sequence[int]) -> int:
        """The sub-entity spanned by ``vertices``."""
        key = tuple(sorted(vertices))
        if key not in self._ids:
            raise ValueError(
                f"vertices must be a nonempty set from 0..{self._ndim}, got "
                f"{tuple(vertices)}"
            )
        return self._ids[key]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ReferenceSimplex) and other._ndim == self._ndim

    def __hash__(self) -> int:
        return hash((ReferenceSimplex, self._ndim))

    def __repr__(self) -> str:
        return f"ReferenceSimplex({self._ndim})"
