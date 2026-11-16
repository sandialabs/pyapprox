r"""Skeleton topology: subdomains, the entities they share, and who owns them.

A partition of a :math:`d`-dimensional domain into subdomains has a
*skeleton*: the union of the subdomain-boundary pieces that lie inside
the domain. Its entities are faces (dimension :math:`d-1`), edges,
vertices and so on down to dimension 0. An entity of codimension
:math:`k` is shared by its *incident* subdomains, whose number is its
multiplicity.

Where the skeleton meets the physical boundary, its faces' closures
contain pieces of the boundary: the ends of an interface line in 2D,
say. These are entities too, flagged :meth:`on_boundary`, since nodes
on them are shared by the faces that meet there; the problem's own
boundary conditions decide what is imposed on them. Pieces of the
physical boundary outside the skeleton's closure are not entities.

Entities and subdomains are numbered from zero, skeleton entities
before boundary ones. Each subdomain has an owner, the rank that
evaluates its local response.
"""

import itertools
from typing import Optional, Protocol, Sequence, runtime_checkable


@runtime_checkable
class SkeletonTopologyProtocol(Protocol):
    """Subdomains, skeleton entities with dimension and incidence, owners."""

    def ndim(self) -> int:
        """The spatial dimension of the partitioned domain."""
        ...

    def nsubdomains(self) -> int:
        """The number of subdomains."""
        ...

    def nentities(self) -> int:
        """The number of skeleton entities."""
        ...

    def entity_dim(self, entity: int) -> int:
        """The dimension of ``entity``, from 0 to ``ndim() - 1``."""
        ...

    def on_boundary(self, entity: int) -> bool:
        """Whether ``entity`` lies on the physical boundary."""
        ...

    def incident_subdomains(self, entity: int) -> tuple[int, ...]:
        """The subdomains whose closure contains ``entity``, increasing.

        The length is the entity's multiplicity, at least 2 off the
        physical boundary.
        """
        ...

    def subdomain_entities(self, subdomain: int) -> tuple[int, ...]:
        """The skeleton entities on ``subdomain``'s boundary, increasing."""
        ...

    def owner(self, subdomain: int) -> int:
        """The rank that owns ``subdomain``."""
        ...


def _check_index(index: int, size: int, name: str) -> None:
    if not 0 <= index < size:
        raise IndexError(f"{name} {index} is out of range [0, {size})")


class TensorBoxPartition:
    r"""A box cut into :math:`n_1 \times \cdots \times n_d` subdomains.

    Subdomains are numbered by their cell multi-index in row-major order
    (the last direction varies fastest). An entity spans some directions,
    over one cell each, and is fixed at a cut in the others, at least one
    of them interior. With :math:`k` fixed directions it has codimension
    :math:`k`; with :math:`k_i` of them interior it is shared by
    :math:`2^{k_i}` cells, :math:`2^k` off the physical boundary.
    Skeleton entities come first, then boundary ones; each group is
    numbered by codimension, then by multi-index.

    Parameters
    ----------
    ncells
        The number of subdomains along each direction, each at least 1.
    owners
        The owning rank of each subdomain. Default: all on rank 0.
    bounds
        The box's ``(lower, upper)`` extent along each direction, cut into
        equal cells. Default: the unit box.
    """

    def __init__(
        self,
        ncells: Sequence[int],
        owners: Optional[Sequence[int]] = None,
        bounds: Optional[Sequence[tuple[float, float]]] = None,
    ) -> None:
        if len(ncells) == 0 or any(n < 1 for n in ncells):
            raise ValueError(f"ncells must be nonempty and positive, got {ncells}")
        self._ncells = tuple(ncells)
        if bounds is None:
            bounds = [(0.0, 1.0)] * len(ncells)
        if len(bounds) != len(ncells) or any(lo >= up for lo, up in bounds):
            raise ValueError(
                f"bounds must give lower < upper for each of {len(ncells)} "
                f"directions, got {bounds}"
            )
        self._lower = tuple(lo for lo, _ in bounds)
        self._widths = tuple(
            (up - lo) / n for (lo, up), n in zip(bounds, self._ncells)
        )
        nsub = 1
        for n in self._ncells:
            nsub *= n
        if owners is None:
            owners = [0] * nsub
        if len(owners) != nsub or any(r < 0 for r in owners):
            raise ValueError(
                f"owners must give a nonnegative rank for each of {nsub} "
                f"subdomains, got {owners}"
            )
        self._owners = tuple(owners)
        self._nsub = nsub
        self._build_entities()

    def _build_entities(self) -> None:
        # Per direction: (False, cell) spans that cell; (True, cut) is
        # fixed at cut between cells cut - 1 and cut, 0 and n being the
        # physical boundary.
        choices = [
            [(False, c) for c in range(n)] + [(True, i) for i in range(n + 1)]
            for n in self._ncells
        ]
        entities = [
            e
            for e in itertools.product(*choices)
            if any(self._interior_cut(j, e[j]) for j in range(len(e)))
        ]
        entities.sort(
            key=lambda e: (self._touches_boundary(e), sum(f for f, _ in e))
        )
        self._entities = tuple(entities)
        self._ids = {e: k for k, e in enumerate(entities)}
        self._dims = tuple(sum(not f for f, _ in e) for e in entities)
        self._boundary = tuple(self._touches_boundary(e) for e in entities)
        self._incident = tuple(self._cells_around(e) for e in entities)
        on_subdomain: list[list[int]] = [[] for _ in range(self._nsub)]
        for entity, cells in enumerate(self._incident):
            for cell in cells:
                on_subdomain[cell].append(entity)
        self._subdomain_entities = tuple(tuple(es) for es in on_subdomain)

    def _interior_cut(self, direction: int, choice: tuple[bool, int]) -> bool:
        fixed, i = choice
        return fixed and 0 < i < self._ncells[direction]

    def _touches_boundary(self, entity: Sequence[tuple[bool, int]]) -> bool:
        return any(
            fixed and not self._interior_cut(j, (fixed, i))
            for j, (fixed, i) in enumerate(entity)
        )

    def _cells_around(self, entity: Sequence[tuple[bool, int]]) -> tuple[int, ...]:
        per_dir = [
            tuple(c for c in (i - 1, i) if 0 <= c < n) if fixed else (i,)
            for (fixed, i), n in zip(entity, self._ncells)
        ]
        return tuple(sorted(self._flat(m) for m in itertools.product(*per_dir)))

    def _flat(self, multi_index: Sequence[int]) -> int:
        flat = 0
        for n, i in zip(self._ncells, multi_index):
            flat = flat * n + i
        return flat

    def ncells(self) -> tuple[int, ...]:
        """The number of subdomains along each direction."""
        return self._ncells

    def ndim(self) -> int:
        return len(self._ncells)

    def nsubdomains(self) -> int:
        return self._nsub

    def nentities(self) -> int:
        return len(self._dims)

    def entity_dim(self, entity: int) -> int:
        _check_index(entity, len(self._dims), "entity")
        return self._dims[entity]

    def on_boundary(self, entity: int) -> bool:
        _check_index(entity, len(self._dims), "entity")
        return self._boundary[entity]

    def closure_entity(self, entity: int, signs: Sequence[int]) -> int:
        r"""The entity a piece of ``entity``'s closure is, given the piece
        as a sub-entity of the reference box :math:`[-1, 1]^m`.

        ``signs`` has one entry per direction ``entity`` spans, in
        increasing axis order: 0 keeps spanning it, -1 and +1 fix it at
        the cell's lower and upper cut. All zeros is ``entity`` itself.
        """
        _check_index(entity, len(self._dims), "entity")
        pattern = list(self._entities[entity])
        spanned = [j for j, (fixed, _) in enumerate(pattern) if not fixed]
        if len(signs) != len(spanned) or any(s not in (-1, 0, 1) for s in signs):
            raise ValueError(
                f"signs must give -1, 0 or 1 for each of {len(spanned)} spanned "
                f"directions, got {tuple(signs)}"
            )
        for j, sign in zip(spanned, signs):
            if sign != 0:
                pattern[j] = (True, pattern[j][1] + (sign + 1) // 2)
        return self._ids[tuple(pattern)]

    def entity_bounds(self, entity: int) -> tuple[tuple[float, float], ...]:
        """The ``(lower, upper)`` extent of ``entity`` along each direction;
        equal along the directions it is fixed in."""
        _check_index(entity, len(self._dims), "entity")
        return tuple(
            (lo + i * w, lo + (i + (not fixed)) * w)
            for (fixed, i), lo, w in zip(
                self._entities[entity], self._lower, self._widths
            )
        )

    def entity_measure(self, entity: int) -> float:
        """The ``entity_dim(entity)``-dimensional measure of ``entity``: the
        product of its cell widths along the directions it spans (1 for a
        vertex)."""
        _check_index(entity, len(self._dims), "entity")
        measure = 1.0
        for (fixed, _), width in zip(self._entities[entity], self._widths):
            if not fixed:
                measure *= width
        return measure

    def incident_subdomains(self, entity: int) -> tuple[int, ...]:
        _check_index(entity, len(self._dims), "entity")
        return self._incident[entity]

    def subdomain_entities(self, subdomain: int) -> tuple[int, ...]:
        _check_index(subdomain, self._nsub, "subdomain")
        return self._subdomain_entities[subdomain]

    def owner(self, subdomain: int) -> int:
        _check_index(subdomain, self._nsub, "subdomain")
        return self._owners[subdomain]
