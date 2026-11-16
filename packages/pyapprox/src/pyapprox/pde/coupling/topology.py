r"""Skeleton topology: subdomains, the entities they share, and who owns them.

A partition of a :math:`d`-dimensional domain into subdomains has a
*skeleton*: the union of the subdomain-boundary pieces that lie inside
the domain. Its entities are faces (dimension :math:`d-1`), edges,
vertices and so on down to dimension 0. An entity of codimension
:math:`k` is shared by its *incident* subdomains, whose number is its
multiplicity. Pieces of a subdomain boundary on the physical boundary
are not skeleton entities: they carry the problem's own boundary
conditions, not coupling data.

Entities and subdomains are numbered from zero. Each subdomain has an
owner, the rank that evaluates its local response.
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

    def incident_subdomains(self, entity: int) -> tuple[int, ...]:
        """The subdomains sharing ``entity``, in increasing order.

        The length is the entity's multiplicity, at least 2.
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
    over one cell each, and is fixed at an interior cut in the others.
    With :math:`k` fixed directions it has codimension :math:`k` and is
    shared by the :math:`2^k` cells on either side of each cut. Entities
    are numbered by codimension, then by their multi-index.

    Parameters
    ----------
    ncells
        The number of subdomains along each direction, each at least 1.
    owners
        The owning rank of each subdomain. Default: all on rank 0.
    """

    def __init__(
        self, ncells: Sequence[int], owners: Optional[Sequence[int]] = None
    ) -> None:
        if len(ncells) == 0 or any(n < 1 for n in ncells):
            raise ValueError(f"ncells must be nonempty and positive, got {ncells}")
        self._ncells = tuple(ncells)
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
        # fixed at an interior cut between cells cut - 1 and cut.
        choices = [
            [(False, c) for c in range(n)] + [(True, i) for i in range(1, n)]
            for n in self._ncells
        ]
        entities = [
            e for e in itertools.product(*choices) if any(f for f, _ in e)
        ]
        entities.sort(key=lambda e: sum(f for f, _ in e))
        self._dims = tuple(sum(not f for f, _ in e) for e in entities)
        self._incident = tuple(self._cells_around(e) for e in entities)
        on_subdomain: list[list[int]] = [[] for _ in range(self._nsub)]
        for entity, cells in enumerate(self._incident):
            for cell in cells:
                on_subdomain[cell].append(entity)
        self._subdomain_entities = tuple(tuple(es) for es in on_subdomain)

    def _cells_around(self, entity: Sequence[tuple[bool, int]]) -> tuple[int, ...]:
        per_dir = [(i - 1, i) if fixed else (i,) for fixed, i in entity]
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

    def incident_subdomains(self, entity: int) -> tuple[int, ...]:
        _check_index(entity, len(self._dims), "entity")
        return self._incident[entity]

    def subdomain_entities(self, subdomain: int) -> tuple[int, ...]:
        _check_index(subdomain, self._nsub, "subdomain")
        return self._subdomain_entities[subdomain]

    def owner(self, subdomain: int) -> int:
        _check_index(subdomain, self._nsub, "subdomain")
        return self._owners[subdomain]
