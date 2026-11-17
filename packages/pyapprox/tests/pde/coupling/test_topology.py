"""Tensor box partitions: entity counts, multiplicity and ownership.

Counts are checked against the combinatorics of a cell complex: an
entity of codimension k picks k fixed directions, an interior cut in each
(n_j - 1 choices) and a cell in each other direction (n_j choices).
"""

import itertools
from math import comb, prod

import pytest

from pyapprox.pde.coupling.topology import (
    SkeletonTopologyProtocol,
    TensorBoxPartition,
    entity_owner,
    local_subdomains,
)


def _expected_counts(ncells: tuple[int, ...]) -> dict[int, int]:
    d = len(ncells)
    counts: dict[int, int] = {}
    for k in range(1, d + 1):
        counts[d - k] = sum(
            prod(ncells[j] - 1 if j in fixed else ncells[j] for j in range(d))
            for fixed in itertools.combinations(range(d), k)
        )
    return counts


def _actual_counts(topo: TensorBoxPartition) -> dict[int, int]:
    """Skeleton (off-boundary) entities per dimension."""
    counts = {m: 0 for m in range(topo.ndim())}
    for e in range(topo.nentities()):
        if not topo.on_boundary(e):
            counts[topo.entity_dim(e)] += 1
    return counts


@pytest.mark.parametrize(
    "ncells", [(2,), (5,), (2, 2), (3, 4), (2, 3, 4), (3, 3, 3), (1, 3)]
)
def test_entity_counts_match_combinatorics(ncells: tuple[int, ...]) -> None:
    topo = TensorBoxPartition(ncells)
    assert topo.nsubdomains() == prod(ncells)
    assert _actual_counts(topo) == _expected_counts(ncells)


@pytest.mark.parametrize("d,n", [(1, 4), (2, 3), (3, 3), (4, 2)])
def test_uniform_counts_closed_form(d: int, n: int) -> None:
    topo = TensorBoxPartition((n,) * d)
    expected = {
        d - k: comb(d, k) * (n - 1) ** k * n ** (d - k) for k in range(1, d + 1)
    }
    assert _actual_counts(topo) == expected


@pytest.mark.parametrize("ncells", [(4,), (3, 2), (3, 3, 2), (2, 2, 2, 2)])
def test_multiplicity_is_two_to_the_codimension(ncells: tuple[int, ...]) -> None:
    topo = TensorBoxPartition(ncells)
    for e in range(topo.nentities()):
        incident = topo.incident_subdomains(e)
        assert len(set(incident)) == len(incident)
        assert list(incident) == sorted(incident)
        full = 2 ** (topo.ndim() - topo.entity_dim(e))
        if topo.on_boundary(e):
            # Some cut is on the boundary, so fewer cells surround it.
            assert len(incident) < full
            assert len(incident) & (len(incident) - 1) == 0
        else:
            assert len(incident) == full


@pytest.mark.parametrize("ncells", [(1,), (3,), (2, 2), (3, 1), (2, 3, 2)])
def test_total_count_with_boundary_pieces(ncells: tuple[int, ...]) -> None:
    # Per direction: span a cell (n), an interior cut (n - 1) or a
    # boundary cut (2); at least one cut must be interior.
    topo = TensorBoxPartition(ncells)
    expected = prod(2 * n + 1 for n in ncells) - prod(n + 2 for n in ncells)
    assert topo.nentities() == expected
    first_boundary = sum(not topo.on_boundary(e) for e in range(expected))
    assert all(topo.on_boundary(e) for e in range(first_boundary, expected))


@pytest.mark.parametrize("ncells", [(3,), (2, 2), (3, 2), (2, 2, 2)])
def test_closure_entities(ncells: tuple[int, ...]) -> None:
    topo = TensorBoxPartition(ncells)
    d = topo.ndim()
    reached = set()
    for face in range(topo.nentities()):
        if topo.entity_dim(face) != d - 1 or topo.on_boundary(face):
            continue
        assert topo.closure_entity(face, (0,) * (d - 1)) == face
        for signs in itertools.product((-1, 0, 1), repeat=d - 1):
            piece = topo.closure_entity(face, signs)
            assert topo.entity_dim(piece) == signs.count(0)
            assert set(topo.incident_subdomains(face)) <= set(
                topo.incident_subdomains(piece)
            )
            reached.add(piece)
    # Every entity lies in the closure of some skeleton face.
    assert reached == set(range(topo.nentities()))


@pytest.mark.parametrize("ncells", [(3,), (3, 2), (2, 3, 2)])
def test_subdomain_entities_invert_incidence(ncells: tuple[int, ...]) -> None:
    topo = TensorBoxPartition(ncells)
    pairs = {
        (e, s) for e in range(topo.nentities()) for s in topo.incident_subdomains(e)
    }
    inverse = {
        (e, s)
        for s in range(topo.nsubdomains())
        for e in topo.subdomain_entities(s)
    }
    assert pairs == inverse


def test_two_by_two_explicit() -> None:
    topo = TensorBoxPartition((2, 2))
    assert topo.nentities() == 9
    assert [topo.on_boundary(e) for e in range(9)] == [False] * 5 + [True] * 4
    vertex = [e for e in range(5) if topo.entity_dim(e) == 0]
    assert len(vertex) == 1
    assert topo.incident_subdomains(vertex[0]) == (0, 1, 2, 3)
    edges = sorted(topo.incident_subdomains(e) for e in range(5) if e != vertex[0])
    assert edges == [(0, 1), (0, 2), (1, 3), (2, 3)]
    # The ends of the two interface lines, on the four sides of the box.
    ends = sorted(topo.incident_subdomains(e) for e in range(5, 9))
    assert ends == [(0, 1), (0, 2), (1, 3), (2, 3)]
    assert all(topo.entity_dim(e) == 0 for e in range(5, 9))


def test_six_dimensional_smoke() -> None:
    topo = TensorBoxPartition((2,) * 6)
    assert topo.nsubdomains() == 64
    assert topo.nentities() == 5**6 - 4**6
    skeleton = [e for e in range(topo.nentities()) if not topo.on_boundary(e)]
    assert len(skeleton) == 3**6 - 2**6
    vertices = [e for e in skeleton if topo.entity_dim(e) == 0]
    assert len(vertices) == 1
    assert topo.incident_subdomains(vertices[0]) == tuple(range(64))
    for s in range(64):
        on_s = topo.subdomain_entities(s)
        # Per direction: span, the interior cut, or this cell's outer cut.
        assert len(on_s) == 3**6 - 2**6
        assert sum(not topo.on_boundary(e) for e in on_s) == 2**6 - 1


def test_ownership() -> None:
    assert TensorBoxPartition((3,)).owner(2) == 0
    topo = TensorBoxPartition((2, 2), owners=[1, 0, 3, 2])
    assert [topo.owner(s) for s in range(4)] == [1, 0, 3, 2]
    assert isinstance(topo, SkeletonTopologyProtocol)


def test_entity_owner_is_the_lowest_incident_owner() -> None:
    # A chain: interface 0 joins subdomains 0 and 1 (owners 2, 5),
    # interface 1 joins 1 and 2 (owners 5, 1).
    chain = TensorBoxPartition((3,), owners=[2, 5, 1])
    assert [entity_owner(chain, e) for e in range(chain.nentities())] == [2, 1]
    # On a 2x2 square the center vertex touches all four subdomains; a
    # boundary piece touches fewer, and takes the lowest of those.
    square = TensorBoxPartition((2, 2), owners=[3, 1, 2, 4])
    for e in range(square.nentities()):
        incident = square.incident_subdomains(e)
        owners = {square.owner(s) for s in incident}
        assert entity_owner(square, e) == min(owners)
        if len(incident) == 4:
            assert entity_owner(square, e) == 1
    uniform = TensorBoxPartition((2, 2, 2), owners=[7] * 8)
    assert {entity_owner(uniform, e) for e in range(uniform.nentities())} == {7}


@pytest.mark.parametrize("ncells", [(4,), (3, 2), (2, 2, 2)])
def test_local_subdomains_partition_the_subdomains(ncells: tuple[int, ...]) -> None:
    n = prod(ncells)
    topo = TensorBoxPartition(ncells, owners=[s % 2 for s in range(n)])
    pieces = [local_subdomains(topo, rank) for rank in range(3)]
    assert sorted(itertools.chain.from_iterable(pieces)) == list(range(n))
    assert all(list(p) == sorted(p) for p in pieces)
    assert pieces[2] == ()


def test_invalid_input() -> None:
    with pytest.raises(ValueError):
        TensorBoxPartition(())
    with pytest.raises(ValueError):
        TensorBoxPartition((2, 0))
    with pytest.raises(ValueError):
        TensorBoxPartition((2,), owners=[0])
    with pytest.raises(ValueError):
        TensorBoxPartition((2,), owners=[0, -1])
    topo = TensorBoxPartition((2,))
    with pytest.raises(IndexError):
        topo.entity_dim(1)
    with pytest.raises(IndexError):
        topo.incident_subdomains(-1)
    with pytest.raises(IndexError):
        topo.subdomain_entities(2)
    with pytest.raises(IndexError):
        topo.owner(2)
    with pytest.raises(IndexError):
        topo.on_boundary(topo.nentities())
    square = TensorBoxPartition((2, 2))
    with pytest.raises(ValueError):
        square.closure_entity(0, (0, 0))
    with pytest.raises(ValueError):
        square.closure_entity(0, (2,))
