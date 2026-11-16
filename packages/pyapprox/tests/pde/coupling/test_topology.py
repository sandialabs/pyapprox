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
    counts = {m: 0 for m in range(topo.ndim())}
    for e in range(topo.nentities()):
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
        assert len(incident) == 2 ** (topo.ndim() - topo.entity_dim(e))
        assert list(incident) == sorted(incident)


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
    assert topo.nentities() == 5
    vertex = [e for e in range(5) if topo.entity_dim(e) == 0]
    assert len(vertex) == 1
    assert topo.incident_subdomains(vertex[0]) == (0, 1, 2, 3)
    edges = sorted(topo.incident_subdomains(e) for e in range(5) if e != vertex[0])
    assert edges == [(0, 1), (0, 2), (1, 3), (2, 3)]


def test_six_dimensional_smoke() -> None:
    topo = TensorBoxPartition((2,) * 6)
    assert topo.nsubdomains() == 64
    assert topo.nentities() == 3**6 - 2**6
    vertices = [e for e in range(topo.nentities()) if topo.entity_dim(e) == 0]
    assert len(vertices) == 1
    assert topo.incident_subdomains(vertices[0]) == tuple(range(64))
    for s in range(64):
        assert len(topo.subdomain_entities(s)) == 2**6 - 1


def test_ownership() -> None:
    assert TensorBoxPartition((3,)).owner(2) == 0
    topo = TensorBoxPartition((2, 2), owners=[1, 0, 3, 2])
    assert [topo.owner(s) for s in range(4)] == [1, 0, 3, 2]
    assert isinstance(topo, SkeletonTopologyProtocol)


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
