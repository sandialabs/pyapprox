"""The reference box's sub-entities and vertex weights.

Checks are derivation-free: sub-entity counts (3^m, with C(m, k) 2^(m-k)
of dimension k), a vertex's weight is 1 on itself, weights sum to 1 and
reproduce the point, and the key of a point on a shared edge is the
same when the edge is read from either end.
"""

import itertools
from math import comb

import pytest

from pyapprox.pde.coupling.closure_numbering import ReferenceFaceProtocol
from pyapprox.pde.coupling.reference_faces import ReferenceBox


@pytest.mark.parametrize("ndim", [0, 1, 2, 3])
def test_subentities(ndim: int) -> None:
    box = ReferenceBox(ndim)
    assert isinstance(box, ReferenceFaceProtocol)
    assert box.nsubentities() == 3**ndim
    dims = [box.subentity_dim(k) for k in range(box.nsubentities())]
    assert dims == sorted(dims)
    for k in range(ndim + 1):
        assert dims.count(k) == comb(ndim, k) * 2 ** (ndim - k)
    for k in range(box.nsubentities()):
        vertices = box.subentity_vertices(k)
        assert len(vertices) == 2 ** box.subentity_dim(k)
        assert all(box.subentity_dim(v) == 0 for v in vertices)
    assert box.subentity_vertices(box.nsubentities() - 1) == tuple(range(2**ndim))


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_weights_reproduce_points(ndim: int) -> None:
    box = ReferenceBox(ndim)
    for k in range(box.nsubentities()):
        signs = box.signs(k)
        for t in itertools.product((-0.6, 0.2), repeat=ndim):
            point = [float(s) if s != 0 else x for s, x in zip(signs, t)]
            weights = box.vertex_weights(k, point)
            assert abs(sum(weights) - 1.0) < 1e-14
            for j in range(ndim):
                mixed = sum(
                    w * box.signs(v)[j]
                    for w, v in zip(weights, box.subentity_vertices(k))
                )
                assert abs(mixed - point[j]) < 1e-14
    for v in range(2**ndim):
        assert box.vertex_weights(v, box.signs(v)) == (1.0,)


def test_edge_key_is_orientation_free() -> None:
    # Two squares sharing an edge, numbered in opposite directions: the
    # edge's vertices are entities 7 and 12 on one, 12 and 7 on the other.
    box = ReferenceBox(2)
    edge = box.subentity((-1, 0))
    low, high = box.subentity_vertices(edge)
    keys = []
    for ends, x in (({low: 7, high: 12}, -0.3), ({low: 12, high: 7}, 0.3)):
        weights = box.vertex_weights(edge, [-1.0, x])
        keys.append(
            sorted((ends[v], w) for v, w in zip(box.subentity_vertices(edge), weights))
        )
    assert keys[0] == pytest.approx(keys[1])


def test_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        ReferenceBox(-1)
    box = ReferenceBox(2)
    with pytest.raises(IndexError):
        box.subentity_vertices(9)
    with pytest.raises(ValueError):
        box.subentity((2, 0))
    with pytest.raises(ValueError):
        box.vertex_weights(8, [0.0])
    assert box == ReferenceBox(2) and box != ReferenceBox(1)
