"""Reference-face bases and the facet-discontinuous skeleton space.

Checks are derivation-free: the nodal (Kronecker) property, partition
of unity, the known mass of linear hat functions, and the total
skeleton measure recovered as 1^T M 1.
"""

import numpy as np
import pytest

from pyapprox.pde.coupling.basis import TensorLagrangeBasis
from pyapprox.pde.coupling.spaces import (
    FacetSkeletonSpace,
    SkeletonBasisProtocol,
)
from pyapprox.pde.coupling.topology import TensorBoxPartition
from pyapprox.util.backends.protocols import Array, Backend


@pytest.mark.parametrize("ndim,npoints", [(0, 1), (1, 1), (1, 2), (1, 4), (2, 3)])
def test_basis_is_nodal_partition_of_unity(
    bkd: Backend[Array], ndim: int, npoints: int
) -> None:
    basis = TensorLagrangeBasis(ndim, npoints, bkd)
    assert isinstance(basis, SkeletonBasisProtocol)
    assert basis.nterms() == npoints**ndim
    nodes = basis.nodes()
    assert tuple(nodes.shape) == (ndim, basis.nterms())
    bkd.assert_allclose(
        basis.evaluate(nodes), bkd.eye(basis.nterms()), rtol=1e-12, atol=1e-12
    )
    points = bkd.asarray(np.random.default_rng(1).uniform(-1, 1, (ndim, 7)))
    bkd.assert_allclose(
        bkd.sum(basis.evaluate(points), axis=1), bkd.ones((7,)), rtol=1e-12
    )
    _, weights = basis.quadrature()
    bkd.assert_allclose(bkd.sum(weights, axis=0), bkd.ones((1,)), rtol=1e-12)


def test_linear_hat_mass(bkd: Backend[Array]) -> None:
    # Hats on [-1, 1], averaged over the interval: [[1/3, 1/6], [1/6, 1/3]].
    basis = TensorLagrangeBasis(1, 2, bkd)
    points, weights = basis.quadrature()
    phi = basis.evaluate(points)
    expected = bkd.asarray([[1 / 3, 1 / 6], [1 / 6, 1 / 3]])
    bkd.assert_allclose(bkd.dot(phi.T, weights * phi), expected, rtol=1e-12)


def test_basis_rejects_bad_input(bkd: Backend[Array]) -> None:
    with pytest.raises(ValueError):
        TensorLagrangeBasis(-1, 2, bkd)
    with pytest.raises(ValueError):
        TensorLagrangeBasis(1, 0, bkd)
    with pytest.raises(ValueError):
        TensorLagrangeBasis(2, 2, bkd).evaluate(bkd.zeros((1, 3)))


@pytest.mark.parametrize(
    "ncells,npoints", [((2,), 1), ((4,), 1), ((2, 2), 3), ((3, 2), 2), ((2, 2, 2), 2)]
)
def test_facet_space_dofs_and_mass(
    bkd: Backend[Array], ncells: tuple[int, ...], npoints: int
) -> None:
    bounds = [(0.0, 2.0)] + [(0.0, 1.0)] * (len(ncells) - 1)
    topo = TensorBoxPartition(ncells, bounds=bounds)
    d = topo.ndim()
    space = FacetSkeletonSpace(topo, TensorLagrangeBasis(d - 1, npoints, bkd))
    faces = [e for e in range(topo.nentities()) if topo.entity_dim(e) == d - 1]
    nterms = npoints ** (d - 1)
    size = space.descriptor().size
    assert size == len(faces) * nterms
    blocks = [space.entity_dofs(e) for e in range(topo.nentities())]
    assert sorted(dof for block in blocks for dof in block) == list(range(size))
    for e, block in enumerate(blocks):
        assert len(block) == (nterms if e in faces else 0)

    matrix = space.mass().format().matrix()
    bkd.assert_allclose(matrix, matrix.T, rtol=1e-12)
    eigenvalues, _ = bkd.eigh(matrix)
    assert bkd.to_float(bkd.min(eigenvalues)) > 0.0
    ones = space.vector_space().zeros() + 1.0
    total = bkd.sum(ones * space.mass().apply(ones))
    expected = sum(topo.entity_measure(f) for f in faces)
    bkd.assert_allclose(
        bkd.reshape(total, (1,)), bkd.asarray([expected]), rtol=1e-12
    )


def test_facet_space_rejects_mismatch(bkd: Backend[Array]) -> None:
    topo = TensorBoxPartition((2, 2))
    with pytest.raises(ValueError):
        FacetSkeletonSpace(topo, TensorLagrangeBasis(2, 2, bkd))
    with pytest.raises(TypeError):
        FacetSkeletonSpace(object(), TensorLagrangeBasis(1, 2, bkd))  # type: ignore[arg-type]
    space = FacetSkeletonSpace(topo, TensorLagrangeBasis(1, 2, bkd))
    with pytest.raises(IndexError):
        space.entity_dofs(topo.nentities())
