"""``BoundarySide`` names the boundary ids the meshes use.

For each side of a 1D (5 nodes), 2D (4 x 5) and 3D (3 x 4 x 5)
transformed mesh, every node of ``boundary_indices(side)`` lies on the
reference side the name states: coordinate ``side // 2`` at -1 for an
even id and +1 for an odd one, exactly. A mesh of dimension d has 2d
sides, the first 2d names.
"""

from typing import List, Union

import pytest

from pyapprox.pde.collocation.mesh import (
    BoundarySide,
    TransformedMesh1D,
    TransformedMesh2D,
    TransformedMesh3D,
)
from pyapprox.util.backends.protocols import Array, Backend

_Mesh = Union[
    TransformedMesh1D[Array], TransformedMesh2D[Array], TransformedMesh3D[Array]
]


def _meshes(bkd: Backend[Array]) -> List[_Mesh[Array]]:
    return [
        TransformedMesh1D(5, bkd),
        TransformedMesh2D(4, 5, bkd),
        TransformedMesh3D(3, 4, 5, bkd),
    ]


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_each_name_is_its_reference_side(bkd: Backend[Array], ndim: int) -> None:
    mesh = _meshes(bkd)[ndim - 1]
    assert mesh.nboundaries() == 2 * ndim
    reference = mesh.reference_points()
    for side in list(BoundarySide)[: 2 * ndim]:
        coordinate = reference[side // 2, mesh.boundary_indices(side)]
        expected = -1.0 if side % 2 == 0 else 1.0
        bkd.assert_allclose(
            coordinate, bkd.full(tuple(coordinate.shape), expected), rtol=0.0, atol=0.0
        )


def test_names_are_the_ids(bkd: Backend[Array]) -> None:
    assert [int(side) for side in BoundarySide] == [0, 1, 2, 3, 4, 5]
    assert [side.name for side in BoundarySide] == [
        "LEFT",
        "RIGHT",
        "BOTTOM",
        "TOP",
        "FRONT",
        "BACK",
    ]
