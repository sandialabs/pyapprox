"""Assembly and refusals of the collocation boundary specs.

- On a 2D mesh (5 x 6 nodes, diffusion 1): Dirichlet on the left and
  right sides, gradient Robin on the bottom and the total flux on the
  top give essential rows exactly the left and right nodes and, in spec
  order, closure rows exactly the bottom and the top nodes; 2D periodic
  left-right rows replace both sides' nodes.
- ``assemble_boundaries`` refuses no specs, two specs on one side
  (including a periodic pair over a Dirichlet side), a side the mesh
  lacks (``TOP`` on an interval), and an object that is not a spec.
- A spec without essential rows gives no essential declaration.
"""

from typing import List, Sequence

import pytest

from pyapprox.pde.collocation.basis import ChebyshevBasis1D, ChebyshevBasis2D
from pyapprox.pde.collocation.mesh import (
    BoundarySide,
    TransformedMesh1D,
    TransformedMesh2D,
)
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox_benchmarks.functions.pde.collocation.boundaries import (
    BoundaryContext,
    BoundarySpecProtocol,
    Dirichlet,
    FluxNeumann,
    GradientRobin,
    Periodic,
    assemble_boundaries,
)


def _context_2d(bkd: Backend[Array]) -> BoundaryContext[Array]:
    mesh = TransformedMesh2D(5, 6, bkd)
    basis = ChebyshevBasis2D(mesh, bkd)
    return BoundaryContext(mesh, basis, AdvectionDiffusionReaction(basis, bkd, 1.0))


def _context_1d(bkd: Backend[Array]) -> BoundaryContext[Array]:
    mesh = TransformedMesh1D(6, bkd)
    basis = ChebyshevBasis1D(mesh, bkd)
    return BoundaryContext(mesh, basis, AdvectionDiffusionReaction(basis, bkd, 1.0))


def _indices(bkd: Backend[Array], array: Array) -> List[int]:
    return sorted(bkd.to_int(index) for index in array)


class TestAssembly:
    def test_rows_cover_the_named_sides_in_2d(self, bkd: Backend[Array]) -> None:
        context = _context_2d(bkd)
        mesh = context.mesh
        essential, closures = assemble_boundaries(
            context,
            (
                Dirichlet(BoundarySide.LEFT, 0.0),
                GradientRobin(BoundarySide.BOTTOM, 1.0, 0.5, 0.0),
                Dirichlet(BoundarySide.RIGHT, 1.0),
                FluxNeumann(BoundarySide.TOP, 0.0),
            ),
        )
        assert essential is not None
        side_nodes = {
            side: _indices(bkd, mesh.boundary_indices(side))
            for side in BoundarySide
            if side < mesh.nboundaries()
        }
        assert sorted(essential.rows().indices()) == sorted(
            set(side_nodes[BoundarySide.LEFT]) | set(side_nodes[BoundarySide.RIGHT])
        )
        assert len(closures) == 2
        assert sorted(closures[0].rows().indices()) == side_nodes[BoundarySide.BOTTOM]
        assert sorted(closures[1].rows().indices()) == side_nodes[BoundarySide.TOP]

    def test_periodic_pair_in_2d(self, bkd: Backend[Array]) -> None:
        context = _context_2d(bkd)
        mesh = context.mesh
        essential, closures = assemble_boundaries(
            context,
            (
                Periodic(BoundarySide.LEFT, BoundarySide.RIGHT, dim=0),
                Dirichlet(BoundarySide.BOTTOM, 0.0),
                Dirichlet(BoundarySide.TOP, 0.0),
            ),
        )
        assert essential is not None and len(closures) == 1
        expected = set(_indices(bkd, mesh.boundary_indices(BoundarySide.LEFT))) | set(
            _indices(bkd, mesh.boundary_indices(BoundarySide.RIGHT))
        )
        assert set(closures[0].rows().indices()) == expected

    def test_no_essential_rows_without_dirichlet(self, bkd: Backend[Array]) -> None:
        essential, closures = assemble_boundaries(_context_1d(bkd), (Periodic(),))
        assert essential is None and len(closures) == 1


class TestRefusals:
    @pytest.mark.parametrize(
        "specs",
        [
            (),
            (Dirichlet(BoundarySide.LEFT, 0.0), Dirichlet(BoundarySide.LEFT, 1.0)),
            (Periodic(), Dirichlet(BoundarySide.LEFT, 0.0)),
            (Dirichlet(BoundarySide.LEFT, 0.0), Dirichlet(BoundarySide.TOP, 0.0)),
        ],
        ids=["empty", "same side", "periodic over dirichlet", "side off the mesh"],
    )
    def test_refuses(
        self, bkd: Backend[Array], specs: Sequence[BoundarySpecProtocol]
    ) -> None:
        with pytest.raises(ValueError):
            assemble_boundaries(_context_1d(bkd), specs)

    def test_refuses_a_non_spec(self, bkd: Backend[Array]) -> None:
        with pytest.raises(TypeError, match="BoundarySpecProtocol"):
            assemble_boundaries(_context_1d(bkd), (0.0,))  # type: ignore[arg-type]
