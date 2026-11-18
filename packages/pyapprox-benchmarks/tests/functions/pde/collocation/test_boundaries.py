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
- On a 2D displacement (5 x 6 nodes, two components): a Dirichlet
  component, a traction component and a pressure give rows at the side's
  nodes offset by ``component * npts``; one side may carry different
  components; the traction and pressure specs refuse a physics of the
  other law, and the flux spec one without a flux.
- ``assemble_boundaries`` refuses a component the state lacks (including
  a pressure on a scalar state) and two specs on one (side, component).
"""

from typing import List, Sequence

import pytest

from pyapprox.pde.collocation.basis import ChebyshevBasis1D, ChebyshevBasis2D
from pyapprox.pde.collocation.mesh import (
    BoundarySide,
    TransformedMesh1D,
    TransformedMesh2D,
)
from pyapprox.pde.collocation.physics import LinearElasticityPhysics
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.collocation.physics.hyperelasticity import (
    HyperelasticityPhysics,
)
from pyapprox.pde.collocation.physics.neo_hookean_law import NeoHookeanLaw
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox_benchmarks.functions.pde.collocation.boundaries import (
    BoundaryContext,
    BoundarySpecProtocol,
    CauchyPressure,
    CauchyTraction,
    Dirichlet,
    FluxNeumann,
    GradientRobin,
    Periodic,
    PiolaPressure,
    PiolaTraction,
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


def _linear_context(bkd: Backend[Array]) -> BoundaryContext[Array]:
    mesh = TransformedMesh2D(5, 6, bkd)
    basis = ChebyshevBasis2D(mesh, bkd)
    return BoundaryContext(
        mesh, basis, LinearElasticityPhysics(basis, bkd, lamda=1.0, mu=1.0)
    )


def _hyperelastic_context(bkd: Backend[Array]) -> BoundaryContext[Array]:
    mesh = TransformedMesh2D(5, 6, bkd)
    basis = ChebyshevBasis2D(mesh, bkd)
    return BoundaryContext(
        mesh,
        basis,
        HyperelasticityPhysics(basis, bkd, NeoHookeanLaw(bkd), mu=1.0, lamda=1.0),
    )


def _indices(bkd: Backend[Array], array: Array) -> List[int]:
    return sorted(bkd.to_int(index) for index in array)


def _side(context: BoundaryContext[Array], side: BoundarySide) -> List[int]:
    return _indices(context.mesh.bkd(), context.mesh.boundary_indices(side))


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


class TestVectorSpecs:
    def test_component_rows(self, bkd: Backend[Array]) -> None:
        """Dirichlet v on the bottom, traction u on the bottom and the
        right, and a pressure on the left."""
        context = _linear_context(bkd)
        npts = context.mesh.npts()
        essential, closures = assemble_boundaries(
            context,
            (
                CauchyTraction(BoundarySide.RIGHT, 0, 0.0),
                CauchyTraction(BoundarySide.BOTTOM, 0, 0.0),
                CauchyPressure(BoundarySide.LEFT, 1.0),
                Dirichlet(BoundarySide.BOTTOM, 0.0, component=1),
            ),
        )
        assert essential is not None
        bottom = _side(context, BoundarySide.BOTTOM)
        left = _side(context, BoundarySide.LEFT)
        assert sorted(essential.rows().indices()) == [i + npts for i in bottom]
        assert len(closures) == 3
        assert sorted(closures[0].rows().indices()) == _side(
            context, BoundarySide.RIGHT
        )
        assert sorted(closures[1].rows().indices()) == bottom
        assert sorted(closures[2].rows().indices()) == sorted(
            left + [i + npts for i in left]
        )

    def test_piola_specs_on_hyperelasticity(self, bkd: Backend[Array]) -> None:
        context = _hyperelastic_context(bkd)
        npts = context.mesh.npts()
        _, closures = assemble_boundaries(
            context,
            (
                PiolaTraction(BoundarySide.TOP, 1, 0.0),
                PiolaPressure(BoundarySide.LEFT, 1.0),
            ),
        )
        assert sorted(closures[0].rows().indices()) == [
            i + npts for i in _side(context, BoundarySide.TOP)
        ]
        assert len(closures[1].rows().indices()) == 2 * len(
            _side(context, BoundarySide.LEFT)
        )

    @pytest.mark.parametrize(
        "specs",
        [
            (Dirichlet(BoundarySide.LEFT, 0.0, component=2),),
            (CauchyTraction(BoundarySide.LEFT, -1, 0.0),),
            (
                CauchyTraction(BoundarySide.LEFT, 1, 0.0),
                CauchyPressure(BoundarySide.LEFT, 1.0),
            ),
            (
                Dirichlet(BoundarySide.TOP, 0.0, component=0),
                Dirichlet(BoundarySide.TOP, 1.0, component=0),
            ),
        ],
        ids=[
            "component past the state",
            "negative component",
            "traction under a pressure",
            "same side and component",
        ],
    )
    def test_refuses(
        self, bkd: Backend[Array], specs: Sequence[BoundarySpecProtocol]
    ) -> None:
        with pytest.raises(ValueError):
            assemble_boundaries(_linear_context(bkd), specs)

    def test_refuses_a_pressure_on_a_scalar_state(self, bkd: Backend[Array]) -> None:
        with pytest.raises(ValueError, match="component"):
            assemble_boundaries(
                _context_2d(bkd), (CauchyPressure(BoundarySide.LEFT, 1.0),)
            )

    @pytest.mark.parametrize(
        "spec, linear",
        [
            (CauchyTraction(BoundarySide.LEFT, 0, 0.0), False),
            (CauchyPressure(BoundarySide.LEFT, 1.0), False),
            (PiolaTraction(BoundarySide.LEFT, 0, 0.0), True),
            (PiolaPressure(BoundarySide.LEFT, 1.0), True),
            (FluxNeumann(BoundarySide.LEFT, 0.0), True),
        ],
        ids=[
            "cauchy traction on hyperelasticity",
            "cauchy pressure on hyperelasticity",
            "piola traction on linear elasticity",
            "piola pressure on linear elasticity",
            "flux on linear elasticity",
        ],
    )
    def test_refuses_the_other_law(
        self, bkd: Backend[Array], spec: BoundarySpecProtocol, linear: bool
    ) -> None:
        context = _linear_context(bkd) if linear else _hyperelastic_context(bkd)
        with pytest.raises(TypeError):
            assemble_boundaries(context, (spec,))
