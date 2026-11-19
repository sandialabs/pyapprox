r"""Two-dimensional skfem subdomains coupled through the skeleton, against
the single-domain skfem solve. NumPy only: skfem is.

The meshes match the skeleton's Lobatto nodes, so the coupled and
single-domain solves are the same discrete equations
(:mod:`tests._helpers.coupling_skfem`).

- Two subdomains (cut at x = 1/2), :math:`-\Delta u = f`, u = 1 on the
  left edge, u = 2 on the right, natural on top and bottom.
- Four subdomains (2 x 2, one cross point),
  :math:`-\Delta u + u [+ u^3] = f`, natural all round.
- Three subdomains, not a box partition: one on the left half, two
  stacked on the right, so the left one's side is two skeleton faces
  and the vertex between them (a hanging vertex) is incident to all
  three subdomains, with one skeleton DOF. Same PDE as the four.

Interface ends lie on natural boundaries, where they are unknowns of
the skeleton like any interface node, except in the Dirichlet case,
where the skeleton pins them. Checks: the skeleton solution and every
subdomain's state equal the single-domain solve at the same points
(1e-10), and the gathered interface fluxes balance (1e-10).
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, Callable, Sequence

import numpy as np

from pyapprox.pde.boundary import ExceptDofs
from pyapprox.pde.constitutive.coefficient_functions import (
    CallableReaction,
    ConstantDiffusion,
    TimeIndependent,
)
from pyapprox.pde.coupling.ports import subdomain_port_dofs
from pyapprox.pde.coupling.reference_faces import ReferenceBox
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.util.backends.numpy import NumpyBkd
from tests._helpers.coupling_skfem import (
    NPOINTS,
    CoupledProblem,
    Grid,
    Np,
    SkeletonPins,
    SystemBuilder,
    TensorQuadMesh,
    Value,
    box_problem,
    check_coupled,
    lobatto,
    union,
)

_Reaction = Callable[[Np, Np], Np]


def _forcing(x: Np) -> Np:
    return np.asarray(1.0 + x[0] + 2.0 * x[1])


def _no_sink(x: Np, u: Np) -> Np:
    return np.zeros_like(u)


def _no_sink_deriv(x: Np, u: Np) -> Np:
    return np.zeros_like(u)


# The skfem physics adds R(u) as a source, so a sink is negative.
def _linear_sink(x: Np, u: Np) -> Np:
    return -u


def _linear_sink_deriv(x: Np, u: Np) -> Np:
    return -np.ones_like(u)


def _cubic_sink(x: Np, u: Np) -> Np:
    sink: Np = -(u + u**3)
    return sink


def _cubic_sink_deriv(x: Np, u: Np) -> Np:
    return -(1.0 + 3.0 * u**2)


def _sink(cubic: bool) -> tuple[_Reaction, _Reaction]:
    if cubic:
        return _cubic_sink, _cubic_sink_deriv
    return _linear_sink, _linear_sink_deriv


def _poisson(sink: tuple[_Reaction, _Reaction]) -> SystemBuilder:
    """Diffusion-reaction systems with the sink ``sink``."""

    def build(
        mesh: TensorQuadMesh, sides: Sequence[tuple[str, Value]], exempt: set[int]
    ) -> Any:
        bkd = mesh.bkd()
        basis = LagrangeBasis(mesh, degree=1)
        physics = AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=ConstantDiffusion(1.0),
            bkd=bkd,
            reaction=CallableReaction(*sink),
            forcing=TimeIndependent(_forcing),
        )
        conditions = []
        for side, v in sides:
            bc = DirichletBC(
                basis, side, v if isinstance(v, float) else TimeIndependent(v), bkd
            )
            on_side = exempt & {int(d) for d in bc.constrained_dofs()}
            conditions.append(ExceptDofs(bc, sorted(on_side)))
        return compose_galerkin_system(physics, conditions)

    return build


# The hanging-vertex partition: subdomain 0 is [0, 1/2] x [0, 1], 1 is
# [1/2, 1] x [0, 1/2], 2 is [1/2, 1] x [1/2, 1]. Faces 0-2, each listed by
# its ends (the reference -1 end first; face 1 runs downward); vertices
# 3-6, of which 4 = (1/2, 1/2) is incident to all three subdomains.
_HANGING_POINTS = {3: (0.5, 0.0), 4: (0.5, 0.5), 5: (0.5, 1.0), 6: (1.0, 0.5)}
_HANGING_FACES = {0: (3, 4), 1: (5, 4), 2: (4, 6)}
_HANGING_SUBDOMAINS = ((0, 1, 3, 4, 5), (0, 2, 3, 4, 6), (1, 2, 4, 5, 6))
_SEGMENT = ReferenceBox(1)


class _HangingPartition:
    """Three subdomains of the unit square, not a box partition: the left
    one's side is two skeleton faces meeting at a hanging vertex."""

    def ndim(self) -> int:
        return 2

    def nsubdomains(self) -> int:
        return 3

    def nentities(self) -> int:
        return 7

    def entity_dim(self, entity: int) -> int:
        return 1 if entity in _HANGING_FACES else 0

    def entity_measure(self, entity: int) -> float:
        return 0.5 if entity in _HANGING_FACES else 1.0

    def on_boundary(self, entity: int) -> bool:
        return entity in (3, 5, 6)

    def face_reference(self, face: int) -> ReferenceBox:
        return _SEGMENT

    def face_closure(self, face: int) -> tuple[int, ...]:
        # The segment's sub-entities: its -1 end, its +1 end, itself.
        return (*_HANGING_FACES[face], face)

    def subdomain_entities(self, subdomain: int) -> tuple[int, ...]:
        return _HANGING_SUBDOMAINS[subdomain]


class _HangingGeometry:
    """Each face as the segment between its ends."""

    def __init__(self, bkd: NumpyBkd) -> None:
        self._bkd = bkd

    def bkd(self) -> NumpyBkd:
        return self._bkd

    def ndim(self) -> int:
        return 2

    def face_points(self, face: int, points: Np) -> Np:
        lo, up = (np.asarray(_HANGING_POINTS[v]) for v in _HANGING_FACES[face])
        t = 0.5 * (points[0] + 1.0)
        mapped: Np = lo[:, None] + (up - lo)[:, None] * t[None, :]
        return mapped


def test_two_subdomains_equal_the_single_domain() -> None:
    build = _poisson((_no_sink, _no_sink_deriv))
    check_coupled(box_problem((2, 1), build, [("left", 1.0), ("right", 2.0)]))


@pytest.mark.parametrize("cubic", [False, True])
def test_four_subdomains_equal_the_single_domain(cubic: bool) -> None:
    problem = box_problem((2, 2), _poisson(_sink(cubic)), [])
    assert len(problem.skeleton_coords()[0]) == 4 * (NPOINTS - 2) + 5
    check_coupled(problem)


def _boundary_data(x: Np) -> Np:
    data: Np = 1.0 + x[0] * x[1] + 0.5 * x[1] ** 2
    return data


def _everywhere(x: Np) -> Np:
    return np.ones(x.shape[1], dtype=bool)


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("cubic", [False, True])
def test_dirichlet_closure_dofs_are_pinned(cubic: bool, dense: bool) -> None:
    sides = [(s, _boundary_data) for s in ("left", "right", "bottom", "top")]
    pins = SkeletonPins(_everywhere, lambda x: _boundary_data(x)[None, :])
    problem = box_problem((2, 2), _poisson(_sink(cubic)), sides, pins=pins)
    # The four interface ends, one on each outer edge.
    assert len(problem.pinned()) == 4
    check_coupled(problem, dense)


@pytest.mark.parametrize("cubic", [False, True])
def test_hanging_vertex_partition_equals_the_single_domain(cubic: bool) -> None:
    low, high = lobatto(0.0, 0.5), lobatto(0.5, 1.0)
    both = union(low, high)
    grids: list[Grid] = [(low, both, []), (high, low, []), (high, high, [])]
    topo = _HangingPartition()
    geometry = _HangingGeometry(NumpyBkd())
    problem = CoupledProblem(
        topo, geometry, grids, (both, both, []), _poisson(_sink(cubic))
    )
    # 3 faces with interior nodes, 4 vertices; one DOF at the hanging
    # vertex, read by all three subdomains.
    assert problem.space.descriptor().size == 3 * (NPOINTS - 2) + 4
    hanging = problem.space.entity_dofs(4)
    assert len(hanging) == 1
    assert all(
        hanging[0] in subdomain_port_dofs(topo, problem.space, s) for s in range(3)
    )
    check_coupled(problem)
