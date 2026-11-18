r"""Two-dimensional skfem subdomains coupled through the skeleton, against
the single-domain skfem solve. NumPy only: skfem is.

Matching nodal transfer needs each subdomain mesh's interface nodes on
the skeleton's face nodes, which are Gauss-Lobatto points. Every
subdomain is meshed by Q1 quadrilaterals on the tensor grid of
:data:`_NPOINTS` Lobatto points per direction of its partition cell, so
the single-domain mesh on the union of those grids has the same
elements: the coupled solve and the single-domain solve are the same
discrete equations, equal to solver tolerance.

- Two subdomains (cut at x = 1/2), :math:`-\Delta u = f`, u = 1 on the
  left edge, u = 2 on the right, natural on top and bottom.
- Four subdomains (2 x 2, one cross point),
  :math:`-\Delta u + u [+ u^3] = f`, natural all round.
- Three subdomains, not a box partition: one on the left half, two
  stacked on the right, so the left one's side is two skeleton faces
  and the vertex between them (a hanging vertex) is incident to all
  three subdomains, with one skeleton DOF. Same PDE as the four.

Interface ends lie on natural boundaries, where they are unknowns of
the skeleton like any interface node. Checks: the skeleton solution and
every subdomain's state equal the single-domain solve at the same
points (1e-10), and the gathered interface fluxes balance (1e-10).
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, Callable, Optional, Protocol, Sequence, Union

import numpy as np
from numpy.typing import NDArray
from skfem import MeshQuad

from pyapprox.implicitfunction.residual import ResidualProtocol
from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.implicitfunction.steady_system import SteadySystem
from pyapprox.pde.constitutive.coefficient_functions import (
    CallableReaction,
    ConstantDiffusion,
    TimeIndependent,
)
from pyapprox.pde.coupling.basis import TensorLagrangeBasis
from pyapprox.pde.coupling.closure_numbering import VertexWeightNumbering
from pyapprox.pde.coupling.geometry import (
    FaceGeometryProtocol,
    TensorBoxGeometry,
    skeleton_dof_points,
)
from pyapprox.pde.coupling.pinned_ports import PortPinnedData
from pyapprox.pde.coupling.pinned_response import (
    SteadyPinnedPoint,
    SteadyPinnedResponse,
)
from pyapprox.pde.coupling.pinning import PinnedSkeletonResidual
from pyapprox.pde.coupling.ports import (
    BackendPortRows,
    MatchingNodalTransfer,
    subdomain_port_dofs,
)
from pyapprox.pde.coupling.reference_faces import ReferenceBox
from pyapprox.pde.coupling.responses import TimeWindow
from pyapprox.pde.coupling.skeleton import (
    CoupledSubdomain,
    MatrixFreeSchur,
    SkeletonResidual,
)
from pyapprox.pde.coupling.spaces import ContinuousSkeletonSpace, UniformBases
from pyapprox.pde.coupling.topology import TensorBoxPartition
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.time_integration import skfem_discretization
from pyapprox.pde.models.galerkin.steady_residual import (
    galerkin_steady_residual,
    solve_galerkin_steady,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.operators import InnerProductSpaceProtocol, MatrixFree
from pyapprox.util.linalg.operators.krylov import GMRES
from pyapprox.util.linalg.scipy_stack import NumpyRowSet, ScipySparseLU
from pyapprox.util.rootfinding.nonlinear import Newton

_Np = NDArray[Any]
_NPOINTS = 5
_STEADY = TimeWindow.steady()
_Reaction = Callable[[_Np, _Np], _Np]
# Essential data on a side: a constant or g(points).
_Value = Union[float, Callable[[_Np], _Np]]
_Piece = SteadyPinnedResponse[_Np, _Np, Any]


class _TensorQuadMesh:
    """Q1 quadrilaterals on the tensor grid of ``xs`` by ``ys``, with
    boundaries named left, right, bottom and top; the nodes at the points
    ``free`` (shape (2, n)) belong to no named boundary."""

    def __init__(
        self, xs: _Np, ys: _Np, bkd: NumpyBkd, free: Optional[_Np] = None
    ) -> None:
        self._bkd = bkd
        self._free = np.zeros((2, 0)) if free is None else free
        self._lower = (float(xs[0]), float(ys[0]))
        self._upper = (float(xs[-1]), float(ys[-1]))
        self._mesh = MeshQuad.init_tensor(xs, ys)
        self._nodes: _Np = np.asarray(self._mesh.p, dtype=np.float64)

    def bkd(self) -> NumpyBkd:
        return self._bkd

    def ndim(self) -> int:
        return 2

    def nelements(self) -> int:
        return int(self._mesh.nelements)

    def nnodes(self) -> int:
        return int(self._mesh.nvertices)

    def nodes(self) -> _Np:
        return self._nodes

    def elements(self) -> _Np:
        return np.asarray(self._mesh.t, dtype=np.int64)

    def skfem_mesh(self) -> MeshQuad:
        return self._mesh

    def boundary_nodes(self, boundary_id: str) -> _Np:
        axis, value = {
            "left": (0, self._lower[0]),
            "right": (0, self._upper[0]),
            "bottom": (1, self._lower[1]),
            "top": (1, self._upper[1]),
        }[boundary_id]
        on_side = np.abs(self._nodes[axis] - value) < 1e-12
        gaps = np.linalg.norm(
            self._nodes[:, :, None] - self._free[:, None, :], axis=0
        )
        freed = np.any(gaps < 1e-12, axis=1)
        return np.where(on_side & ~freed)[0]


def _forcing(x: _Np) -> _Np:
    return np.asarray(1.0 + x[0] + 2.0 * x[1])


def _no_sink(x: _Np, u: _Np) -> _Np:
    return np.zeros_like(u)


def _no_sink_deriv(x: _Np, u: _Np) -> _Np:
    return np.zeros_like(u)


# The skfem physics adds R(u) as a source, so a sink is negative.
def _linear_sink(x: _Np, u: _Np) -> _Np:
    return -u


def _linear_sink_deriv(x: _Np, u: _Np) -> _Np:
    return -np.ones_like(u)


def _cubic_sink(x: _Np, u: _Np) -> _Np:
    sink: _Np = -(u + u**3)
    return sink


def _cubic_sink_deriv(x: _Np, u: _Np) -> _Np:
    return -(1.0 + 3.0 * u**2)


def _newton() -> Newton[_Np, _Np]:
    return Newton(atol=1e-13, rtol=0.0, maxiters=30)


class _Partition(Protocol):
    """What the harness asks of a partition of the square."""

    def ndim(self) -> int: ...

    def nsubdomains(self) -> int: ...

    def nentities(self) -> int: ...

    def entity_dim(self, entity: int) -> int: ...

    def entity_measure(self, entity: int) -> float: ...

    def on_boundary(self, entity: int) -> bool: ...

    def face_reference(self, face: int) -> ReferenceBox: ...

    def face_closure(self, face: int) -> tuple[int, ...]: ...

    def subdomain_entities(self, subdomain: int) -> tuple[int, ...]: ...


# A subdomain's mesh coordinates along x and y, and its essential sides.
_Grid = tuple[_Np, _Np, Sequence[tuple[str, "_Value"]]]
_REFERENCE = np.asarray(TensorLagrangeBasis(1, _NPOINTS, NumpyBkd()).nodes()[0])


def _lobatto(lo: float, up: float) -> _Np:
    """The skeleton's face nodes on [lo, up]."""
    points: _Np = lo + 0.5 * (_REFERENCE + 1.0) * (up - lo)
    return points


def _union(*pieces: _Np) -> _Np:
    """Consecutive pieces' points, each shared end once."""
    return np.concatenate([pieces[0]] + [p[1:] for p in pieces[1:]])


class _Problem:
    """A partition of the unit square, its skeleton, the subdomain and
    single-domain meshes, and the PDE data."""

    def __init__(
        self,
        topo: _Partition,
        geometry: FaceGeometryProtocol[_Np],
        grids: Sequence[_Grid],
        whole: _Grid,
        sink: tuple[_Reaction, _Reaction],
        boundary: Optional[Callable[[_Np], _Np]] = None,
    ) -> None:
        self.bkd = NumpyBkd()
        self._boundary = boundary
        self.topo, self._geometry = topo, geometry
        self._grids, self._whole = grids, whole
        self._bases = UniformBases(TensorLagrangeBasis(1, _NPOINTS, self.bkd))
        self.skeleton = ContinuousSkeletonSpace(
            topo, self._bases, VertexWeightNumbering(topo, self._bases)
        )
        self.sink = sink

    def skeleton_coords(self) -> _Np:
        """The point of each skeleton DOF. Shape: (2, ndofs)."""
        return skeleton_dof_points(self.skeleton, self._geometry, self._bases)

    def pinned(self) -> list[int]:
        """The skeleton DOFs pinned to the boundary data: every DOF on the
        physical boundary when there is boundary data, else none."""
        if self._boundary is None:
            return []
        return [
            dof
            for e in range(self.topo.nentities())
            if self.topo.on_boundary(e)
            for dof in self.skeleton.entity_dofs(e)
        ]

    def system(
        self,
        xs: _Np,
        ys: _Np,
        sides: Sequence[tuple[str, _Value]],
        free: Optional[_Np] = None,
    ) -> Any:
        """The system on the grid ``xs`` by ``ys``; essential rows on
        ``sides`` except at the points ``free`` (shape (2, n))."""
        mesh = _TensorQuadMesh(xs, ys, self.bkd, free)
        basis = LagrangeBasis(mesh, degree=1)
        physics = AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=ConstantDiffusion(1.0),
            bkd=self.bkd,
            reaction=CallableReaction(*self.sink),
            forcing=TimeIndependent(_forcing),
        )
        conditions = [
            DirichletBC(
                basis,
                side,
                v if isinstance(v, float) else TimeIndependent(v),
                self.bkd,
            )
            for side, v in sides
        ]
        return compose_galerkin_system(physics, conditions), mesh

    def piece(self, subdomain: int) -> tuple[_Piece, _Np, tuple[int, ...]]:
        """Subdomain ``subdomain``'s response, its node points and its
        skeleton DOFs."""
        dofs = subdomain_port_dofs(self.topo, self.skeleton, subdomain)
        skel = self.skeleton_coords()
        # Pinned port nodes take g through the skeleton, not as essentials.
        system, mesh = self.system(*self._grids[subdomain], skel[:, list(dofs)])
        disc = skfem_discretization(system)
        ports = [_node_at(mesh.nodes(), skel[:, d]) for d in dofs]
        states, theta = disc.space(), disc.parameter_space(len(ports))
        # The stack's space has an inner product; its declared type does not.
        assert isinstance(states, InnerProductSpaceProtocol)
        data: PortPinnedData[_Np, _Np] = PortPinnedData(
            NumpyRowSet(states.descriptor(), ports),
            BackendPortRows(states.descriptor(), ports, self.bkd),
            BackendPortRows(theta.descriptor(), list(range(len(ports))), self.bkd),
            states,
            theta,
        )
        response: _Piece = SteadyPinnedResponse(
            galerkin_steady_residual(disc),
            data,
            states,
            RecomputeAlways(ScipySparseLU()),
            _newton(),
            self.bkd,
        )
        return response, mesh.nodes(), dofs

    def single_domain(self) -> tuple[_Np, _Np]:
        """The single-domain solution and its node points."""
        system, mesh = self.system(*self._whole)
        _, point = solve_galerkin_steady(
            skfem_discretization(system), RecomputeAlways(ScipySparseLU()), _newton()
        )
        assert point.converged()
        return point.state(), mesh.nodes()

    def solve_skeleton(self, pieces: Sequence[_Piece]) -> _Np:
        bkd, space = self.bkd, self.skeleton.vector_space()
        subdomains = [
            CoupledSubdomain(
                piece,
                MatchingNodalTransfer(
                    BackendPortRows(
                        self.skeleton.descriptor(),
                        subdomain_port_dofs(self.topo, self.skeleton, s),
                        bkd,
                    )
                ),
            )
            for s, piece in enumerate(pieces)
        ]
        schur: MatrixFreeSchur[_Np, _Np] = MatrixFreeSchur()
        residual: ResidualProtocol[_Np, MatrixFree] = SkeletonResidual(
            space, subdomains, schur
        )
        pinned = self.pinned()
        if self._boundary is not None:
            g = self._boundary(self.skeleton_coords()[:, pinned])
            residual = PinnedSkeletonResidual(
                residual,
                BackendPortRows(self.skeleton.descriptor(), pinned, bkd),
                bkd.reshape(g, (len(pinned), 1)),
            )
        setups: RecomputeAlways[_Np, MatrixFree] = RecomputeAlways(
            GMRES(space, bkd, tol=1e-13)
        )
        newton: Newton[_Np, _Np] = Newton(atol=1e-11, rtol=1e-11, maxiters=20)
        point = SteadySystem(residual, space, setups).solve(newton, space.zeros(1))
        assert point.converged()
        return point.solution()


def _node_at(nodes: _Np, point: _Np) -> int:
    distances = np.linalg.norm(nodes - point[:, None], axis=0)
    node = int(np.argmin(distances))
    if distances[node] > 1e-12:
        raise ValueError(f"no mesh node at {point}")
    return node


def _check(problem: _Problem) -> None:
    bkd = problem.bkd
    reference, ref_nodes = problem.single_domain()
    built = [problem.piece(s) for s in range(problem.topo.nsubdomains())]
    lam = problem.solve_skeleton([piece for piece, _, _ in built])
    skel = problem.skeleton_coords()
    at_skeleton = [_node_at(ref_nodes, skel[:, d]) for d in range(skel.shape[1])]
    bkd.assert_allclose(lam, reference[at_skeleton], atol=1e-10)
    balance = np.zeros_like(lam)
    for piece, nodes, dofs in built:
        point = piece.linearize(lam[list(dofs)], bkd.zeros((0, 1)), _STEADY)
        assert isinstance(point, SteadyPinnedPoint)
        np.add.at(balance, list(dofs), point.output())
        rows = [_node_at(ref_nodes, nodes[:, k]) for k in range(nodes.shape[1])]
        bkd.assert_allclose(point.state(), reference[rows], atol=1e-10)
    # Pinned DOFs carry the boundary's reaction, not a balance.
    balance[problem.pinned()] = 0.0
    bkd.assert_allclose(balance, bkd.zeros(balance.shape), atol=1e-10)


def _box(
    ncells: tuple[int, int],
    sink: tuple[_Reaction, _Reaction],
    essentials: Sequence[tuple[str, _Value]],
    boundary: Optional[Callable[[_Np], _Np]] = None,
) -> _Problem:
    """A box partition of the unit square, each cell meshed on its
    Lobatto grid; ``boundary`` pins the skeleton's boundary DOFs."""
    topo = TensorBoxPartition(ncells)
    ex, ey = (np.linspace(0.0, 1.0, n + 1) for n in ncells)
    grids: list[_Grid] = []
    for s in range(topo.nsubdomains()):
        ix, iy = divmod(s, ncells[1])
        # Only the outer edges a subdomain touches carry its essentials.
        touched = {
            "left": ix == 0,
            "right": ix == ncells[0] - 1,
            "bottom": iy == 0,
            "top": iy == ncells[1] - 1,
        }
        sides = [(side, v) for side, v in essentials if touched[side]]
        grids.append(
            (_lobatto(ex[ix], ex[ix + 1]), _lobatto(ey[iy], ey[iy + 1]), sides)
        )
    whole = (
        _union(*(_lobatto(ex[i], ex[i + 1]) for i in range(ncells[0]))),
        _union(*(_lobatto(ey[i], ey[i + 1]) for i in range(ncells[1]))),
        essentials,
    )
    geometry = TensorBoxGeometry(topo, NumpyBkd())
    return _Problem(topo, geometry, grids, whole, sink, boundary)


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

    def face_points(self, face: int, points: _Np) -> _Np:
        lo, up = (np.asarray(_HANGING_POINTS[v]) for v in _HANGING_FACES[face])
        t = 0.5 * (points[0] + 1.0)
        mapped: _Np = lo[:, None] + (up - lo)[:, None] * t[None, :]
        return mapped


def test_two_subdomains_equal_the_single_domain() -> None:
    _check(_box((2, 1), (_no_sink, _no_sink_deriv), [("left", 1.0), ("right", 2.0)]))


@pytest.mark.parametrize("cubic", [False, True])
def test_four_subdomains_equal_the_single_domain(cubic: bool) -> None:
    sink = (_cubic_sink, _cubic_sink_deriv) if cubic else (
        _linear_sink, _linear_sink_deriv
    )
    problem = _box((2, 2), sink, [])
    assert len(problem.skeleton_coords()[0]) == 4 * (_NPOINTS - 2) + 5
    _check(problem)


def _boundary_data(x: _Np) -> _Np:
    data: _Np = 1.0 + x[0] * x[1] + 0.5 * x[1] ** 2
    return data


@pytest.mark.parametrize("cubic", [False, True])
def test_dirichlet_closure_dofs_are_pinned(cubic: bool) -> None:
    sink = (_cubic_sink, _cubic_sink_deriv) if cubic else (
        _linear_sink, _linear_sink_deriv
    )
    sides = [(s, _boundary_data) for s in ("left", "right", "bottom", "top")]
    problem = _box((2, 2), sink, sides, boundary=_boundary_data)
    # The four interface ends, one on each outer edge.
    assert len(problem.pinned()) == 4
    _check(problem)


def test_pinning_rejects_bad_input() -> None:
    problem = _box((2, 2), (_linear_sink, _linear_sink_deriv), [])
    space = problem.skeleton.vector_space()
    rows = BackendPortRows(problem.skeleton.descriptor(), [0, 1], problem.bkd)
    residual: SkeletonResidual[_Np, _Np, _Np, MatrixFree] = SkeletonResidual(
        space, [], MatrixFreeSchur()
    )
    g, wrong = problem.bkd.zeros((2, 1)), problem.bkd.zeros((3, 1))
    with pytest.raises(ValueError, match="shape"):
        PinnedSkeletonResidual(residual, rows, wrong)
    with pytest.raises(TypeError, match="ResidualProtocol"):
        PinnedSkeletonResidual(object(), rows, g)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="PortRowsProtocol"):
        PinnedSkeletonResidual(residual, object(), g)  # type: ignore[arg-type]


@pytest.mark.parametrize("cubic", [False, True])
def test_hanging_vertex_partition_equals_the_single_domain(cubic: bool) -> None:
    sink = (_cubic_sink, _cubic_sink_deriv) if cubic else (
        _linear_sink, _linear_sink_deriv
    )
    low, high = _lobatto(0.0, 0.5), _lobatto(0.5, 1.0)
    both = _union(low, high)
    grids: list[_Grid] = [(low, both, []), (high, low, []), (high, high, [])]
    topo = _HangingPartition()
    geometry = _HangingGeometry(NumpyBkd())
    problem = _Problem(topo, geometry, grids, (both, both, []), sink)
    # 3 faces with interior nodes, 4 vertices; one DOF at the hanging
    # vertex, read by all three subdomains.
    assert problem.skeleton.descriptor().size == 3 * (_NPOINTS - 2) + 4
    hanging = problem.skeleton.entity_dofs(4)
    assert len(hanging) == 1
    assert all(
        hanging[0] in subdomain_port_dofs(topo, problem.skeleton, s) for s in range(3)
    )
    _check(problem)
