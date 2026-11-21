r"""Harness for skfem subdomains of the unit square coupled through a
skeleton, compared with the single-domain skfem solve.

Matching nodal transfer needs each subdomain mesh's interface nodes on
the skeleton's face nodes, which are Gauss-Lobatto points. Every
subdomain is meshed by Q1 quadrilaterals on the tensor grid of
:data:`NPOINTS` Lobatto points per direction of its partition cell, so
the single-domain mesh on the union of those grids has the same
elements: the coupled solve and the single-domain solve are the same
discrete equations, equal to solver tolerance.

The field has ``k`` components per mesh node, interleaved as skfem's
vector basis numbers them (node ``n``, component ``c`` is DOF
``k * n + c``); the skeleton is :class:`ComponentSkeletonSpace` over the
continuous scalar space, interleaved the same way (``k = 1`` is the
scalar space). The physics is injected as a :data:`SystemBuilder`.

Importers must skip when skfem is not installed before importing this
module.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, Sequence, TypeVar, Union

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
from pyapprox.pde.coupling.basis import lobatto_lagrange_basis
from pyapprox.pde.coupling.closure_numbering import VertexWeightNumbering
from pyapprox.pde.coupling.geometry import (
    FaceGeometryProtocol,
    TensorBoxGeometry,
    skeleton_dof_points,
)
from pyapprox.pde.coupling.pinned_response import (
    SteadyPinnedPoint,
    SteadyPinnedResponse,
)
from pyapprox.pde.coupling.pinning import PinnedSkeletonResidual
from pyapprox.pde.coupling.ports import (
    BackendPortRows,
    MatchingNodalTransfer,
    PortTransferProtocol,
    subdomain_port_dofs,
)
from pyapprox.pde.coupling.reference_faces import ReferenceBox
from pyapprox.pde.coupling.responses import TimeWindow
from pyapprox.pde.coupling.skeleton import (
    CoupledSubdomain,
    DenseSchur,
    MatrixFreeSchur,
    SkeletonResidual,
)
from pyapprox.pde.coupling.spaces import (
    ComponentSkeletonSpace,
    ContinuousSkeletonSpace,
    UniformBases,
)
from pyapprox.pde.coupling.stacks.galerkin import skfem_pinned_response
from pyapprox.pde.coupling.topology import TensorBoxPartition
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC, NeumannBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.time_integration import skfem_discretization
from pyapprox.pde.models.galerkin.steady_residual import (
    solve_galerkin_steady,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.backend_dense import (
    BackendDenseLU,
    BackendRowSet,
    BackendVectorSpace,
    DenseFormat,
)
from pyapprox.util.linalg.operators import (
    FormatProtocol,
    MatrixFree,
    SpaceDescriptor,
)
from pyapprox.util.linalg.operators.krylov import GMRES
from pyapprox.util.linalg.scipy_stack import ScipySparseLU
from pyapprox.util.rootfinding.nonlinear import Newton

Np = NDArray[Any]
NPOINTS = 5
STEADY = TimeWindow.steady()
# Essential data on a side: a constant or g(points).
Value = Union[float, Callable[[Np], Np]]
Piece = SteadyPinnedResponse[Np, Np, Any]
_F = TypeVar("_F", bound=FormatProtocol)


class TensorQuadMesh:
    """Q1 quadrilaterals on the tensor grid of ``xs`` by ``ys``, with
    boundaries named left, right, bottom and top."""

    def __init__(self, xs: Np, ys: Np, bkd: NumpyBkd) -> None:
        self._bkd = bkd
        self._lower = (float(xs[0]), float(ys[0]))
        self._upper = (float(xs[-1]), float(ys[-1]))
        (x0, y0), (x1, y1) = self._lower, self._upper
        # Vector bases look boundaries up by name on the skfem mesh.
        self._mesh = MeshQuad.init_tensor(xs, ys).with_boundaries(
            {
                "left": lambda x: np.abs(x[0] - x0) < 1e-12,
                "right": lambda x: np.abs(x[0] - x1) < 1e-12,
                "bottom": lambda x: np.abs(x[1] - y0) < 1e-12,
                "top": lambda x: np.abs(x[1] - y1) < 1e-12,
            }
        )
        self._nodes: Np = np.asarray(self._mesh.p, dtype=np.float64)

    def bkd(self) -> NumpyBkd:
        return self._bkd

    def ndim(self) -> int:
        return 2

    def nelements(self) -> int:
        return int(self._mesh.nelements)

    def nnodes(self) -> int:
        return int(self._mesh.nvertices)

    def nodes(self) -> Np:
        return self._nodes

    def elements(self) -> Np:
        return np.asarray(self._mesh.t, dtype=np.int64)

    def skfem_mesh(self) -> MeshQuad:
        return self._mesh

    def boundary_nodes(self, boundary_id: str) -> Np:
        axis, value = {
            "left": (0, self._lower[0]),
            "right": (0, self._upper[0]),
            "bottom": (1, self._lower[1]),
            "top": (1, self._upper[1]),
        }[boundary_id]
        return np.where(np.abs(self._nodes[axis] - value) < 1e-12)[0]


# Builds the system on a mesh: its essential conditions on the named
# sides, each dropping the listed DOFs (skeleton-pinned port nodes).
SystemBuilder = Callable[[TensorQuadMesh, Sequence[tuple[str, Value]], set[int]], Any]


class Partition(Protocol):
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


@dataclass(frozen=True)
class SkeletonPins:
    """Skeleton nodes on the physical boundary held at given values:
    ``where`` selects them by point (a bool mask over shape (2, n)),
    ``values`` gives their components, shape (k, n)."""

    where: Callable[[Np], Np]
    values: Callable[[Np], Np]


# A subdomain's mesh coordinates along x and y, and its essential sides.
Grid = tuple[Np, Np, Sequence[tuple[str, Value]]]
_REFERENCE = np.asarray(lobatto_lagrange_basis(1, NPOINTS, NumpyBkd()).nodes()[0])


def lobatto(lo: float, up: float) -> Np:
    """The skeleton's face nodes on [lo, up]."""
    points: Np = lo + 0.5 * (_REFERENCE + 1.0) * (up - lo)
    return points


def union(*pieces: Np) -> Np:
    """Consecutive pieces' points, each shared end once."""
    return np.concatenate([pieces[0]] + [p[1:] for p in pieces[1:]])


def node_at(nodes: Np, point: Np) -> int:
    """The mesh node at ``point``."""
    distances = np.linalg.norm(nodes - point[:, None], axis=0)
    node = int(np.argmin(distances))
    if distances[node] > 1e-12:
        raise ValueError(f"no mesh node at {point}")
    return node


def newton() -> Newton[Np, Np]:
    return Newton(atol=1e-13, rtol=0.0, maxiters=30)


def pinned_piece(system: Any, ports: Sequence[int]) -> Piece:
    """The steady response of a composed skfem ``system`` whose field
    DOFs ``ports`` are pinned to the port data, in that order, solved by
    Newton with a sparse LU."""
    return skfem_pinned_response(
        system, ports, RecomputeAlways(ScipySparseLU()), newton()
    )


def _zero(x: Np) -> Np:
    return np.zeros(x.shape[1])


def _no_sink(x: Np, u: Np) -> Np:
    return np.zeros_like(u)


def poisson_system(
    mesh: TensorQuadMesh,
    dirichlet: Sequence[tuple[str, Callable[[Np], Np]]],
    neumann: Sequence[tuple[str, float]] = (),
) -> Any:
    r""":math:`-\Delta u = 0` on ``mesh`` with Q1 elements: u = g on each
    Dirichlet side, the outward normal derivative on each Neumann side."""
    bkd = mesh.bkd()
    basis = LagrangeBasis(mesh, degree=1)
    physics = AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=ConstantDiffusion(1.0),
        bkd=bkd,
        reaction=CallableReaction(_no_sink, _no_sink),
        forcing=TimeIndependent(_zero),
    )
    conditions: list[Any] = [
        DirichletBC(basis, side, TimeIndependent(g), bkd) for side, g in dirichlet
    ]
    conditions += [NeumannBC(basis, side, value, bkd) for side, value in neumann]
    return compose_galerkin_system(physics, conditions)


def solve_dense_skeleton(
    descriptor: SpaceDescriptor,
    subdomains: Sequence[CoupledSubdomain[Np, Np]],
    pinned: Sequence[int] = (),
    values: Optional[Np] = None,
) -> Np:
    """The skeleton solution by the assembled Schur complement and dense
    LU, with the DOFs ``pinned`` held at ``values`` (a full skeleton
    column) when there are any."""
    bkd = NumpyBkd()
    vectors = BackendVectorSpace(descriptor, bkd)
    skeleton: SkeletonResidual[Np, Np, Np, DenseFormat[Np]] = SkeletonResidual(
        vectors, subdomains, DenseSchur(bkd)
    )
    residual: ResidualProtocol[Np, DenseFormat[Np]] = skeleton
    if pinned:
        if values is None:
            raise ValueError("pinned DOFs need values")
        residual = PinnedSkeletonResidual(
            skeleton, vectors, BackendRowSet(descriptor, list(pinned), bkd), values
        )
    lu: RecomputeAlways[Np, DenseFormat[Np]] = RecomputeAlways(BackendDenseLU(bkd))
    outer: Newton[Np, Np] = Newton(atol=1e-11, rtol=1e-11, maxiters=20)
    solved = SteadySystem(residual, vectors, lu).solve(outer, vectors.zeros(1))
    assert solved.converged()
    return solved.solution()


def max_state_error(
    pieces: Sequence[tuple[Piece, PortTransferProtocol[Np, Np], Np]],
    lam: Np,
    exact: Callable[[Np], Np],
    where: Optional[Callable[[Np], Np]] = None,
) -> float:
    """The largest nodal error of the pieces' states, given each piece's
    transfer and node points, against ``exact``; only at the nodes
    ``where`` selects (a bool mask over the points) when given."""
    bkd, error = NumpyBkd(), 0.0
    for piece, transfer, nodes in pieces:
        point = piece.linearize(transfer.scatter(lam), bkd.zeros((0, 1)), STEADY)
        assert isinstance(point, SteadyPinnedPoint)
        diff = np.abs(point.state()[:, 0] - exact(nodes))
        if where is not None:
            diff = diff[where(nodes)]
        error = max(error, float(diff.max()))
    return error


class CoupledProblem:
    """A partition of the unit square, its skeleton, the subdomain and
    single-domain meshes, and the physics.

    Parameters
    ----------
    topo, geometry
        The partition and where its faces are.
    grids, whole
        Each subdomain's grid and the single domain's.
    build
        Builds a system on a mesh.
    ncomponents
        Field components per node.
    pins
        Skeleton boundary nodes held at given values, or none.
    """

    def __init__(
        self,
        topo: Partition,
        geometry: FaceGeometryProtocol[Np],
        grids: Sequence[Grid],
        whole: Grid,
        build: SystemBuilder,
        ncomponents: int = 1,
        pins: Optional[SkeletonPins] = None,
    ) -> None:
        self.bkd = NumpyBkd()
        self.topo, self._geometry = topo, geometry
        self._grids, self._whole = grids, whole
        self._build, self._pins = build, pins
        self.k = ncomponents
        self._bases = UniformBases(lobatto_lagrange_basis(1, NPOINTS, self.bkd))
        self.scalar = ContinuousSkeletonSpace(
            topo, self._bases, VertexWeightNumbering(topo, self._bases)
        )
        self.space = ComponentSkeletonSpace(self.scalar, ncomponents)

    def skeleton_coords(self) -> Np:
        """The point of each scalar skeleton DOF. Shape: (2, nscalar)."""
        return skeleton_dof_points(self.scalar, self._geometry, self._bases)

    def _pinned_nodes(self) -> list[int]:
        if self._pins is None:
            return []
        coords = self.skeleton_coords()
        boundary = [
            d
            for e in range(self.topo.nentities())
            if self.topo.on_boundary(e)
            for d in self.scalar.entity_dofs(e)
        ]
        mask = self._pins.where(coords[:, boundary])
        return [d for d, keep in zip(boundary, mask) if keep]

    def pinned(self) -> list[int]:
        """The skeleton DOFs held by the pins: every component of each
        selected boundary node."""
        return [dof for d in self._pinned_nodes() for dof in self.space.node_dofs(d)]

    def mesh_dofs(self, nodes: Sequence[int]) -> list[int]:
        """The field DOFs of mesh nodes ``nodes``, interleaved."""
        return [self.k * n + c for n in nodes for c in range(self.k)]

    def system(self, grid: Grid, free: Optional[Np] = None) -> Any:
        """The system on ``grid``; no essential rows at the nodes at the
        points ``free`` (shape (2, n))."""
        xs, ys, sides = grid
        mesh = TensorQuadMesh(xs, ys, self.bkd)
        exempt = (
            set()
            if free is None
            else set(
                self.mesh_dofs(
                    [node_at(mesh.nodes(), free[:, j]) for j in range(free.shape[1])]
                )
            )
        )
        return self._build(mesh, sides, exempt), mesh

    def port_system(self, subdomain: int) -> tuple[Any, Np, list[int], tuple[int, ...]]:
        """Subdomain ``subdomain``'s system without essential rows at its
        port nodes, its node points, its port field DOFs and its skeleton
        DOFs, both in skeleton port order."""
        nodes = subdomain_port_dofs(self.topo, self.scalar, subdomain)
        dofs = subdomain_port_dofs(self.topo, self.space, subdomain)
        skel = self.skeleton_coords()
        # Port nodes take their data through the skeleton, not as essentials.
        system, mesh = self.system(self._grids[subdomain], skel[:, list(nodes)])
        # Skeleton port DOFs are node-major, as are the mesh DOFs.
        ports = self.mesh_dofs([node_at(mesh.nodes(), skel[:, d]) for d in nodes])
        if len(ports) != len(dofs):
            raise ValueError("port DOFs and mesh DOFs differ in number")
        return system, mesh.nodes(), ports, dofs

    def piece(self, subdomain: int) -> tuple[Piece, Np, tuple[int, ...]]:
        """Subdomain ``subdomain``'s response, its node points and its
        skeleton DOFs."""
        system, nodes, ports, dofs = self.port_system(subdomain)
        return pinned_piece(system, ports), nodes, dofs

    def single_domain(self) -> tuple[Np, Np]:
        """The single-domain solution and its node points."""
        system, mesh = self.system(self._whole)
        _, point = solve_galerkin_steady(
            skfem_discretization(system), RecomputeAlways(ScipySparseLU()), newton()
        )
        assert point.converged()
        return point.state(), mesh.nodes()

    def _pin(self, residual: ResidualProtocol[Np, _F]) -> ResidualProtocol[Np, _F]:
        """``residual`` with the pinned DOFs held, if there are pins."""
        if self._pins is None:
            return residual
        bkd, nodes = self.bkd, self._pinned_nodes()
        values = bkd.zeros((self.space.descriptor().size, 1))
        given = self._pins.values(self.skeleton_coords()[:, nodes])
        for j, d in enumerate(nodes):
            for c, dof in enumerate(self.space.node_dofs(d)):
                values[dof, 0] = given[c, j]
        return PinnedSkeletonResidual(
            residual,
            BackendVectorSpace(self.space.descriptor(), bkd),
            BackendRowSet(self.space.descriptor(), self.pinned(), bkd),
            values,
        )

    def subdomains(self, pieces: Sequence[Piece]) -> list[CoupledSubdomain[Np, Np]]:
        """Each piece with its matching nodal transfer."""
        return [
            CoupledSubdomain(
                piece,
                MatchingNodalTransfer(
                    BackendPortRows(
                        self.space.descriptor(),
                        subdomain_port_dofs(self.topo, self.space, s),
                        self.bkd,
                    )
                ),
            )
            for s, piece in enumerate(pieces)
        ]

    def residual(
        self, pieces: Sequence[Piece]
    ) -> SkeletonResidual[Np, Np, Np, MatrixFree]:
        """The unpinned skeleton residual with a matrix-free Schur."""
        space = BackendVectorSpace(self.space.descriptor(), self.bkd)
        schur: MatrixFreeSchur[Np, Np] = MatrixFreeSchur()
        return SkeletonResidual(space, self.subdomains(pieces), schur)

    def solve_skeleton(self, pieces: Sequence[Piece], dense: bool = False) -> Np:
        """The skeleton solution: matrix-free Schur and GMRES, or with
        ``dense`` the assembled Schur complement and dense LU."""
        bkd = self.bkd
        space = BackendVectorSpace(self.space.descriptor(), bkd)
        outer: Newton[Np, Np] = Newton(atol=1e-11, rtol=1e-11, maxiters=20)
        if dense:
            assembled = self._pin(
                SkeletonResidual(space, self.subdomains(pieces), DenseSchur(bkd))
            )
            lu: RecomputeAlways[Np, DenseFormat[Np]] = RecomputeAlways(
                BackendDenseLU(bkd)
            )
            point = SteadySystem(assembled, space, lu).solve(outer, space.zeros(1))
        else:
            applied = self._pin(self.residual(pieces))
            krylov: RecomputeAlways[Np, MatrixFree] = RecomputeAlways(
                GMRES(space, bkd, tol=1e-13)
            )
            point = SteadySystem(applied, space, krylov).solve(outer, space.zeros(1))
        assert point.converged()
        return point.solution()


def check_coupled(problem: CoupledProblem, dense: bool = False) -> None:
    """The skeleton solution and every subdomain's state equal the single
    domain's at the same points, and the free interface fluxes balance
    per component, all to 1e-10."""
    bkd = problem.bkd
    reference, ref_nodes = problem.single_domain()
    built = [problem.piece(s) for s in range(problem.topo.nsubdomains())]
    lam = problem.solve_skeleton([piece for piece, _, _ in built], dense)
    skel = problem.skeleton_coords()
    at_skeleton = problem.mesh_dofs(
        [node_at(ref_nodes, skel[:, d]) for d in range(skel.shape[1])]
    )
    bkd.assert_allclose(lam, reference[at_skeleton], atol=1e-10)
    balance = np.zeros_like(lam)
    for piece, nodes, dofs in built:
        point = piece.linearize(lam[list(dofs)], bkd.zeros((0, 1)), STEADY)
        assert isinstance(point, SteadyPinnedPoint)
        np.add.at(balance, list(dofs), point.output())
        rows = problem.mesh_dofs(
            [node_at(ref_nodes, nodes[:, j]) for j in range(nodes.shape[1])]
        )
        bkd.assert_allclose(point.state(), reference[rows], atol=1e-10)
    # Pinned DOFs carry the boundary's reaction, not a balance.
    balance[problem.pinned()] = 0.0
    bkd.assert_allclose(balance, bkd.zeros(balance.shape), atol=1e-10)


def box_problem(
    ncells: tuple[int, int],
    build: SystemBuilder,
    essentials: Sequence[tuple[str, Value]],
    ncomponents: int = 1,
    pins: Optional[SkeletonPins] = None,
) -> CoupledProblem:
    """A box partition of the unit square, each cell meshed on its
    Lobatto grid with the essentials on the outer sides it touches."""
    topo = TensorBoxPartition(ncells)
    ex, ey = (np.linspace(0.0, 1.0, n + 1) for n in ncells)
    grids: list[Grid] = []
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
        grids.append((lobatto(ex[ix], ex[ix + 1]), lobatto(ey[iy], ey[iy + 1]), sides))
    whole = (
        union(*(lobatto(ex[i], ex[i + 1]) for i in range(ncells[0]))),
        union(*(lobatto(ey[i], ey[i + 1]) for i in range(ncells[1]))),
        essentials,
    )
    geometry = TensorBoxGeometry(topo, NumpyBkd())
    return CoupledProblem(topo, geometry, grids, whole, build, ncomponents, pins)
