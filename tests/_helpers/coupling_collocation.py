r"""Harness for collocation subdomains of the unit box coupled through a
skeleton, compared with a single-domain collocation solve.

The partition is a :class:`~pyapprox.pde.coupling.topology.
TensorBoxPartition` of :math:`[0, 1]^d` (d = 2 or 3). Each subdomain is a
Chebyshev collocation mesh with ``n`` nodes per direction; the skeleton's
face basis is the tensor Lagrange basis on the same Chebyshev-Gauss-Lobatto
nodes, so every subdomain's interface nodes are skeleton nodes and the
transfer is matching. Each subdomain is a
:func:`~pyapprox.pde.coupling.stacks.collocation.
collocation_pinned_response`.

The physics is the steady scalar advection-diffusion-reaction equation
:math:`D \Delta u - \nabla\cdot(v u) + r u + f = 0` with constant
coefficients, f manufactured from an exact solution, which also gives the
Dirichlet data on the whole outer boundary.

Interface nodes on the outer boundary stay skeleton DOFs: the skeleton
holds them at the exact data, and each subdomain reads them as port nodes,
not essential rows. Their flux is computed with the interface side's
normal and discarded with the pinned skeleton rows.
"""

import itertools
from dataclasses import dataclass
from typing import Callable, Generic, Optional, Sequence, Union

from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.implicitfunction.steady_system import SteadySystem
from pyapprox.pde.collocation.basis.chebyshev.basis_2d import ChebyshevBasis2D
from pyapprox.pde.collocation.basis.chebyshev.basis_3d import ChebyshevBasis3D
from pyapprox.pde.collocation.boundary.essential_rows import (
    DirichletPart,
    DirichletRows,
)
from pyapprox.pde.collocation.mesh.transformed import (
    TransformedMesh2D,
    TransformedMesh3D,
)
from pyapprox.pde.collocation.mesh.transforms.affine import (
    AffineTransform2D,
    AffineTransform3D,
)
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.collocation.physics.coefficients import NodalField
from pyapprox.pde.constitutive.advection import ConservativeAdvection
from pyapprox.pde.coupling.basis import TensorLagrangeBasis
from pyapprox.pde.coupling.closure_numbering import VertexWeightNumbering
from pyapprox.pde.coupling.geometry import TensorBoxGeometry, skeleton_dof_points
from pyapprox.pde.coupling.pinned_response import (
    SteadyPinnedPoint,
    SteadyPinnedResponse,
)
from pyapprox.pde.coupling.pinning import PinnedSkeletonResidual
from pyapprox.pde.coupling.ports import BackendPortRows, MatchingNodalTransfer
from pyapprox.pde.coupling.responses import TimeWindow
from pyapprox.pde.coupling.skeleton import (
    CoupledSubdomain,
    DenseSchur,
    SkeletonResidual,
)
from pyapprox.pde.coupling.spaces import ContinuousSkeletonSpace, UniformBases
from pyapprox.pde.coupling.stacks.collocation import (
    PortSide,
    collocation_pinned_response,
)
from pyapprox.pde.coupling.topology import TensorBoxPartition
from pyapprox.pde.models.collocation.closed_system import CollocationSystem
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import (
    BackendDenseLU,
    BackendRowSet,
    BackendVectorSpace,
    DenseFormat,
)
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.physics_test_utils import solve_steady_dirichlet

STEADY = TimeWindow.steady()
ATOL = 1e-11

# u, its gradient and its Laplacian at points of shape (d, n): shapes
# (n,), (d, n), (n,).
Exact = Callable[[Array, Backend[Array]], tuple[Array, Array, Array]]
Bounds = Sequence[tuple[float, float]]
Mesh = Union[TransformedMesh2D[Array], TransformedMesh3D[Array]]
# A side of a box: the direction normal to it and whether at its upper end.
Side = tuple[int, bool]
Response = SteadyPinnedResponse[Array, Array, DenseFormat[Array]]


@dataclass(frozen=True)
class Coefficients:
    """Constant coefficients: diffusion, velocity (one entry per
    direction) or none, reaction or none."""

    diffusion: float
    velocity: Optional[tuple[float, ...]]
    reaction: Optional[float]


def _mesh(bounds: Bounds, n: int, bkd: Backend[Array]) -> Mesh[Array]:
    flat = tuple(v for pair in bounds for v in pair)
    if len(bounds) == 2:
        ax, bx, ay, by = flat
        return TransformedMesh2D(n, n, bkd, AffineTransform2D((ax, bx, ay, by), bkd))
    if len(bounds) == 3:
        ax, bx, ay, by, az, bz = flat
        return TransformedMesh3D(
            n, n, n, bkd, AffineTransform3D((ax, bx, ay, by, az, bz), bkd)
        )
    raise ValueError(f"the harness covers 2D and 3D, got {len(bounds)}D")


def reference_nodes(n: int, bkd: Backend[Array]) -> Array:
    """The ``n`` Chebyshev-Gauss-Lobatto nodes on [-1, 1], increasing."""
    return _mesh([(-1.0, 1.0)] * 2, n, bkd).reference_nodes_1d(0)


def _indices(bkd: Backend[Array], array: Array) -> list[int]:
    return [bkd.to_int(array[k]) for k in range(array.shape[0])]


class Box(Generic[Array]):
    """The physics on a box with ``n`` nodes per direction, its forcing
    manufactured from ``exact``."""

    def __init__(
        self,
        bkd: Backend[Array],
        bounds: Bounds,
        n: int,
        coefs: Coefficients,
        exact: Exact[Array],
    ) -> None:
        self._bkd, self._exact = bkd, exact
        self._mesh = _mesh(bounds, n, bkd)
        ndim = len(bounds)
        basis = (
            ChebyshevBasis2D(self._mesh, bkd)
            if ndim == 2
            else ChebyshevBasis3D(self._mesh, bkd)
        )
        npts = self._mesh.npts()
        u, grad, lap = exact(self.points(), bkd)
        forcing = -coefs.diffusion * lap
        velocity = None
        if coefs.velocity is not None:
            if len(coefs.velocity) != ndim:
                raise ValueError(f"velocity needs {ndim} components")
            for d, v in enumerate(coefs.velocity):
                forcing = forcing + v * grad[d]
            velocity = [bkd.full((npts,), v) for v in coefs.velocity]
        if coefs.reaction is not None:
            forcing = forcing - coefs.reaction * u
        self._physics = AdvectionDiffusionReaction(
            basis,
            bkd,
            diffusion=coefs.diffusion,
            velocity=velocity,
            reaction=coefs.reaction,
            forcing=NodalField(forcing),
            advection_form=ConservativeAdvection(),
        )

    def mesh(self) -> Mesh[Array]:
        return self._mesh

    def physics(self) -> AdvectionDiffusionReaction[Array]:
        return self._physics

    def points(self) -> Array:
        """Shape: (d, npts)"""
        return self._mesh.points()

    def side(self, side: Side) -> list[int]:
        """The node indices of ``side``."""
        direction, upper = side
        return _indices(self._bkd, self._mesh.boundary_indices(2 * direction + upper))

    def dirichlet(
        self, sides: Sequence[Side], free: set[int]
    ) -> Optional[DirichletPart[Array]]:
        """The exact data on ``sides``, except at the nodes ``free``; None
        when no node remains."""
        bkd = self._bkd
        nodes = sorted({i for side in sides for i in self.side(side)} - free)
        if not nodes:
            return None
        index = bkd.asarray(nodes, dtype=bkd.int64_dtype())
        return DirichletPart(
            index, NodalField(self._exact(self.points()[:, index], bkd)[0])
        )

    def essential(
        self, sides: Sequence[Side], free: set[int]
    ) -> Optional[DirichletRows[Array]]:
        """:meth:`dirichlet` as essential rows; None without any."""
        part = self.dirichlet(sides, free)
        if part is None:
            return None
        return DirichletRows(self._bkd, self._mesh.npts(), 1, [part])


def node_at(points: Array, point: Array, bkd: Backend[Array]) -> int:
    """The index of the column of ``points`` at ``point``."""
    distances = bkd.sqrt(bkd.sum((points - point[:, None]) ** 2, axis=0))
    node = bkd.to_int(bkd.argmin(distances))
    if bkd.to_float(distances[node]) > 1e-12:
        raise ValueError(f"no node at {bkd.to_numpy(point)}")
    return node


@dataclass(frozen=True)
class Piece(Generic[Array]):
    """A subdomain: its box, its response, and the skeleton DOF each port
    reads, in port order."""

    box: Box[Array]
    response: Response[Array]
    dofs: list[int]

    def state(self, lam: Array) -> Array:
        """The subdomain's state for the skeleton values ``lam``. Shape:
        (npts,)"""
        bkd = self.box.mesh().bkd()
        point = self.response.linearize(lam[self.dofs], bkd.zeros((0, 1)), STEADY)
        if not isinstance(point, SteadyPinnedPoint):
            raise TypeError("expected a SteadyPinnedPoint")
        return point.state()[:, 0]


class CoupledCollocation(Generic[Array]):
    """A box partition of the unit box into collocation subdomains with
    ``n`` nodes per direction, its continuous skeleton on the same nodes,
    and the coupled and single-domain solves."""

    def __init__(
        self,
        bkd: Backend[Array],
        ncells: Sequence[int],
        n: int,
        coefs: Coefficients,
        exact: Exact[Array],
    ) -> None:
        self._bkd, self._ncells, self._n = bkd, tuple(ncells), n
        self._coefs, self._exact = coefs, exact
        self._topo = TensorBoxPartition(ncells)
        bases = UniformBases(
            TensorLagrangeBasis(len(ncells) - 1, reference_nodes(n, bkd), True, bkd)
        )
        self._space = ContinuousSkeletonSpace(
            self._topo, bases, VertexWeightNumbering(self._topo, bases)
        )
        self._coords = skeleton_dof_points(
            self._space, TensorBoxGeometry(self._topo, bkd), bases
        )
        self._pieces = [self._piece(s) for s in range(self._topo.nsubdomains())]

    def _cell(self, subdomain: int) -> tuple[int, ...]:
        """The subdomain's cell multi-index, the last direction fastest."""
        index = []
        for m in reversed(self._ncells):
            subdomain, i = divmod(subdomain, m)
            index.append(i)
        return tuple(reversed(index))

    def _piece(self, subdomain: int) -> Piece[Array]:
        bkd, ncells = self._bkd, self._ncells
        cell = self._cell(subdomain)
        bounds = [(c / m, (c + 1) / m) for c, m in zip(cell, ncells)]
        box = Box(bkd, bounds, self._n, self._coefs, self._exact)
        sides: list[Side] = list(itertools.product(range(len(ncells)), (False, True)))
        interface = [
            (d, upper)
            for d, upper in sides
            if (cell[d] < ncells[d] - 1 if upper else cell[d] > 0)
        ]
        on_side = {side: set(box.side(side)) for side in interface}
        by_side: dict[Side, list[tuple[int, int]]] = {side: [] for side in interface}
        dofs = sorted(
            dof
            for entity in self._topo.subdomain_entities(subdomain)
            for dof in self._space.entity_dofs(entity)
        )
        for dof in dofs:
            node = node_at(box.points(), self._coords[:, dof], bkd)
            # A node on two interface sides goes to both; the builder
            # refuses the cross point.
            for side in interface:
                if node in on_side[side]:
                    by_side[side].append((dof, node))
        port_sides = [
            PortSide(2 * d + upper, [node for _, node in pairs])
            for (d, upper), pairs in by_side.items()
            if pairs
        ]
        ports = {node for side in port_sides for node in side.nodes}
        outer = [side for side in sides if side not in interface]
        response = collocation_pinned_response(
            CollocationSystem(box.physics(), box.essential(outer, ports)),
            box.physics(),
            box.mesh(),
            port_sides,
            RecomputeAlways(BackendDenseLU(bkd)),
            Newton(atol=1e-13, rtol=0.0, maxiters=30),
        )
        order = [dof for pairs in by_side.values() for dof, _ in pairs]
        return Piece(box, response, order)

    def pieces(self) -> list[Piece[Array]]:
        return self._pieces

    def skeleton_coords(self) -> Array:
        """Shape: (d, ndofs)"""
        return self._coords

    def pinned(self) -> list[int]:
        """The skeleton DOFs on the outer boundary."""
        return sorted(
            dof
            for entity in range(self._topo.nentities())
            if self._topo.on_boundary(entity)
            for dof in self._space.entity_dofs(entity)
        )

    def residual(self) -> SkeletonResidual[Array, Array, Array, DenseFormat[Array]]:
        """The unpinned skeleton residual: the gathered tested fluxes."""
        bkd, descriptor = self._bkd, self._space.descriptor()
        subdomains = [
            CoupledSubdomain(
                piece.response,
                MatchingNodalTransfer(BackendPortRows(descriptor, piece.dofs, bkd)),
            )
            for piece in self._pieces
        ]
        return SkeletonResidual(
            BackendVectorSpace(descriptor, bkd), subdomains, DenseSchur(bkd)
        )

    def solve_skeleton(self) -> Array:
        """The skeleton values, the outer-boundary DOFs held at the exact
        data. Shape: (ndofs, 1)"""
        bkd, descriptor = self._bkd, self._space.descriptor()
        space = BackendVectorSpace(descriptor, bkd)
        pinned = self.pinned()
        mask = bkd.zeros((descriptor.size,))
        mask[pinned] = 1.0
        values = bkd.reshape(mask * self._exact(self._coords, bkd)[0], (-1, 1))
        residual: PinnedSkeletonResidual[Array, DenseFormat[Array]] = (
            PinnedSkeletonResidual(
                self.residual(), space, BackendRowSet(descriptor, pinned, bkd), values
            )
        )
        lu: RecomputeAlways[Array, DenseFormat[Array]] = RecomputeAlways(
            BackendDenseLU(bkd)
        )
        outer: Newton[Array, Array] = Newton(atol=ATOL, rtol=0.0, maxiters=20)
        point = SteadySystem(residual, space, lu).solve(outer, space.zeros(1))
        if not point.converged():
            raise RuntimeError("the skeleton solve did not converge")
        return point.solution()

    def single_domain(self, n: int) -> tuple[Box[Array], Array]:
        """The single-domain solution with ``n`` nodes per direction: its
        box and nodal values. Shape of the values: (npts,)"""
        bkd, ndim = self._bkd, len(self._ncells)
        box = Box(bkd, [(0.0, 1.0)] * ndim, n, self._coefs, self._exact)
        sides = list(itertools.product(range(ndim), (False, True)))
        part = box.dirichlet(sides, set())
        if part is None:
            raise ValueError("the single domain has boundary nodes")
        state = solve_steady_dirichlet(
            box.physics(), 1, [part], bkd.zeros((box.mesh().npts(),)), atol=ATOL
        )
        return box, state


def interpolate(values: Array, n: int, points: Array, bkd: Backend[Array]) -> Array:
    """The tensor Chebyshev interpolant of nodal ``values`` on the unit box
    with ``n`` nodes per direction (the first varying fastest), at
    ``points`` of shape (d, npoints)."""
    ndim = int(points.shape[0])
    basis = TensorLagrangeBasis(ndim, reference_nodes(n, bkd), True, bkd)
    # The basis varies its last direction fastest, the mesh its first.
    flipped = 2.0 * points[list(reversed(range(ndim)))] - 1.0
    return basis.evaluate(flipped) @ values
