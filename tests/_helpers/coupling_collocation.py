r"""Harness for collocation subdomains of the unit box coupled through a
skeleton, compared with a single-domain collocation solve.

The partition is a :class:`~pyapprox.pde.coupling.topology.
TensorBoxPartition` of :math:`[0, 1]^d` (d = 2 or 3). Each subdomain is a
Chebyshev collocation mesh with its own number of nodes per direction,
and a :func:`~pyapprox.pde.coupling.stacks.collocation.
collocation_pinned_response`; the skeleton's face basis is the tensor
Lagrange basis on Chebyshev-Gauss-Lobatto nodes. The transfer is matching
when every subdomain has the skeleton's nodes, or interpolating
(:class:`CoupledCollocation`).

The physics is the steady scalar advection-diffusion-reaction equation
:math:`D \Delta u - \nabla\cdot(v u) + r u + f = 0` with constant
coefficients, f manufactured from an exact solution, which also gives the
Dirichlet data on the whole outer boundary.

Skeleton DOFs on the outer boundary are held at the exact data.
"""

import itertools
from dataclasses import dataclass
from typing import Callable, Generic, Optional, Sequence, Union

from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.implicitfunction.steady_system import SteadySystem
from pyapprox.pde.assembly.backend_dense_sink import BackendDenseMatrixAssembly
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
from pyapprox.pde.collocation.quadrature.boundary_quadrature import (
    CollocationNodeRule,
    GaussLegendreSideRule,
    SideRuleProtocol,
)
from pyapprox.pde.constitutive.advection import ConservativeAdvection
from pyapprox.pde.coupling.basis import TensorLagrangeBasis
from pyapprox.pde.coupling.closure_numbering import VertexWeightNumbering
from pyapprox.pde.coupling.geometry import TensorBoxGeometry, skeleton_dof_points
from pyapprox.pde.coupling.pinned_response import (
    SteadyPinnedPoint,
    SteadyPinnedResponse,
)
from pyapprox.pde.coupling.pinning import PinnedSkeletonResidual
from pyapprox.pde.coupling.ports import (
    AverageIncidentFaces,
    BackendPortRows,
    MatchingNodalTransfer,
    PortMatrixTransfer,
    PortTransferProtocol,
    port_interpolation,
)
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
    """A subdomain: its box, its response, and the transfer from the
    skeleton to its ports."""

    box: Box[Array]
    response: Response[Array]
    transfer: PortTransferProtocol[Array, Array]

    def state(self, lam: Array) -> Array:
        """The subdomain's state for the skeleton values ``lam``. Shape:
        (npts,)"""
        bkd = self.box.mesh().bkd()
        port_data = self.transfer.scatter(lam)
        point = self.response.linearize(port_data, bkd.zeros((0, 1)), STEADY)
        if not isinstance(point, SteadyPinnedPoint):
            raise TypeError("expected a SteadyPinnedPoint")
        return point.state()[:, 0]


class CoupledCollocation(Generic[Array]):
    """A box partition of the unit box into collocation subdomains, its
    continuous skeleton, and the coupled and single-domain solves.

    Subdomain ``s`` has ``ns[s]`` nodes per direction; the skeleton's face
    basis has ``nskel`` Chebyshev-Gauss-Lobatto nodes per direction.

    Matching (``interpolate`` False): every ``ns[s]`` equals ``nskel``, so
    each subdomain's interface nodes are skeleton nodes, the outer-boundary
    ones among them pinned skeleton DOFs read as port nodes.

    Interpolating (``interpolate`` True): a subdomain's port nodes are its
    interface nodes off the outer boundary, which keep their own Dirichlet
    rows, and each reads the skeleton through
    :func:`~pyapprox.pde.coupling.ports.port_interpolation`.

    ``rule`` tests each side's flux; omitted, lumped
    (``CollocationNodeRule``) when matching and exact
    (``GaussLegendreSideRule``) when interpolating.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        ncells: Sequence[int],
        ns: Sequence[int],
        nskel: int,
        coefs: Coefficients,
        exact: Exact[Array],
        interpolate: bool = False,
        rule: Optional[SideRuleProtocol] = None,
    ) -> None:
        self._topo = TensorBoxPartition(ncells)
        if len(ns) != self._topo.nsubdomains():
            raise ValueError(f"ns needs one count per subdomain, got {len(ns)}")
        if not interpolate and any(n != nskel for n in ns):
            raise ValueError("a matching transfer needs every count equal to nskel")
        self._bkd, self._ncells, self._ns = bkd, tuple(ncells), tuple(ns)
        self._coefs, self._exact = coefs, exact
        self._interpolate = interpolate
        default: SideRuleProtocol = (
            GaussLegendreSideRule() if interpolate else CollocationNodeRule()
        )
        self._rule = default if rule is None else rule
        bases = UniformBases(
            TensorLagrangeBasis(len(ncells) - 1, reference_nodes(nskel, bkd), True, bkd)
        )
        self._space = ContinuousSkeletonSpace(
            self._topo, bases, VertexWeightNumbering(self._topo, bases)
        )
        self._geometry = TensorBoxGeometry(self._topo, bkd)
        self._coords = skeleton_dof_points(self._space, self._geometry, bases)
        self._pieces = [self._piece(s) for s in range(self._topo.nsubdomains())]

    def _cell(self, subdomain: int) -> tuple[int, ...]:
        """The subdomain's cell multi-index, the last direction fastest."""
        index = []
        for m in reversed(self._ncells):
            subdomain, i = divmod(subdomain, m)
            index.append(i)
        return tuple(reversed(index))

    def _dofs(self, subdomain: int) -> list[int]:
        """The skeleton DOFs on the subdomain's closure, increasing."""
        return sorted(
            dof
            for entity in self._topo.subdomain_entities(subdomain)
            for dof in self._space.entity_dofs(entity)
        )

    def _response(
        self,
        box: Box[Array],
        port_sides: Sequence[PortSide],
        outer: Sequence[Side],
        ports: set[int],
    ) -> Response[Array]:
        """The box's response with ports ``port_sides`` and the exact data
        on its ``outer`` sides except at the nodes ``ports``."""
        return collocation_pinned_response(
            CollocationSystem(box.physics(), box.essential(outer, ports)),
            box.physics(),
            box.mesh(),
            port_sides,
            self._rule,
            RecomputeAlways(BackendDenseLU(self._bkd)),
            Newton(atol=1e-13, rtol=0.0, maxiters=30),
        )

    def _piece(self, subdomain: int) -> Piece[Array]:
        bkd, ncells = self._bkd, self._ncells
        cell = self._cell(subdomain)
        bounds = [(c / m, (c + 1) / m) for c, m in zip(cell, ncells)]
        box = Box(bkd, bounds, self._ns[subdomain], self._coefs, self._exact)
        sides: list[Side] = list(itertools.product(range(len(ncells)), (False, True)))
        interface = [
            (d, upper)
            for d, upper in sides
            if (cell[d] < ncells[d] - 1 if upper else cell[d] > 0)
        ]
        outer = [side for side in sides if side not in interface]
        if self._interpolate:
            return self._interpolated_piece(subdomain, box, interface, outer)
        return self._matching_piece(subdomain, box, interface, outer)

    def _interpolated_piece(
        self,
        subdomain: int,
        box: Box[Array],
        interface: Sequence[Side],
        outer: Sequence[Side],
    ) -> Piece[Array]:
        bkd = self._bkd
        on_outer = {i for side in outer for i in box.side(side)}
        port_sides = [
            PortSide(
                2 * d + upper, [i for i in box.side((d, upper)) if i not in on_outer]
            )
            for d, upper in interface
        ]
        ports = [node for side in port_sides for node in side.nodes]
        dofs = self._dofs(subdomain)
        faces = [
            f
            for f in self._topo.subdomain_entities(subdomain)
            if f in self._space.faces()
        ]
        matrix = port_interpolation(
            self._space,
            box.points()[:, ports],
            self._geometry,
            faces,
            dofs,
            AverageIncidentFaces(),
            BackendDenseMatrixAssembly(bkd),
        )
        transfer = PortMatrixTransfer(
            BackendPortRows(self._space.descriptor(), dofs, bkd), matrix
        )
        return Piece(box, self._response(box, port_sides, outer, set()), transfer)

    def _matching_piece(
        self,
        subdomain: int,
        box: Box[Array],
        interface: Sequence[Side],
        outer: Sequence[Side],
    ) -> Piece[Array]:
        bkd = self._bkd
        on_side = {side: set(box.side(side)) for side in interface}
        by_side: dict[Side, list[tuple[int, int]]] = {side: [] for side in interface}
        for dof in self._dofs(subdomain):
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
        order = [dof for pairs in by_side.values() for dof, _ in pairs]
        transfer = MatchingNodalTransfer(
            BackendPortRows(self._space.descriptor(), order, bkd)
        )
        return Piece(box, self._response(box, port_sides, outer, ports), transfer)

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def exact(self) -> Exact[Array]:
        """The exact solution."""
        return self._exact

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
            CoupledSubdomain(piece.response, piece.transfer) for piece in self._pieces
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


def cubic(points: Array, bkd: Backend[Array]) -> tuple[Array, Array, Array]:
    """u = 1 + x - 2x^2 + x^3 + xy + y^3/2 [+ yz + z^2 in 3D]."""
    x, y = points[0], points[1]
    u = 1.0 + x - 2.0 * x**2 + x**3 + x * y + 0.5 * y**3
    grad = [1.0 - 4.0 * x + 3.0 * x**2 + y, x + 1.5 * y**2]
    lap = -4.0 + 6.0 * x + 3.0 * y
    if points.shape[0] == 3:
        z = points[2]
        u = u + y * z + z**2
        grad = [grad[0], grad[1] + z, y + 2.0 * z]
        lap = lap + 2.0
    return u, bkd.stack(grad, axis=0), lap


def smooth(points: Array, bkd: Backend[Array]) -> tuple[Array, Array, Array]:
    """u = e^x sin(3x) cos(2y), in 2D."""
    x, y = points[0], points[1]
    ex, s, c = bkd.exp(x), bkd.sin(3.0 * x), bkd.cos(3.0 * x)
    cy, sy = bkd.cos(2.0 * y), bkd.sin(2.0 * y)
    u = ex * s * cy
    grad = bkd.stack([ex * (s + 3.0 * c) * cy, -2.0 * ex * s * sy], axis=0)
    return u, grad, ex * cy * (6.0 * c - 12.0 * s)


def max_state_error(problem: CoupledCollocation[Array], lam: Array) -> float:
    """The largest nodal error of the subdomains' states for the skeleton
    values ``lam``, against the problem's exact solution."""
    bkd, exact = problem.bkd(), problem.exact()
    return max(
        bkd.to_float(
            bkd.max(bkd.abs(piece.state(lam) - exact(piece.box.points(), bkd)[0]))
        )
        for piece in problem.pieces()
    )


def interpolate(values: Array, n: int, points: Array, bkd: Backend[Array]) -> Array:
    """The tensor Chebyshev interpolant of nodal ``values`` on the unit box
    with ``n`` nodes per direction (the first varying fastest), at
    ``points`` of shape (d, npoints)."""
    ndim = int(points.shape[0])
    basis = TensorLagrangeBasis(ndim, reference_nodes(n, bkd), True, bkd)
    # The basis varies its last direction fastest, the mesh its first.
    flipped = 2.0 * points[list(reversed(range(ndim)))] - 1.0
    return basis.evaluate(flipped) @ values
