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

The partition and the skeleton live in the parameter box; a
:class:`Domain` maps it to the physical domain, the box itself
(:class:`UnitBox`) or a curved one (:class:`QuarterAnnulus`). Skeleton
nodes are matched and interpolated in parameter coordinates; physics,
normals and quadrature weights are physical.
"""

import itertools
import math
from dataclasses import dataclass
from typing import Callable, Generic, Optional, Protocol, Sequence, Union

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
from pyapprox.pde.collocation.mesh.transforms import PolarTransform
from pyapprox.pde.collocation.mesh.transforms.affine import (
    AffineTransform2D,
    AffineTransform3D,
)
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.collocation.physics.coefficients import NodalField
from pyapprox.pde.collocation.protocols.physics import (
    CollocationInteriorOperatorProtocol,
)
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
from pyapprox.pde.coupling.spaces import (
    ComponentSkeletonSpace,
    ContinuousSkeletonSpace,
    UniformBases,
)
from pyapprox.pde.coupling.stacks.collocation import (
    ComponentFluxes,
    NormalFluxesProtocol,
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


class Domain(Protocol[Array]):
    """The physical domain over the parameter box :math:`[0, 1]^d` that
    the partition and the skeleton live in."""

    def mesh(self, bounds: Bounds, n: int, bkd: Backend[Array]) -> Mesh[Array]:
        """The collocation mesh of the parameter box ``bounds``."""
        ...

    def to_physical(self, params: Array, bkd: Backend[Array]) -> Array:
        """Physical points of parameter points. Shape: (d, n)"""
        ...


class UnitBox(Generic[Array]):
    """The parameter box itself."""

    def mesh(self, bounds: Bounds, n: int, bkd: Backend[Array]) -> Mesh[Array]:
        return _mesh(bounds, n, bkd)

    def to_physical(self, params: Array, bkd: Backend[Array]) -> Array:
        return params


class QuarterAnnulus(Generic[Array]):
    r"""The quarter annulus :math:`1 \le r \le 2`, :math:`0 \le \theta \le
    \pi/2`: parameter s maps to r = 1 + s and t to theta = t pi / 2, so a
    cut at constant s is an arc."""

    def mesh(self, bounds: Bounds, n: int, bkd: Backend[Array]) -> Mesh[Array]:
        if len(bounds) != 2:
            raise ValueError(f"the quarter annulus is 2D, got {len(bounds)}D")
        (s0, s1), (t0, t1) = bounds
        quarter = math.pi / 2.0
        transform = PolarTransform(
            (1.0 + s0, 1.0 + s1), (quarter * t0, quarter * t1), bkd
        )
        return TransformedMesh2D(n, n, bkd, transform)

    def to_physical(self, params: Array, bkd: Backend[Array]) -> Array:
        r, theta = 1.0 + params[0], (math.pi / 2.0) * params[1]
        return bkd.stack([r * bkd.cos(theta), r * bkd.sin(theta)], axis=0)


def reference_nodes(n: int, bkd: Backend[Array]) -> Array:
    """The ``n`` Chebyshev-Gauss-Lobatto nodes on [-1, 1], increasing."""
    return _mesh([(-1.0, 1.0)] * 2, n, bkd).reference_nodes_1d(0)


def _indices(bkd: Backend[Array], array: Array) -> list[int]:
    return [bkd.to_int(array[k]) for k in range(array.shape[0])]


def basis(
    mesh: Mesh[Array], bkd: Backend[Array]
) -> Union[ChebyshevBasis2D[Array], ChebyshevBasis3D[Array]]:
    """The Chebyshev collocation basis of a 2D or 3D mesh."""
    if isinstance(mesh, TransformedMesh2D):
        return ChebyshevBasis2D(mesh, bkd)
    return ChebyshevBasis3D(mesh, bkd)


class Model(Protocol[Array]):
    """A physics of ``k`` components with a manufactured exact solution:
    what a subdomain solves and what its ports return."""

    def ncomponents(self) -> int: ...

    def exact(self, points: Array, bkd: Backend[Array]) -> Array:
        """The exact solution at physical ``points``. Shape: (k, n)"""
        ...

    def build(
        self, mesh: Mesh[Array], bkd: Backend[Array]
    ) -> tuple[CollocationInteriorOperatorProtocol[Array], NormalFluxesProtocol[Array]]:
        """The physics on ``mesh``, its forcing manufactured from the
        exact solution, and the fluxes its ports return."""
        ...


class Adr(Generic[Array]):
    """Scalar advection-diffusion-reaction with constant ``coefs``,
    :math:`D \\Delta u - \\nabla\\cdot(v u) + r u + f = 0`, f manufactured
    from ``exact``; the ports return the total flux."""

    def __init__(self, coefs: Coefficients, exact: Exact[Array]) -> None:
        self._coefs, self._exact = coefs, exact

    def ncomponents(self) -> int:
        return 1

    def exact(self, points: Array, bkd: Backend[Array]) -> Array:
        return self._exact(points, bkd)[0][None, :]

    def build(
        self, mesh: Mesh[Array], bkd: Backend[Array]
    ) -> tuple[AdvectionDiffusionReaction[Array], ComponentFluxes[Array]]:
        coefs, ndim, npts = self._coefs, mesh.ndim(), mesh.npts()
        u, grad, lap = self._exact(mesh.points(), bkd)
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
        physics = AdvectionDiffusionReaction(
            basis(mesh, bkd),
            bkd,
            diffusion=coefs.diffusion,
            velocity=velocity,
            reaction=coefs.reaction,
            forcing=NodalField(forcing),
            advection_form=ConservativeAdvection(),
        )
        return physics, ComponentFluxes([physics])


class Box(Generic[Array]):
    """A model's physics on the image of a parameter box with ``n`` nodes
    per direction."""

    def __init__(
        self,
        bkd: Backend[Array],
        bounds: Bounds,
        n: int,
        model: Model[Array],
        domain: Optional[Domain[Array]] = None,
    ) -> None:
        self._bkd, self._model, self._bounds = bkd, model, bounds
        self._mesh = (UnitBox() if domain is None else domain).mesh(bounds, n, bkd)
        self._physics, self._fluxes = model.build(self._mesh, bkd)

    def mesh(self) -> Mesh[Array]:
        return self._mesh

    def physics(self) -> CollocationInteriorOperatorProtocol[Array]:
        return self._physics

    def fluxes(self) -> NormalFluxesProtocol[Array]:
        return self._fluxes

    def ncomponents(self) -> int:
        return self._model.ncomponents()

    def points(self) -> Array:
        """Shape: (d, npts)"""
        return self._mesh.points()

    def parameters(self) -> Array:
        """The nodes' parameter points, from the mesh's reference points.
        Shape: (d, npts)"""
        reference = self._mesh.reference_points()
        rows = [
            lo + 0.5 * (reference[d] + 1.0) * (hi - lo)
            for d, (lo, hi) in enumerate(self._bounds)
        ]
        return self._bkd.stack(rows, axis=0)

    def side(self, side: Side) -> list[int]:
        """The node indices of ``side``."""
        direction, upper = side
        return _indices(self._bkd, self._mesh.boundary_indices(2 * direction + upper))

    def dirichlet(
        self, sides: Sequence[Side], free: set[int]
    ) -> list[DirichletPart[Array]]:
        """The exact data on ``sides``, every component, except at the
        nodes ``free``; empty when no node remains."""
        bkd = self._bkd
        nodes = sorted({i for side in sides for i in self.side(side)} - free)
        if not nodes:
            return []
        index = bkd.asarray(nodes, dtype=bkd.int64_dtype())
        values = self._model.exact(self.points()[:, index], bkd)
        return [
            DirichletPart(index, NodalField(values[c]), c)
            for c in range(self.ncomponents())
        ]

    def essential(
        self, sides: Sequence[Side], free: set[int]
    ) -> Optional[DirichletRows[Array]]:
        """:meth:`dirichlet` as essential rows; None without any."""
        parts = self.dirichlet(sides, free)
        if not parts:
            return None
        return DirichletRows(self._bkd, self._mesh.npts(), self.ncomponents(), parts)


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
        """The subdomain's state for the skeleton values ``lam``, its
        components stacked. Shape: (k * npts,)"""
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

    A ``model`` of k components has a skeleton of k components per node
    (:class:`~pyapprox.pde.coupling.spaces.ComponentSkeletonSpace`, DOF
    ``k d + c``).
    """

    def __init__(
        self,
        bkd: Backend[Array],
        ncells: Sequence[int],
        ns: Sequence[int],
        nskel: int,
        model: Model[Array],
        interpolate: bool = False,
        rule: Optional[SideRuleProtocol] = None,
        domain: Optional[Domain[Array]] = None,
    ) -> None:
        self._domain: Domain[Array] = UnitBox() if domain is None else domain
        self._topo = TensorBoxPartition(ncells)
        if len(ns) != self._topo.nsubdomains():
            raise ValueError(f"ns needs one count per subdomain, got {len(ns)}")
        if not interpolate and any(n != nskel for n in ns):
            raise ValueError("a matching transfer needs every count equal to nskel")
        self._bkd, self._ncells, self._ns = bkd, tuple(ncells), tuple(ns)
        self._model, self._k = model, model.ncomponents()
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
        # The scalar space locates nodes; the field space numbers DOFs.
        self._fields = ComponentSkeletonSpace(self._space, self._k)
        self._geometry = TensorBoxGeometry(self._topo, bkd)
        self._coords = skeleton_dof_points(self._space, self._geometry, bases)
        self._pieces = [self._piece(s) for s in range(self._topo.nsubdomains())]

    def _field_dofs(self, scalar: Sequence[int]) -> list[int]:
        """The field DOFs of scalar DOFs, node-major."""
        return [self._k * d + c for d in scalar for c in range(self._k)]

    def _cell(self, subdomain: int) -> tuple[int, ...]:
        """The subdomain's cell multi-index, the last direction fastest."""
        index = []
        for m in reversed(self._ncells):
            subdomain, i = divmod(subdomain, m)
            index.append(i)
        return tuple(reversed(index))

    def _dofs(self, subdomain: int) -> list[int]:
        """The scalar skeleton DOFs on the subdomain's closure, increasing."""
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
            box.fluxes(),
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
        box = Box(bkd, bounds, self._ns[subdomain], self._model, self._domain)
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
        # Each node once, first appearance first: the response's port order.
        ports = list(dict.fromkeys(node for side in port_sides for node in side.nodes))
        dofs = self._field_dofs(self._dofs(subdomain))
        faces = [
            f
            for f in self._topo.subdomain_entities(subdomain)
            if f in self._space.faces()
        ]
        matrix = port_interpolation(
            self._fields,
            box.parameters()[:, ports],
            self._geometry,
            faces,
            dofs,
            AverageIncidentFaces(),
            BackendDenseMatrixAssembly(bkd),
        )
        transfer = PortMatrixTransfer(
            BackendPortRows(self._fields.descriptor(), dofs, bkd), matrix
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
            node = node_at(box.parameters(), self._coords[:, dof], bkd)
            # A node on two interface sides (a cross point) goes to both;
            # the response tests it on each.
            for side in interface:
                if node in on_side[side]:
                    by_side[side].append((dof, node))
        port_sides = [
            PortSide(2 * d + upper, [node for _, node in pairs])
            for (d, upper), pairs in by_side.items()
            if pairs
        ]
        ports = {node for side in port_sides for node in side.nodes}
        # Each node once, first appearance first: the response's port order.
        order = list(
            dict.fromkeys(dof for pairs in by_side.values() for dof, _ in pairs)
        )
        transfer = MatchingNodalTransfer(
            BackendPortRows(self._fields.descriptor(), self._field_dofs(order), bkd)
        )
        return Piece(box, self._response(box, port_sides, outer, ports), transfer)

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def model(self) -> Model[Array]:
        return self._model

    def pieces(self) -> list[Piece[Array]]:
        return self._pieces

    def skeleton_coords(self) -> Array:
        """The scalar skeleton DOFs' physical points. Shape: (d, nscalar)"""
        return self._domain.to_physical(self._coords, self._bkd)

    def skeleton_exact(self) -> Array:
        """The exact solution at the skeleton's field DOFs, node-major.
        Shape: (k * nscalar,)"""
        exact = self._model.exact(self.skeleton_coords(), self._bkd)
        return self._bkd.reshape(self._bkd.transpose(exact), (-1,))

    def pinned(self) -> list[int]:
        """The skeleton field DOFs on the outer boundary."""
        return sorted(
            dof
            for entity in range(self._topo.nentities())
            if self._topo.on_boundary(entity)
            for dof in self._fields.entity_dofs(entity)
        )

    def residual(self) -> SkeletonResidual[Array, Array, Array, DenseFormat[Array]]:
        """The unpinned skeleton residual: the gathered tested fluxes."""
        bkd, descriptor = self._bkd, self._fields.descriptor()
        subdomains = [
            CoupledSubdomain(piece.response, piece.transfer) for piece in self._pieces
        ]
        return SkeletonResidual(
            BackendVectorSpace(descriptor, bkd), subdomains, DenseSchur(bkd)
        )

    def solve_skeleton(self) -> Array:
        """The skeleton values, the outer-boundary DOFs held at the exact
        data. Shape: (ndofs, 1)"""
        bkd, descriptor = self._bkd, self._fields.descriptor()
        space = BackendVectorSpace(descriptor, bkd)
        pinned = self.pinned()
        index = bkd.asarray(pinned, dtype=bkd.int64_dtype())
        mask = bkd.index_update(
            bkd.zeros((descriptor.size,)), index, bkd.ones((len(pinned),))
        )
        values = bkd.reshape(mask * self.skeleton_exact(), (-1, 1))
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
        box and nodal values, components stacked. Shape of the values:
        (k * npts,)"""
        bkd, ndim = self._bkd, len(self._ncells)
        box = Box(bkd, [(0.0, 1.0)] * ndim, n, self._model, self._domain)
        sides = list(itertools.product(range(ndim), (False, True)))
        parts = box.dirichlet(sides, set())
        if not parts:
            raise ValueError("the single domain has boundary nodes")
        state = solve_steady_dirichlet(
            box.physics(),
            self._k,
            parts,
            bkd.zeros((self._k * box.mesh().npts(),)),
            atol=ATOL,
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
    values ``lam``, against the problem's exact solution, over every
    component."""
    bkd, model = problem.bkd(), problem.model()
    return max(
        bkd.to_float(
            bkd.max(
                bkd.abs(
                    piece.state(lam)
                    - bkd.reshape(model.exact(piece.box.points(), bkd), (-1,))
                )
            )
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
