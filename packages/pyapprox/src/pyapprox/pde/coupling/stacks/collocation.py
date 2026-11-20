r"""A spectral collocation subdomain as a Dirichlet-to-Neumann response.

The subdomain's port nodes lie on sides of its mesh. Its rows are the PDE
evaluated pointwise, so the residual left on a pinned port row is not a
flux (:mod:`~pyapprox.pde.coupling.normal_flux`); the response instead
returns the outward normal flux :math:`\mathbf{F}(u)\cdot n`, interpolated
on each port side from its nodes and tested against the side's Lagrange
polynomials by an injected rule (:func:`~pyapprox.pde.collocation.
quadrature.boundary_quadrature.boundary_side_quadrature`).

Scalar fluxes only. A port node on several port sides is tested on each
with that side's normal.
"""

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, runtime_checkable

from pyapprox.implicitfunction.setups import SetupStrategyProtocol, SteadyRequest
from pyapprox.ode.residual_derivatives import backend_state_space
from pyapprox.pde.collocation.boundary.normal_operators import FluxNormalOperator
from pyapprox.pde.collocation.protocols.boundary import FluxProviderProtocol
from pyapprox.pde.collocation.protocols.mesh import MeshWithTransformProtocol
from pyapprox.pde.collocation.quadrature.boundary_quadrature import (
    SideRuleProtocol,
    boundary_side_quadrature,
)
from pyapprox.pde.coupling.normal_flux import NormalFluxMoments, NormalFluxPart
from pyapprox.pde.coupling.pinned_ports import PortPinnedData
from pyapprox.pde.coupling.pinned_response import SteadyPinnedResponse
from pyapprox.pde.coupling.ports import BackendPortRows
from pyapprox.pde.coupling.traces import TabulatedTrace
from pyapprox.pde.models.collocation.closed_system import CollocationSystem
from pyapprox.pde.models.collocation.steady_residual import (
    collocation_steady_residual,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import (
    BackendRowSet,
    BackendVectorSpace,
    DenseFormat,
    backend_dtype_name,
)
from pyapprox.util.linalg.operators import SpaceDescriptor
from pyapprox.util.rootfinding.nonlinear import NonlinearSolverProtocol


@runtime_checkable
class PortMeshProtocol(MeshWithTransformProtocol[Array], Protocol[Array]):
    """What :func:`collocation_pinned_response` asks of a mesh: a
    transformed tensor-product mesh with its sides' outward normals."""

    def boundary_normals(self, boundary_id: int) -> Array:
        """The outward unit normals at the side's nodes, in
        ``boundary_indices`` order. Shape: (nboundary, ndim)"""
        ...


@dataclass(frozen=True)
class PortSide:
    """Port nodes on one side of a mesh: the side's ``boundary_id`` and
    the mesh node indices of its port nodes, in port order."""

    boundary_id: int
    nodes: Sequence[int]


def _placed(
    blocks: Sequence[Array],
    columns: Sequence[Sequence[int]],
    ncols: int,
    bkd: Backend[Array],
) -> Array:
    """Each side's block in the port columns of its nodes, the sides'
    rows stacked: a node on two sides has entries in both blocks' rows.
    Placed by a constant 0/1 product, so a torch graph survives."""
    rows = []
    for block, cols in zip(blocks, columns):
        placement = bkd.asarray(
            [[1.0 if c == j else 0.0 for j in range(ncols)] for c in cols]
        )
        rows.append(block @ placement)
    return bkd.concatenate(rows, axis=0)


def collocation_pinned_response(
    system: CollocationSystem[Array],
    flux: FluxProviderProtocol[Array],
    mesh: PortMeshProtocol[Array],
    sides: Sequence[PortSide],
    rule: SideRuleProtocol,
    setups: SetupStrategyProtocol[
        SteadyRequest[Array, DenseFormat[Array]], Array, DenseFormat[Array]
    ],
    nonlinear: NonlinearSolverProtocol[Array, Array],
    time: Optional[float] = None,
) -> SteadyPinnedResponse[Array, Array, DenseFormat[Array]]:
    r"""A collocation subdomain pinned to the port data at the nodes of
    ``sides``, returning the outward flux there, tested against the
    port's trace functions.

    The ports are the sides' nodes in order, the first side's first, each
    node once. Port node ``a``'s trace function on a side is that side's
    Lagrange polynomial; the flux's nodal interpolant on the side (every
    side node, ports or not), with the side's normal, is tested against it
    by the tensor product of ``rule``. A node on several port sides (a
    cross point, or an edge in 3D) has a trace function on each, and
    returns the sum of their moments: the flux tested against a skeleton
    function supported on all of them.

    Parameters
    ----------
    system
        The subdomain's physics and essential rows. A port node must not
        be an essential row: on a side carrying both, leave the port nodes
        out of the essential declaration.
    flux
        The flux whose normal component the ports return, e.g. the
        physics itself or an ``AdvectionDiffusionFlux``; scalar.
    mesh
        The subdomain's mesh.
    sides
        The port nodes, by side.
    rule
        The 1D rule per tangent direction of a side.
        :class:`~pyapprox.pde.collocation.quadrature.boundary_quadrature.
        CollocationNodeRule` lumps: where every subdomain sharing an
        interface has the same nodes there, the coupled solve is then the
        single-domain collocation solve. :class:`~pyapprox.pde.collocation.
        quadrature.boundary_quadrature.GaussLegendreSideRule` tests the
        interpolant exactly, which subdomains of different nodes need: a
        lumped rule's errors cancel only between equal node sets.
    setups
        Sets up the Jacobian of the port-pinned residual.
    nonlinear
        Solves the subdomain problem.
    time
        The time the data and the flux's coefficients are frozen at;
        omitted, everything must be declared time-invariant
        (:func:`~pyapprox.pde.models.collocation.steady_residual.
        collocation_steady_residual`).

    Raises
    ------
    TypeError
        If ``mesh`` does not satisfy :class:`PortMeshProtocol`.
    ValueError
        If there are no port nodes, a node is not on its side, a side
        appears twice, or a side repeats a node.
    """
    if not isinstance(mesh, PortMeshProtocol):
        raise TypeError(
            f"mesh must satisfy PortMeshProtocol, got {type(mesh).__name__}"
        )
    bkd = mesh.bkd()
    # Each node once, in order of first appearance.
    ports = list(dict.fromkeys(node for side in sides for node in side.nodes))
    if not ports:
        raise ValueError("a collocation response needs at least one port node")
    ids = [side.boundary_id for side in sides]
    if len(set(ids)) != len(ids):
        raise ValueError(f"a side appears more than once among {ids}")
    for side in sides:
        if len(set(side.nodes)) != len(side.nodes):
            raise ValueError(f"side {side.boundary_id} repeats a node")
    column = {node: k for k, node in enumerate(ports)}
    parts, points, weights, blocks = [], [], [], []
    for side in sides:
        indices = mesh.boundary_indices(side.boundary_id)
        position = {bkd.to_int(indices[k]): k for k in range(indices.shape[0])}
        outside = sorted(set(side.nodes) - position.keys())
        if outside:
            raise ValueError(f"nodes {outside} are not on side {side.boundary_id}")
        quadrature = boundary_side_quadrature(mesh, side.boundary_id, rule)
        operator = FluxNormalOperator(
            bkd, indices, mesh.boundary_normals(side.boundary_id), flux
        )
        parts.append(NormalFluxPart(operator, quadrature.interpolation))
        points.append(quadrature.points)
        weights.append(quadrature.weights)
        # The side's port nodes' Lagrange polynomials at the rule's points.
        selected = bkd.asarray(
            [position[node] for node in side.nodes], dtype=bkd.int64_dtype()
        )
        blocks.append(quadrature.interpolation[:, selected])
    trace = TabulatedTrace(
        bkd.concatenate(points, axis=1),
        bkd.concatenate(weights, axis=0),
        _placed(
            blocks,
            [[column[node] for node in side.nodes] for side in sides],
            len(ports),
            bkd,
        ),
        bkd,
    )
    moments = NormalFluxMoments(parts, trace, 0.0 if time is None else time)
    states = backend_state_space(bkd, mesh.npts())
    theta = BackendVectorSpace(
        SpaceDescriptor("params", len(ports), backend_dtype_name(bkd)), bkd
    )
    data: PortPinnedData[Array, Array] = PortPinnedData(
        BackendRowSet(states.descriptor(), ports, bkd),
        BackendPortRows(states.descriptor(), ports, bkd),
        BackendPortRows(theta.descriptor(), list(range(len(ports))), bkd),
        states,
        theta,
    )
    return SteadyPinnedResponse(
        collocation_steady_residual(system, time=time),
        data,
        moments,
        states,
        setups,
        nonlinear,
        bkd,
    )
