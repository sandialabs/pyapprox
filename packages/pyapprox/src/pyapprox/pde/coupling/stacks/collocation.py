r"""A spectral collocation subdomain as a Dirichlet-to-Neumann response.

The subdomain's port nodes lie on sides of its mesh. Its rows are the PDE
evaluated pointwise, so the residual left on a pinned port row is not a
flux (:mod:`~pyapprox.pde.coupling.normal_flux`); the response instead
returns the outward normal flux :math:`\mathbf{F}(u)\cdot n` at the port
nodes, each weighted by its side's collocation quadrature weight
(:func:`~pyapprox.pde.collocation.quadrature.boundary_quadrature.
boundary_quadrature_weights`): the flux tested against lumped nodal trace
functions. Summed over the subdomains sharing an interface node, the
weighted fluxes vanish exactly when the pointwise fluxes balance there.

Scalar fluxes only. A port node on two port sides (a cross point, where
the normal is ambiguous) is refused.
"""

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, runtime_checkable

from pyapprox.implicitfunction.setups import SetupStrategyProtocol, SteadyRequest
from pyapprox.ode.residual_derivatives import backend_state_space
from pyapprox.pde.collocation.boundary.normal_operators import FluxNormalOperator
from pyapprox.pde.collocation.protocols.boundary import FluxProviderProtocol
from pyapprox.pde.collocation.protocols.mesh import MeshWithTransformProtocol
from pyapprox.pde.collocation.quadrature.boundary_quadrature import (
    boundary_quadrature_weights,
)
from pyapprox.pde.coupling.normal_flux import NormalFluxMoments, NormalFluxPart
from pyapprox.pde.coupling.pinned_ports import PortPinnedData
from pyapprox.pde.coupling.pinned_response import SteadyPinnedResponse
from pyapprox.pde.coupling.ports import BackendPortRows
from pyapprox.pde.coupling.traces import LumpedNodalTrace
from pyapprox.pde.models.collocation.closed_system import CollocationSystem
from pyapprox.pde.models.collocation.steady_residual import (
    collocation_steady_residual,
)
from pyapprox.util.backends.protocols import Array
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


def collocation_pinned_response(
    system: CollocationSystem[Array],
    flux: FluxProviderProtocol[Array],
    mesh: PortMeshProtocol[Array],
    sides: Sequence[PortSide],
    setups: SetupStrategyProtocol[
        SteadyRequest[Array, DenseFormat[Array]], Array, DenseFormat[Array]
    ],
    nonlinear: NonlinearSolverProtocol[Array, Array],
    time: Optional[float] = None,
) -> SteadyPinnedResponse[Array, Array, DenseFormat[Array]]:
    r"""A collocation subdomain pinned to the port data at the nodes of
    ``sides``, returning the outward flux there, tested against the
    port's lumped trace functions.

    The ports are the sides' nodes in order, the first side's first.

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
        If there are no port nodes, a node is not on its side, or a node
        is on two port sides.
    """
    if not isinstance(mesh, PortMeshProtocol):
        raise TypeError(
            f"mesh must satisfy PortMeshProtocol, got {type(mesh).__name__}"
        )
    bkd = mesh.bkd()
    ports = [node for side in sides for node in side.nodes]
    if not ports:
        raise ValueError("a collocation response needs at least one port node")
    repeated = sorted({node for node in ports if ports.count(node) > 1})
    if repeated:
        raise ValueError(
            f"nodes {repeated} lie on more than one port side: a cross point "
            "has no single normal"
        )
    parts, weights = [], []
    for side in sides:
        indices = mesh.boundary_indices(side.boundary_id)
        position = {bkd.to_int(indices[k]): k for k in range(indices.shape[0])}
        outside = sorted(set(side.nodes) - position.keys())
        if outside:
            raise ValueError(f"nodes {outside} are not on side {side.boundary_id}")
        selected = [position[node] for node in side.nodes]
        operator = FluxNormalOperator(
            bkd, indices, mesh.boundary_normals(side.boundary_id), flux
        )
        parts.append(NormalFluxPart(operator, selected))
        rows = bkd.asarray(selected, dtype=bkd.int64_dtype())
        weights.append(boundary_quadrature_weights(mesh, side.boundary_id)[rows])
    trace = LumpedNodalTrace(
        mesh.points()[:, ports], bkd.concatenate(weights, axis=0), bkd
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
