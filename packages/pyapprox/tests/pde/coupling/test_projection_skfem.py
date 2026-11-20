r"""Projection port transfers coupling skfem subdomains whose meshes do
not match each other or the skeleton. NumPy only: skfem is.

The unit square is cut at x = 1/2; each half has its own uniform Q1 mesh,
with different element counts along the interface, so only the interface
ends coincide. :math:`-\Delta u = 0` with u = g on the left and right
sides and exact Neumann data :math:`\partial u / \partial n` of g on the
top and bottom, g linear. The interface ends lie on the Neumann sides, so
every mesh node on x = 1/2 is a port node and the skeleton has no pinned
DOF.

- Patch test: a degree-4 continuous skeleton. Projection
  (:math:`L = M_p^{-1} B` on each side's linear hat trace) tests the
  returned fluxes against the skeleton functions exactly, so the linear
  solution is reproduced (1e-10).
- Control: interpolation on the same meshes tests against each mesh's
  interpolant of the skeleton functions instead and misses (> 1e-3).
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Callable, Sequence

import numpy as np

from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.implicitfunction.steady_system import SteadySystem
from pyapprox.pde.assembly.scipy_stack import ScipyAssemblyStack
from pyapprox.pde.coupling.basis import lobatto_lagrange_basis
from pyapprox.pde.coupling.closure_numbering import VertexWeightNumbering
from pyapprox.pde.coupling.geometry import TensorBoxGeometry
from pyapprox.pde.coupling.ports import (
    AverageIncidentFaces,
    BackendPortRows,
    PortAssemblyProtocol,
    PortMapProtocol,
    PortMatrixTransfer,
    PortTransferProtocol,
    port_interpolation,
    port_projection,
    port_trace_coupling,
    port_trace_mass,
    subdomain_port_dofs,
)
from pyapprox.pde.coupling.skeleton import CoupledSubdomain
from pyapprox.pde.coupling.spaces import ContinuousSkeletonSpace, UniformBases
from pyapprox.pde.coupling.topology import TensorBoxPartition
from pyapprox.pde.coupling.traces import LinearSegmentTrace
from pyapprox.pde.galerkin.time_integration import skfem_discretization
from pyapprox.pde.models.galerkin.steady_residual import galerkin_steady_residual
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.scipy_stack import CSRFormat, ScipySparseLU
from tests._helpers.coupling_skfem import (
    Np,
    Piece,
    TensorQuadMesh,
    max_state_error,
    newton,
    pinned_piece,
    poisson_system,
    solve_dense_skeleton,
)

_TOL = 1e-12
_LocalMap = Callable[
    [
        ContinuousSkeletonSpace[Np],
        TensorBoxGeometry[Np],
        Sequence[int],
        Sequence[int],
        Np,
    ],
    PortMapProtocol[Np],
]
# The outward normal derivative of g on the top and bottom.
_NEUMANN = [("top", -3.0), ("bottom", 3.0)]


def _exact(x: Np) -> Np:
    return np.asarray(1.0 + 2.0 * x[0] - 3.0 * x[1])


def test_single_domain_reproduces_the_linear_solution() -> None:
    """Checks the Neumann data and its sign that the coupled tests use."""
    bkd = NumpyBkd()
    mesh = TensorQuadMesh(np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 4), bkd)
    system = poisson_system(mesh, [("left", _exact), ("right", _exact)], _NEUMANN)
    disc = skfem_discretization(system)
    states = disc.space()
    solved = SteadySystem(
        galerkin_steady_residual(disc), states, RecomputeAlways(ScipySparseLU())
    ).solve(newton(), states.zeros(1))
    assert solved.converged()
    error = np.abs(solved.solution()[:, 0] - _exact(mesh.nodes())).max()
    assert error < 1e-10


def _projection(
    space: ContinuousSkeletonSpace[Np],
    geometry: TensorBoxGeometry[Np],
    faces: Sequence[int],
    dofs: Sequence[int],
    points: Np,
) -> PortMapProtocol[Np]:
    stack: PortAssemblyProtocol[Np, CSRFormat, Np, Np] = ScipyAssemblyStack(np.float64)
    chain = [int(i) for i in np.argsort(points[1])]
    trace = LinearSegmentTrace(points, chain, 3, NumpyBkd())
    mass = port_trace_mass(trace, 1, stack)
    coupling = port_trace_coupling(
        space, trace, geometry, faces, dofs, AverageIncidentFaces(), stack
    )
    return port_projection(mass, coupling, ScipySparseLU())


def _interpolation(
    space: ContinuousSkeletonSpace[Np],
    geometry: TensorBoxGeometry[Np],
    faces: Sequence[int],
    dofs: Sequence[int],
    points: Np,
) -> PortMapProtocol[Np]:
    return port_interpolation(
        space, geometry.locate(points, faces), dofs, AverageIncidentFaces()
    )


def _solve(counts: Sequence[tuple[int, int]], local_map: _LocalMap) -> float:
    """Coupled solve with half ``s`` meshed by ``counts[s]`` elements and
    each port's local map from ``local_map``; the max nodal error."""
    bkd = NumpyBkd()
    topo = TensorBoxPartition((2, 1))
    geometry = TensorBoxGeometry(topo, bkd)
    bases = UniformBases(lobatto_lagrange_basis(1, 5, bkd))
    space = ContinuousSkeletonSpace(topo, bases, VertexWeightNumbering(topo, bases))
    descriptor = space.descriptor()
    edges = np.linspace(0.0, 1.0, 3)
    pieces: list[tuple[Piece, PortTransferProtocol[Np, Np], Np]] = []
    for s in range(topo.nsubdomains()):
        nx, ny = counts[s]
        mesh = TensorQuadMesh(
            np.linspace(edges[s], edges[s + 1], nx + 1),
            np.linspace(0.0, 1.0, ny + 1),
            bkd,
        )
        nodes = mesh.nodes()
        ports = [int(n) for n in np.where(np.abs(nodes[0] - 0.5) < _TOL)[0]]
        outer = "left" if s == 0 else "right"
        system = poisson_system(mesh, [(outer, _exact)], _NEUMANN)
        faces = [f for f in topo.subdomain_entities(s) if f in space.faces()]
        dofs = subdomain_port_dofs(topo, space, s)
        local = local_map(space, geometry, faces, dofs, nodes[:, ports])
        transfer = PortMatrixTransfer(BackendPortRows(descriptor, dofs, bkd), local)
        pieces.append((pinned_piece(system, ports, bkd), transfer, nodes))
    lam = solve_dense_skeleton(
        descriptor, [CoupledSubdomain(piece, t) for piece, t, _ in pieces]
    )
    return max_state_error(pieces, lam, _exact)


_COUNTS = [(3, 4), (4, 7)]


def test_projection_reproduces_the_linear_solution() -> None:
    assert _solve(_COUNTS, _projection) < 1e-10


def test_interpolation_misses_it_on_the_same_meshes() -> None:
    assert _solve(_COUNTS, _interpolation) > 1e-3
