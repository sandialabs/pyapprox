r"""Skfem subdomains whose meshes do not match each other or the skeleton,
coupled through interpolation port transfers. NumPy only: skfem is.

The unit square is cut 2 x 2; each subdomain has its own uniform Q1 mesh
(different element counts, so no interface node of one side lies on the
other's, except at the ends of faces). :math:`-\Delta u = 0` with
:math:`u = g` on the whole outer boundary, g a harmonic function, so
:math:`u = g` everywhere. Port nodes are each mesh's interface nodes off
the outer boundary; nodes on it keep their own Dirichlet rows, and the
skeleton DOFs there are pinned to g.

- Patch test: g linear and a degree-1 continuous skeleton. Each mesh
  represents the skeleton basis exactly, so the gathered fluxes
  :math:`E_i^\top q_i` of the two sides test the same functions and
  the linear solution is reproduced (1e-10).
- Convergence: g = e^x sin(y), a degree-4 skeleton; the meshes refine
  by 2 and 4, the nodal error falls at order >= 1.5.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import math
from typing import Any, Callable, Sequence

import numpy as np

from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.implicitfunction.steady_system import SteadySystem
from pyapprox.pde.constitutive.coefficient_functions import (
    CallableReaction,
    ConstantDiffusion,
    TimeIndependent,
)
from pyapprox.pde.coupling.basis import lobatto_lagrange_basis
from pyapprox.pde.coupling.closure_numbering import VertexWeightNumbering
from pyapprox.pde.coupling.geometry import TensorBoxGeometry, skeleton_dof_points
from pyapprox.pde.coupling.pinned_ports import PortPinnedData
from pyapprox.pde.coupling.pinned_response import (
    SteadyPinnedPoint,
    SteadyPinnedResponse,
)
from pyapprox.pde.coupling.pinning import PinnedSkeletonResidual
from pyapprox.pde.coupling.ports import (
    AverageIncidentFaces,
    BackendPortRows,
    InterpolationTransfer,
    port_interpolation,
    subdomain_port_dofs,
)
from pyapprox.pde.coupling.skeleton import (
    CoupledSubdomain,
    DenseSchur,
    SkeletonResidual,
)
from pyapprox.pde.coupling.spaces import ContinuousSkeletonSpace, UniformBases
from pyapprox.pde.coupling.topology import TensorBoxPartition
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.time_integration import skfem_discretization
from pyapprox.pde.models.galerkin.steady_residual import galerkin_steady_residual
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.backend_dense import (
    BackendDenseLU,
    BackendRowSet,
    BackendVectorSpace,
    DenseFormat,
)
from pyapprox.util.linalg.operators import InnerProductSpaceProtocol
from pyapprox.util.linalg.scipy_stack import NumpyRowSet, ScipySparseLU
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.coupling_skfem import STEADY, Np, TensorQuadMesh

_Exact = Callable[[Np], Np]
_Piece = SteadyPinnedResponse[Np, Np, Any]
_TOL = 1e-12


def _linear(x: Np) -> Np:
    return np.asarray(1.0 + 2.0 * x[0] - 3.0 * x[1])


def _smooth(x: Np) -> Np:
    return np.asarray(np.exp(x[0]) * np.sin(x[1]))


def _zero(x: Np) -> Np:
    return np.zeros(x.shape[1])


def _no_sink(x: Np, u: Np) -> Np:
    return np.zeros_like(u)


def _system(mesh: TensorQuadMesh, sides: Sequence[str], exact: _Exact) -> Any:
    """:math:`-\\Delta u = 0` with u = ``exact`` on ``sides``."""
    bkd = mesh.bkd()
    basis = LagrangeBasis(mesh, degree=1)
    physics = AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=ConstantDiffusion(1.0),
        bkd=bkd,
        reaction=CallableReaction(_no_sink, _no_sink),
        forcing=TimeIndependent(_zero),
    )
    conditions = [
        DirichletBC(basis, side, TimeIndependent(exact), bkd) for side in sides
    ]
    return compose_galerkin_system(physics, conditions)


def _on_square_boundary(x: Np) -> Np:
    return np.asarray(
        (np.abs(x[0]) < _TOL)
        | (np.abs(x[0] - 1.0) < _TOL)
        | (np.abs(x[1]) < _TOL)
        | (np.abs(x[1] - 1.0) < _TOL)
    )


def _solve(
    counts: Sequence[tuple[int, int]], npoints: int, exact: _Exact
) -> tuple[float, float]:
    """Coupled solve on the 2 x 2 partition with subdomain ``s`` meshed by
    ``counts[s]`` elements; the max nodal error over all subdomains, and
    over the outer-boundary nodes alone."""
    bkd = NumpyBkd()
    topo = TensorBoxPartition((2, 2))
    geometry = TensorBoxGeometry(topo, bkd)
    bases = UniformBases(lobatto_lagrange_basis(1, npoints, bkd))
    space = ContinuousSkeletonSpace(topo, bases, VertexWeightNumbering(topo, bases))
    descriptor = space.descriptor()
    edges = np.linspace(0.0, 1.0, 3)
    pieces: list[tuple[_Piece, InterpolationTransfer[Np, Np], Np]] = []
    subdomains: list[CoupledSubdomain[Np, Np]] = []
    for s in range(topo.nsubdomains()):
        ix, iy = divmod(s, 2)
        nx, ny = counts[s]
        mesh = TensorQuadMesh(
            np.linspace(edges[ix], edges[ix + 1], nx + 1),
            np.linspace(edges[iy], edges[iy + 1], ny + 1),
            bkd,
        )
        sides = [
            side
            for side, touched in (
                ("left", ix == 0),
                ("right", ix == 1),
                ("bottom", iy == 0),
                ("top", iy == 1),
            )
            if touched
        ]
        nodes = mesh.nodes()
        on_cell = np.zeros(nodes.shape[1], dtype=bool)
        for side in ("left", "right", "bottom", "top"):
            on_cell[mesh.boundary_nodes(side)] = True
        ports = [int(n) for n in np.where(on_cell & ~_on_square_boundary(nodes))[0]]
        disc = skfem_discretization(_system(mesh, sides, exact))
        states, theta = disc.space(), disc.parameter_space(len(ports))
        assert isinstance(states, InnerProductSpaceProtocol)
        data: PortPinnedData[Np, Np] = PortPinnedData(
            NumpyRowSet(states.descriptor(), ports),
            BackendPortRows(states.descriptor(), ports, bkd),
            BackendPortRows(theta.descriptor(), list(range(len(ports))), bkd),
            states,
            theta,
        )
        piece: SteadyPinnedResponse[Np, Np, Any] = SteadyPinnedResponse(
            galerkin_steady_residual(disc),
            data,
            states,
            RecomputeAlways(ScipySparseLU()),
            Newton(atol=1e-13, rtol=0.0, maxiters=30),
            bkd,
        )
        faces = [f for f in topo.subdomain_entities(s) if f in space.faces()]
        dofs = subdomain_port_dofs(topo, space, s)
        matrix = port_interpolation(
            space,
            geometry.locate(nodes[:, ports], faces),
            dofs,
            AverageIncidentFaces(),
        )
        transfer = InterpolationTransfer(BackendPortRows(descriptor, dofs, bkd), matrix)
        pieces.append((piece, transfer, nodes))
        subdomains.append(CoupledSubdomain(piece, transfer))

    coords = skeleton_dof_points(space, geometry, bases)
    pinned = [
        d
        for e in range(topo.nentities())
        if topo.on_boundary(e)
        for d in space.entity_dofs(e)
    ]
    values: Np = np.zeros((descriptor.size, 1))
    values[pinned, 0] = exact(coords[:, pinned])
    vectors = BackendVectorSpace(descriptor, bkd)
    skeleton: SkeletonResidual[Np, Np, Np, DenseFormat[Np]] = SkeletonResidual(
        vectors, subdomains, DenseSchur(bkd)
    )
    residual: PinnedSkeletonResidual[Np, DenseFormat[Np]] = PinnedSkeletonResidual(
        skeleton,
        vectors,
        BackendRowSet(descriptor, pinned, bkd),
        values,
    )
    lu: RecomputeAlways[Np, DenseFormat[Np]] = RecomputeAlways(BackendDenseLU(bkd))
    outer: Newton[Np, Np] = Newton(atol=1e-11, rtol=1e-11, maxiters=20)
    solved = SteadySystem(residual, vectors, lu).solve(outer, vectors.zeros(1))
    assert solved.converged()
    lam = solved.solution()

    error, boundary_error = 0.0, 0.0
    for piece, transfer, nodes in pieces:
        point = piece.linearize(transfer.scatter(lam), bkd.zeros((0, 1)), STEADY)
        assert isinstance(point, SteadyPinnedPoint)
        diff = np.abs(point.state()[:, 0] - exact(nodes))
        error = max(error, float(diff.max()))
        boundary_error = max(
            boundary_error, float(diff[_on_square_boundary(nodes)].max())
        )
    return error, boundary_error


_COUNTS = [(3, 4), (5, 2), (4, 6), (2, 3)]


def test_patch_test_with_a_linear_skeleton() -> None:
    error, boundary_error = _solve(_COUNTS, 2, _linear)
    assert error < 1e-10
    assert boundary_error < 1e-13


def test_converges_under_refinement() -> None:
    base = [(4, 5), (6, 4), (5, 6), (4, 4)]
    errors = [
        _solve([(m * nx, m * ny) for nx, ny in base], 5, _smooth)[0] for m in (1, 2, 4)
    ]
    assert errors[0] > errors[1] > errors[2]
    assert math.log2(errors[1] / errors[2]) >= 1.5
